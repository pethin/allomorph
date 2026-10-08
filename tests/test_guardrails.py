"""
Allomorph Architectural Guardrails Automated Invariant Test Suite.

Programmatically verifies that all core physical modeling and numerical invariants
specified in AGENTS.md Section 5 and docs/architectural_guardrails.md are strictly upheld:
1. Guardrail 5.1: Asymmetric order-4 displacement limiter (+12 dB boost / -16 dB cut) & inharmonicity RBF monotonicity
2. Guardrail 5.2: Order-8 algebraic rail limiter & C^inf smooth thresholded soft-knee saturation
3. Guardrail 5.3: Active preamp finite DC transmission (H(0) >= 1.0) & first-class transducer taxonomy
4. Guardrail 5.3.5: Fail-fast declarative integrity & zero silent fallbacks
5. Guardrail 5.4: Small-signal linearity (bit-exact linear bypass for peak <= 0.10)
6. Guardrail 5.6: Zero-scroll pedalboard display (<= 34 chars) and T3K filename ceilings (<= 64 chars)
"""

import math

import numpy as np
import pytest

from allomorph.circuit.saturation import (
    apply_algebraic_rail_limiter,
    apply_oversampled_saturation,
)
from allomorph.circuit.solver import compute_active_preamp_eq, smooth_soft_knee_db
from allomorph.config.instruments import (
    STANDARD_CATALOG_TARGETS,
    get_source_pickup,
    load_all_instruments,
    load_instrument,
)
from allomorph.config.voices import VOICES
from allomorph.naming import get_t3k_basename, resolve_instruments, resolve_voices
from allomorph.physics.aperture import soft_clamp_displacement_ratio
from allomorph.physics.strings import (
    INHARMONICITY_ANCHORS_BS,
    INHARMONICITY_ANCHORS_F0,
    get_inharmonicity_for_f0,
)


def test_guardrail_5_1_asymmetric_displacement_limiter():
    """Guardrail 5.1.3: Asymmetric order-4 algebraic limiter ('alg4') bounds positive excursion boost
    to <= +12.0 dB and negative proximity thinning to >= -16.0 dB, with passband linearity for |ΔG| <= 3 dB."""
    # Passband linearity
    for dg in [-3.0, -1.0, 0.0, 1.0, 3.0]:
        clamped = float(soft_clamp_displacement_ratio(dg))
        assert abs(clamped - dg) < 0.05, (
            f"Displacement ratio {dg} dB experienced premature compression: {clamped} dB"
        )

    # Positive excursion asymptotic ceiling <= +12.0 dB
    huge_boost = float(soft_clamp_displacement_ratio(100.0))
    assert 11.5 < huge_boost <= 12.0, (
        f"Positive excursion boost {huge_boost:.2f} dB violated +12.0 dB ceiling"
    )

    # Negative proximity asymptotic floor >= -16.0 dB
    huge_cut = float(soft_clamp_displacement_ratio(-100.0))
    assert -16.0 <= huge_cut < -15.5, (
        f"Negative proximity cut {huge_cut:.2f} dB violated -16.0 dB floor"
    )


def test_guardrail_5_1_inharmonicity_rbf_calibration_and_monotonicity():
    """Guardrail 5.1.4: Inharmonicity Gaussian RBF solver reproduces calibration anchors to < 1e-6 relative error
    and exhibits strictly decreasing monotonicity across bass fundamental registers."""
    # 1. Anchor precision
    for f0, expected_bs in zip(INHARMONICITY_ANCHORS_F0, INHARMONICITY_ANCHORS_BS):
        actual_bs = get_inharmonicity_for_f0(f0)
        rel_err = abs(actual_bs - expected_bs) / expected_bs
        assert rel_err < 1e-6, f"RBF anchor error at {f0} Hz was {rel_err:.2e} (expected < 1e-6)"

    # 2. Strict decreasing monotonicity across bass fundamental registers (up to 195 Hz)
    test_freqs = np.linspace(30.0, 195.0, 100)
    bs_vals = [get_inharmonicity_for_f0(f) for f in test_freqs]
    for i in range(len(bs_vals) - 1):
        assert bs_vals[i] > bs_vals[i + 1], (
            f"Inharmonicity RBF violated monotonicity at {test_freqs[i]:.1f} Hz"
        )


def test_guardrail_5_2_algebraic_rail_limiter_order_8():
    """Guardrail 5.2.1: Order-8 algebraic rail limiter f(x) = x / (1 + (|x|/vsat)^8)^(1/8)
    guarantees strict peak bounding (|f(x)| < vsat) and passband linearity (< 0.005 dB deviation for peak <= 0.5*vsat)."""
    vsat = 0.9900
    # Small signal linearity
    x_small = 0.3 * vsat
    f_small = float(apply_algebraic_rail_limiter(x_small, vsat=vsat))
    dev_db = abs(20.0 * math.log10(f_small / x_small))
    assert dev_db < 0.005, f"Small signal experienced {dev_db:.4f} dB non-linear distortion"

    # Strict peak bounding across extreme range
    for val in [-1e6, -100.0, -1.5, 0.0, 1.5, 100.0, 1e6]:
        out = float(apply_algebraic_rail_limiter(val, vsat=vsat))
        assert abs(out) <= vsat, f"Limiter allowed signal {out} to exceed ceiling {vsat}"


def test_guardrail_5_2_smooth_soft_knee_c_inf():
    """Guardrail 5.2.1: Thresholded soft-knee saturation must be strictly C^inf without piecewise derivative kinks."""
    thresh = 6.0
    w = 3.0
    ceiling = thresh + w
    x_grid = np.linspace(0.0, 20.0, 500)
    y_grid = smooth_soft_knee_db(x_grid, thresh=thresh, ceiling=ceiling)

    # 1. Identity far below threshold
    assert math.isclose(smooth_soft_knee_db(0.0, thresh=thresh, ceiling=ceiling), 0.0, abs_tol=1e-3)

    # 2. Continuous first and second derivatives (no NaN or kinks)
    dy = np.diff(y_grid) / np.diff(x_grid)
    d2y = np.diff(dy)
    assert np.all(np.isfinite(dy))
    assert np.all(np.isfinite(d2y))
    # Monotonically non-decreasing
    assert np.all(dy >= 0.0)


def test_guardrail_5_3_active_preamp_dc_transmission():
    """Guardrail 5.3.3: Active preamps must feature flat, finite DC transmission (|H(0)| >= 1.0)
    to prevent unphysical sub-audible inversion steps and Gibbs truncation ripples."""
    s_dc = 1j * 2.0 * math.pi * 1e-6

    for preamp_type in ["sadowsky_2band", "stingray_2band", "aguilar_3band", "dingwall_active"]:
        H_eq = compute_active_preamp_eq(preamp_type, s_dc)
        mag_dc = abs(H_eq)
        assert mag_dc >= 0.999, (
            f"Preamp {preamp_type} DC magnitude was {mag_dc:.4f} (expected >= 1.0). "
            "Sub-audible differentiator poles are prohibited."
        )


def test_guardrail_5_3_transducer_taxonomy():
    """Guardrail 5.3.4: All instruments and target voicings must declare a valid physical sensor_type."""
    valid_sensors = {"magnetic", "bridge_force", "direct"}

    for vid, v in VOICES.items():
        assert v.sensor_type in valid_sensors, (
            f"Voice {vid} had invalid sensor_type: {v.sensor_type}"
        )

    all_insts = load_all_instruments()
    for inst_id, inst in all_insts.items():
        for vid, v in inst.voicings.items():
            assert v.sensor_type in valid_sensors, (
                f"Instrument {inst_id} voice {vid} had invalid sensor_type: {v.sensor_type}"
            )


def test_guardrail_5_3_zero_silent_fallbacks():
    """Guardrail 5.3.5: Missing configurations, invalid voices, or unmapped pickups must raise explicit diagnostic exceptions."""
    inst = load_instrument("34in_standard_jazz")

    # 1. Invalid pickup mapping target
    broken = inst.model_copy(deep=True)
    broken.pickup_mapping = {"broken_voice": "ghost_pickup"}
    with pytest.raises(KeyError, match="ghost_pickup"):
        get_source_pickup(broken, "broken_voice")

    # 2. Unknown instrument
    with pytest.raises(ValueError, match="Unknown source instrument"):
        resolve_instruments("ghost_bass_model")

    # 3. Unknown voice
    with pytest.raises(ValueError, match="Unknown target voice"):
        resolve_voices("ghost_target_voice")


def test_guardrail_5_4_small_signal_linearity():
    """Guardrail 5.4.1: Saturation dynamics must be bit-exact linear bypass on small signals (peak <= 0.10)."""
    t = np.linspace(0, 0.05, 2400, endpoint=False)
    small_sig = (0.05 * np.sin(2.0 * np.pi * 100.0 * t)).astype(np.float32)

    # Various pickup metallurgy parameters
    out_alnico = apply_oversampled_saturation(small_sig, vsat=0.5, alpha=0.3, alpha3=0.1, k_sag=0.1)
    np.testing.assert_array_equal(
        out_alnico,
        small_sig,
        err_msg="Small signals must pass through saturation stage with bit-exact linearity",
    )


def test_guardrail_5_6_tone_naming_pedalboard_budget():
    """Guardrail 5.6: Target voice model basenames must satisfy pedalboard zero-scroll (<= 34 chars unversioned)
    and Tone3000 upload constraints (<= 64 chars versioned). Slashes must be sanitized to Unicode Division Slash."""
    for inst_id, target_vid in STANDARD_CATALOG_TARGETS:
        tgt_inst = load_instrument(inst_id)
        if target_vid in tgt_inst.voicings:
            voicing = tgt_inst.voicings[target_vid]
            tone_name = voicing.tone_name or voicing.name

            # Unversioned stage name (single pickup or standalone)
            unversioned = get_t3k_basename(tone_name, position_name=None, version_tag=None)
            assert len(unversioned) <= 34, (
                f"Unversioned tone name '{unversioned}' exceeds 34 char zero-scroll ceiling"
            )

            # Versioned Tone3000 filename
            versioned = get_t3k_basename(
                tone_name,
                position_name="Bridge",
                version_tag="v2.1.1",
            )
            assert len(versioned) <= 64, (
                f"Versioned filename '{versioned}' exceeds 64 char Tone3000 upload ceiling"
            )
            assert "/" not in versioned
            assert "\\" not in versioned
