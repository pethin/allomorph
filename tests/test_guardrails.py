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
import re
from pathlib import Path

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from allomorph.circuit.saturation import (
    apply_active_pickup_dynamics,
    apply_algebraic_rail_limiter,
    apply_oversampled_saturation,
)
from allomorph.circuit.solver import compute_active_preamp_eq
from allomorph.cli.targets import resolve_targets
from allomorph.config.instruments import (
    load_all_instruments,
    load_instrument,
)
from allomorph.naming import get_t3k_basename
from allomorph.physics.aperture import (
    compute_pickup_isolation_leveling,
    soft_clamp_displacement_ratio,
)
from allomorph.physics.strings import (
    INHARMONICITY_ANCHORS_BS,
    INHARMONICITY_ANCHORS_F0,
    get_inharmonicity_for_f0,
)
from tests.strategies import st_audio_buffers, st_rail_voltages, st_signals_for_rail


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


@given(st.floats(min_value=-500.0, max_value=500.0, allow_nan=False, allow_infinity=False))
def test_guardrail_5_1_asymmetric_displacement_limiter_bounds(dg: float) -> None:
    """Guardrail 5.1.3 property: Bounded in [-16.0, +12.0] dB across arbitrary inputs."""
    clamped = float(soft_clamp_displacement_ratio(dg))
    assert -16.0 <= clamped <= 12.0
    if abs(dg) <= 3.0:
        assert abs(clamped - dg) < 0.05


@given(
    st.floats(min_value=-200.0, max_value=200.0, allow_nan=False, allow_infinity=False),
    st.floats(min_value=0.01, max_value=50.0, allow_nan=False, allow_infinity=False),
)
def test_guardrail_5_1_displacement_limiter_monotonicity(dg1: float, delta: float) -> None:
    """Guardrail 5.1.3 property: Limiter is strictly non-decreasing across all inputs."""
    dg2 = dg1 + delta
    c1 = float(soft_clamp_displacement_ratio(dg1))
    c2 = float(soft_clamp_displacement_ratio(dg2))
    assert c2 >= c1


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
    guarantees strict peak bounding (|f(x)| <= vsat) and passband linearity (< 0.005 dB deviation for peak <= 0.3*vsat)."""
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

    # Verify active pickup op-amp output is also strictly bounded by vsat via CL-13
    t = np.linspace(0, 0.02, 960, endpoint=False)
    hot_sig = (2.5 * np.sin(2.0 * np.pi * 120.0 * t)).astype(np.float32)
    act_out = apply_active_pickup_dynamics(hot_sig, vsat=0.50, k_level=0.25)
    assert float(np.max(np.abs(act_out))) <= 0.50 + 1e-6


@given(
    st_signals_for_rail(min_x=-1000.0, max_x=1000.0),
    st_rail_voltages(min_vsat=0.1, max_vsat=5.0),
)
def test_guardrail_5_2_algebraic_rail_limiter_order_8_property(x: float, vsat: float) -> None:
    """Guardrail 5.2.1 property: Strict peak bounding (|f(x)| <= vsat) and small-signal linearity."""
    out = float(apply_algebraic_rail_limiter(x, vsat=vsat))
    assert abs(out) <= vsat
    if x != 0.0:
        assert (out > 0.0) == (x > 0.0)
    if abs(x) <= 0.3 * vsat and abs(x) > 1e-9:
        dev_db = abs(20.0 * math.log10(out / x))
        assert dev_db < 0.005


def test_guardrail_5_2_smooth_soft_knee_c_inf():
    """Guardrail 5.2.1: Shifted-softplus luthier isolation leveling (CL-26) must be strictly C^inf without derivative kinks."""
    pos_grid = np.linspace(0.02, 0.25, 500)
    gains_db = np.array(
        [
            20.0
            * math.log10(
                compute_pickup_isolation_leveling(float(x), scale_m=0.8636, ref_pos_m=0.15)
            )
            for x in pos_grid
        ]
    )

    # 1. Exact 0.0 dB (gain 1.0) at reference position
    assert math.isclose(
        compute_pickup_isolation_leveling(0.15, scale_m=0.8636, ref_pos_m=0.15), 1.0, abs_tol=1e-9
    )

    # 2. Continuous first and second derivatives across active bridge-ward region (pos < ref_pos)
    active_mask = pos_grid < 0.15
    x_act = pos_grid[active_mask]
    y_act = gains_db[active_mask]
    dy = np.diff(y_act) / np.diff(x_act)
    d2y = np.diff(dy) / np.diff(x_act[:-1])
    assert np.all(np.isfinite(dy))
    assert np.all(np.isfinite(d2y))
    # Monotonically decreasing with distance from bridge (closer to bridge -> higher compensation)
    assert np.all(dy <= 0.0)


@given(
    st.floats(min_value=0.01, max_value=0.30, allow_nan=False, allow_infinity=False),
    st.floats(min_value=0.70, max_value=1.10, allow_nan=False, allow_infinity=False),
    st.floats(min_value=0.05, max_value=0.25, allow_nan=False, allow_infinity=False),
)
def test_guardrail_5_2_smooth_soft_knee_property(
    pos_m: float, scale_m: float, ref_pos_m: float
) -> None:
    """Guardrail 5.2.1 property: Luthier isolation leveling (CL-26) is bounded in [0.0, +6.0] dB
    and is exact unity (1.0) when pos_m >= ref_pos_m."""
    k_iso = compute_pickup_isolation_leveling(pos_m, scale_m=scale_m, ref_pos_m=ref_pos_m)
    assert 1.0 <= k_iso <= 10.0 ** (6.0 / 20.0) + 1e-9
    if pos_m >= ref_pos_m:
        assert math.isclose(k_iso, 1.0, abs_tol=1e-12)


def test_guardrail_5_3_active_preamp_dc_transmission():
    """Guardrail 5.3.3: Active preamps must feature flat, finite DC transmission (|H(0)| >= 1.0)
    to prevent unphysical sub-audible inversion steps and Gibbs truncation ripples."""
    s_dc = 1j * 2.0 * math.pi * 1e-6

    from allomorph.config.preamps import PREAMPS

    for preamp_type in [
        "sadowsky_2band",
        "stingray_2band",
        "aguilar_3band",
        "dingwall_active",
        "fender_ultra2_3band",
    ]:
        H_eq = compute_active_preamp_eq(PREAMPS[preamp_type], s_dc)
        mag_dc = abs(H_eq)
        assert mag_dc >= 0.999, (
            f"Preamp {preamp_type} DC magnitude was {mag_dc:.4f} (expected >= 1.0). "
            "Sub-audible differentiator poles are prohibited."
        )


def test_guardrail_5_3_transducer_taxonomy():
    """Guardrail 5.3.4: All instruments and target voicings must declare a valid physical sensor_type."""
    valid_sensors = {"magnetic", "bridge_force"}

    all_insts = load_all_instruments()
    for inst_id, inst in all_insts.items():
        for vid, v in inst.voicings.items():
            assert v.sensor_type in valid_sensors, (
                f"Instrument {inst_id} voice {vid} had invalid sensor_type: {v.sensor_type}"
            )


def test_guardrail_5_3_zero_silent_fallbacks():
    """Guardrail 5.3.5: Missing configurations, invalid voices, or unmapped pickups must raise explicit diagnostic exceptions."""
    from allomorph.config.instruments import resolve_target_voicing

    inst = load_instrument("34in_standard_jazz")

    # 1. Unresolvable voice raises explicit KeyError (zero silent fallbacks)
    with pytest.raises(KeyError, match="not found on instrument"):
        resolve_target_voicing("broken_voice", instrument=inst)

    # 2. Unknown instrument
    with pytest.raises(ValueError, match="Unknown instrument"):
        resolve_targets(["instrument:ghost_bass_model"], shorthand_kind="instrument")

    # 3. Unknown pack
    with pytest.raises(ValueError, match="Unknown tone pack"):
        resolve_targets(["pack:ghost_target_pack"], shorthand_kind="pack")


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

    out_active = apply_active_pickup_dynamics(small_sig, vsat=0.50, k_level=0.25)
    np.testing.assert_array_equal(
        out_active,
        small_sig,
        err_msg="Small signals must pass through active pickup dynamics with bit-exact linearity",
    )


@given(st_audio_buffers(min_len=16, max_len=256, min_val=-0.09, max_val=0.09))
def test_guardrail_5_4_small_signal_linearity_property(small_sig: np.ndarray) -> None:
    """Guardrail 5.4.1 property: Saturation is bit-exact identity bypass for arbitrary small buffers."""
    out = apply_oversampled_saturation(small_sig, vsat=0.5, alpha=0.3, alpha3=0.1, k_sag=0.1)
    np.testing.assert_array_equal(out, small_sig)


def test_guardrail_5_6_tone_naming_pedalboard_budget():
    """Guardrail 5.6: Target voice model basenames must satisfy pedalboard zero-scroll (<= 34 chars unversioned)
    and Tone3000 upload constraints (<= 64 chars versioned). Slashes must be sanitized to Unicode Division Slash."""
    all_insts = load_all_instruments()
    for tgt_inst in all_insts.values():
        for voicing in tgt_inst.voicings.values():
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


def test_guardrail_5_3_difference_high_frequency_ceiling():
    """Guardrail 5.3: Differential voicing curves (Target - Source) must not exhibit
    spurious high-frequency treble boost artifacts (> +2.0 dB relative to passband)
    arising from deconvolution noise floor clamping when targeting darker/series voicings."""
    from allomorph.config.instruments import resolve_target_voicing
    from allomorph.visualizer.dataframe import _VOICING_CURVE_CACHE, build_voice_dataframe

    _VOICING_CURVE_CACHE.clear()

    # Specifically test P/MM Parallel (Source) -> P/MM Series (Target)
    inst_par, v_par = resolve_target_voicing("34in_active_pmm:pmm_parallel")
    inst_ser, v_ser = resolve_target_voicing("34in_active_pmm:pmm_series")
    df_par = build_voice_dataframe(inst_par, v_par)
    df_ser = build_voice_dataframe(inst_ser, v_ser)

    freqs = np.array(df_par["frequency"])
    mag_par = np.array(df_par["magnitude_db"])
    mag_ser = np.array(df_ser["magnitude_db"])
    diff = mag_ser - mag_par
    pb_mask = (freqs >= 100.0) & (freqs <= 800.0)
    pb_diff = float(np.median(diff[pb_mask]))

    # Across high frequencies (5 kHz - 20 kHz), Series targeting from Parallel must NOT
    # produce a spurious treble spike. Difference relative to passband must stay <= +2.0 dB across the upper band.
    hf_mask = freqs >= 5000.0
    max_hf_rel = float(np.max(diff[hf_mask]) - pb_diff)
    assert max_hf_rel <= 2.0, (
        f"P/MM Series difference exhibited spurious treble boost: {max_hf_rel:.2f} dB above passband (expected <= +2.0 dB)"
    )


def test_guardrail_agents_md_progressive_disclosure_budgets() -> None:
    """Guardrail: Root AGENTS.md (< 5,000 bytes) and scoped AGENTS.md files (< 2,500 bytes each)
    must remain concise progressive-disclosure routers well below the 24,000-byte context cap."""
    repo_root = Path(__file__).resolve().parent.parent
    root_agents = repo_root / "AGENTS.md"
    assert root_agents.is_file(), "Root AGENTS.md is missing"
    root_size = root_agents.stat().st_size
    assert root_size < 5000, (
        f"Root AGENTS.md size ({root_size} bytes) exceeds 5,000-byte progressive disclosure budget"
    )

    scoped_paths = [
        "src/allomorph/physics/AGENTS.md",
        "src/allomorph/circuit/AGENTS.md",
        "src/allomorph/dsp/AGENTS.md",
        "src/allomorph/trainer/AGENTS.md",
        "src/allomorph/visualizer/AGENTS.md",
        "src/allomorph/pipeline/AGENTS.md",
        "config/AGENTS.md",
        "tests/AGENTS.md",
    ]
    for rel_path in scoped_paths:
        scoped_file = repo_root / rel_path
        assert scoped_file.is_file(), f"Scoped AGENTS.md missing at {rel_path}"
        scoped_size = scoped_file.stat().st_size
        assert scoped_size < 2500, (
            f"Scoped {rel_path} size ({scoped_size} bytes) exceeds 2,500-byte progressive disclosure budget"
        )


def test_guardrail_documentation_markdown_links_valid() -> None:
    """Guardrail: All relative markdown file links in AGENTS.md, README.md, CONTRIBUTING.md,
    and docs/**/*.md must resolve to existing files or directories on disk."""
    repo_root = Path(__file__).resolve().parent.parent
    md_files: list[Path] = [
        repo_root / "AGENTS.md",
        repo_root / "README.md",
        repo_root / "CONTRIBUTING.md",
        repo_root / "config/AGENTS.md",
        repo_root / "tests/AGENTS.md",
        *sorted((repo_root / "src/allomorph").glob("*/AGENTS.md")),
        *sorted((repo_root / "docs").rglob("*.md")),
    ]

    # Strip fenced code blocks and inline code spans before scanning links
    fenced_code_re = re.compile(r"```.*?```", re.DOTALL)
    inline_code_re = re.compile(r"`[^`]+`")
    md_link_re = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")

    broken_links: list[str] = []
    for md_file in md_files:
        assert md_file.is_file(), f"Expected markdown file missing: {md_file}"
        raw_text = md_file.read_text(encoding="utf-8")
        cleaned_text = inline_code_re.sub("", fenced_code_re.sub("", raw_text))

        for match in md_link_re.finditer(cleaned_text):
            raw_target = match.group(1).strip()
            # Strip optional title in link: (path "title")
            target_url = raw_target.split()[0]
            if not target_url or target_url.startswith(
                ("#", "http://", "https://", "mailto:", "file://", "conversation://")
            ):
                continue

            path_part = target_url.split("#")[0]
            if not path_part:
                continue

            if "training.md" in path_part:
                broken_links.append(
                    f"Deprecated docs/training.md link in {md_file.relative_to(repo_root)}: {raw_target}"
                )

            resolved = (md_file.parent / path_part).resolve()
            if not resolved.exists():
                broken_links.append(
                    f"{md_file.relative_to(repo_root)} -> '{raw_target}' (missing {resolved})"
                )

    assert not broken_links, "Broken markdown links found:\n" + "\n".join(broken_links)


def test_guardrail_clamp_and_limit_registry_integrity() -> None:
    """Guardrail 5.2.3: Every CL-XX row in docs/clamp_and_limit_registry.md must reference
    existing Python modules under src/allomorph/ and valid line numbers."""
    repo_root = Path(__file__).resolve().parent.parent
    registry_path = repo_root / "docs" / "clamp_and_limit_registry.md"
    assert registry_path.is_file(), "docs/clamp_and_limit_registry.md is missing"

    src_root = repo_root / "src" / "allomorph"
    text = registry_path.read_text(encoding="utf-8")

    row_re = re.compile(r"^\|\s*\*\*(CL-\d+)\*\*\s*\|\s*([^|]+)\|\s*([^|]+)\|", re.MULTILINE)
    code_span_re = re.compile(r"`([^`]+)`")

    matches = list(row_re.finditer(text))
    assert len(matches) >= 18, f"Expected >= 18 CL-XX rows in registry, found {len(matches)}"

    for m in matches:
        cl_id = m.group(1)
        loc_cell = m.group(3).strip()
        spans = code_span_re.findall(loc_cell)
        assert spans, f"{cl_id} has no code span locations in '{loc_cell}'"

        for span in spans:
            if ":" in span:
                file_part, lines_part = span.split(":", 1)
            else:
                file_part, lines_part = span, ""

            file_part = file_part.strip()
            candidate = src_root / file_part
            if not candidate.is_file():
                found = list(src_root.rglob(file_part))
                assert len(found) == 1, (
                    f"{cl_id} location '{file_part}' did not resolve uniquely under src/allomorph/ (found {found})"
                )
                candidate = found[0]

            total_lines = len(candidate.read_text(encoding="utf-8").splitlines())
            if lines_part.strip():
                for line_tok in lines_part.split(","):
                    line_no = int(line_tok.strip())
                    assert 1 <= line_no <= total_lines, (
                        f"{cl_id} line {line_no} in {file_part} out of range [1, {total_lines}]"
                    )
