"""
Property-based tests for forward digital twin simulation in allomorph.circuit.forward.
Verifies mathematical invariants:
1. Length invariance: output buffer length strictly matches input buffer length.
2. Zero latency: causal minimum-phase FIR onset starts strictly at sample 0.
3. True-peak ceiling: normalized output true peak never exceeds CALIBRATION_PEAK_CEILING.
4. Spatial blend DC normalization: spatial aperture ratio filter evaluates to exact 1.0 at DC.
5. Transducer branch finiteness: pickup branch evaluation produces all-finite output.
"""

import numpy as np
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from allomorph.circuit.forward import (
    CALIBRATION_PEAK_CEILING,
    simulate_pickup_transducer_branch,
    simulate_voicing_dsp,
    synthesize_multi_pickup_spatial_blend,
)
from allomorph.dsp import (
    compute_true_peak,
    synthesize_minimum_phase_fir,
)
from tests.conftest import (
    make_generic_dual_pickup_instrument,
    make_generic_instrument_config,
)
from tests.strategies import st_audio_buffers


@settings(suppress_health_check=[HealthCheck.large_base_example])
@given(buf=st_audio_buffers(min_len=128, max_len=512, min_val=-0.8, max_val=0.8, dtype=np.float64))
def test_simulate_voicing_dsp_length_invariance(buf: np.ndarray) -> None:
    """Forward simulation preserves input audio length and produces finite output."""
    inst = make_generic_instrument_config()
    voicing = inst.voicings["generic_voice"]

    out = simulate_voicing_dsp(
        raw_audio=buf,
        instrument=inst,
        voicing=voicing,
        num_taps=256,
        apply_dither=False,
    )
    assert len(out) == len(buf)
    assert np.all(np.isfinite(out))


def test_simulate_voicing_dsp_zero_latency() -> None:
    """Simulated digital twin FIR begins strictly at sample 0 without artificial delay taps."""
    inst = make_generic_instrument_config()
    voicing = inst.voicings["generic_voice"]

    impulse = np.zeros(512, dtype=np.float64)
    impulse[0] = 1.0

    out = simulate_voicing_dsp(
        raw_audio=impulse,
        instrument=inst,
        voicing=voicing,
        num_taps=256,
        apply_dither=False,
        dc_block=False,
        normalize="none",
    )
    # Energy must be present at sample 0 (causal minimum phase)
    assert abs(out[0]) > 1e-4, f"Expected non-zero onset at sample 0, got {out[0]}"


@settings(suppress_health_check=[HealthCheck.large_base_example])
@given(
    buf=st_audio_buffers(min_len=256, max_len=512, min_val=-1.0, max_val=1.0, dtype=np.float64),
    scale=st.floats(min_value=0.1, max_value=5.0, allow_nan=False),
)
def test_simulate_voicing_dsp_true_peak_ceiling(buf: np.ndarray, scale: float) -> None:
    """Simulated output true peak never exceeds CALIBRATION_PEAK_CEILING under auto normalization."""
    inst = make_generic_instrument_config()
    voicing = inst.voicings["generic_voice"]

    input_audio = buf * scale
    out = simulate_voicing_dsp(
        raw_audio=input_audio,
        instrument=inst,
        voicing=voicing,
        num_taps=256,
        normalize="auto",
    )
    tp = compute_true_peak(out)
    assert tp <= CALIBRATION_PEAK_CEILING + 1e-3, (
        f"Output true peak {tp:.4f} exceeded ceiling {CALIBRATION_PEAK_CEILING}"
    )


@given(
    w1=st.floats(min_value=0.2, max_value=2.0, allow_nan=False),
    w2=st.floats(min_value=0.2, max_value=2.0, allow_nan=False),
    delta_samples=st.integers(min_value=1, max_value=30),
)
def test_spatial_blend_dc_normalization(w1: float, w2: float, delta_samples: int) -> None:
    """Universal spatial ratio filter evaluates to exact sum at DC (0.0 dB gain ratio)."""
    n_bins = 513
    f_bins = np.linspace(0.0, 24000.0, n_bins)
    H1 = np.ones(n_bins, dtype=np.complex128) * w1
    H2 = np.ones(n_bins, dtype=np.complex128) * w2
    raw_sum = np.zeros(256, dtype=np.float64)

    _composite, mag_spectrum = synthesize_multi_pickup_spatial_blend(
        f_bins=f_bins,
        H_channels=[H1, H2],
        H_channels_delayed=[H1, H2],
        delta_samples=delta_samples,
        raw_sum=raw_sum,
        num_taps=256,
    )
    expected_dc = w1 + w2
    assert np.isclose(mag_spectrum[0], expected_dc, rtol=1e-3), (
        f"Expected DC {expected_dc}, got {mag_spectrum[0]}"
    )


@settings(suppress_health_check=[HealthCheck.large_base_example])
@given(buf=st_audio_buffers(min_len=64, max_len=256, min_val=-0.9, max_val=0.9, dtype=np.float64))
def test_simulate_pickup_transducer_branch_finite(buf: np.ndarray) -> None:
    """Transducer branch simulation preserves length and produces finite numbers."""
    inst = make_generic_dual_pickup_instrument()
    voicing = inst.voicings["bridge_solo"]
    pickup = inst.pickups["bridge"]

    fir_ac = synthesize_minimum_phase_fir(np.ones(128), num_taps=128, normalize=False)
    fir_circ = synthesize_minimum_phase_fir(np.ones(128), num_taps=128, normalize=False)

    out = simulate_pickup_transducer_branch(
        input_mono=buf,
        fir_ac=fir_ac,
        fir_circ=fir_circ,
        pickup=pickup,
        voicing=voicing,
        apply_saturation=True,
    )
    assert len(out) == len(buf)
    assert np.all(np.isfinite(out))
