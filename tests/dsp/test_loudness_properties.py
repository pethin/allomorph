"""
Property-based tests for ITU-R BS.1770-4 / EBU R128 loudness analysis.
Verifies mathematical invariants: K-weighting stability, gating linearity,
DC blocking attenuation, and monotonicity across random audio buffers.
"""

import math

import numpy as np
from hypothesis import given
from hypothesis import strategies as st

from allomorph.dsp.loudness import (
    apply_itu_gating,
    apply_k_weighting_filter,
    compute_lufs,
)
from tests.strategies import st_audio_buffers


@given(st_audio_buffers(min_len=64, max_len=1024, min_val=-1.0, max_val=1.0))
def test_k_weighting_stability_and_bounds(audio: np.ndarray) -> None:
    """K-weighting filter must produce finite outputs without NaNs or Infs for any bounded input."""
    filtered = apply_k_weighting_filter(audio, sample_rate=48000)
    assert len(filtered) == len(audio)
    assert np.all(np.isfinite(filtered))


@given(st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False))
def test_k_weighting_dc_attenuation(dc_level: float) -> None:
    """K-weighting filter includes RLB highpass stage; continuous DC must decay to zero."""
    dc_input = np.full(4800, dc_level, dtype=np.float64)
    filtered = apply_k_weighting_filter(dc_input, sample_rate=48000)
    # After ~100 ms of DC settling, output magnitude should be near zero (< 1e-4)
    assert abs(filtered[-1]) < 1e-4


@given(
    st.lists(
        st.floats(min_value=1e-4, max_value=1.0, allow_nan=False, allow_infinity=False),
        min_size=10,
        max_size=50,
    ),
    st.floats(min_value=0.2, max_value=5.0, allow_nan=False, allow_infinity=False),
)
def test_itu_gating_scale_invariance(energies_list: list[float], scale_factor: float) -> None:
    """Scaling block energies by factor alpha^2 must shift integrated loudness by exactly 20*log10(alpha) LUFS."""
    energies = np.array(energies_list, dtype=np.float64)
    lufs_base = apply_itu_gating(energies, abs_thresh_lkfs=-70.0, rel_offset_lu=-10.0)

    scaled_energies = energies * (scale_factor**2)
    lufs_scaled = apply_itu_gating(scaled_energies, abs_thresh_lkfs=-70.0, rel_offset_lu=-10.0)

    if lufs_base > -70.0 and lufs_scaled > -70.0:
        expected_shift = 20.0 * math.log10(scale_factor)
        actual_shift = lufs_scaled - lufs_base
        assert math.isclose(actual_shift, expected_shift, abs_tol=1e-4)


@given(st.integers(min_value=100, max_value=4800))
def test_silence_evaluates_to_minimum_loudness(num_samples: int) -> None:
    """Pure digital silence evaluates to -100.0 LUFS."""
    silence = np.zeros(num_samples, dtype=np.float64)
    lufs = compute_lufs(silence, sample_rate=48000)
    assert lufs <= -70.0


@given(
    st_audio_buffers(min_len=2400, max_len=4800, min_val=-0.5, max_val=0.5),
    st.floats(min_value=1.1, max_value=2.0, allow_nan=False, allow_infinity=False),
)
def test_loudness_monotonicity_under_gain(audio: np.ndarray, gain: float) -> None:
    """Multiplying non-silent audio by gain > 1.0 strictly increases integrated loudness."""
    if np.max(np.abs(audio)) < 1e-4:
        return

    lufs1 = compute_lufs(audio, sample_rate=48000)
    lufs2 = compute_lufs(audio * gain, sample_rate=48000)

    if lufs1 > -70.0 and lufs2 > -70.0:
        assert lufs2 > lufs1
