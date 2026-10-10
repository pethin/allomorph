"""
Property-based tests for audio conditioning primitives.
Verifies mathematical invariants: DC offset nulling, Johnson-Nyquist thermal noise RMS,
and true-peak headroom ceiling enforcement across arbitrary audio buffers.
"""

import math

import numpy as np
from hypothesis import given
from hypothesis import strategies as st

from allomorph.dsp.conditioning import (
    apply_calibrated_normalization,
    apply_dc_block,
    apply_johnson_dither,
)
from allomorph.dsp.io import compute_true_peak
from tests.strategies import st_audio_buffers


@given(
    st_audio_buffers(min_len=2400, max_len=4800, min_val=-0.5, max_val=0.5),
    st.floats(min_value=-0.5, max_value=0.5, allow_nan=False, allow_infinity=False),
)
def test_dc_block_nulls_constant_offset(audio: np.ndarray, dc_offset: float) -> None:
    """Applying DC-block filter subtracts any continuous DC offset to near zero (< 1e-6)."""
    offset_audio = audio + dc_offset
    cleaned = apply_dc_block(offset_audio, sample_rate=48000, cutoff_hz=8.0)

    assert len(cleaned) == len(audio)
    assert np.all(np.isfinite(cleaned))
    assert abs(float(np.mean(cleaned))) < 1e-6


@given(st.integers(min_value=2400, max_value=4800))
def test_johnson_dither_rms_tolerance(n_samples: int) -> None:
    """Johnson-Nyquist dither added to digital silence matches target RMS within +/- 1.0 dB."""
    silence = np.zeros(n_samples, dtype=np.float64)
    flat_h = np.ones(512, dtype=np.float64)

    target_rms_dbfs = -108.0
    dithered = apply_johnson_dither(
        silence, flat_h, target_rms_dbfs=target_rms_dbfs, num_taps=512, seed=42
    )

    assert len(dithered) == n_samples
    assert np.all(np.isfinite(dithered))

    actual_rms = float(np.sqrt(np.mean(dithered**2)))
    actual_rms_db = 20.0 * math.log10(max(actual_rms, 1e-12))
    assert math.isclose(actual_rms_db, target_rms_dbfs, abs_tol=1.0)


@given(
    st_audio_buffers(min_len=2400, max_len=4800, min_val=-10.0, max_val=10.0),
    st.floats(min_value=0.5, max_value=0.99, allow_nan=False, allow_infinity=False),
)
def test_calibrated_normalization_enforces_peak_ceiling(audio: np.ndarray, ceiling: float) -> None:
    """Normalized audio must strictly satisfy the requested peak ceiling."""
    if np.max(np.abs(audio)) < 1e-4:
        return

    normalized = apply_calibrated_normalization(
        audio,
        mode="rms",
        target_dbfs=-18.0,
        sample_rate=48000,
        peak_ceiling=ceiling,
    )

    assert len(normalized) == len(audio)
    assert np.all(np.isfinite(normalized))

    tp = compute_true_peak(normalized, sample_rate=48000)
    assert tp <= ceiling + 1e-4
