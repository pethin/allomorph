"""
Property-based tests for dry excitation segment generators.
Verifies mathematical invariants: calibration blip heights, velocity tier monotonicity,
smooth boundary micro-fades, and overall excitation peak ceilings.
"""

import math

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from allomorph.dsp.dry import (
    generate_optimal_bass_dry,
    synth_calibration_blips,
    synth_log_chirps,
    synth_velocity_ladder,
)


@settings(max_examples=10)
@given(st.floats(min_value=0.2, max_value=2.0, allow_nan=False, allow_infinity=False))
def test_calibration_blips_heights_and_bounds(scale: float) -> None:
    """Calibration blips must contain exact +/-0.89 impulses within bounded ranges."""
    blips = synth_calibration_blips(sample_rate=48000, scale=scale)
    assert len(blips) > 0
    assert np.all(np.isfinite(blips))
    assert math.isclose(float(np.max(blips)), 0.89, abs_tol=1e-5)
    assert math.isclose(float(np.min(blips)), -0.89, abs_tol=1e-5)


@settings(max_examples=5)
@given(st.floats(min_value=0.2, max_value=1.5, allow_nan=False, allow_infinity=False))
def test_log_chirps_boundary_fades(scale: float) -> None:
    """Log chirps must have smooth C^inf boundary fades to avoid spectral leakage."""
    chirps = synth_log_chirps(sample_rate=24000, scale=scale)
    assert len(chirps) == 5

    for seg, pause in chirps:
        assert len(seg) > 0
        assert np.all(np.isfinite(seg))
        assert abs(seg[0]) < 1e-4
        assert abs(seg[-1]) < 1e-4
        assert pause > 0.0


@settings(max_examples=5)
@given(st.floats(min_value=0.5, max_value=1.0, allow_nan=False, allow_infinity=False))
def test_velocity_ladder_monotonic_peak_growth(scale: float) -> None:
    """Velocity ladder segments must exhibit strictly increasing peak amplitudes."""
    ladder = synth_velocity_ladder(sample_rate=24000, scale=scale)
    assert len(ladder) == 7

    peaks = [float(np.max(np.abs(seg))) for seg, _ in ladder]
    for i in range(1, len(peaks)):
        assert peaks[i] > peaks[i - 1]


@settings(max_examples=10)
@given(
    st.floats(min_value=1.0, max_value=4.0, allow_nan=False, allow_infinity=False),
    st.floats(min_value=-6.0, max_value=-0.5, allow_nan=False, allow_infinity=False),
)
def test_optimal_bass_dry_duration_and_peak_ceiling(duration_sec: float, peak_dbfs: float) -> None:
    """Dry excitation audio must match requested duration and respect the peak ceiling."""
    sample_rate = 48000
    expected_samples = int(duration_sec * sample_rate)

    dry = generate_optimal_bass_dry(
        duration_sec=duration_sec,
        sample_rate=sample_rate,
        peak_dbfs=peak_dbfs,
        target_rms_dbfs=None,
    )

    assert len(dry) == expected_samples
    assert np.all(np.isfinite(dry))

    actual_peak = float(np.max(np.abs(dry)))
    ceiling = 10.0 ** (peak_dbfs / 20.0)
    assert actual_peak <= ceiling + 1e-4
