"""
Property-based tests for homomorphic real-cepstrum Hilbert transform minimum-phase FIR synthesis.
Verifies mathematical invariants: causal energy concentration, peak normalization ceiling,
and smooth tail boundary conditions across arbitrary positive magnitude curves.
"""

import math

import numpy as np
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from allomorph.dsp.fir import (
    compute_minimum_phase_spectrum,
    synthesize_minimum_phase_fir,
    window_fir_impulse,
)
from tests.strategies import st_audio_buffers


@given(
    st.lists(
        st.floats(min_value=0.01, max_value=10.0, allow_nan=False, allow_infinity=False),
        min_size=16,
        max_size=64,
    ),
    st.sampled_from([512, 1024, 2048]),
)
def test_minimum_phase_fir_causality_and_peak_ceiling(
    mag_points: list[float], num_taps: int
) -> None:
    """Synthesized FIR must have exact requested length, be finite, and peak-bounded to 0.9900."""
    grid_x = np.linspace(0.0, 1.0, len(mag_points))
    target_x = np.linspace(0.0, 1.0, num_taps)
    curve = np.interp(target_x, grid_x, mag_points)

    fir = synthesize_minimum_phase_fir(curve, num_taps=num_taps, normalize=True)
    assert len(fir) == num_taps

    fir_np = np.asarray(fir, dtype=np.float64)
    assert np.all(np.isfinite(fir_np))

    max_peak = float(np.max(np.abs(fir_np)))
    assert max_peak <= 0.990001
    assert math.isfinite(max_peak)


@given(
    st.lists(
        st.floats(min_value=0.1, max_value=2.0, allow_nan=False, allow_infinity=False),
        min_size=32,
        max_size=64,
    )
)
def test_minimum_phase_energy_concentration(mag_points: list[float]) -> None:
    """A minimum-phase filter concentrates the majority of its total energy in the early samples."""
    num_taps = 1024
    grid_x = np.linspace(0.0, 1.0, len(mag_points))
    target_x = np.linspace(0.0, 1.0, num_taps)
    curve = np.interp(target_x, grid_x, mag_points)

    fir = np.asarray(
        synthesize_minimum_phase_fir(curve, num_taps=num_taps, normalize=False), dtype=np.float64
    )
    half = num_taps // 2
    early_energy = np.sum(fir[:half] ** 2)
    late_energy = np.sum(fir[half:] ** 2)

    # Minimum phase guarantees early energy dominates
    assert early_energy >= late_energy


@given(
    st.lists(
        st.floats(min_value=0.01, max_value=5.0, allow_nan=False, allow_infinity=False),
        min_size=33,
        max_size=33,
    )
)
def test_minimum_phase_spectrum_modulus_consistency(mag_points: list[float]) -> None:
    """The modulus of H_min(omega) matches the input magnitude curve closely."""
    grid = np.asarray(mag_points, dtype=np.float64)
    n_fft = 2 * (len(grid) - 1)
    h_min = compute_minimum_phase_spectrum(grid, n_fft=n_fft)

    mag_recovered = np.abs(h_min)
    # Check consistency above DC (where Guardrail 5.3 DC zero extrapolation does not apply)
    rel_error = np.abs(mag_recovered[1:] - grid[1:]) / (grid[1:] + 1e-4)
    assert float(np.mean(rel_error)) < 0.05
    assert np.all(np.isfinite(mag_recovered))
    assert mag_recovered[0] > 0.0


@settings(suppress_health_check=[HealthCheck.large_base_example])
@given(st_audio_buffers(min_len=512, max_len=512, min_val=-1.0, max_val=1.0, dtype=np.float64))
def test_window_fir_tail_taper(raw_impulse: np.ndarray) -> None:
    """Windowed impulse response smoothly decays to zero at the final sample."""
    windowed = window_fir_impulse(raw_impulse, num_taps=512, normalize=True, peak_limit=0.99)
    # Final sample must be zero due to C^inf smoothstep tail
    assert abs(windowed[-1]) < 1e-6
