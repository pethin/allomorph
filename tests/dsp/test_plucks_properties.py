"""
Property-based tests for physical pluck synthesis primitives.
Verifies mathematical invariants: modal frequency monotonicity, inharmonic dispersion,
attack pitch sag decay, and exponential ringout envelope bounds.
"""

import math

import numpy as np
from hypothesis import given
from hypothesis import strategies as st

from allomorph.dsp.plucks import (
    apply_attack_pitch_sag,
    compute_modal_frequencies,
    generate_exponential_ringout_envelope,
)


@given(
    st.floats(min_value=20.0, max_value=400.0, allow_nan=False, allow_infinity=False),
    st.floats(min_value=0.0, max_value=0.001, allow_nan=False, allow_infinity=False),
    st.integers(min_value=1, max_value=64),
)
def test_modal_frequencies_monotonicity_and_positivity(
    f0: float, b_inharm: float, num_h: int
) -> None:
    """Modal frequencies must be strictly positive and strictly increasing."""
    fns = compute_modal_frequencies(f0, b_inharm, num_h)
    assert len(fns) == num_h
    assert np.all(fns > 0.0)

    if num_h > 1:
        diffs = np.diff(fns)
        assert np.all(diffs > 0.0)


@given(
    st.floats(min_value=20.0, max_value=200.0, allow_nan=False, allow_infinity=False),
    st.integers(min_value=1, max_value=32),
)
def test_zero_inharmonicity_exact_harmonics(f0: float, num_h: int) -> None:
    """When B = 0, modal frequencies must be exact integer harmonics fn = n * f0."""
    fns = compute_modal_frequencies(f0, 0.0, num_h)
    expected = np.arange(1, num_h + 1, dtype=np.float64) * f0
    assert np.allclose(fns, expected, rtol=1e-7, atol=1e-7)


@given(
    st.floats(min_value=0.5, max_value=10.0, allow_nan=False, allow_infinity=False),
    st.floats(min_value=0.01, max_value=0.5, allow_nan=False, allow_infinity=False),
)
def test_attack_pitch_sag_monotonic_decay(sag_hz: float, tau: float) -> None:
    """Pitch sag dynamic frequency shift is strictly positive and decays toward zero."""
    t = np.linspace(0.0, 1.0, 480, endpoint=False)
    # The instantaneous frequency shift is Delta f(t) = sag_hz * exp(-t / tau)
    freq_shift = sag_hz * np.exp(-t / tau)
    assert np.all(freq_shift > 0.0)
    assert np.all(np.diff(freq_shift) <= 0.0)

    phase = apply_attack_pitch_sag(t, pitch_sag_hz=sag_hz, tau_sag=tau)
    assert np.all(np.isfinite(phase))


@given(
    st.floats(min_value=0.1, max_value=5.0, allow_nan=False, allow_infinity=False),
    st.floats(min_value=0.5, max_value=10.0, allow_nan=False, allow_infinity=False),
)
def test_exponential_ringout_envelope_bounds_and_monotonicity(duration: float, t60: float) -> None:
    """Exponential ringout envelope starts at 1.0, stays within (0, 1], and decays monotonically."""
    env = generate_exponential_ringout_envelope(duration, sample_rate=48000, t60_sec=t60)
    assert len(env) > 0
    assert math.isclose(env[0], 1.0, abs_tol=1e-5)
    assert np.all(env > 0.0)
    assert np.all(env <= 1.0)
    assert np.all(np.diff(env) <= 0.0)
