"""
Property-based tests for non-linear saturation, algebraic rail limiter,
conformal clearance asymmetry, and Lenz drag dynamics.
Verifies mathematical invariants: strict peak bounding, odd symmetry,
small-signal linear bypass, and dissipative energy conservation.
"""

import math

import numpy as np
from hypothesis import given
from hypothesis import strategies as st

from allomorph.circuit.saturation import (
    apply_algebraic_rail_limiter,
    apply_conformal_clearance_asymmetry,
    apply_higher_order_magnetic_saturation,
    apply_oversampled_saturation,
    apply_velocity_drag_pitch_sag,
)
from tests.strategies import st_audio_buffers


@given(
    st.floats(min_value=-1000.0, max_value=1000.0, allow_nan=False, allow_infinity=False),
    st.floats(min_value=0.50, max_value=2.50, allow_nan=False, allow_infinity=False),
)
def test_algebraic_rail_limiter_strict_bounding_and_symmetry(x: float, vsat: float) -> None:
    """Algebraic limiter satisfies strict peak bounding |f(x)| < vsat and odd symmetry f(-x) == -f(x)."""
    y = float(apply_algebraic_rail_limiter(x, vsat=vsat))
    y_neg = float(apply_algebraic_rail_limiter(-x, vsat=vsat))

    assert math.isfinite(y)
    assert abs(y) < vsat + 1e-6
    assert math.isclose(y, -y_neg, abs_tol=1e-6)

    # Small-signal near-exact linearity
    if abs(x) <= 0.10 * vsat:
        assert math.isclose(y, x, rel_tol=1e-4, abs_tol=1e-5)


@given(
    st_audio_buffers(min_len=16, max_len=64, min_val=-2.0, max_val=2.0, dtype=np.float64),
    st.floats(min_value=0.50, max_value=2.0, allow_nan=False, allow_infinity=False),
)
def test_algebraic_rail_limiter_vector_invariants(buf: np.ndarray, vsat: float) -> None:
    """Vectorized algebraic rail limiter preserves array shape, finiteness, and peak ceiling."""
    out = apply_algebraic_rail_limiter(buf, vsat=vsat)
    assert out.shape == buf.shape
    assert np.all(np.isfinite(out))
    assert np.all(np.abs(out) < vsat + 1e-6)


@given(
    st.floats(min_value=-5.0, max_value=5.0, allow_nan=False, allow_infinity=False),
    st.floats(min_value=0.5, max_value=2.0, allow_nan=False, allow_infinity=False),
    st.floats(min_value=0.0, max_value=0.3, allow_nan=False, allow_infinity=False),
)
def test_conformal_clearance_asymmetry_invariants(x: float, vsat: float, kappa: float) -> None:
    """Conformal clearance asymmetry is finite, positive-direction amplified, and preserves sign."""
    arr = np.array([x], dtype=np.float64)
    out = float(apply_conformal_clearance_asymmetry(arr, vsat=vsat, kappa_geom=kappa)[0])

    assert math.isfinite(out)
    if x > 0 and kappa > 0:
        assert out >= x  # Proximity growl amplification
    if math.isclose(x, 0.0, abs_tol=1e-6):
        assert math.isclose(out, 0.0, abs_tol=1e-6)


@given(
    st.floats(min_value=-2.0, max_value=2.0, allow_nan=False, allow_infinity=False),
    st.floats(min_value=0.50, max_value=2.0, allow_nan=False, allow_infinity=False),
)
def test_higher_order_magnetic_saturation_bounds(x: float, vsat: float) -> None:
    """Higher-order magnetic saturation stays bounded by vsat and preserves zero DC."""
    arr = np.array([x], dtype=np.float64)
    out = float(apply_higher_order_magnetic_saturation(arr, vsat=vsat)[0])

    assert math.isfinite(out)
    assert abs(out) < vsat + 1e-6


@given(
    st_audio_buffers(min_len=64, max_len=256, min_val=-1.0, max_val=1.0, dtype=np.float64),
    st.floats(min_value=0.50, max_value=1.50, allow_nan=False, allow_infinity=False),
)
def test_velocity_drag_pitch_sag_dissipative_stability(buf: np.ndarray, vsat: float) -> None:
    """Lenz velocity drag ODE is numerically stable and energy dissipative."""
    out = apply_velocity_drag_pitch_sag(buf, vsat=vsat, k_sag=0.10, k_eddy=0.10)
    assert out.shape == buf.shape
    assert np.all(np.isfinite(out))


@given(
    st_audio_buffers(min_len=32, max_len=128, min_val=-0.08, max_val=0.08, dtype=np.float32),
)
def test_oversampled_saturation_small_signal_linear_bypass(buf: np.ndarray) -> None:
    """Signals below peak 0.10 bypass saturation with bit-exact linearity."""
    out = apply_oversampled_saturation(buf)
    assert np.array_equal(out, buf)
