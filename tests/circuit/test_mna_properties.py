"""
Property-based tests for MNA linear system solver, smooth soft-knee saturation,
core impedance passivity, and biquad stability.
Verifies mathematical invariants: Cramer solver accuracy, strictly monotonic soft-knee,
positive-real impedance passivity, and pole stability of synthesized biquads.
"""

import math
from typing import Literal

import numpy as np
from hypothesis import given
from hypothesis import strategies as st

from allomorph.circuit.mna import (
    compute_active_preamp_biquads,
    compute_core_impedance,
    smooth_soft_knee_db,
    solve_mna_linear_system,
)
from allomorph.config.schema import PreampBandConfig
from tests.strategies import st_frequency_arrays


@given(
    st.sampled_from([1, 2, 3, 4]),
    st.integers(min_value=2, max_value=8),
)
def test_mna_linear_system_solver_consistency_and_inversion(n_nodes: int, n_freqs: int) -> None:
    """MNA linear solver Y * V = I must reproduce I exactly and match np.linalg.solve."""
    rng = np.random.default_rng(42)

    # Construct strictly diagonally dominant complex admittance matrices
    A_real = rng.uniform(0.1, 2.0, (n_freqs, n_nodes, n_nodes))
    A_imag = rng.uniform(-1.0, 1.0, (n_freqs, n_nodes, n_nodes))
    Y = A_real + 1j * A_imag

    # Ensure strong diagonal dominance for non-singular passive circuit matrices
    for i in range(n_nodes):
        Y[:, i, i] += n_nodes * 5.0 + 1.0

    I_real = rng.uniform(-1.0, 1.0, (n_freqs, n_nodes))
    I_imag = rng.uniform(-1.0, 1.0, (n_freqs, n_nodes))
    I_vec = I_real + 1j * I_imag

    V_sol = solve_mna_linear_system(Y, I_vec)
    assert V_sol.shape == (n_freqs, n_nodes)
    assert np.all(np.isfinite(V_sol))

    # Verify Y * V ≈ I
    I_recovered = np.einsum("fij,fj->fi", Y, V_sol)
    rel_error = np.abs(I_recovered - I_vec) / (np.abs(I_vec) + 1e-6)
    assert np.max(rel_error) < 1e-5

    # Verify closed-form Cramer rule matches NumPy reference solve
    V_np = np.linalg.solve(Y, I_vec[:, :, np.newaxis])[:, :, 0]
    cramer_err = np.abs(V_sol - V_np)
    assert np.max(cramer_err) < 1e-5


@given(
    st.floats(min_value=-50.0, max_value=50.0, allow_nan=False, allow_infinity=False),
    st.floats(min_value=2.0, max_value=12.0, allow_nan=False, allow_infinity=False),
    st.floats(min_value=1.0, max_value=6.0, allow_nan=False, allow_infinity=False),
)
def test_smooth_soft_knee_db_invariants(x_db: float, thresh: float, headroom: float) -> None:
    """Soft knee saturation must be strictly bounded below ceiling, linear below thresh, and monotonic."""
    ceiling = thresh + headroom
    y = float(smooth_soft_knee_db(x_db, thresh=thresh, ceiling=ceiling, alpha=2.0))

    assert math.isfinite(y)
    assert y < ceiling + 1e-6

    # Linear passband transparency for inputs well below threshold
    if x_db < thresh - 10.0:
        assert math.isclose(y, x_db, rel_tol=1e-4, abs_tol=1e-4)

    # Monotonicity: non-decreasing everywhere, strictly increasing below asymptotic saturation
    eps = 0.01
    y_next = float(smooth_soft_knee_db(x_db + eps, thresh=thresh, ceiling=ceiling, alpha=2.0))
    assert y_next >= y
    if x_db < ceiling + 2.0:
        assert y_next > y


@given(
    st_frequency_arrays(min_len=5, max_len=20, min_f=20.0, max_f=20000.0),
    st.floats(min_value=1.0, max_value=8.0, allow_nan=False, allow_infinity=False),
    st.floats(min_value=3000.0, max_value=12000.0, allow_nan=False, allow_infinity=False),
)
def test_core_impedance_passivity_invariants(freqs: np.ndarray, L: float, Rdc: float) -> None:
    """Core inductor impedance Z_L(omega) must be positive-real (passive inductive load)."""
    w = 2.0 * np.pi * freqs
    s = 1j * w

    Z = compute_core_impedance(
        s,
        L=L,
        L_core=0.08 * L,
        R_core=2.0 * math.pi * 2500.0 * (0.08 * L),
        chi_mu=0.035,
        k_skin=0.10,
        Rdc=Rdc,
    )

    assert np.all(np.isfinite(Z))

    # Passivity: Re(Z) >= 0 and Im(Z) >= 0 for an inductor across positive frequencies
    assert np.all(np.real(Z) >= -1e-6)
    assert np.all(np.imag(Z) >= -1e-6)

    # At DC (s = 0), reactive impedance must be zero
    Z_dc = compute_core_impedance(0.0 + 0j, L=L, Rdc=Rdc)
    assert math.isclose(abs(complex(Z_dc)), 0.0, abs_tol=1e-6)


@given(
    st.sampled_from(["low_shelf", "high_shelf", "bell"]),
    st.floats(min_value=40.0, max_value=8000.0, allow_nan=False, allow_infinity=False),
    st.floats(min_value=-12.0, max_value=12.0, allow_nan=False, allow_infinity=False),
)
def test_active_preamp_biquads_stability_and_finite_coefficients(
    b_type: Literal["low_shelf", "high_shelf", "bell"], freq_hz: float, gain_db: float
) -> None:
    """Synthesized Direct-Form II biquad poles must strictly lie inside the unit circle (|p| < 1)."""
    band = PreampBandConfig(type=b_type, freq_hz=freq_hz, gain_db=gain_db, q=1.0)
    biquads = compute_active_preamp_biquads([band], fs=48000.0)

    for b0, b1, b2, a0, a1, a2 in biquads:
        assert math.isclose(a0, 1.0, abs_tol=1e-5)
        assert math.isfinite(b0)
        assert math.isfinite(b1)
        assert math.isfinite(b2)
        assert math.isfinite(a1)
        assert math.isfinite(a2)

        # Solve quadratic poles: z^2 + a1*z + a2 = 0
        disc = a1 * a1 - 4.0 * a2
        if disc >= 0:
            p1 = abs((-a1 + math.sqrt(disc)) / 2.0)
            p2 = abs((-a1 - math.sqrt(disc)) / 2.0)
        else:
            p1 = p2 = math.sqrt(complex((-a1) / 2.0, math.sqrt(-disc) / 2.0).__abs__())

        # Strict stability for audio filters
        assert p1 < 1.0001
        assert p2 < 1.0001
