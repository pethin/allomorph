"""
Tests for circuit dielectric absorption, cable losses, and Jordan permeability relaxation.
"""

import math

import numpy as np

from allomorph.circuit import (
    compute_core_impedance,
    solve_mna_harness,
)
from allomorph.circuit.forward import resolve_target_voicing
from allomorph.dsp import FREQS


def test_cable_dielectric_and_capacitive_loss():
    """
    Verify instrument cable loading and dielectric loss:
    1. At DC (f=0), dielectric conductance is strictly zero, preserving exact DC transfer.
    2. Cable capacitance downshifts resonant peak and softens high-frequency response.
    """
    inst, voicing = resolve_target_voicing("precision_vintage")
    harness = inst.harnesses[voicing.harness]

    v_low = voicing.model_copy(deep=True)
    v_low.components["cable_pf"] = 500.0

    v_high = voicing.model_copy(deep=True)
    v_high.components["cable_pf"] = 1500.0

    c_low = np.abs(next(iter(solve_mna_harness(inst, harness, v_low, freqs=FREQS).values())))
    c_high = np.abs(next(iter(solve_mna_harness(inst, harness, v_high, freqs=FREQS).values())))

    # 1. Exact DC unity preservation (DC is unaffected by shunt cable capacitance)
    assert math.isclose(c_low[0], c_high[0], abs_tol=1e-4)
    assert 0.95 <= c_low[0] <= 1.0

    # 2. Resonant peak shift downwards
    f_arr = np.asarray(FREQS)
    pk_low = f_arr[np.argmax(c_low)]
    pk_high = f_arr[np.argmax(c_high)]
    assert pk_high < pk_low, f"Expected downshift: {pk_low} -> {pk_high}"

    # 3. High-frequency roll-off at 4 kHz is strictly greater with higher cable capacitance
    idx_4k = int(np.argmin(np.abs(f_arr - 4000.0)))
    assert c_high[idx_4k] < c_low[idx_4k]
    assert c_high[-1] < 0.20


def test_complex_magnetic_permeability_dispersion():
    """Verify causal Jordan complex permeability dispersion provides midrange core loss."""
    omega = 2.0 * np.pi * np.array([400.0, 800.0, 1200.0, 2400.0], dtype=np.float64)
    s = 1j * omega

    # 1. Complex core impedance with chi_mu > 0
    Z_ideal = compute_core_impedance(s, L=5.0, R_core=25000.0, chi_mu=0.0)
    Z_dispersive = compute_core_impedance(s, L=5.0, R_core=25000.0, chi_mu=0.04)

    # Real part (resistive loss) must be enhanced by Jordan relaxation
    assert np.all(np.real(Z_dispersive) > np.real(Z_ideal)), (
        "Complex permeability must add core relaxation losses"
    )
    assert not np.any(np.isnan(Z_dispersive))

    # 2. Transfer functions of identical model must be bit-exact
    inst, voicing = resolve_target_voicing("precision_vintage")
    harness = inst.harnesses[voicing.harness]
    c1 = solve_mna_harness(inst, harness, voicing, freqs=FREQS)
    c2 = solve_mna_harness(inst, harness, voicing, freqs=FREQS)
    for k in c1:
        assert np.allclose(c1[k], c2[k], atol=1e-6)


def test_dc_regularization_exact_unity():
    """Verify DC gain is strictly finite, non-zero, and near unity (~0.95 due to volume pot load)."""
    inst, voicing = resolve_target_voicing("precision_vintage")
    harness = inst.harnesses[voicing.harness]
    curves = solve_mna_harness(inst, harness, voicing, freqs=FREQS)
    c = np.abs(next(iter(curves.values())))
    assert 0.95 <= c[0] <= 1.0
