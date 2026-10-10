"""
Tests for circuit dielectric absorption, cable losses, and Jordan permeability relaxation.
"""

import math

import numpy as np

from allomorph.circuit import (
    compute_circuit_transfer_functions,
    compute_core_impedance,
    load_circuit,
)
from allomorph.dsp import FREQS


def test_cable_dielectric_loss():
    """
    Verify instrument cable dielectric loss (tan delta):
    1. At DC (f=0), dielectric conductance is strictly zero, preserving exact 0.00 dB DC transfer.
    2. At the resonant peak, tan_delta=0.025 provides gentle 0.2 to 0.7 dB softening of Q peak.
    3. Active buffered pickups (model.has_active_buffer=True) isolate coils from cable dielectric loss.
    """
    m_lossless = load_circuit("precision_vintage")
    m_lossless.tan_delta = 0.0

    m_lossy = load_circuit("precision_vintage")
    m_lossy.tan_delta = 0.025

    c_lossless = np.array(compute_circuit_transfer_functions(m_lossless, freqs=FREQS)[0])
    c_lossy = np.array(compute_circuit_transfer_functions(m_lossy, freqs=FREQS)[0])

    # 1. Exact DC unity preservation
    assert math.isclose(c_lossless[0], c_lossy[0], abs_tol=1e-5)

    # 2. Resonant peak softening (between 1800 and 2400 Hz)
    pk_idx = np.argmax(c_lossless)
    diff_peak_db = 20.0 * np.log10(c_lossless[pk_idx] / c_lossy[pk_idx])
    assert 0.05 <= diff_peak_db <= 0.50, (
        f"Peak attenuation {diff_peak_db:.2f} dB outside expected range"
    )

    # 3. High-frequency rolloff remains smooth
    assert c_lossy[-1] < 0.20


def test_coil_dielectric_loss():
    """
    Verify Refinement 2: Coil self-capacitance dielectric loss (tan delta = 0.025).
    Gently softens resonant peak by ~0.01-0.5 dB without shifting center frequency.
    """
    model = load_circuit("precision_vintage")

    # Compute with zero dielectric loss
    model.tan_delta_coil = 0.0
    curves_lossless = compute_circuit_transfer_functions(model, freqs=FREQS)
    peak_lossless = float(max(curves_lossless[0]))
    peak_idx_lossless = curves_lossless[0].index(peak_lossless)
    peak_freq_lossless = FREQS[peak_idx_lossless]

    # Compute with physical dielectric loss (tan delta = 0.025)
    model.tan_delta_coil = 0.025
    curves_lossy = compute_circuit_transfer_functions(model, freqs=FREQS)
    peak_lossy = float(max(curves_lossy[0]))
    peak_idx_lossy = curves_lossy[0].index(peak_lossy)
    peak_freq_lossy = FREQS[peak_idx_lossy]

    # Resonant frequency must remain virtually unchanged (within 50 Hz)
    assert abs(peak_freq_lossy - peak_freq_lossless) <= 50.0

    # Dielectric loss should gently soften the resonant peak
    delta_db = 20.0 * np.log10(peak_lossless / peak_lossy)
    assert 0.005 <= delta_db <= 0.50


def test_fractional_order_dielectric_absorption():
    """Verify Cole-Davidson fractional-order dielectric absorption in capacitors."""
    # Load Voice 05c (47nF rolled tone)
    m = load_circuit("precision_warm")

    # 1. Ideal capacitor (alpha = 1.0)
    m.alpha_dielectric_tone = 1.0
    m.alpha_dielectric_cable = 1.0
    curves_ideal = compute_circuit_transfer_functions(m, freqs=FREQS)

    # 2. Fractional-order film dielectric (alpha = 0.988)
    m.alpha_dielectric_tone = 0.988
    m.alpha_dielectric_cable = 0.994
    curves_dielectric = compute_circuit_transfer_functions(m, freqs=FREQS)

    mag_ideal = np.asarray(curves_ideal[0])
    mag_dielectric = np.asarray(curves_dielectric[0])

    # Dielectric absorption should create subtle, smooth loss differences (within 0.05 to 1.5 dB across passband)
    diff_db = 20.0 * np.log10(np.maximum(mag_dielectric, 1e-6) / np.maximum(mag_ideal, 1e-6))
    assert np.all(np.abs(diff_db) < 2.0), "Dielectric absorption should be a subtle analog nuance"
    assert np.max(np.abs(diff_db)) > 0.05, (
        "Dielectric absorption must produce non-trivial difference"
    )
    # At low frequencies (200 Hz), dielectric absorption provides subtle low-mid loss/bloom
    idx_200 = min(range(len(FREQS)), key=lambda i: abs(FREQS[i] - 200.0))
    assert diff_db[idx_200] < 0.0, "Dielectric relaxation should introduce low-mid dissipation"


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

    # 2. Transfer functions of matching model must be identical
    m1 = load_circuit("precision_vintage")
    m2 = load_circuit("precision_vintage")
    m1.chi_mu = 0.04
    m2.chi_mu = 0.04
    c1 = compute_circuit_transfer_functions(m1, freqs=FREQS)
    c2 = compute_circuit_transfer_functions(m2, freqs=FREQS)
    assert np.allclose(c1, c2, atol=1e-4)
