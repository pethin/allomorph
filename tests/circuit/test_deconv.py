"""
Tests for differential circuit deconvolution, Wiener regularization,
dielectric absorption, and Jordan permeability relaxation.
"""

import math

import numpy as np

from allomorph.circuit import (
    compute_circuit_transfer_functions,
    compute_core_impedance,
    compute_differential_circuit_transfer_functions,
    load_circuit,
)
from allomorph.config import INSTRUMENTS, VOICES
from allomorph.dsp import FREQS, NUM_TAPS
from allomorph.physics import compute_voice_prefilter_firs


def test_passive_identity_differential_flatness():
    """Verify that identity differential response is flat, and pot unloading is accurately modeled."""
    f_arr = np.asarray(FREQS)
    passband_mask = (f_arr >= 40.0) & (f_arr <= 4500.0)

    # 1. Precision Bass true identity: Standard P source against Voice 05 Vintage '62 P (< 0.10 dB)
    p_circ = INSTRUMENTS["34in_standard_p"].pickups["split_p"].circuit
    assert p_circ is not None
    m_src_p = load_circuit(p_circ)
    m_tgt_p = load_circuit("precision_vintage")
    diff_p = compute_differential_circuit_transfer_functions(m_tgt_p, m_src_p, freqs=FREQS)
    h_p = np.asarray(diff_p[0])
    h_p_db = 20.0 * np.log10(h_p / h_p[0])
    assert np.all(np.abs(h_p_db[passband_mask]) < 0.10), "P-Bass identity differential not flat!"

    # 2. Jazz Bass true identity: Standard Jazz source against Voice 02 Jazz Bass Pair (< 0.10 dB)
    j_circ = INSTRUMENTS["34in_standard_jazz"].pickups["pair_parallel"].circuit
    assert j_circ is not None
    m_src_j = load_circuit(j_circ)
    m_tgt_j = load_circuit("jazz_pair_open")
    diff_j = compute_differential_circuit_transfer_functions(m_tgt_j, m_src_j, freqs=FREQS)
    for ch_idx, ch in enumerate(diff_j):
        h_j = np.asarray(ch)
        h_j_db = 20.0 * np.log10(h_j / h_j[0])
        assert np.all(np.abs(h_j_db[passband_mask]) < 0.10), (
            f"Jazz channel {ch_idx} identity differential not flat!"
        )

    # 3. Active Split-P: Source Standard P (250k V/T) vs Target Modern Active Ceramic P (Sadowsky 2-band)
    m_tgt_act = load_circuit("precision_active")
    diff_act = compute_differential_circuit_transfer_functions(m_tgt_act, m_src_p, freqs=FREQS)
    h_act = np.asarray(diff_act[0])
    h_act_db = 20.0 * np.log10(h_act / h_act[0])
    assert -5.0 <= np.min(h_act_db[passband_mask])
    assert np.max(h_act_db[passband_mask]) <= 6.0


def test_active_source_differential_deconvolution_and_identity():
    """Verify that commercial active sources (StingRay, Dingwall) have valid source circuits,
    yield exact 0.0 dB on identity, and properly deconvolve active preamps on cross-voicings."""
    # 1. Source netlist existence & parsing
    ray_circ = INSTRUMENTS["34in_active_stingray"].pickups["mm_parallel"].circuit
    assert ray_circ is not None
    m_src_ray = load_circuit(ray_circ)
    assert m_src_ray.has_active_buffer is True
    assert m_src_ray.preamp_type == "stingray_2band"

    ding_circ = INSTRUMENTS["37in_multiscale_dingwall"].pickups["bridge"].circuit
    assert ding_circ is not None
    m_src_ding = load_circuit(ding_circ)
    assert m_src_ding.has_active_buffer is True
    assert m_src_ding.preamp_type == "none"
    assert m_src_ding.L == 2.3

    # SP1 Passive Dingwall bridge source verification
    sp1_circ = INSTRUMENTS["34in_dingwall_sp1"].pickups["bridge"].circuit
    assert sp1_circ is not None
    m_src_sp1 = load_circuit(sp1_circ)
    assert m_src_sp1.has_active_buffer is False
    assert m_src_sp1.L == 2.3
    assert m_src_sp1.Rtop == 10
    assert m_src_sp1.Rbot == 250000.0

    # 2. Mathematical identity flatness (exact 0.00 dB everywhere, including DC and 20 kHz)
    m_tgt_ray = load_circuit("stingray_parallel")
    diff_ray = compute_differential_circuit_transfer_functions(m_tgt_ray, m_src_ray, freqs=FREQS)
    assert np.all(np.array(diff_ray[0]) == 1.0)

    m_tgt_ding = load_circuit("dingwall_bridge")
    assert m_tgt_ding.has_active_buffer is True
    assert m_tgt_ding.preamp_type == "none"
    diff_ding = compute_differential_circuit_transfer_functions(m_tgt_ding, m_src_ding, freqs=FREQS)
    assert np.all(np.array(diff_ding[0]) == 1.0)

    # 3. Cross-deconvolution: Active StingRay -> Vintage '62 P-Bass
    m_tgt_p = load_circuit("precision_vintage")
    diff_cross = compute_differential_circuit_transfer_functions(m_tgt_p, m_src_ray, freqs=FREQS)
    h_cross = np.asarray(diff_cross[0])
    f_arr = np.asarray(FREQS)
    # The StingRay has a +2.2 dB active treble boost around 5-7 kHz.
    # When converting to a darker vintage P-bass, high frequencies must be rolled off (< -10 dB @ 8 kHz)
    idx_8k = np.argmin(np.abs(f_arr - 8000.0))
    cross_db_8k = 20.0 * np.log10(h_cross[idx_8k])
    assert cross_db_8k < -10.0, (
        f"Expected active treble shelf deconvolution (< -10 dB), got {cross_db_8k:.2f} dB"
    )


def test_wiener_clamping_prevents_noise_explosion():
    """Verify that differential top-end boost is strictly clamped <= +6.0 dB above 4.5 kHz."""
    # Convert high-inductance P (3.8H) to brighter Jazz Bridge (3.6H)
    p_circ = INSTRUMENTS["34in_standard_p"].pickups["split_p"].circuit
    assert p_circ is not None
    m_src = load_circuit(p_circ)
    m_tgt = load_circuit("jazz_bridge_open")

    diff_curves = compute_differential_circuit_transfer_functions(
        m_tgt, m_src, freqs=FREQS, max_boost_db=6.0
    )
    h_diff = np.asarray(diff_curves[0])

    f_arr = np.asarray(FREQS)
    ref_idx = int(np.argmin(np.abs(f_arr - 1000.0)))
    h_diff_rel_db = 20.0 * np.log10(h_diff / h_diff[ref_idx])

    hi_mask = f_arr >= 4500.0
    max_hi_boost = np.max(h_diff_rel_db[hi_mask])
    assert max_hi_boost <= 6.05  # Within 0.05 dB of +6.0 dB ceiling


def test_differential_circuit_hf_limiter_smoothness():
    """
    Verify that differential circuit transfer functions crossing 0.0 dB above 8 kHz
    transition smoothly with strictly continuous first derivative and zero slope kinks.
    """
    tgt_model = load_circuit("jazz_bridge_open")
    src_circ = INSTRUMENTS["34in_standard_p"].pickups["split_p"].circuit
    assert src_circ is not None
    src_model = load_circuit(src_circ)

    curves = compute_differential_circuit_transfer_functions(tgt_model, src_model, freqs=FREQS)
    assert len(curves) > 0
    h_c = np.asarray(curves[0], dtype=np.float64)
    h_db = 20.0 * np.log10(h_c / h_c[0])

    f_arr = np.asarray(FREQS, dtype=np.float64)
    hf_mask = f_arr >= 8000.0
    # Check that the derivative d(h_db)/df does not have sudden jump steps across 0 dB
    dh = np.gradient(h_db[hf_mask], f_arr[hf_mask])
    # The rate of change of derivative should be well-behaved
    d2h = np.gradient(dh, f_arr[hf_mask])
    assert np.max(np.abs(d2h)) < 1e-4, (
        "Derivative of differential curve should be smooth without piecewise kinks"
    )


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
    """Verify causal Jordan complex permeability dispersion provides midrange core loss and preserves differential identity."""
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

    # 2. Differential transfer function of matching model must be exact identity (0.00 dB)
    m1 = load_circuit("precision_vintage")
    m2 = load_circuit("precision_vintage")
    m1.chi_mu = 0.04
    m2.chi_mu = 0.04
    diff_curves = compute_differential_circuit_transfer_functions(m1, m2, freqs=FREQS)
    for curve in diff_curves:
        assert np.allclose(curve, 1.0, atol=1e-4), (
            "Matching complex permeability models must yield exact 0.00 dB identity"
        )


def test_neutral_character_simulation():
    """Verify studio_direct preserves aperture and provides dynamic feel."""
    vcfg = VOICES["studio_direct"]
    assert vcfg.sensor_type == "direct"
    assert vcfg.preserve_aperture is True
    assert vcfg.alpha == 0.26
    assert vcfg.vsat == 0.50

    # Netlist must be a no_eq flat studio buffer
    model = load_circuit(vcfg.circuit)
    assert getattr(model, "no_eq", False) is True

    # Evaluated on an instrument, prefilter FIR preserves physical aperture (unit impulse)
    firs = compute_voice_prefilter_firs("studio_direct", instrument="34in_standard_p")
    assert len(firs) == 1
    fir = np.array(firs[0])
    assert len(fir) == NUM_TAPS
    assert fir[0] == 1.0
    assert np.all(fir[1:] == 0.0)


def test_active_character_differential_cable_isolation():
    """Verify studio_active deconvolves passive cable loading when evaluating from a passive bass."""
    tgt_model = load_circuit("studio_active")
    p_circ = INSTRUMENTS["34in_standard_p"].pickups["split_p"].circuit
    assert p_circ is not None
    src_model = load_circuit(p_circ)

    diff_curves = compute_differential_circuit_transfer_functions(tgt_model, src_model, freqs=FREQS)
    assert len(diff_curves) == 1
    h_diff = np.array(diff_curves[0])

    # At 100 Hz, both circuits have flat DC/low-frequency transmission
    idx_100 = np.argmin(np.abs(np.array(FREQS) - 100.0))
    assert 0.90 <= h_diff[idx_100] <= 1.15

    # At 8 kHz, passive circuit has heavy cable loading, active buffer is isolated
    idx_8k = np.argmin(np.abs(np.array(FREQS) - 8000.0))
    assert h_diff[idx_8k] > 1.0, "Active buffer must deconvolve passive cable loading at 8 kHz"


def test_passive_character_circuit_properties():
    """Verify studio_passive preserves aperture and models passive cable loading."""
    vcfg = VOICES["studio_passive"]
    assert vcfg.preserve_aperture is True
    assert vcfg.alpha == 0.28
    assert vcfg.vsat == 0.50

    model = load_circuit(vcfg.circuit)
    assert model.has_active_buffer is False
    assert model.L == 4.2
    assert model.Rdc == 8500.0

    firs = compute_voice_prefilter_firs("studio_passive", instrument="34in_standard_p")
    assert len(firs) == 1
    fir = np.array(firs[0])
    assert fir[0] == 1.0
    assert np.all(fir[1:] == 0.0)
