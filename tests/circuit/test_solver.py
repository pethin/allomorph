"""
Tests for analytical SPICE nodal RLC solver, potentiometer wiper math,
active preamp buffers, mutual coupling matrix, and core dispersion.
"""

import math

import numpy as np
import pytest

from allomorph.circuit import (
    CircuitModel,
    compute_circuit_transfer_functions,
    compute_core_impedance,
    compute_differential_circuit_transfer_functions,
    load_circuit,
)
from allomorph.circuit.schema import CircuitConfig
from allomorph.config import INSTRUMENTS, load_instrument
from allomorph.dsp import FREQS


def test_single_pickup_transfer_function():
    model = load_circuit("04_modern_p_ceramic")
    curves = compute_circuit_transfer_functions(model, freqs=FREQS)

    assert len(curves) == 1
    mag = curves[0]

    # DC Gain should be near 0 dB (~0.97 due to 10-ohm pot + load divider)
    assert 0.95 < mag[0] < 1.0

    # Resonant peak should occur between 1850 Hz and 2300 Hz
    max_val = max(mag)
    peak_idx = mag.index(max_val)
    peak_freq = FREQS[peak_idx]
    assert 1850.0 <= peak_freq <= 2300.0
    assert max_val > 1.10  # Under pot/cable load, Q peak > 1.10

    # High-frequency rolloff (at 20 kHz, gain should be < 0.20)
    assert mag[-1] < 0.20


def test_tone_rolloff_transfer_function():
    model = load_circuit("05c_vintage_62_p_47nf")
    assert model.Ctone == pytest.approx(47e-9)

    curves = compute_circuit_transfer_functions(model, freqs=FREQS)
    mag = curves[0]

    # With 47nF shunt, resonant peak collapses into low-mids (180-500 Hz)
    max_val = max(mag)
    peak_idx = mag.index(max_val)
    peak_freq = FREQS[peak_idx]
    assert 180.0 <= peak_freq <= 500.0

    # Treble above 3 kHz is completely rolled off
    idx_3k = min(range(len(FREQS)), key=lambda i: abs(FREQS[i] - 3000.0))
    assert mag[idx_3k] < 0.10


def test_series_hpf_transfer_function():
    model = load_circuit("10_rickenbacker_bridge_hpf")
    assert model.Crick == pytest.approx(4.7e-9)

    curves = compute_circuit_transfer_functions(model, freqs=FREQS)
    mag = curves[0]

    # DC should be 0 (blocked by series 4.7nF capacitor)
    assert mag[0] == 0.0

    # Upper mids / treble should pass cleanly
    idx_2k = min(range(len(FREQS)), key=lambda i: abs(FREQS[i] - 2000.0))
    assert mag[idx_2k] > 0.8


def test_parallel_dual_pickup_transfer_function():
    model = load_circuit("02_jazz_bass_pair")
    assert model.topology == "parallel"

    curves = compute_circuit_transfer_functions(model, freqs=FREQS)
    assert len(curves) == 2  # Neck and Bridge curves

    mag_n, mag_b = curves
    # Both channels under authentic dual 250k vol (125k net) + 250k tone have peak in 2.4 - 3.2 kHz
    peak_n = FREQS[mag_n.index(max(mag_n))]
    peak_b = FREQS[mag_b.index(max(mag_b))]
    assert 2400.0 <= peak_n <= 3200.0
    assert 2500.0 <= peak_b <= 3200.0


def test_active_preamp_buffer_transfer_function():
    # 1. Voice 01 Modern Active Jazz Bass
    m01 = load_circuit("01_modern_jazz_active")
    assert m01.has_active_buffer is True
    assert m01.preamp_type == "sadowsky_2band"
    assert m01.topology == "parallel"

    curves01 = compute_circuit_transfer_functions(m01, freqs=FREQS)
    assert len(curves01) == 2
    mag_n01, mag_b01 = curves01
    peak_n01 = FREQS[mag_n01.index(max(mag_n01))]
    peak_b01 = FREQS[mag_b01.index(max(mag_b01))]
    # Coils are isolated from 750pF cable load, and boosted by Sadowsky active treble shelf (6.5 - 9 kHz)
    assert 6500.0 <= peak_n01 <= 8500.0
    assert 7000.0 <= peak_b01 <= 9000.0

    # 2. Voice 07 Modern Active P/J 2-Band
    m07 = load_circuit("07_modern_pj_active")
    assert m07.has_active_buffer is True
    assert m07.preamp_type == "sadowsky_2band"
    assert m07.topology == "parallel"
    curves07 = compute_circuit_transfer_functions(m07, freqs=FREQS)
    assert len(curves07) == 2

    # 3. Voice 09 Music Man StingRay Active 2-Band
    m09 = load_circuit("09_stingray_mm_parallel")
    assert m09.has_active_buffer is True
    assert m09.preamp_type == "stingray_2band"
    assert m09.topology == "single"

    curves09 = compute_circuit_transfer_functions(m09, freqs=FREQS)
    assert len(curves09) == 1
    mag09 = curves09[0]
    peak09 = FREQS[mag09.index(max(mag09))]
    # Isolated from cable capacitance, peak is in 6.5 - 9.0 kHz clank & sizzle region
    assert 6500.0 <= peak09 <= 9000.0


def test_series_dual_pickup_transfer_function():
    model = load_circuit("11b_pmm_hybrid_series")
    assert model.topology == "series"
    assert model.has_active_buffer is True
    assert model.preamp_type == "none"

    curves = compute_circuit_transfer_functions(model, freqs=FREQS)
    assert len(curves) == 2

    mag_n, mag_b = curves
    # Both channels sum at DC with equal weight
    assert math.isclose(mag_n[0], mag_b[0], rel_tol=1e-3)
    # Active buffer isolates coils from cable capacitance, preserving high resonance (>= 2800 Hz)
    peak_n = FREQS[mag_n.index(max(mag_n))]
    assert 2800.0 <= peak_n <= 3600.0


def test_active_pmm_transfer_function():
    model = load_circuit("11_modern_pmm_active")
    assert model.topology == "parallel"
    assert model.has_active_buffer is True
    assert model.preamp_type == "none"

    curves = compute_circuit_transfer_functions(model, freqs=FREQS)
    assert len(curves) == 2

    mag_n, mag_b = curves
    # Finite DC transmission balanced between neck and bridge
    assert 0.40 < mag_n[0] < 0.95
    assert 0.50 < mag_b[0] < 0.95

    # Active buffer isolates coils from cable capacitance, preserving high resonance (>= 2800 Hz)
    peak_b = FREQS[mag_b.index(max(mag_b))]
    assert peak_b >= 2800.0


def test_tone_pot_series_admittance():
    """Verify that series Rtone allows wide-open tone pots to preserve pickup resonance."""
    # 1. Voice 05c: Rtone = 3.3 Ohm ESR floor, Ctone = 47nF -> collapses peak to 180-500 Hz
    m_rolled = load_circuit("05c_vintage_62_p_47nf")
    assert m_rolled.Rtone == pytest.approx(3.3)
    assert m_rolled.Ctone == pytest.approx(47e-9)
    curves_rolled = compute_circuit_transfer_functions(m_rolled, freqs=FREQS)
    peak_rolled = FREQS[curves_rolled[0].index(max(curves_rolled[0]))]
    assert 180.0 <= peak_rolled <= 500.0

    # 2. Source Standard P: Rtone = 250k, Ctone = 47nF -> loaded peak stays in 2000-2400 Hz range
    p_circ = INSTRUMENTS["34in_standard_p"].pickups["split_p"].circuit
    assert p_circ is not None
    m_open = load_circuit(p_circ)
    assert m_open.Rtone == pytest.approx(250000.0)
    assert m_open.Ctone == pytest.approx(47e-9)
    curves_open = compute_circuit_transfer_functions(m_open, freqs=FREQS)
    peak_open = FREQS[curves_open[0].index(max(curves_open[0]))]
    assert 2000.0 <= peak_open <= 2400.0


def test_tonestyler_p_bass_progression_transfer_functions():
    """
    Verify ToneStyler P-Bass sequence (05, 05b, 05c, 05d):
    1. Netlists parse R_tone = 3.3 ohms (zero pot wiper damping) with 22nF, 47nF, 100nF.
    2. Resonant peak physically glides down through the spectrum:
       - 05 (Tone Open): peak ~ 2128 Hz (+0.5 dB)
       - 05b (22nF): peak ~ 440 Hz (+1.5 dB)
       - 05c (47nF): peak ~ 205 Hz (-0.1 dB)
       - 05d (100nF): deep sub-bass rolloff with -3 dB cutoff at ~240 Hz
    3. Each step preserves undamped Q factor without muddy pot wiper damping.
    """
    m05 = load_circuit("05_vintage_62_p_alnico")
    m05b = load_circuit("05b_vintage_62_p_22nf")
    m05c = load_circuit("05c_vintage_62_p_47nf")
    m05d = load_circuit("05d_vintage_50s_p_100nf")

    assert m05b.Ctone == pytest.approx(22e-9)
    assert m05b.Rtone == pytest.approx(3.3)
    assert m05c.Ctone == pytest.approx(47e-9)
    assert m05c.Rtone == pytest.approx(3.3)
    assert m05d.Ctone == pytest.approx(100e-9)
    assert m05d.Rtone == pytest.approx(3.3)

    c05 = compute_circuit_transfer_functions(m05, freqs=FREQS)[0]
    c05b = compute_circuit_transfer_functions(m05b, freqs=FREQS)[0]
    c05c = compute_circuit_transfer_functions(m05c, freqs=FREQS)[0]
    c05d = compute_circuit_transfer_functions(m05d, freqs=FREQS)[0]

    # Peak frequencies glide downward
    peak_f05 = FREQS[c05.index(max(c05))]
    peak_f05b = FREQS[c05b.index(max(c05b))]
    peak_f05c = FREQS[c05c.index(max(c05c))]

    assert 1900.0 <= peak_f05 <= 2400.0
    assert 400.0 <= peak_f05b <= 480.0
    assert 180.0 <= peak_f05c <= 240.0
    assert peak_f05 > peak_f05b > peak_f05c

    # 05d has deepest cutoff: at 500 Hz, 05d is significantly more attenuated than 05c
    idx_500 = min(range(len(FREQS)), key=lambda i: abs(FREQS[i] - 500.0))
    assert c05d[idx_500] < c05c[idx_500] < c05b[idx_500]

    # Differential transfer function verification against standard P source:
    p_circ = INSTRUMENTS["34in_standard_p"].pickups["split_p"].circuit
    assert p_circ is not None
    src_p = load_circuit(p_circ)
    diff_05b = compute_differential_circuit_transfer_functions(m05b, src_p, freqs=FREQS)[0]
    diff_05d = compute_differential_circuit_transfer_functions(m05d, src_p, freqs=FREQS)[0]

    idx_440 = min(range(len(FREQS)), key=lambda i: abs(FREQS[i] - 440.0))
    db_05b_440 = 20.0 * math.log10(diff_05b[idx_440] / diff_05b[0])
    db_05d_440 = 20.0 * math.log10(diff_05d[idx_440] / diff_05d[0])

    # 05b (22nF) has resonant boost at 440 Hz (> 1.5 dB)
    assert db_05b_440 > 1.5
    # 05d (100nF) has deep rolloff at 440 Hz (< -8.0 dB)
    assert db_05d_440 < -8.0
    # 05b and 05d must be distinct and separated by > 10 dB at 440 Hz
    assert (db_05b_440 - db_05d_440) > 10.0


def test_voice_02b_transfer_function():
    """
    Verify Voice 02b (Vintage '60s Jazz Bass Pair with 22nF ToneStyler Detent):
    1. Netlist parses topology = parallel, R_tone = 3.3, C_tone = 22nF.
    2. Resonant peak occurs in the 700-850 Hz region (Jaco vocal bridge burp).
    3. Treble at 3 kHz is attenuated by > 6 dB relative to wide-open Voice 02.
    """
    m02b = load_circuit("02b_jazz_bass_pair_22nf")
    assert m02b.topology == "parallel"
    assert m02b.Ctone == pytest.approx(22e-9)
    assert m02b.Rtone == pytest.approx(3.3)

    m02 = load_circuit("02_jazz_bass_pair")

    curves_02b = compute_circuit_transfer_functions(m02b, freqs=FREQS)
    curves_02 = compute_circuit_transfer_functions(m02, freqs=FREQS)

    for ch in [0, 1]:
        peak_f = FREQS[curves_02b[ch].index(max(curves_02b[ch]))]
        assert 700.0 <= peak_f <= 850.0

        idx_3k = min(range(len(FREQS)), key=lambda i: abs(FREQS[i] - 3000.0))
        diff_3k_db = 20.0 * math.log10(curves_02b[ch][idx_3k] / curves_02[ch][idx_3k])
        assert diff_3k_db < -6.0


def test_voice_09b_series_netlist_and_transfer():
    """
    Verify Voice 09b Music Man StingRay Series netlist and transfer function:
    1. Standalone SPICE netlist parses with active buffer, 1.9x series gain, and stingray_2band preamp.
    2. Physical 4:1 impedance scaling: L_ser = 4.8H, Rdc_ser = 8.8k vs L_par = 1.2H, Rdc_par = 2.2k.
    3. Active series resonance sits at authentic ~4.1 kHz.
    4. Series connection delivers +5.6 dB output boost over parallel.
    """
    m09 = load_circuit("09_stingray_mm_parallel")
    m09b = load_circuit("09b_stingray_mm_series")

    assert m09.L == pytest.approx(1.20)
    assert m09.Rdc == pytest.approx(2200.0)
    assert m09.Reddy == pytest.approx(75000.0)
    assert m09.Ccoil == pytest.approx(180e-12)

    assert m09b.has_active_buffer is True
    assert m09b.preamp_type == "stingray_2band"
    assert m09b.preamp_gain == pytest.approx(1.9)
    assert m09b.topology == "single"
    assert m09b.L == pytest.approx(4.80)
    assert m09b.Rdc == pytest.approx(8800.0)
    assert m09b.Reddy == pytest.approx(150000.0)
    assert m09b.Ccoil == pytest.approx(210e-12)

    # Physical 4:1 series/parallel impedance scaling
    assert m09b.L / m09.L == pytest.approx(4.0)
    assert m09b.Rdc / m09.Rdc == pytest.approx(4.0)

    c09 = np.array(compute_circuit_transfer_functions(m09, freqs=FREQS)[0])
    c09b = np.array(compute_circuit_transfer_functions(m09b, freqs=FREQS)[0])

    peak_09 = FREQS[np.argmax(c09)]
    peak_09b = FREQS[np.argmax(c09b)]

    # Authentic active series resonance sits around ~4.1 kHz
    assert 3900.0 <= peak_09b <= 4300.0
    assert peak_09b < peak_09

    # Series open-circuit output gain delivers +5.0 to +6.0 dB boost over parallel across passband
    idx_100 = FREQS.index(100.0) if 100.0 in FREQS else np.argmin(np.abs(np.array(FREQS) - 100.0))
    series_boost_db = 20.0 * np.log10(c09b[idx_100] / c09[idx_100])
    assert 5.0 <= series_boost_db <= 6.0

    # Relative treble rolloff: normalized to low frequencies, series has less treble sizzle than parallel
    idx_7k = FREQS.index(7000.0) if 7000.0 in FREQS else np.argmin(np.abs(np.array(FREQS) - 7000.0))
    norm_treble_09 = c09[idx_7k] / c09[idx_100]
    norm_treble_09b = c09b[idx_7k] / c09b[idx_100]
    assert norm_treble_09b < norm_treble_09


def test_dingwall_composite_source_circuit():
    """
    Verify Vector 2: 37in_multiscale_dingwall pair_parallel declares source circuit
    and computes differential SPICE transfer functions without falling back to generic RLC.
    """
    inst = load_instrument("37in_multiscale_dingwall")
    pair_pickup = inst.pickups["pair_parallel"]
    assert pair_pickup.circuit is not None
    assert isinstance(pair_pickup.circuit, CircuitConfig)

    # Differential SPICE transfer functions evaluate cleanly
    tgt_model = load_circuit("02_jazz_bass_pair")
    src_model = load_circuit(pair_pickup.circuit)
    diff_curves = compute_differential_circuit_transfer_functions(tgt_model, src_model, freqs=FREQS)
    assert len(diff_curves) == 2
    for c in diff_curves:
        assert len(c) == len(FREQS)
        assert np.all(np.isfinite(c))
        assert np.all(np.array(c) > 0.0)


def test_inter_coil_mutual_coupling_matrix():
    """Verify coupled 2x2 nodal transfer matrix for parallel dual-coil configurations."""
    # Voice 02: Jazz Bass Pair (parallel topology)
    m = load_circuit("02_jazz_bass_pair")
    assert m.topology == "parallel"

    # 1. Zero mutual coupling (k=0, C=0)
    m.k_mutual = 0.0
    m.C_mutual = 0.0
    curves_uncoupled = compute_circuit_transfer_functions(m, freqs=FREQS)

    # 2. Authentic mutual coupling (k=0.05, C=20pF)
    m.k_mutual = 0.05
    m.C_mutual = 20e-12
    curves_coupled = compute_circuit_transfer_functions(m, freqs=FREQS)

    assert len(curves_coupled) == 2
    # Check that coupled responses are smooth and well-behaved
    for ch in range(2):
        uncoupled_ch = np.asarray(curves_uncoupled[ch])
        coupled_ch = np.asarray(curves_coupled[ch])
        ratio_db = 20.0 * np.log10(coupled_ch / uncoupled_ch)
        # Subtle acoustic coupling within +/- 2.5 dB
        assert np.all(np.abs(ratio_db) < 2.5), f"Coupling on ch {ch} must be realistic and bounded"
        assert not np.any(np.isnan(coupled_ch)), "Coupled matrix must not produce NaNs"


def test_distributed_coil_transmission_line():
    """Verify distributed coil admittance softens the lumped LC cliff and preserves high-end sheen."""
    m = load_circuit("04_modern_p_ceramic")

    # 1. Lumped model (k_dist = 0.0)
    m.k_dist = 0.0
    lumped_curve = np.asarray(compute_circuit_transfer_functions(m, freqs=FREQS)[0])

    # 2. Distributed transmission line model (k_dist = 0.035)
    m.k_dist = 0.035
    dist_curve = np.asarray(compute_circuit_transfer_functions(m, freqs=FREQS)[0])

    assert not np.any(np.isnan(dist_curve)), "Distributed curve must not contain NaNs"
    assert np.all(dist_curve > 0.0), "Distributed curve must be strictly positive"

    # In the high-frequency band (8 kHz to 18 kHz), distributed factor smooths the impedance
    freqs_arr = np.asarray(FREQS)
    hf_idx = np.where((freqs_arr >= 8000.0) & (freqs_arr <= 18000.0))[0]
    # Ratio between distributed and lumped should be smooth and bounded within +/- 3 dB
    ratio_db = 20.0 * np.log10(dist_curve[hf_idx] / lumped_curve[hf_idx])
    assert np.all(np.abs(ratio_db) < 3.0), (
        "Distributed transmission factor must be bounded and physically realistic"
    )


def test_potentiometer_wiper_positions():
    """Verify dynamic Volume and Tone pot wiper positions and cable interaction."""
    # 1. 100% open matches default baseline bit-exact
    m_default = load_circuit("05_vintage_62_p_alnico")
    m_open = load_circuit("05_vintage_62_p_alnico")
    m_open.apply_pot_positions(vol_pos=1.0, tone_pos=1.0)

    h_def = compute_circuit_transfer_functions(m_default, freqs=FREQS)[0]
    h_open = compute_circuit_transfer_functions(m_open, freqs=FREQS)[0]
    assert np.allclose(h_def, h_open, atol=1e-6), (
        "Wiper at 1.0, 1.0 must match default netlist exactly"
    )

    # 2. Tone rolled off (tone_pos = 0.2) increases roll-off around 1-3 kHz
    m_tone_rolled = load_circuit("05_vintage_62_p_alnico")
    m_tone_rolled.apply_pot_positions(vol_pos=1.0, tone_pos=0.2)
    h_rolled = compute_circuit_transfer_functions(m_tone_rolled, freqs=FREQS)[0]

    freqs_arr = np.asarray(FREQS)
    idx_3k = np.argmin(np.abs(freqs_arr - 3000.0))
    assert h_rolled[idx_3k] < h_open[idx_3k], "Tone pot rolled off must attenuate 3 kHz resonance"

    # 3. Volume rolled off (vol_pos = 0.7) inserts series resistance loading cable capacitance
    m_vol_rolled = load_circuit("05_vintage_62_p_alnico")
    m_vol_rolled.apply_pot_positions(vol_pos=0.7, tone_pos=1.0)
    h_vol = compute_circuit_transfer_functions(m_vol_rolled, freqs=FREQS)[0]
    assert np.max(h_vol) < np.max(h_open), "Volume attenuation must reduce overall output gain"


def test_solid_pole_eddy_skin_dispersion():
    """Verify solid Alnico pole eddy skin-effect fractional dispersion (sqrt(omega)) roll-off vs flat Ceramic."""
    f_arr = np.asarray(FREQS)

    # 1. Direct impedance check: at DC, Z_skin is identically 0.0
    z_skin_dc = compute_core_impedance(
        0.0, L=4.0, k_skin=0.10, omega_skin=2.0 * math.pi * 3200.0, Rdc=9000.0
    )
    assert abs(z_skin_dc) == 0.0, "Skin impedance at DC must be exactly 0.0"

    # 2. Circuit model: Alnico V (k_skin=0.10, f_skin=3200) vs Ceramic (k_skin=0.0)
    m_alnico = CircuitModel()
    m_alnico.k_skin = 0.10
    m_alnico.f_skin = 3200.0

    m_ceramic = CircuitModel()
    m_ceramic.k_skin = 0.0
    m_ceramic.f_skin = 0.0

    h_alnico = np.asarray(compute_circuit_transfer_functions(m_alnico, freqs=FREQS)[0])
    h_ceramic = np.asarray(compute_circuit_transfer_functions(m_ceramic, freqs=FREQS)[0])

    diff_db = 20.0 * np.log10(np.maximum(h_alnico / h_ceramic, 1e-6))

    # At low frequencies (50-200 Hz), diff must be negligible (< 0.01 dB)
    low_mask = (f_arr >= 50.0) & (f_arr <= 200.0)
    assert np.all(np.abs(diff_db[low_mask]) < 0.01), "Low frequency skin effect must be negligible"

    # Above 2 kHz, Alnico exhibits fractional roll-off without peaking
    idx_3k = np.argmin(np.abs(f_arr - 3000.0))
    assert diff_db[idx_3k] <= 0.0, "Alnico skin effect must damp 3 kHz resonance"
    assert np.all(diff_db <= 0.05), (
        "Skin effect must never cause un-damped high-frequency resonance boost"
    )


def test_generic_analog_preamp_bands():
    """Verify that evaluate_analog_band and compute_active_preamp_transfer evaluate continuous s-domain filters."""
    from allomorph.circuit.solver import compute_active_preamp_transfer, evaluate_analog_band
    from allomorph.config.schema import PreampBandConfig

    # 1. Low shelf boost: +4.0 dB @ 60 Hz
    s_dc = 1j * 2.0 * math.pi * 1e-4
    s_hf = 1j * 2.0 * math.pi * 10000.0
    band_low = PreampBandConfig(type="low_shelf", freq_hz=60.0, gain_db=4.0)

    h_dc = abs(evaluate_analog_band(band_low, s_dc))
    h_hf = abs(evaluate_analog_band(band_low, s_hf))

    expected_boost = 10.0 ** (4.0 / 20.0)
    assert h_dc == pytest.approx(expected_boost, rel=1e-3)
    assert h_hf == pytest.approx(1.0, rel=1e-3)

    # 2. High shelf boost: +3.0 dB @ 4000 Hz
    band_high = PreampBandConfig(type="high_shelf", freq_hz=4000.0, gain_db=3.0)
    s_inf = 1j * 2.0 * math.pi * 1e7
    h_high_dc = abs(evaluate_analog_band(band_high, s_dc))
    h_high_inf = abs(evaluate_analog_band(band_high, s_inf))

    expected_treble = 10.0 ** (3.0 / 20.0)
    assert h_high_dc == pytest.approx(1.0, rel=1e-3)
    assert h_high_inf == pytest.approx(expected_treble, rel=1e-3)

    # 3. Composite preamp transfer
    h_comp = compute_active_preamp_transfer([band_low, band_high], np.array([s_dc, s_inf]))
    assert abs(h_comp[0]) == pytest.approx(expected_boost, rel=1e-3)
    assert abs(h_comp[1]) == pytest.approx(expected_treble, rel=1e-3)


def test_smooth_soft_knee_db_cinf_properties():
    """Verify that smooth_soft_knee_db is strictly C^inf smooth, monotonic, and linearly transparent."""
    from allomorph.circuit.solver import smooth_soft_knee_db

    # 1. Scalar and vector type safety
    s_val = smooth_soft_knee_db(5.0, thresh=6.0, ceiling=8.0)
    assert isinstance(s_val, float)
    assert s_val == pytest.approx(5.0, abs=1e-4)

    arr = np.array([0.0, 3.0, 5.0, 6.0, 6.91, 8.0, 20.0])
    v_val = smooth_soft_knee_db(arr, thresh=6.0, ceiling=8.0)
    assert isinstance(v_val, np.ndarray)

    # 2. Linear passband transparency below threshold (<= 5.0 dB)
    for x in [-20.0, -6.0, 0.0, 2.0, 4.0]:
        y = smooth_soft_knee_db(x, thresh=6.0, ceiling=8.0)
        assert abs(y - x) < 1e-4, f"Passband linearity violated at {x} dB: diff was {abs(y-x):.6f} dB"

    # 3. Threshold and deconvolution headroom
    y_thresh = smooth_soft_knee_db(6.0, thresh=6.0, ceiling=8.0)
    assert abs(y_thresh - 6.0) < 0.01

    y_p_peak = smooth_soft_knee_db(6.91, thresh=6.0, ceiling=8.0)
    assert 6.75 <= y_p_peak <= 6.91, f"Expected P-Bass peak to pass unclipped: got {y_p_peak:.3f} dB"

    y_ceil = smooth_soft_knee_db(50.0, thresh=6.0, ceiling=8.0)
    assert y_ceil == pytest.approx(8.0, abs=1e-4)

    # 4. Strict monotonicity across wide dynamic range: dy/dx > 0
    xs = np.linspace(-10.0, 20.0, 3001)
    ys = smooth_soft_knee_db(xs, thresh=6.0, ceiling=8.0)
    diffs = np.diff(ys)
    assert np.all(diffs > 0.0), "Monotonicity violated: dy/dx must be strictly positive"

    # 5. C^inf smoothness across knee: continuous 1st, 2nd, and 3rd derivatives without kinks
    dy_dx = np.gradient(ys, xs)
    d2y_dx2 = np.gradient(dy_dx, xs)
    d3y_dx3 = np.gradient(d2y_dx2, xs)

    assert not np.any(np.isnan(dy_dx))
    assert not np.any(np.isnan(d2y_dx2))
    assert not np.any(np.isnan(d3y_dx3))
    assert np.all(dy_dx <= 1.00001)
    assert np.all(dy_dx > 0.0)
    # Second derivative must be continuous and bounded everywhere
    assert np.max(np.abs(d2y_dx2)) < 1.0

