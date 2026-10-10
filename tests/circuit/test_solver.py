"""
Tests for analytical SPICE nodal RLC solver, potentiometer wiper math,
active preamp buffers, mutual coupling matrix, and core dispersion.
"""

import math

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from allomorph.circuit import (
    compute_core_impedance,
    resolve_target_voicing,
    solve_mna_harness,
)
from allomorph.config import load_instrument
from allomorph.dsp import FREQS
from tests.strategies import st_pot_wipers


def test_single_pickup_transfer_function():
    inst, v = resolve_target_voicing("precision_vintage")
    harness = inst.harnesses[v.harness]
    curves = solve_mna_harness(inst, harness, v, freqs=FREQS)

    assert len(curves) >= 1
    mag = next(iter(curves.values()))

    # DC Gain should be near 0 dB (~0.97 due to pot + load divider)
    assert 0.90 < mag[0] < 1.0

    # Resonant peak should occur between 1850 Hz and 2300 Hz
    max_val = max(mag)
    peak_idx = int(np.argmax(mag))
    peak_freq = FREQS[peak_idx]
    assert 1850.0 <= peak_freq <= 2300.0
    assert max_val > 1.0  # Under CTS 250k pot/cable load, Q peak > 1.0

    # High-frequency rolloff (at 20 kHz, gain should be < 0.20)
    assert mag[-1] < 0.20


def test_tone_rolloff_transfer_function():
    inst, v = resolve_target_voicing("precision_warm")
    harness = inst.harnesses[v.harness]
    curves = solve_mna_harness(inst, harness, v, freqs=FREQS)
    mag = next(iter(curves.values()))

    # With 47nF shunt, resonant peak collapses into low-mids (180-500 Hz)
    max(mag)
    peak_idx = int(np.argmax(mag))
    peak_freq = FREQS[peak_idx]
    assert 180.0 <= peak_freq <= 500.0

    # Treble above 3 kHz is completely rolled off
    idx_3k = min(range(len(FREQS)), key=lambda i: abs(FREQS[i] - 3000.0))
    assert mag[idx_3k] < 0.10


def test_series_hpf_transfer_function():
    inst, v = resolve_target_voicing("rickenbacker_clank")
    harness = inst.harnesses[v.harness]
    curves = solve_mna_harness(inst, harness, v, freqs=FREQS)
    mag = curves.get("bridge.bridge", curves.get("bridge"))
    assert mag is not None

    # DC should be 0 (blocked by series 4.7nF capacitor)
    assert mag[0] == 0.0

    # Upper mids / treble should pass cleanly
    idx_2k = min(range(len(FREQS)), key=lambda i: abs(FREQS[i] - 2000.0))
    assert mag[idx_2k] > 0.8


def test_parallel_dual_pickup_transfer_function():
    inst, v = resolve_target_voicing("jazz_pair_open")
    harness = inst.harnesses[v.harness]
    curves = solve_mna_harness(inst, harness, v, freqs=FREQS)
    assert "neck" in curves and "bridge" in curves

    mag_n = curves["neck"]
    mag_b = curves["bridge"]
    # Both channels under authentic dual 250k vol (125k net) + 250k tone have peak in 2.4 - 3.2 kHz
    peak_n = FREQS[int(np.argmax(mag_n))]
    peak_b = FREQS[int(np.argmax(mag_b))]
    assert 2400.0 <= peak_n <= 3200.0
    assert 2500.0 <= peak_b <= 3200.0


def test_active_preamp_buffer_transfer_function():
    # 1. Voice 01 Modern Active Jazz Bass
    inst01, v01 = resolve_target_voicing("jazz_pair_active")
    curves01 = solve_mna_harness(
        inst01, inst01.harnesses[v01.harness], v01, freqs=FREQS
    )
    assert "neck" in curves01 and "bridge" in curves01
    mag_n01 = curves01["neck"]
    mag_b01 = curves01["bridge"]
    peak_n01 = FREQS[int(np.argmax(mag_n01))]
    peak_b01 = FREQS[int(np.argmax(mag_b01))]
    # Coils are isolated from 750pF cable load, and boosted by Sadowsky active treble shelf (6.5 - 10 kHz)
    assert 6500.0 <= peak_n01 <= 9000.0
    assert 7000.0 <= peak_b01 <= 10000.0

    # 2. Voice 07 Modern Active P/J 2-Band
    inst07, v07 = resolve_target_voicing("pj_active")
    curves07 = solve_mna_harness(
        inst07, inst07.harnesses[v07.harness], v07, freqs=FREQS
    )
    assert len(curves07) >= 2

    # 3. Voice 09 Music Man StingRay Active 2-Band
    inst09, v09 = resolve_target_voicing("stingray_parallel")
    curves09 = solve_mna_harness(
        inst09, inst09.harnesses[v09.harness], v09, freqs=FREQS
    )
    mag09 = curves09.get("mm_humbucker", next(iter(curves09.values())))
    peak09 = FREQS[int(np.argmax(mag09))]
    # Isolated from cable capacitance, peak is in 5.5 - 9.0 kHz clank & sizzle region
    assert 5500.0 <= peak09 <= 9000.0


def test_series_dual_pickup_transfer_function():
    inst, v = resolve_target_voicing("p_mm_series")
    curves = solve_mna_harness(inst, inst.harnesses[v.harness], v, freqs=FREQS)
    assert "split_p" in curves
    assert "mm" in curves
    mag_n = curves["split_p.forward"]
    mag_b = curves["mm"]

    # Both channels sum at DC with equal weight per string pair
    assert math.isclose(mag_n[0], mag_b[0], rel_tol=1e-3)
    # Output rolls off smoothly at high frequencies (at 20 kHz, < 20% of DC)
    assert mag_n[-1] < mag_n[0] * 0.20
    assert mag_b[-1] < mag_b[0] * 0.20


def test_active_pmm_transfer_function():
    inst, v = resolve_target_voicing("p_mm_parallel")
    curves = solve_mna_harness(inst, inst.harnesses[v.harness], v, freqs=FREQS)
    mag_n = curves.get("split_p", curves.get("split", next(iter(curves.values()))))
    mag_b = curves.get("mm", curves.get("bridge", list(curves.values())[1]))

    # Finite DC transmission balanced between neck and bridge
    assert 0.40 < mag_n[0] < 0.95
    assert 0.50 < mag_b[0] < 0.95

    # Active buffer isolates coils from cable capacitance, preserving high resonance (>= 2800 Hz)
    peak_b = FREQS[int(np.argmax(mag_b))]
    assert peak_b >= 2800.0


def test_tone_pot_series_admittance():
    """Verify that series Rtone allows wide-open tone pots to preserve pickup resonance."""
    # 1. precision_warm: tone rolled off -> collapses peak to 180-500 Hz
    inst_rolled, v_rolled = resolve_target_voicing("precision_warm")
    c_rolled = solve_mna_harness(inst_rolled, inst_rolled.harnesses[v_rolled.harness], v_rolled, freqs=FREQS)
    mag_rolled = next(iter(c_rolled.values()))
    peak_rolled = FREQS[int(np.argmax(mag_rolled))]
    assert 180.0 <= peak_rolled <= 500.0

    # 2. precision_vintage: tone open -> loaded peak stays in 2000-2400 Hz range
    inst_open, v_open = resolve_target_voicing("precision_vintage")
    c_open = solve_mna_harness(inst_open, inst_open.harnesses[v_open.harness], v_open, freqs=FREQS)
    mag_open = next(iter(c_open.values()))
    peak_open = FREQS[int(np.argmax(mag_open))]
    assert 2000.0 <= peak_open <= 2400.0


def test_tonestyler_p_bass_progression_transfer_functions():
    """
    Verify P-Bass tone sequence (vintage, mids, warm, dub):
    Resonant peak physically glides down through the spectrum.
    """
    inst05, v05 = resolve_target_voicing("precision_vintage")
    inst05b, v05b = resolve_target_voicing("precision_mids")
    inst05c, v05c = resolve_target_voicing("precision_warm")
    inst05d, v05d = resolve_target_voicing("precision_dub")

    c05 = next(iter(solve_mna_harness(inst05, inst05.harnesses[v05.harness], v05, freqs=FREQS).values()))
    c05b = next(iter(solve_mna_harness(inst05b, inst05b.harnesses[v05b.harness], v05b, freqs=FREQS).values()))
    c05c = next(iter(solve_mna_harness(inst05c, inst05c.harnesses[v05c.harness], v05c, freqs=FREQS).values()))
    c05d = next(iter(solve_mna_harness(inst05d, inst05d.harnesses[v05d.harness], v05d, freqs=FREQS).values()))

    # Peak frequencies glide downward
    peak_f05 = FREQS[int(np.argmax(c05))]
    peak_f05b = FREQS[int(np.argmax(c05b))]
    peak_f05c = FREQS[int(np.argmax(c05c))]

    assert 1900.0 <= peak_f05 <= 2400.0
    assert 400.0 <= peak_f05b <= 480.0
    assert 180.0 <= peak_f05c <= 240.0
    assert peak_f05 > peak_f05b > peak_f05c

    # dub has deepest cutoff: at 500 Hz, dub is significantly more attenuated than warm
    idx_500 = min(range(len(FREQS)), key=lambda i: abs(FREQS[i] - 500.0))
    assert c05d[idx_500] < c05c[idx_500]

    # Ratio verification against open source:
    diff_05d = c05d / c05
    idx_440 = min(range(len(FREQS)), key=lambda i: abs(FREQS[i] - 440.0))
    db_05d_440 = 20.0 * math.log10(diff_05d[idx_440] / diff_05d[0])

    # dub (100nF) has deep rolloff at 440 Hz (< -8.0 dB)
    assert db_05d_440 < -8.0


def test_voice_02b_transfer_function():
    """Verify Voice 02b (60s Jazz Bass Pair with 22nF Tone Cap)."""
    inst, v = resolve_target_voicing("jazz_pair_mids")
    assert v.harness == "passive"
    assert v.components.get("controls.tone.cap") == pytest.approx(2.2e-8)
    curves = solve_mna_harness(inst, inst.harnesses[v.harness], v, freqs=FREQS)
    assert "neck" in curves and "bridge" in curves


def test_voice_09b_series_netlist_and_transfer():
    """
    Verify Voice 09b Music Man StingRay Series netlist and transfer function:
    1. Active series resonance sits lower than parallel resonance.
    2. Series connection delivers output boost over parallel.
    """
    inst09, v09 = resolve_target_voicing("stingray_parallel")
    inst09b, v09b = resolve_target_voicing("stingray_series")

    c09_dict = solve_mna_harness(inst09, inst09.harnesses[v09.harness], v09, freqs=FREQS)
    c09b_dict = solve_mna_harness(inst09b, inst09b.harnesses[v09b.harness], v09b, freqs=FREQS)

    c09 = c09_dict.get("mm_humbucker", next(iter(c09_dict.values())))
    c09b = c09b_dict.get("mm_humbucker", next(iter(c09b_dict.values())))

    peak_09 = FREQS[int(np.argmax(c09))]
    peak_09b = FREQS[int(np.argmax(c09b))]

    assert peak_09b < peak_09

    # Series open-circuit output gain delivers +5.0 to +6.5 dB boost over parallel across passband
    idx_100 = int(np.argmin(np.abs(np.array(FREQS) - 100.0)))
    series_boost_db = 20.0 * np.log10(c09b[idx_100] / c09[idx_100])
    assert 5.0 <= series_boost_db <= 6.5

    # Relative treble rolloff: normalized to low frequencies, series has less treble sizzle than parallel
    idx_7k = int(np.argmin(np.abs(np.array(FREQS) - 7000.0)))
    norm_treble_09 = c09[idx_7k] / c09[idx_100]
    norm_treble_09b = c09b[idx_7k] / c09b[idx_100]
    assert norm_treble_09b < norm_treble_09


def test_dingwall_composite_source_circuit():
    """
    Verify Vector 2: 37in_multiscale_dingwall pair_parallel declares source circuit
    and evaluates circuit transfer functions cleanly without falling back to generic RLC.
    """
    inst = load_instrument("37in_multiscale_dingwall")
    assert "pair_parallel" in inst.voicings
    v = inst.voicings["pair_parallel"]
    from allomorph.config.voices import voicing_to_voice_config

    vcfg = voicing_to_voice_config(inst, v)
    assert vcfg.id == "dingwall_parallel"

    # MNA transfer functions evaluate cleanly across all active branches
    from allomorph.circuit.solver import solve_mna_harness

    curves = solve_mna_harness(inst, inst.harnesses[v.harness], v, freqs=FREQS)
    assert "middle" in curves and "bridge" in curves
    for c in curves.values():
        assert len(c) == len(FREQS)
        assert np.all(np.isfinite(c))
        assert np.all(c > 0.0)


def test_inter_coil_mutual_coupling_matrix():
    """Verify coupled 2x2 nodal transfer matrix for parallel dual-coil configurations."""
    inst, v = resolve_target_voicing("jazz_pair_open")
    harness = inst.harnesses[v.harness]

    # 1. Zero mutual coupling (k=0)
    h_uncoupled = harness.model_copy(update={"k_mutual": 0.0})
    curves_uncoupled = solve_mna_harness(inst, h_uncoupled, v, freqs=FREQS)

    # 2. Authentic mutual coupling (k=0.05)
    h_coupled = harness.model_copy(update={"k_mutual": 0.05})
    curves_coupled = solve_mna_harness(inst, h_coupled, v, freqs=FREQS)

    assert "neck" in curves_coupled and "bridge" in curves_coupled
    for ch in ("neck", "bridge"):
        uncoupled_ch = curves_uncoupled[ch]
        coupled_ch = curves_coupled[ch]
        ratio_db = 20.0 * np.log10(coupled_ch / uncoupled_ch)
        assert np.all(np.abs(ratio_db) < 2.5), f"Coupling on ch {ch} must be realistic and bounded"
        assert not np.any(np.isnan(coupled_ch)), "Coupled matrix must not produce NaNs"


def test_potentiometer_wiper_positions():
    """Verify dynamic Volume and Tone pot wiper positions and cable interaction via MNA."""
    inst, v = resolve_target_voicing("precision_vintage")
    harness = inst.harnesses[v.harness]

    # 1. 100% open matches default baseline bit-exact
    h_def = solve_mna_harness(inst, harness, v, freqs=FREQS)
    v_open = v.model_copy(update={"controls": {**v.controls, "vol": 1.0, "tone": 1.0}})
    h_open = solve_mna_harness(inst, harness, v_open, freqs=FREQS)
    curve_def = next(iter(h_def.values()))
    curve_open = next(iter(h_open.values()))
    assert np.allclose(curve_def, curve_open, atol=1e-6)

    # 2. Tone rolled off (tone = 0.2) attenuates 3 kHz resonance
    v_rolled = v.model_copy(update={"controls": {**v.controls, "vol": 1.0, "tone": 0.2}})
    h_rolled = solve_mna_harness(inst, harness, v_rolled, freqs=FREQS)
    curve_rolled = next(iter(h_rolled.values()))
    freqs_arr = np.asarray(FREQS)
    idx_3k = np.argmin(np.abs(freqs_arr - 3000.0))
    assert curve_rolled[idx_3k] < curve_open[idx_3k]

    # 3. Volume rolled off (vol = 0.7) inserts series resistance
    v_vol = v.model_copy(update={"controls": {**v.controls, "vol": 0.7, "tone": 1.0}})
    h_vol = solve_mna_harness(inst, harness, v_vol, freqs=FREQS)
    curve_vol = next(iter(h_vol.values()))
    assert np.max(curve_vol) < np.max(curve_open)


def test_solid_pole_eddy_skin_dispersion():
    """Verify solid Alnico pole eddy skin-effect fractional dispersion (sqrt(omega)) roll-off vs flat Ceramic."""
    f_arr = np.asarray(FREQS)

    # 1. Direct impedance check: at DC, Z_skin is identically 0.0
    z_skin_dc = compute_core_impedance(
        0.0, L=4.0, k_skin=0.10, omega_skin=2.0 * math.pi * 3200.0, Rdc=9000.0
    )
    assert abs(z_skin_dc) == 0.0, "Skin impedance at DC must be exactly 0.0"

    # 2. At audio frequencies, Alnico V exhibits fractional skin resistance
    s_arr = 1j * 2.0 * math.pi * f_arr
    z_alnico = np.asarray(
        compute_core_impedance(
            s_arr, L=4.0, k_skin=0.10, omega_skin=2.0 * math.pi * 3200.0, Rdc=9000.0
        ),
        dtype=np.complex128,
    )
    z_ceramic = np.asarray(
        compute_core_impedance(
            s_arr, L=4.0, k_skin=0.0, omega_skin=0.0, Rdc=9000.0
        ),
        dtype=np.complex128,
    )

    # At low frequencies (50-200 Hz), difference in skin impedance is negligible
    low_mask = (f_arr >= 50.0) & (f_arr <= 200.0)
    assert np.all(np.abs(z_alnico[low_mask] - z_ceramic[low_mask]) < 50.0)

    # Above 2 kHz, Alnico exhibits fractional skin resistance Re(Z_skin) > 0
    idx_3k = int(np.argmin(np.abs(f_arr - 3000.0)))
    assert z_alnico[idx_3k].real > z_ceramic[idx_3k].real + 50.0


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
        assert abs(y - x) < 1e-4, (
            f"Passband linearity violated at {x} dB: diff was {abs(y - x):.6f} dB"
        )

    # 3. Threshold and deconvolution headroom
    y_thresh = smooth_soft_knee_db(6.0, thresh=6.0, ceiling=8.0)
    assert abs(y_thresh - 6.0) < 0.01

    y_p_peak = smooth_soft_knee_db(6.91, thresh=6.0, ceiling=8.0)
    assert 6.75 <= y_p_peak <= 6.91, (
        f"Expected P-Bass peak to pass unclipped: got {y_p_peak:.3f} dB"
    )

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


def test_compute_active_preamp_biquads():
    """Verify bilinear transform biquad coefficient generation for active preamp bands."""
    from allomorph.circuit.solver import compute_active_preamp_biquads
    from allomorph.config.schema import PreampBandConfig

    bands = [
        PreampBandConfig(type="low_shelf", freq_hz=100.0, gain_db=6.0),
        PreampBandConfig(type="high_shelf", freq_hz=4000.0, gain_db=-4.0),
        PreampBandConfig(type="bell", freq_hz=1000.0, gain_db=3.0, q=1.5),
        PreampBandConfig(type="bell", freq_hz=800.0, gain_db=0.0),  # Should be bypassed
        PreampBandConfig(type="low_pass", freq_hz=650.0, gain_db=0.0, q=1.6),  # 2nd order resonant
        PreampBandConfig(type="low_pass", freq_hz=2000.0, gain_db=0.0),  # 1st order
        PreampBandConfig(type="high_pass", freq_hz=80.0, gain_db=0.0, q=0.707),  # 2nd order
        PreampBandConfig(type="high_pass", freq_hz=30.0, gain_db=0.0),  # 1st order
    ]

    biquads = compute_active_preamp_biquads(bands, fs=48000.0)
    # Flat band is bypassed, so 7 biquads generated
    assert len(biquads) == 7
    for b0, b1, b2, a0, a1, a2 in biquads:
        assert a0 == 1.0
        assert all(math.isfinite(val) for val in (b0, b1, b2, a1, a2))


def test_evaluate_analog_band_types():
    """Verify continuous s-domain evaluation for low-pass, high-pass, and flat bands."""
    from allomorph.circuit.solver import evaluate_analog_band
    from allomorph.config.schema import PreampBandConfig

    s = 2.0 * np.pi * 1000.0 * 1j

    # Low pass (1st order)
    lp_band = PreampBandConfig(type="low_pass", freq_hz=500.0, gain_db=0.0)
    h_lp = evaluate_analog_band(lp_band, s)
    assert abs(h_lp) < 1.0

    # Low pass (2nd order resonant bump with Q)
    lp_res_band = PreampBandConfig(type="low_pass", freq_hz=1000.0, gain_db=0.0, q=2.0)
    h_lp_res = evaluate_analog_band(lp_res_band, s)
    assert abs(abs(h_lp_res) - 2.0) < 1e-6  # Peak resonance at f0 equals Q

    # High pass (1st order)
    hp_band = PreampBandConfig(type="high_pass", freq_hz=2000.0, gain_db=0.0)
    h_hp = evaluate_analog_band(hp_band, s)
    assert abs(h_hp) < 1.0

    # High pass (2nd order resonant bump with Q)
    hp_res_band = PreampBandConfig(type="high_pass", freq_hz=1000.0, gain_db=0.0, q=2.0)
    h_hp_res = evaluate_analog_band(hp_res_band, s)
    assert abs(abs(h_hp_res) - 2.0) < 1e-6  # Peak resonance at f0 equals Q

    # Flat gain (bypassed)
    flat_band = PreampBandConfig(type="bell", freq_hz=1000.0, gain_db=0.0)
    h_flat = evaluate_analog_band(flat_band, s)
    assert abs(h_flat - 1.0) < 1e-6


def test_compute_active_preamp_eq_variants():
    """Verify compute_active_preamp_eq with PreampConfig, list of bands, and preset strings."""
    from allomorph.circuit.solver import compute_active_preamp_eq
    from allomorph.config.schema import PreampBandConfig, PreampConfig

    s = 2.0 * np.pi * 1000.0 * 1j

    # 1. Preset name
    h_preset = compute_active_preamp_eq("sadowsky_2band", s)
    assert isinstance(h_preset, (np.ndarray, complex))

    # 2. PreampConfig object
    cfg = PreampConfig(
        id="custom_pre",
        name="Custom Preamp",
        description="Custom preamp description",
        input_impedance_meg=1.0,
        output_impedance_ohm=100.0,
        gain_db=3.0,
        bands=[PreampBandConfig(type="low_shelf", freq_hz=100.0, gain_db=4.0)],
    )
    h_cfg = compute_active_preamp_eq(cfg, s)
    assert np.all(np.isfinite(h_cfg))

    # 3. List of PreampBandConfig
    h_list = compute_active_preamp_eq(cfg.bands, s)
    assert np.all(np.isfinite(h_list))


def test_compute_core_impedance_jacobians_branches():
    """Verify core impedance sensitivity Jacobians across chi_mu and k_skin branches."""
    from allomorph.circuit.solver import compute_core_impedance_jacobians

    s = 2.0 * np.pi * 1000.0 * 1j

    # Both chi_mu > 0 and k_skin > 0
    jac = compute_core_impedance_jacobians(
        s,
        L=4.0,
        L_core=1.5,
        R_core=80000.0,
        chi_mu=0.15,
        k_skin=0.08,
        omega_skin=2.0 * math.pi * 3200.0,
    )
    assert "dZ_dL" in jac
    assert "dZ_dchi_mu" in jac
    assert "dZ_dk_skin" in jac
    assert abs(jac["dZ_dchi_mu"]) > 0.0
    assert abs(jac["dZ_dk_skin"]) > 0.0

    # Zero core diffusion
    jac_zero = compute_core_impedance_jacobians(s, L=4.0, chi_mu=0.0, k_skin=0.0)
    assert jac_zero["dZ_dchi_mu"] == 0.0
    assert jac_zero["dZ_dk_skin"] == 0.0


def test_series_topology_and_blend_pot():
    """Verify series circuit topology transfer functions and blend potentiometer current division via MNA."""
    inst, v_series = resolve_target_voicing("p_mm_series")
    harness = inst.harnesses[v_series.harness]

    # Center detent: both pickups present
    v_mid = v_series.model_copy(update={"controls": {**v_series.controls, "blend": 0.5}})
    curves_mid = solve_mna_harness(
        inst, harness, v_mid, freqs=np.array([100.0, 1000.0, 3000.0])
    )
    assert len(curves_mid) >= 2
    for h in curves_mid.values():
        assert np.all(np.isfinite(h))

    # Neck favored (blend = 0.2 < 0.5)
    v_neck = v_series.model_copy(update={"controls": {**v_series.controls, "blend": 0.2}})
    curves_neck = solve_mna_harness(
        inst, harness, v_neck, freqs=np.array([1000.0])
    )
    # Bridge favored (blend = 0.8 > 0.5)
    v_bridge = v_series.model_copy(update={"controls": {**v_series.controls, "blend": 0.8}})
    curves_bridge = solve_mna_harness(
        inst, harness, v_bridge, freqs=np.array([1000.0])
    )
    mag_n_fav = curves_neck.get("split_p", next(iter(curves_neck.values())))
    mag_b_fav = curves_bridge.get("mm", list(curves_bridge.values())[-1])
    assert np.all(np.isfinite(mag_n_fav))
    assert np.all(np.isfinite(mag_b_fav))


def test_eval_pot_taper_curves():
    """Verify evaluation of audio, linear, and reverse-audio potentiometer tapers."""
    from allomorph.circuit.solver import eval_pot_taper

    # Endpoints
    assert math.isclose(eval_pot_taper(0.0, "audio"), 0.0, abs_tol=1e-12)
    assert math.isclose(eval_pot_taper(1.0, "audio"), 1.0, abs_tol=1e-12)
    assert math.isclose(eval_pot_taper(0.0, "linear"), 0.0, abs_tol=1e-12)
    assert math.isclose(eval_pot_taper(1.0, "linear"), 1.0, abs_tol=1e-12)

    # Audio taper at midpoint ~ 10-15% resistance
    mid_audio = eval_pot_taper(0.5, "audio")
    assert 0.05 < mid_audio < 0.25

    # Linear taper at midpoint == 50%
    mid_linear = eval_pot_taper(0.5, "linear")
    assert math.isclose(mid_linear, 0.5, abs_tol=1e-4)

    # Reverse audio taper at midpoint ~ 85-90% resistance
    mid_rev = eval_pot_taper(0.5, "reverse_audio")
    assert 0.75 < mid_rev < 0.95


@given(
    st_pot_wipers(),
    st.sampled_from(["audio", "linear", "reverse_audio", "audio10", "audio15", "mn_blend"]),
)
def test_eval_pot_taper_property(pos: float, taper: str) -> None:
    """Property test verifying potentiometer tapers are smoothly bounded in [0.0, 1.0]."""
    from allomorph.circuit.solver import eval_pot_taper

    val = float(eval_pot_taper(pos, taper))
    assert 0.0 <= val <= 1.0
    if pos == 0.0:
        assert math.isclose(val, 0.0, abs_tol=1e-9)
    elif pos == 1.0:
        assert math.isclose(val, 1.0, abs_tol=1e-9)


# ==============================================================================
# MNA NODAL HARNESS SOLVER TESTS
# ==============================================================================


def test_mna_precision_bass_open_and_warm():
    """Verify 1V/1T Precision Bass solves vintage open resonance (~2.0 kHz) and warm tone roll-off (~180 Hz)."""
    import time

    from allomorph.circuit.solver import solve_mna_harness
    from allomorph.config.schema import (
        CoilConfig,
        ControlElementConfig,
        HarnessConfig,
        InstrumentConfig,
        PickupConfig,
        VoicingConfig,
    )

    p_bass = InstrumentConfig(
        id="test_p",
        name="Test Precision",
        scale_length_in=34.0,
        pickups={
            "p": PickupConfig(
                name="Precision Split-Coil",
                type="split_coil",
                coils=[
                    CoilConfig(
                        id="split",
                        position_from_bridge_m=0.125,
                        L=3.8,
                        Rdc=10500.0,
                        Reddy=180000.0,
                        Ccoil=60e-12,
                    )
                ],
            )
        },
        harnesses={
            "passive": HarnessConfig(
                name="1V/1T Passive",
                wiring=[
                    ["pickups.p.split.hot", "controls.volume.in"],
                    ["controls.volume.wiper", "out"],
                    ["controls.volume.gnd", "GND"],
                    ["pickups.p.split.cold", "GND"],
                    ["out", "controls.tone.in"],
                    ["controls.tone.gnd", "GND"],
                ],
                controls={
                    "volume": ControlElementConfig(
                        name="Volume", resistance=250000.0, taper="audio_15", default=1.0
                    ),
                    "tone": ControlElementConfig(
                        name="Tone", resistance=250000.0, taper="audio_15", cap=47e-9, default=1.0
                    ),
                },
                cable_pf=750.0,
                load_resistance=1000000.0,
            )
        },
    )

    v_open = VoicingConfig(name="Precision Open", harness="passive", controls={"volume": 1.0, "tone": 1.0})
    v_warm = VoicingConfig(name="Precision Warm", harness="passive", controls={"volume": 1.0, "tone": 0.0})

    # Benchmark solve speed
    t0 = time.perf_counter()
    res_open = solve_mna_harness(p_bass, p_bass.harnesses["passive"], v_open)
    t_single_ms = (time.perf_counter() - t0) * 1000.0
    assert t_single_ms < 5.0, f"Single MNA solve must be < 5.0 ms, took {t_single_ms:.2f} ms"

    mag_open = res_open["p.split"]
    peak_freq = FREQS[np.argmax(mag_open)]
    assert 1850.0 <= peak_freq <= 2300.0, f"Open peak at {peak_freq:.1f} Hz outside [1850, 2300]"
    assert np.max(mag_open) > 1.0

    res_warm = solve_mna_harness(p_bass, p_bass.harnesses["passive"], v_warm)
    mag_warm = res_warm["p.split"]
    peak_warm_freq = FREQS[np.argmax(mag_warm)]
    assert (
        150.0 <= peak_warm_freq <= 500.0
    ), f"Warm peak at {peak_warm_freq:.1f} Hz outside [150, 500]"


def test_mna_stingray_parallel_vs_series():
    """Verify StingRay active preamp with coil mode switch (series shifts peak down by factor of ~sqrt(2) to 2)."""
    from allomorph.circuit.solver import solve_mna_harness
    from allomorph.config.schema import (
        CoilConfig,
        ControlElementConfig,
        HarnessConfig,
        InstrumentConfig,
        PickupConfig,
        SwitchConfig,
        SwitchPositionConfig,
        VoicingConfig,
    )

    stingray = InstrumentConfig(
        id="test_stingray",
        name="Test StingRay",
        scale_length_in=34.0,
        pickups={
            "mm_humbucker": PickupConfig(
                name="MM Humbucker",
                type="dual_coil",
                coils=[
                    CoilConfig(
                        id="neck",
                        position_from_bridge_m=0.0771,
                        L=2.4,
                        Rdc=4400.0,
                        Reddy=75000.0,
                        Ccoil=180e-12,
                    ),
                    CoilConfig(
                        id="bridge",
                        position_from_bridge_m=0.0549,
                        L=2.4,
                        Rdc=4400.0,
                        Reddy=75000.0,
                        Ccoil=180e-12,
                    ),
                ],
            )
        },
        harnesses={
            "active": HarnessConfig(
                name="StingRay Active Preamp",
                type="active_preamp",
                preamp="stingray_2band",
                wiring=[
                    ["preamp.out", "controls.volume.in"],
                    ["controls.volume.wiper", "out"],
                    ["controls.volume.gnd", "GND"],
                    ["preamp.gnd", "GND"],
                ],
                controls={
                    "volume": ControlElementConfig(
                        name="Master Volume", resistance=25000.0, taper="audio_15", default=1.0
                    ),
                    "treble": ControlElementConfig(
                        name="Treble", type="preamp_band", band="high_shelf", default=0.5
                    ),
                    "bass": ControlElementConfig(
                        name="Bass", type="preamp_band", band="low_shelf", default=0.5
                    ),
                },
                switches={
                    "coil_mode": SwitchConfig(
                        name="Coil Mode",
                        default="parallel",
                        positions={
                            "parallel": SwitchPositionConfig(
                                connect=[
                                    ["pickups.mm_humbucker.neck.hot", "preamp.in"],
                                    ["pickups.mm_humbucker.neck.cold", "GND"],
                                    ["pickups.mm_humbucker.bridge.hot", "preamp.in"],
                                    ["pickups.mm_humbucker.bridge.cold", "GND"],
                                ],
                                k_mutual=0.20,
                            ),
                            "series": SwitchPositionConfig(
                                connect=[
                                    ["pickups.mm_humbucker.neck.hot", "preamp.in"],
                                    ["pickups.mm_humbucker.neck.cold", "mid"],
                                    ["pickups.mm_humbucker.bridge.hot", "mid"],
                                    ["pickups.mm_humbucker.bridge.cold", "GND"],
                                ],
                                k_mutual=0.20,
                            ),
                        },
                    )
                },
            )
        },
    )

    v_par = VoicingConfig(
        name="StingRay Parallel",
        harness="active",
        switch="parallel",
        controls={"volume": 1.0, "treble": 0.5, "bass": 0.5},
    )
    v_ser = VoicingConfig(
        name="StingRay Series",
        harness="active",
        switch="series",
        controls={"volume": 1.0, "treble": 0.5, "bass": 0.5},
    )

    res_par = solve_mna_harness(stingray, stingray.harnesses["active"], v_par)
    res_ser = solve_mna_harness(stingray, stingray.harnesses["active"], v_ser)

    par_n = res_par["mm_humbucker.neck"]
    ser_n = res_ser["mm_humbucker.neck"]

    peak_par = FREQS[np.argmax(par_n)]
    peak_ser = FREQS[np.argmax(ser_n)]
    assert peak_ser < peak_par, f"Series peak {peak_ser:.1f} must be lower than parallel {peak_par:.1f}"
    assert np.max(ser_n) > np.max(par_n), "Series connection must have higher peak voltage gain"


def test_mna_jazz_bass_independent_volumes():
    """Verify 2V/1T Jazz Bass circuit independent volume controls and mutual crosstalk."""
    from allomorph.circuit.solver import solve_mna_harness
    from allomorph.config.schema import (
        CoilConfig,
        ControlElementConfig,
        HarnessConfig,
        InstrumentConfig,
        PickupConfig,
        VoicingConfig,
    )

    jazz_bass = InstrumentConfig(
        id="test_jazz",
        name="Test Jazz Bass",
        scale_length_in=34.0,
        pickups={
            "neck": PickupConfig(
                name="Jazz Neck",
                coils=[
                    CoilConfig(
                        id="neck",
                        position_from_bridge_m=0.1556,
                        L=3.2,
                        Rdc=7200.0,
                        Reddy=120000.0,
                        Ccoil=75e-12,
                    )
                ],
            ),
            "bridge": PickupConfig(
                name="Jazz Bridge",
                coils=[
                    CoilConfig(
                        id="bridge",
                        position_from_bridge_m=0.0635,
                        L=3.6,
                        Rdc=7800.0,
                        Reddy=125000.0,
                        Ccoil=70e-12,
                    )
                ],
            ),
        },
        harnesses={
            "passive": HarnessConfig(
                name="Standard 60s Passive V/V/T",
                wiring=[
                    ["pickups.neck.hot", "controls.neck_vol.wiper"],
                    ["controls.neck_vol.in", "out"],
                    ["controls.neck_vol.gnd", "GND"],
                    ["pickups.neck.cold", "GND"],
                    ["pickups.bridge.hot", "controls.bridge_vol.wiper"],
                    ["controls.bridge_vol.in", "out"],
                    ["controls.bridge_vol.gnd", "GND"],
                    ["pickups.bridge.cold", "GND"],
                    ["out", "controls.tone.in"],
                    ["controls.tone.gnd", "GND"],
                ],
                k_mutual=0.04,
                controls={
                    "neck_vol": ControlElementConfig(
                        name="Neck Volume", resistance=250000.0, taper="audio_15", default=1.0
                    ),
                    "bridge_vol": ControlElementConfig(
                        name="Bridge Volume", resistance=250000.0, taper="audio_15", default=1.0
                    ),
                    "tone": ControlElementConfig(
                        name="Master Tone", resistance=250000.0, taper="audio_15", cap=47e-9, default=1.0
                    ),
                },
            )
        },
    )

    v_neck = VoicingConfig(
        name="Jazz Solo Neck", harness="passive", controls={"neck_vol": 1.0, "bridge_vol": 0.0, "tone": 1.0}
    )
    v_bridge = VoicingConfig(
        name="Jazz Solo Bridge", harness="passive", controls={"neck_vol": 0.0, "bridge_vol": 1.0, "tone": 1.0}
    )

    res_neck = solve_mna_harness(jazz_bass, jazz_bass.harnesses["passive"], v_neck)
    res_bridge = solve_mna_harness(jazz_bass, jazz_bass.harnesses["passive"], v_bridge)

    # When neck is soloed, bridge must be attenuated by > 26 dB (peak < 0.05 vs ~0.94)
    assert np.max(res_neck["bridge.bridge"]) < 0.05
    assert np.max(res_neck["neck.neck"]) > 0.80

    # When bridge is soloed, neck must be attenuated by > 26 dB
    assert np.max(res_bridge["neck.neck"]) < 0.05
    assert np.max(res_bridge["bridge.bridge"]) > 0.80


def test_mna_rickenbacker_vintage_push_pull_hpf():
    """Verify 2V/2T Rickenbacker 4003 with push-pull vintage 4.7nF HPF capacitor switch."""
    from allomorph.circuit.solver import solve_mna_harness
    from allomorph.config.schema import (
        CoilConfig,
        ControlElementConfig,
        HarnessConfig,
        InstrumentConfig,
        PickupConfig,
        SwitchConfig,
        SwitchPositionConfig,
        VoicingConfig,
    )

    rick = InstrumentConfig(
        id="test_rick",
        name="Test Rickenbacker 4003",
        scale_length_in=33.25,
        pickups={
            "neck": PickupConfig(
                name="Toaster Neck",
                coils=[
                    CoilConfig(
                        id="neck",
                        position_from_bridge_m=0.170,
                        L=3.5,
                        Rdc=7400.0,
                        Reddy=90000.0,
                        Ccoil=95e-12,
                    )
                ],
            ),
            "bridge": PickupConfig(
                name="High-Gain Bridge",
                coils=[
                    CoilConfig(
                        id="bridge",
                        position_from_bridge_m=0.060,
                        L=4.2,
                        Rdc=11000.0,
                        Reddy=70000.0,
                        Ccoil=110e-12,
                    )
                ],
            ),
        },
        harnesses={
            "passive": HarnessConfig(
                name="2V/2T Rickenbacker Cavity",
                wiring=[
                    ["pickups.neck.hot", "controls.neck_vol.wiper"],
                    ["controls.neck_vol.in", "neck_bus"],
                    ["controls.neck_vol.gnd", "GND"],
                    ["pickups.neck.cold", "GND"],
                    ["neck_bus", "controls.neck_tone.in"],
                    ["controls.neck_tone.gnd", "GND"],
                    ["controls.bridge_vol.in", "bridge_bus"],
                    ["controls.bridge_vol.gnd", "GND"],
                    ["pickups.bridge.cold", "GND"],
                    ["bridge_bus", "controls.bridge_tone.in"],
                    ["controls.bridge_tone.gnd", "GND"],
                ],
                controls={
                    "neck_vol": ControlElementConfig(
                        name="Neck Volume", resistance=330000.0, taper="audio_15", default=1.0
                    ),
                    "neck_tone": ControlElementConfig(
                        name="Neck Tone", resistance=330000.0, taper="audio_15", cap=47e-9, default=1.0
                    ),
                    "bridge_vol": ControlElementConfig(
                        name="Bridge Volume", resistance=330000.0, taper="audio_15", default=1.0
                    ),
                    "bridge_tone": ControlElementConfig(
                        name="Bridge Tone", resistance=330000.0, taper="audio_15", cap=47e-9, default=1.0
                    ),
                },
                switches={
                    "pickup_selector": SwitchConfig(
                        name="Pickup Selector",
                        default="both",
                        positions={
                            "neck": SwitchPositionConfig(connect=[["neck_bus", "out"]]),
                            "bridge": SwitchPositionConfig(connect=[["bridge_bus", "out"]]),
                            "both": SwitchPositionConfig(
                                connect=[["neck_bus", "out"], ["bridge_bus", "out"]], k_mutual=0.04
                            ),
                        },
                    ),
                    "vintage_tone": SwitchConfig(
                        name="Push-Pull Vintage Tone",
                        default="bypass",
                        positions={
                            "bypass": SwitchPositionConfig(
                                connect=[["pickups.bridge.hot", "controls.bridge_vol.wiper"]]
                            ),
                            "vintage": SwitchPositionConfig(
                                connect=[
                                    ["pickups.bridge.hot", "vintage_cap.in"],
                                    ["vintage_cap.out", "controls.bridge_vol.wiper"],
                                ],
                                components={"vintage_cap": 4.7e-9},
                            ),
                        },
                    ),
                },
            )
        },
    )

    v_rick_modern = VoicingConfig(
        name="Rick Modern Bridge",
        harness="passive",
        switches={"pickup_selector": "bridge", "vintage_tone": "bypass"},
        controls={"bridge_vol": 1.0, "bridge_tone": 1.0},
    )
    v_rick_vintage = VoicingConfig(
        name="Rick Vintage Bridge",
        harness="passive",
        switches={"pickup_selector": "bridge", "vintage_tone": "vintage"},
        controls={"bridge_vol": 1.0, "bridge_tone": 1.0},
    )

    res_rick_mod = solve_mna_harness(rick, rick.harnesses["passive"], v_rick_modern)
    res_rick_vin = solve_mna_harness(rick, rick.harnesses["passive"], v_rick_vintage)

    mod_b = res_rick_mod["bridge.bridge"]
    vin_b = res_rick_vin["bridge.bridge"]

    gain_40_mod = mod_b[min(range(len(FREQS)), key=lambda i: abs(FREQS[i] - 40.0))]
    gain_40_vin = vin_b[min(range(len(FREQS)), key=lambda i: abs(FREQS[i] - 40.0))]

    # Vintage 4.7nF cap cuts 40Hz sub-bass by > 10 dB (< 30% of modern gain)
    assert gain_40_vin < gain_40_mod * 0.30, f"HPF at 40Hz {gain_40_vin:.4f} vs modern {gain_40_mod:.4f}"

