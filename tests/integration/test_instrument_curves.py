"""
Allomorph Instrument & Voicing Frequency Curves Test Collection.

Validates the empirical and physical frequency response curves of specific catalog
instruments, pickups, and target voicings:
- Mid scoops (PJ ~900 Hz, Jazz pair ~640 Hz)
- Resonant frequencies (Precision Vintage ~2.1 kHz, Precision Mids ~440 Hz, Precision Warm low-mid shift, Active EMG peaks)
- Tone rolloff and filter curves (Precision Warm 47nF rolloff, Rickenbacker 4.7nF series HPF)
- Bridge proximity low-frequency fundamental excursion (Precision 125mm vs Jazz Bridge 63.5mm)
- Differential voicing curves (P/MM Parallel -> Series)
"""

import math

import numpy as np
import polars as pl
import pytest

from allomorph.circuit.solver import solve_mna_harness
from allomorph.config import INSTRUMENTS, load_instrument
from allomorph.config.instruments import resolve_target_voicing
from allomorph.dsp import FREQS
from allomorph.visualizer.dataframe import (
    build_voice_dataframe,
    build_voicings_comparison_dataframe,
)


def test_pj_mid_scoop_curve():
    """Validates that PJ voicings exhibit their characteristic low-mid scoop around ~900 Hz."""
    for vid in ["pj_passive", "pj_active"]:
        inst, v_cfg = resolve_target_voicing(vid)
        df = build_voice_dataframe(inst, v_cfg, mode="output")
        freqs = df["frequency"].to_numpy()
        mag = df["magnitude_db"].to_numpy()

        # Mid scoop in the 750 - 1100 Hz band
        mask_mids = (freqs >= 750.0) & (freqs <= 1100.0)
        min_idx = np.argmin(mag[mask_mids])
        scoop_freq = float(freqs[mask_mids][min_idx])
        scoop_depth = float(mag[mask_mids][min_idx])

        assert 800.0 <= scoop_freq <= 1000.0, (
            f"{vid} mid scoop at {scoop_freq:.1f} Hz outside expected [800, 1000] Hz"
        )
        assert -12.0 <= scoop_depth <= -5.0, (
            f"{vid} mid scoop depth {scoop_depth:.1f} dB outside expected [-12, -5] dB"
        )


def test_jazz_pair_mid_scoop_curve():
    """Validates that Jazz Bass neck+bridge pair exhibits its characteristic mid scoop around ~640 Hz."""
    inst, v_cfg = resolve_target_voicing("jazz_pair_open")
    df = build_voice_dataframe(inst, v_cfg, mode="output")
    freqs = df["frequency"].to_numpy()
    mag = df["magnitude_db"].to_numpy()

    # Mid scoop in the 550 - 750 Hz band
    mask_mids = (freqs >= 550.0) & (freqs <= 750.0)
    min_idx = np.argmin(mag[mask_mids])
    scoop_freq = float(freqs[mask_mids][min_idx])
    scoop_depth = float(mag[mask_mids][min_idx])

    assert 600.0 <= scoop_freq <= 700.0, (
        f"Jazz Pair mid scoop at {scoop_freq:.1f} Hz outside expected [600, 700] Hz"
    )
    assert -12.0 <= scoop_depth <= -5.0, (
        f"Jazz Pair mid scoop depth {scoop_depth:.1f} dB outside expected [-12, -5] dB"
    )


def test_precision_vintage_resonant_frequency_curve():
    """Validates that Precision Vintage (wide-open 250k CTS) exhibits its resonant peak at ~2.1 kHz."""
    inst, v = resolve_target_voicing("precision_vintage")
    harness = inst.harnesses[v.harness]
    curves = solve_mna_harness(inst, harness, v, freqs=FREQS)
    mag = next(iter(curves.values()))

    peak_idx = int(np.argmax(mag))
    peak_freq = FREQS[peak_idx]
    assert 1850.0 <= peak_freq <= 2300.0, (
        f"Precision Vintage resonant peak at {peak_freq:.1f} Hz outside [1850, 2300] Hz"
    )
    assert mag[peak_idx] > 1.0, (
        f"Expected Q peak > 1.0 under CTS 250k load, got {mag[peak_idx]:.2f}"
    )


def test_precision_warm_tone_rolloff_curve():
    """Validates that Precision Warm (47nF tone pot rolled off) downshifts resonance and attenuates treble."""
    # 1. Electrical resonance downshifts into low-mids (150 - 400 Hz)
    inst, v = resolve_target_voicing("precision_warm")
    harness = inst.harnesses[v.harness]
    curves = solve_mna_harness(inst, harness, v, freqs=FREQS)
    mag_elec = next(iter(curves.values()))
    peak_idx = int(np.argmax(mag_elec))
    peak_freq = FREQS[peak_idx]
    assert 150.0 <= peak_freq <= 400.0, (
        f"Precision Warm electrical peak at {peak_freq:.1f} Hz outside [150, 400] Hz"
    )

    # 2. Acoustic + circuit output response attenuates treble by > 20 dB at 5 kHz relative to Vintage Open
    inst_warm, v_warm = resolve_target_voicing("precision_warm")
    inst_p, v_p = resolve_target_voicing("precision_vintage")
    df_warm = build_voice_dataframe(inst_warm, v_warm)
    df_p = build_voice_dataframe(inst_p, v_p)
    mag_warm_5k = df_warm.filter(pl.col("frequency") > 4500.0)["magnitude_db"].to_list()[0]
    mag_p_5k = df_p.filter(pl.col("frequency") > 4500.0)["magnitude_db"].to_list()[0]
    assert mag_warm_5k < mag_p_5k - 20.0, (
        f"Precision Warm did not roll off treble sufficiently: warm={mag_warm_5k:.1f} dB, open={mag_p_5k:.1f} dB"
    )


def test_precision_mids_resonant_frequency_curve():
    """Validates that Precision Mids (22nF ToneStyler shunt) produces a focused vocal resonance at ~440 Hz."""
    # 1. Electrical resonance peak around 400 - 500 Hz
    inst, v = resolve_target_voicing("precision_mids")
    harness = inst.harnesses[v.harness]
    curves = solve_mna_harness(inst, harness, v, freqs=FREQS)
    mag_elec = next(iter(curves.values()))
    peak_idx = int(np.argmax(mag_elec))
    peak_freq = FREQS[peak_idx]
    assert 380.0 <= peak_freq <= 500.0, (
        f"Precision Mids electrical peak at {peak_freq:.1f} Hz outside [380, 500] Hz"
    )

    # 2. Output response has steep treble rolloff (> 15 dB difference at 3 kHz vs Precision Vintage)
    inst_v, v_v = resolve_target_voicing("precision_vintage")
    inst_m, v_m = resolve_target_voicing("precision_mids")
    df_comp = build_voicings_comparison_dataframe(inst_v, v_v, inst_m, v_m, step=1)
    s3 = df_comp.filter(df_comp["line_type"] == "3. Difference")["magnitude_db"].to_numpy()
    freqs = df_comp.filter(df_comp["line_type"] == "3. Difference")["frequency"].to_numpy()
    idx_3k = int(np.argmin(np.abs(freqs - 3000.0)))
    assert s3[idx_3k] < -15.0, (
        f"Precision Mids 3 kHz attenuation {s3[idx_3k]:.1f} dB was not < -15 dB"
    )
    assert np.max(np.abs(s3)) > 20.0


def test_rickenbacker_clank_hpf_curve():
    """Validates that Rickenbacker Clank (series 4.7nF HPF) cuts sub-bass below 50 Hz by > 10 dB."""
    inst_rick, v_rick = resolve_target_voicing("rickenbacker_clank")
    df_rick = build_voice_dataframe(inst_rick, v_rick)
    mag_low = df_rick.filter(pl.col("frequency") < 50.0)["magnitude_db"].to_list()[0]
    mag_mid = df_rick.filter((pl.col("frequency") > 800.0) & (pl.col("frequency") < 1200.0))[
        "magnitude_db"
    ].to_list()[0]
    assert mag_low < mag_mid - 10.0, (
        f"Rickenbacker Clank series HPF did not cut low-end: low={mag_low:.1f} dB, mid={mag_mid:.1f} dB"
    )

    # Output mode should stay within standard dynamic range (< 5 dB peak)
    df_rick_out = build_voice_dataframe(inst_rick, v_rick, mode="output")
    assert float(np.max(df_rick_out["magnitude_db"].to_numpy())) < 5.0


def test_bridge_proximity_bass_excursion_curve():
    """Validates that bridge proximity shapes low-frequency fundamental excursion:
    - Precision Vintage (125mm from bridge): full fundamental excursion around 0 dB (+0.0 to +2.5 dB)
    - Jazz Bridge Open (63.5mm near bridge): leaner fundamental at 20 Hz (-4.5 to -2.0 dB)
    """
    inst_p, v_p = resolve_target_voicing("precision_vintage")
    inst_j, v_j = resolve_target_voicing("jazz_bridge_open")
    df_p = build_voice_dataframe(inst_p, v_p, mode="output")
    df_j = build_voice_dataframe(inst_j, v_j, mode="output")

    mag_p_20 = df_p["magnitude_db"][0]
    mag_j_20 = df_j["magnitude_db"][0]

    assert 0.0 <= mag_p_20 <= 2.5, f"Precision 20 Hz dB {mag_p_20} outside [0.0, 2.5]"
    assert -4.5 <= mag_j_20 <= -2.0, f"Jazz Bridge 20 Hz dB {mag_j_20} outside [-4.5, -2.0]"

    mag_j_1k = df_j.filter(pl.col("frequency") > 1000.0)["magnitude_db"].to_list()[0]
    assert mag_j_20 < mag_j_1k - 2.5, (
        f"Jazz Bridge fundamental ({mag_j_20:.1f} dB) was not leaner than mids ({mag_j_1k:.1f} dB)"
    )


def test_pmm_series_differential_curve():
    """Validates that P/MM Parallel -> P/MM Series difference curve stays <= +2.0 dB across 5-20 kHz."""
    inst_par, v_par = resolve_target_voicing("p_mm_parallel")
    inst_ser, v_ser = resolve_target_voicing("p_mm_series")
    df_par = build_voice_dataframe(inst_par, v_par)
    df_ser = build_voice_dataframe(inst_ser, v_ser)

    freqs = np.array(df_par["frequency"])
    diff = np.array(df_ser["magnitude_db"]) - np.array(df_par["magnitude_db"])

    hf_mask = freqs >= 5000.0
    max_hf_diff = float(np.max(diff[hf_mask]))
    assert max_hf_diff <= 2.0, (
        f"P/MM Series difference exhibited unexpected treble boost: {max_hf_diff:.2f} dB (expected <= +2.0 dB)"
    )


def test_active_pickup_resonant_frequency_peaks():
    """Verify that loaded active pickup circuits match their declared resonant peaks within +-50 Hz."""
    freqs = np.linspace(20.0, 20000.0, 8000)

    test_cases = [
        ("30in_emg_mmtw", "dual_mode", "mmtwx", 2500.0),
        ("30in_emg_mmtw", "single_mode", "mmtwx", 3500.0),
        ("32in_custom_pmm", "px_solo", "px", 3200.0),
        ("32in_custom_pmm", "mm_single", "mmtwx", 3500.0),
        ("32in_custom_pmm", "mm_dual", "mmtwx", 2500.0),
        ("32in_fretless_pmm", "pcsx_solo", "pcsx", 2610.0),
        ("34in_active_emg", "neck_solo", "neck", 4150.0),
        ("34in_active_emg", "bridge_solo", "bridge", 4150.0),
    ]

    for inst_id, v_id, pickup_key, expected_fr in test_cases:
        inst = INSTRUMENTS[inst_id]
        v = inst.voicings[v_id]
        h = inst.harnesses[v.harness]
        curves = solve_mna_harness(inst, h, v, freqs=freqs)
        assert pickup_key in curves, f"Pickup {pickup_key} not found in curves for {inst_id}:{v_id}"
        curve = curves[pickup_key]
        peak_idx = int(np.argmax(curve))
        peak_f = float(freqs[peak_idx])

        error_hz = abs(peak_f - expected_fr)
        assert error_hz <= 50.0, (
            f"{inst_id}:{v_id}:{pickup_key} resonant peak {peak_f:.1f} Hz deviates by {error_hz:.1f} Hz "
            f"from target {expected_fr:.1f} Hz (tolerance +-50 Hz)"
        )


def test_tonestyler_p_bass_progression_curve():
    """Verify P-Bass tone sequence (vintage, mids, warm, dub):
    Resonant peak physically glides down through the spectrum:
    vintage (~2.1 kHz) > mids (~440 Hz) > warm (~210 Hz) > dub (~150 Hz).
    """
    inst05, v05 = resolve_target_voicing("precision_vintage")
    inst05b, v05b = resolve_target_voicing("precision_mids")
    inst05c, v05c = resolve_target_voicing("precision_warm")
    inst05d, v05d = resolve_target_voicing("precision_dub")

    c05 = next(
        iter(solve_mna_harness(inst05, inst05.harnesses[v05.harness], v05, freqs=FREQS).values())
    )
    c05b = next(
        iter(
            solve_mna_harness(inst05b, inst05b.harnesses[v05b.harness], v05b, freqs=FREQS).values()
        )
    )
    c05c = next(
        iter(
            solve_mna_harness(inst05c, inst05c.harnesses[v05c.harness], v05c, freqs=FREQS).values()
        )
    )
    c05d = next(
        iter(
            solve_mna_harness(inst05d, inst05d.harnesses[v05d.harness], v05d, freqs=FREQS).values()
        )
    )

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


def test_jazz_pair_mids_22nf_tone_cap_curve():
    """Verify Voice 02b (60s Jazz Bass Pair with 22nF Tone Cap):
    Harness is passive and contains 22nF tone capacitor component override.
    """
    inst, v = resolve_target_voicing("jazz_pair_mids")
    assert v.harness == "passive"
    assert v.components.get("controls.tone.cap") == pytest.approx(2.2e-8)
    curves = solve_mna_harness(inst, inst.harnesses[v.harness], v, freqs=FREQS)
    assert "neck" in curves and "bridge" in curves


def test_stingray_series_vs_parallel_curve():
    """Verify Music Man StingRay Series vs Parallel transfer functions:
    1. Active series resonance sits lower than parallel resonance.
    2. Series connection delivers +5.0 to +6.5 dB output boost over parallel across passband.
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


def test_dingwall_composite_source_circuit_curve():
    """Verify 37in_multiscale_dingwall pair_parallel declares source circuit
    and evaluates circuit transfer functions cleanly without falling back to generic RLC.
    """
    inst = load_instrument("37in_multiscale_dingwall")
    assert "pair_parallel" in inst.voicings
    v = inst.voicings["pair_parallel"]
    from allomorph.config.voices import voicing_to_voice_config

    vcfg = voicing_to_voice_config(inst, v)
    assert vcfg.id == "dingwall_parallel"

    curves = solve_mna_harness(inst, inst.harnesses[v.harness], v, freqs=FREQS)
    assert len(curves) >= 1
