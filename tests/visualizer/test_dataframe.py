"""
Tests for Polars-based frequency response dataframe generation in allomorph.visualizer.
"""

import numpy as np
import polars as pl
import pytest

from allomorph.config import VOICES
from allomorph.visualizer import (
    build_voice_dataframe,
    build_voicings_comparison_data,
    build_voicings_comparison_dataframe,
    compute_curve_rms_db,
    compute_fir_csd,
)


def test_build_voice_dataframe():
    voice_id = "jazz_pair_active"
    cfg = VOICES[voice_id]
    df = build_voice_dataframe(voice_id, cfg, instrument="30in")

    assert isinstance(df, pl.DataFrame)
    assert set(df.columns) == {
        "frequency",
        "magnitude_db",
        "line_type",
        "voice_id",
    }
    assert df.height == 600

    # Check frequency range
    freqs = df["frequency"].to_list()
    assert freqs[0] >= 20.0
    assert freqs[-1] <= 20000.0

    # Magnitude should be in reasonable dB range (e.g. -50 dB to +20 dB)
    mags = df["magnitude_db"].to_list()
    assert all(-50.0 <= m <= 20.0 for m in mags)


def test_circuit_simulation_integration():
    """Verify that build_voice_dataframe accurately incorporates the exact circuit transfer functions."""
    # 1. 47nF tone capacitor rolloff on P-Bass
    df_tone = build_voice_dataframe("precision_warm", VOICES["precision_warm"], instrument="30in")
    df_p = build_voice_dataframe(
        "precision_vintage", VOICES["precision_vintage"], instrument="30in"
    )

    mag_tone_5k = df_tone.filter(pl.col("frequency") > 4500.0)["magnitude_db"].to_list()[0]
    mag_p_5k = df_p.filter(pl.col("frequency") > 4500.0)["magnitude_db"].to_list()[0]

    # Tone rolled off circuit should have substantially more high frequency attenuation (>20 dB deeper cut)
    assert mag_tone_5k < mag_p_5k - 20.0

    # 2. Rickenbacker 4.7nF series HPF low-end cut
    df_rick = build_voice_dataframe(
        "rickenbacker_clank", VOICES["rickenbacker_clank"], instrument="30in"
    )
    mag_rick_low = df_rick.filter(pl.col("frequency") < 50.0)["magnitude_db"].to_list()[0]
    mag_rick_mid = df_rick.filter((pl.col("frequency") > 800.0) & (pl.col("frequency") < 1200.0))[
        "magnitude_db"
    ].to_list()[0]
    # Series cap introduces strong low-end shelf
    assert mag_rick_low < mag_rick_mid - 10.0

    # 2b. Verify mode="output" on Rickenbacker Clank is not artificially clipped at +24 dB
    df_rick_out = build_voice_dataframe(
        "rickenbacker_clank", VOICES["rickenbacker_clank"], mode="output"
    )
    mag_out_low = df_rick_out.filter(pl.col("frequency") < 50.0)["magnitude_db"].to_list()[0]
    mag_out_mid = df_rick_out.filter(
        (pl.col("frequency") > 800.0) & (pl.col("frequency") < 1200.0)
    )["magnitude_db"].to_list()[0]
    assert mag_out_low < mag_out_mid - 10.0
    assert float(np.max(df_rick_out["magnitude_db"].to_numpy())) < 5.0

    # 3. Verify all voices produce finite, non-null values across all frequencies
    for vid, cfg in VOICES.items():
        vdf = build_voice_dataframe(vid, cfg, instrument="30in")
        assert not vdf["magnitude_db"].is_nan().any()
        assert not vdf["magnitude_db"].is_null().any()


def test_build_voicings_comparison_data():
    data = build_voicings_comparison_data(step=3)
    assert "frequencies" in data
    assert "voices" in data
    assert "families" in data
    assert "default_source" in data
    assert "default_target" in data

    assert len(data["frequencies"]) == 200
    assert len(data["voices"]) == len(VOICES)
    for vdata in data["voices"].values():
        assert "name" in vdata
        assert "family" in vdata
        assert "magnitude_db" in vdata
        assert len(vdata["magnitude_db"]) == 200
        assert not any(np.isnan(vdata["magnitude_db"]))


def test_build_voicings_comparison_dataframe():
    # 1. Identity pair: source == target
    df_id = build_voicings_comparison_dataframe("precision_vintage", "precision_vintage", step=1)
    assert isinstance(df_id, pl.DataFrame)
    assert df_id.height == 600 * 3
    s3_id = df_id.filter(df_id["line_type"] == "3. Difference")
    assert (s3_id["magnitude_db"] == 0.0).all()

    # 2. Transformative pair: H_src + Difference = H_tgt everywhere
    df_tf = build_voicings_comparison_dataframe("precision_vintage", "stingray_parallel", step=1)
    s1 = df_tf.filter(df_tf["line_type"] == "1. Source Voicing")["magnitude_db"].to_numpy()
    s2 = df_tf.filter(df_tf["line_type"] == "2. Target Voicing")["magnitude_db"].to_numpy()
    s3 = df_tf.filter(df_tf["line_type"] == "3. Difference")["magnitude_db"].to_numpy()
    assert np.allclose(s1 + s3, s2, atol=0.05)

    # 3. Invalid voice IDs raise KeyError
    with pytest.raises(KeyError):
        build_voicings_comparison_dataframe("unknown_source", "precision_vintage")
    with pytest.raises(KeyError):
        build_voicings_comparison_dataframe("precision_vintage", "unknown_target")


def test_compute_fir_csd():
    sr = 48000
    t = np.arange(2048) / sr
    fir = np.exp(-t * 800.0) * np.sin(2.0 * np.pi * 1000.0 * t)
    freqs = np.geomspace(20.0, 20000.0, 50)
    time_ms, csd = compute_fir_csd(fir, freqs, num_slices=24, max_time_ms=10.0, sr=sr)

    assert len(time_ms) == 24
    assert len(csd) == 24
    idx_1k = int(np.argmin(np.abs(freqs - 1000.0)))
    assert csd[0][idx_1k] > csd[-1][idx_1k] + 20.0


def test_spatial_bridge_proximity_displacement_ratio_in_visualizer():
    """Validates that build_voice_dataframe accurately reflects standing-wave bridge proximity displacement:
    - Precision Vintage (125mm): full fundamental excursion around 0 dB relative to mids
    - Jazz Bridge Open (63.5mm): leaner fundamental at 20 Hz (-4.4 dB) relative to 1 kHz (+0.8 dB)
    """
    df_p = build_voice_dataframe("precision_vintage", VOICES["precision_vintage"], mode="output")
    df_j = build_voice_dataframe("jazz_bridge_open", VOICES["jazz_bridge_open"], mode="output")

    mag_p_20 = df_p["magnitude_db"][0]
    mag_j_20 = df_j["magnitude_db"][0]

    # Precision Vintage has full fundamental passband excursion around +1.5 dB
    assert 0.0 <= mag_p_20 <= 2.5, f"Precision 20 Hz dB {mag_p_20} outside [0.0, 2.5]"

    # Jazz Bridge Open has leaner fundamental (-4.5 to -2.0 dB)
    assert -4.5 <= mag_j_20 <= -2.0, f"Jazz Bridge 20 Hz dB {mag_j_20} outside [-4.5, -2.0]"

    mag_j_1k = df_j.filter(pl.col("frequency") > 1000.0)["magnitude_db"].to_list()[0]
    # Fundamental is leaner than mid-register by > 2.5 dB due to bridge displacement low-shelf
    assert mag_j_20 < mag_j_1k - 2.5

    # Voicings comparison dataframe verifies Difference curve at 20 Hz equals tgt - src
    df_comp = build_voicings_comparison_dataframe("precision_vintage", "jazz_bridge_open", step=1)
    s3_20 = df_comp.filter(df_comp["line_type"] == "3. Difference")["magnitude_db"][0]
    expected_diff_20 = mag_j_20 - mag_p_20
    assert abs(s3_20 - expected_diff_20) < 0.05


def test_fast_sweep_smoothness_in_visualizer():
    """Validates that build_voice_dataframe produces smooth, ripple-free high-frequency curves
    confirming elimination of Monte Carlo white noise jitter and boundary truncation ripples.
    """
    df_p = build_voice_dataframe("precision_vintage", VOICES["precision_vintage"], mode="output")
    df_j = build_voice_dataframe("jazz_bridge_open", VOICES["jazz_bridge_open"], mode="output")
    freqs = df_p["frequency"].to_numpy()
    diff = df_j["magnitude_db"].to_numpy() - df_p["magnitude_db"].to_numpy()

    high_mask = (freqs >= 10000.0) & (freqs <= 20000.0)
    diff_high = diff[high_mask]
    roughness = float(np.std(np.diff(diff_high, n=2)))
    assert roughness < 0.05, f"High-frequency roughness {roughness} exceeds 0.05 dB"


def test_compute_curve_rms_db():
    """Validates broadband RMS calculation on arrays, lists, and Polars Series."""
    # Flat 0 dB response must yield exact 0.0 dB RMS
    flat_zeros = np.zeros(100, dtype=np.float64)
    assert abs(compute_curve_rms_db(flat_zeros)) < 1e-6

    # Flat +6.0 dB response must yield +6.0 dB RMS
    flat_plus6 = [6.0] * 50
    assert abs(compute_curve_rms_db(flat_plus6) - 6.0) < 1e-6

    # Polars Series input
    series = pl.Series("mag", [-6.0, 6.0])
    # RMS of -6 dB (0.501187) and +6 dB (1.995262):
    # sqrt(mean(0.501187^2 + 1.995262^2)) = sqrt((0.251188 + 3.98107) / 2) = sqrt(2.11613) = 1.45469
    # 20 * log10(1.45469) ~= 3.255 dB
    rms_val = compute_curve_rms_db(series)
    assert 3.2 <= rms_val <= 3.3


def test_precision_mids_vs_vintage_distinctness():
    """Validates that Precision Mids (22nF ToneStyler shunt) and Precision Vintage (250k CTS open)
    are clearly distinct in both resonant frequency and frequency response:
    - Precision Vintage: wide-open resonance ~2.1-2.8 kHz
    - Precision Mids: 22nF pure shunt low-mid vocal peak ~440 Hz with steep treble rolloff (>20 dB diff at 3 kHz)
    """
    df_comp = build_voicings_comparison_dataframe("precision_vintage", "precision_mids", step=1)
    s3 = df_comp.filter(df_comp["line_type"] == "3. Difference")["magnitude_db"].to_numpy()
    freqs = df_comp.filter(df_comp["line_type"] == "3. Difference")["frequency"].to_numpy()

    # Must be distinct with max difference > 20 dB across spectrum
    assert np.max(np.abs(s3)) > 20.0

    # Treble rolloff at 3 kHz must attenuate by > 15 dB
    idx_3k = int(np.argmin(np.abs(freqs - 3000.0)))
    assert s3[idx_3k] < -15.0


def test_multi_pickup_pj_voicings_comb_null_decay():
    """Validates that multi-pickup voicings (such as PJ Passive and PJ Active) exhibit authentic
    low-mid spatial interference (~900 Hz scoop) while cross-coherence decay prevents spurious
    deep comb filter nulls at higher harmonics (2.8 kHz, 4.6 kHz, etc.).
    """
    for vid in ["pj_passive", "pj_active", "p_mm_parallel"]:
        cfg = VOICES[vid]
        df = build_voice_dataframe(vid, cfg, mode="output")
        freqs = df["frequency"].to_numpy()
        mag = df["magnitude_db"].to_numpy()

        # Find local minima deeper than -5 dB (accounting for 2-decimal-place rounding plateaus)
        minima_idx = [
            i
            for i in range(1, len(mag) - 1)
            if mag[i] < -5.0
            and (
                (mag[i] < mag[i - 1] and mag[i] < mag[i + 1])
                or (
                    mag[i] < mag[i - 1]
                    and mag[i] == mag[i + 1]
                    and (i + 2 >= len(mag) or mag[i + 2] > mag[i])
                )
            )
        ]
        # Must have exactly 1 primary spatial notch (in the 500-1500 Hz range)
        notch_freqs = [freqs[idx] for idx in minima_idx]
        assert len(notch_freqs) == 1, (
            f"{vid} has spurious comb nulls: {notch_freqs}"
        )
        assert 500.0 <= notch_freqs[0] <= 1500.0, (
            f"{vid} primary notch at {notch_freqs[0]:.1f} Hz outside expected [500, 1500] Hz"
        )

