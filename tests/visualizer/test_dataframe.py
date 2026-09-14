"""
Tests for Polars-based frequency response dataframe generation in allomorph.visualizer.
"""

import numpy as np
import polars as pl

from allomorph.config import VOICES
from allomorph.visualizer import (
    build_voice_dataframe,
    build_voicing_ir_diff_3d_data,
    build_voicings_comparison_data,
    build_voicings_comparison_dataframe,
    compute_curve_rms_db,
    compute_fir_csd,
)


def test_build_voice_dataframe():
    voice_id = "jazz_pair_active"
    cfg = VOICES[voice_id]
    df = build_voice_dataframe(voice_id, cfg, src_scale="30in")

    assert isinstance(df, pl.DataFrame)
    assert set(df.columns) == {
        "frequency",
        "magnitude_db",
        "voice_id",
        "voice_name",
        "topology",
        "description",
    }
    assert df.height == 600

    # Check frequency range
    freqs = df["frequency"].to_list()
    assert freqs[0] >= 20.0
    assert freqs[-1] <= 20000.0

    # Magnitude should be in reasonable dB range (e.g. -35 dB to +15 dB)
    mags = df["magnitude_db"].to_list()
    assert all(-50.0 <= m <= 20.0 for m in mags)


def test_circuit_simulation_integration():
    """Verify that build_voice_dataframe accurately incorporates the exact circuit transfer functions."""
    # 1. 47nF tone capacitor rolloff on P-Bass
    df_tone = build_voice_dataframe("precision_warm", VOICES["precision_warm"], src_scale="30in")
    df_p = build_voice_dataframe("precision_vintage", VOICES["precision_vintage"], src_scale="30in")

    mag_tone_5k = df_tone.filter(pl.col("frequency") > 4500.0)["magnitude_db"].to_list()[0]
    mag_p_5k = df_p.filter(pl.col("frequency") > 4500.0)["magnitude_db"].to_list()[0]

    # Tone rolled off circuit should have substantially more high frequency attenuation (>20 dB deeper cut)
    assert mag_tone_5k < mag_p_5k - 20.0

    # 2. Rickenbacker 4.7nF series HPF low-end cut
    df_rick = build_voice_dataframe(
        "rickenbacker_clank", VOICES["rickenbacker_clank"], src_scale="30in"
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
        vdf = build_voice_dataframe(vid, cfg, src_scale="30in")
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
    s3_id = df_id.filter(df_id["line_type"] == "3. Normalized Difference (Norm. Diff)")
    assert (s3_id["magnitude_db"] == 0.0).all()

    # 2. Transformative pair: H_src,norm + Norm. Diff = H_tgt,norm in the passband (< 3.5 kHz)
    df_tf = build_voicings_comparison_dataframe("precision_vintage", "stingray_parallel", step=1)
    s1 = df_tf.filter(df_tf["line_type"] == "1. Source Voicing")["magnitude_db"].to_numpy()
    s2 = df_tf.filter(df_tf["line_type"] == "2. Target Voicing")["magnitude_db"].to_numpy()
    s3 = df_tf.filter(df_tf["line_type"] == "3. Normalized Difference (Norm. Diff)")[
        "magnitude_db"
    ].to_numpy()
    freqs = df_tf.filter(df_tf["line_type"] == "1. Source Voicing")["frequency"].to_numpy()
    idx_pass = freqs <= 3000.0
    src_rms = compute_curve_rms_db(s1)
    tgt_rms = compute_curve_rms_db(s2)
    s1_norm = s1 - src_rms
    s2_norm = s2 - tgt_rms
    assert np.allclose((s1_norm + s3)[idx_pass], s2_norm[idx_pass], atol=0.10)
    # Beyond 5 kHz, response rolls off smoothly according to input power spectrum
    # Boost is strictly soft-knee bounded at <= +7.05 dB
    assert np.max(s3) <= 7.05
    assert s3[-1] <= 0.0


def test_build_voicing_ir_diff_3d_data():
    data = build_voicing_ir_diff_3d_data(num_freqs=50, num_slices=24, max_time_ms=10.0)
    assert "frequencies" in data
    assert "time_ms" in data
    assert "voices" in data
    assert "responses" in data
    assert len(data["frequencies"]) == 50
    assert len(data["time_ms"]) == 24

    # Check identity pair in 3D data: precision_vintage -> precision_vintage
    id_entry = data["responses"]["precision_vintage"]["precision_vintage"]
    assert id_entry["fir_waveform"][0] == 1.0
    assert all(val == 0.0 for val in id_entry["fir_waveform"][1:])
    # Initial CSD slice is flat 0.00 dB
    assert all(val == 0.0 for val in id_entry["csd_matrix"][0])


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
    """Validates that build_voice_dataframe and build_voicings_comparison_dataframe accurately
    reflect standing-wave bridge proximity fundamental displacement ratios:
      - Precision Vintage (125mm): ~ +2.3 to +2.5 dB at 20 Hz
      - Jazz Bridge Open (63.5mm): ~ -3.3 to -3.6 dB at 20 Hz
      - Differential (Jazz Bridge - Precision): ~ -5.8 dB at 20 Hz
    """
    df_p = build_voice_dataframe("precision_vintage", VOICES["precision_vintage"], mode="output")
    df_j = build_voice_dataframe("jazz_bridge_open", VOICES["jazz_bridge_open"], mode="output")

    mag_p_20 = df_p["magnitude_db"][0]
    mag_j_20 = df_j["magnitude_db"][0]

    # Precision Vintage has warm fundamental excursion (~ +2.27 dB) loaded by passive pot (-0.42 dB)
    assert 1.7 <= mag_p_20 <= 2.5, f"Precision 20 Hz dB {mag_p_20} outside [1.7, 2.5]"

    # Jazz Bridge Open has lean fundamental attenuation (~ -3.42 dB) loaded by passive pot (-0.31 dB)
    assert -4.0 <= mag_j_20 <= -3.0, f"Jazz Bridge 20 Hz dB {mag_j_20} outside [-4.0, -3.0]"

    # Differential comparison matches theoretical -5.88 dB displacement ratio
    diff_20 = mag_j_20 - mag_p_20
    assert abs(diff_20 - (-5.88)) < 0.3, (
        f"Differential 20 Hz cut {diff_20:.2f} dB deviated from -5.88 dB"
    )

    # Voicings comparison dataframe verifies Norm. Diff curve at 20 Hz
    df_comp = build_voicings_comparison_dataframe("precision_vintage", "jazz_bridge_open", step=1)
    s3_20 = df_comp.filter(df_comp["line_type"] == "3. Normalized Difference (Norm. Diff)")[
        "magnitude_db"
    ][0]
    p_rms = compute_curve_rms_db(df_p["magnitude_db"].to_numpy())
    j_rms = compute_curve_rms_db(df_j["magnitude_db"].to_numpy())
    expected_norm_diff_20 = (mag_j_20 - j_rms) - (mag_p_20 - p_rms)
    assert abs(s3_20 - expected_norm_diff_20) < 0.05, (
        f"Comparison dataframe 20 Hz cut {s3_20:.2f} dB deviated from {expected_norm_diff_20:.2f} dB"
    )
