"""
Tests for Polars-based frequency response dataframe generation in allomorph.visualizer.
All tests in this module check the behavior of the SUT using generic synthetic data.
(Specific instrument curves are tested in the separate tests/integration/ test collection).
"""

from typing import TYPE_CHECKING

import numpy as np
import polars as pl

from allomorph.visualizer import (
    build_voice_dataframe,
    build_voicings_comparison_data,
    build_voicings_comparison_dataframe,
    compute_curve_rms_db,
    compute_fir_csd,
)

if TYPE_CHECKING:
    from allomorph.config.schema import InstrumentConfig


def test_build_voice_dataframe(
    generic_instrument_config: InstrumentConfig,
):
    """Verifies build_voice_dataframe structure, schema, frequency bounds, and finite values using generic data."""
    voicing = generic_instrument_config.voicings["generic_voice"]
    df = build_voice_dataframe(generic_instrument_config, voicing)

    assert isinstance(df, pl.DataFrame)
    assert set(df.columns) == {
        "frequency",
        "magnitude_db",
        "line_type",
        "voice_id",
        "voice_name",
        "sensor_type",
        "description",
    }
    assert df.height == 600

    # Check frequency range
    freqs = df["frequency"].to_list()
    assert freqs[0] >= 20.0
    assert freqs[-1] <= 20000.0

    # Magnitude should be in reasonable dB range (e.g. -60 dB to +20 dB) and strictly finite
    mags = df["magnitude_db"].to_list()
    assert all(-60.0 <= m <= 20.0 for m in mags)
    assert not df["magnitude_db"].is_nan().any()
    assert not df["magnitude_db"].is_null().any()


def test_build_voice_dataframe_modes(
    generic_instrument_config: InstrumentConfig,
):
    """Verifies build_voice_dataframe mode options and mode column inclusion using generic data."""
    voicing = generic_instrument_config.voicings["generic_voice"]
    df_diff = build_voice_dataframe(
        generic_instrument_config,
        voicing,
        mode="difference",
        include_mode_col=True,
    )

    assert "mode" in df_diff.columns
    assert (df_diff["mode"] == "Difference").all()
    assert df_diff.height == 600


def test_build_voice_dataframe_sweep_smoothness_generic(
    generic_instrument_config: InstrumentConfig,
):
    """Validates that build_voice_dataframe produces smooth, ripple-free high-frequency curves
    on generic synthetic data, confirming elimination of Monte Carlo jitter and truncation ripples.
    """
    voicing = generic_instrument_config.voicings["generic_voice"]
    df = build_voice_dataframe(
        generic_instrument_config,
        voicing,
        mode="output",
    )
    freqs = df["frequency"].to_numpy()
    mag = df["magnitude_db"].to_numpy()

    # Second difference standard deviation across high frequencies (10 kHz - 20 kHz)
    high_mask = (freqs >= 10000.0) & (freqs <= 20000.0)
    mag_high = mag[high_mask]
    roughness = float(np.std(np.diff(mag_high, n=2)))
    assert roughness < 0.05, f"High-frequency roughness {roughness} exceeds 0.05 dB"


def test_build_voicings_comparison_data(
    generic_instrument_config: InstrumentConfig,
):
    """Verifies that build_voicings_comparison_data generates valid downsampled client payloads."""
    from allomorph.config.schema import VoiceConfig

    voices = {
        "generic_voice": VoiceConfig(
            id="generic_voice",
            name="Generic Voicing",
            description="Synthetic voice for testing",
            fr=2500.0,
            Q=1.5,
            instrument_id=generic_instrument_config.id,
            coils=[],
        )
    }
    instruments = {generic_instrument_config.id: generic_instrument_config}
    data = build_voicings_comparison_data(voices=voices, instruments=instruments, step=3)
    assert "frequencies" in data
    assert "voices" in data
    assert "families" in data
    assert "default_source" in data
    assert "default_target" in data

    assert len(data["frequencies"]) == 200
    assert len(data["voices"]) > 0
    for vdata in data["voices"].values():
        assert "name" in vdata
        assert "family" in vdata
        assert "magnitude_db" in vdata
        assert len(vdata["magnitude_db"]) == 200
        assert not any(np.isnan(vdata["magnitude_db"]))


def test_build_voicings_comparison_dataframe(
    generic_instrument_config: InstrumentConfig,
    generic_dual_pickup_instrument: InstrumentConfig,
):
    """Verifies identity pairing (exact 0.00 dB) and additive difference math."""
    v_src = generic_instrument_config.voicings["generic_voice"]
    # 1. Identity pair: source == target -> difference is 0.00 dB everywhere
    df_id = build_voicings_comparison_dataframe(
        generic_instrument_config, v_src, generic_instrument_config, v_src, step=1
    )
    assert isinstance(df_id, pl.DataFrame)
    assert df_id.height == 600 * 3
    s3_id = df_id.filter(df_id["line_type"] == "3. Difference")
    assert (s3_id["magnitude_db"] == 0.0).all()

    # 2. Transformative pair: H_src + Difference = H_tgt everywhere
    v_dual = generic_dual_pickup_instrument.voicings["blend"]
    df_tf = build_voicings_comparison_dataframe(
        generic_instrument_config, v_src, generic_dual_pickup_instrument, v_dual, step=1
    )
    s1 = df_tf.filter(df_tf["line_type"] == "1. Source Voicing")["magnitude_db"].to_numpy()
    s2 = df_tf.filter(df_tf["line_type"] == "2. Target Voicing")["magnitude_db"].to_numpy()
    s3 = df_tf.filter(df_tf["line_type"] == "3. Difference")["magnitude_db"].to_numpy()
    assert np.allclose(s1 + s3, s2, atol=0.05)


def test_compute_fir_csd():
    """Verifies cumulative spectral decay waterfall calculation from a synthetic FIR."""
    sr = 48000
    t = np.arange(2048) / sr
    fir = np.exp(-t * 800.0) * np.sin(2.0 * np.pi * 1000.0 * t)
    freqs = np.geomspace(20.0, 20000.0, 50)
    time_ms, csd = compute_fir_csd(fir, freqs, num_slices=24, max_time_ms=10.0, sr=sr)

    assert len(time_ms) == 24
    assert len(csd) == 24
    idx_1k = int(np.argmin(np.abs(freqs - 1000.0)))
    assert csd[0][idx_1k] > csd[-1][idx_1k] + 20.0


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
    rms_val = compute_curve_rms_db(series)
    assert 3.2 <= rms_val <= 3.3
