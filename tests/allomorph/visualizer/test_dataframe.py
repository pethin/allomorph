"""
Tests for Polars-based frequency response dataframe generation in allomorph.visualizer.
All tests in this module check the behavior of the SUT using generic synthetic data.
(Specific instrument curves are tested in the separate tests/integration/ test collection).
"""

from typing import TYPE_CHECKING

import numpy as np
import polars as pl

from allomorph.visualizer.dataframe import (
    build_voice_dataframe,
    build_voicings_comparison_data,
    build_voicings_comparison_dataframe,
    compute_curve_rms_db,
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

    # Magnitude should be in reasonable dB range (-120 dB floor to +20 dB) and strictly finite
    mags = df["magnitude_db"].to_list()
    assert all(-120.0 <= m <= 20.0 for m in mags)
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
    """Verifies that build_voicings_comparison_data generates valid comparison payload directly from instruments."""
    instruments = {generic_instrument_config.id: generic_instrument_config}
    data = build_voicings_comparison_data(instruments=instruments, step=3)
    assert len(data["frequencies"]) == 200
    assert len(data["voices"]) == len(generic_instrument_config.voicings)
    for v_meta in data["voices"].values():
        assert len(v_meta["magnitude_db"]) == 200
        assert not np.isnan(v_meta["magnitude_db"]).any()


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
