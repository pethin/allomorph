"""
Allomorph - Baked Voicing IR 3D Waterfall Visualizer Test Suite.
Validates the Cumulative Spectral Decay (CSD) waterfall matrix generation,
3D multi-voicing topography, time-domain FIR waveform extraction, and standalone HTML generation.
"""

import tempfile
import time
from pathlib import Path

import numpy as np

from allomorph.config import VOICES, load_all_instruments
from allomorph.visualizer.charts import generate_baked_waterfall_3d_page
from allomorph.visualizer.dataframe import (
    build_baked_waterfall_3d_data,
    compute_fir_csd,
)


def test_compute_fir_csd_structure_and_decay():
    """Validates CSD matrix calculation, time decay progression, and numerical bounds."""
    # Synthetic impulse response: 1000 Hz damped sinusoid
    sr = 48000
    t = np.arange(2048) / sr
    fir = np.exp(-t * 800.0) * np.sin(2.0 * np.pi * 1000.0 * t)

    freqs = np.geomspace(20.0, 20000.0, 50)
    time_ms, csd = compute_fir_csd(fir, freqs, num_slices=24, max_time_ms=10.0, sr=sr)

    # 1. Structure assertions
    assert len(time_ms) == 24
    assert time_ms[0] == 0.0
    assert 9.9 <= time_ms[-1] <= 10.1

    assert len(csd) == 24
    for row in csd:
        assert len(row) == 50
        assert not any(np.isnan(row))
        assert not any(np.isinf(row))

    # 2. Time decay verification: energy at resonance (1000 Hz) must decay over time
    idx_1k = int(np.argmin(np.abs(freqs - 1000.0)))
    mag_initial = csd[0][idx_1k]
    mag_decayed = csd[-1][idx_1k]
    assert mag_initial > mag_decayed + 20.0, (
        f"Expected decay at 1kHz, got {mag_initial} -> {mag_decayed}"
    )


def test_build_baked_waterfall_3d_data_coverage_and_bounds():
    """Validates full catalog coverage, identity pair behavior, and FIR waveform integrity."""
    t0 = time.perf_counter()
    data = build_baked_waterfall_3d_data(num_freqs=50, num_slices=24, max_time_ms=10.0)
    duration_ms = (time.perf_counter() - t0) * 1000.0

    # 1. Performance assertion (cached should be sub-millisecond, cold < 3500 ms)
    assert duration_ms < 3500.0, f"build_baked_waterfall_3d_data took {duration_ms:.2f} ms"

    # 2. Top-level keys
    assert "frequencies" in data
    assert "time_ms" in data
    assert "instruments" in data
    assert "voices" in data
    assert "responses" in data

    assert len(data["frequencies"]) == 50
    assert len(data["time_ms"]) == 24

    # 3. Coverage of playable instruments and target voices
    all_insts = load_all_instruments()
    expected_inst_ids = {iid for iid in all_insts if iid != "canonical_intermediate"}
    expected_voice_ids = {vid for vid in VOICES if vid != "00_canonical_intermediate"}

    assert set(data["instruments"].keys()) == expected_inst_ids
    assert set(data["voices"].keys()) == expected_voice_ids

    responses = data["responses"]
    for iid in expected_inst_ids:
        assert iid in responses
        for vid in expected_voice_ids:
            entry = responses[iid][vid]
            assert "pickup_key" in entry
            assert "pickup_name" in entry
            assert "magnitude_db" in entry
            assert "csd_matrix" in entry
            assert "fir_waveform" in entry

            mags = entry["magnitude_db"]
            csd = entry["csd_matrix"]
            fir_head = entry["fir_waveform"]

            assert len(mags) == 50
            assert len(csd) == 24
            assert len(csd[0]) == 50
            assert len(fir_head) == 128

            # Physical magnitude bounds [-60 dB to +30 dB]
            csd_arr = np.array(csd)
            assert np.max(csd_arr) < 30.0, f"{iid} -> {vid} peak {np.max(csd_arr)} exceeds 30 dB"
            assert np.min(csd_arr) >= -60.0, f"{iid} -> {vid} floor {np.min(csd_arr)} below -60 dB"

    # 4. Identity pair verification: 34in_standard_p -> 05_vintage_62_p_alnico
    p_identity = responses["34in_standard_p"]["05_vintage_62_p_alnico"]
    assert p_identity["fir_waveform"][0] == 1.0
    assert p_identity["fir_waveform"][1] == 0.0
    assert p_identity["csd_matrix"][0] == [0.0] * 50
    assert p_identity["csd_matrix"][-1] == [-60.0] * 50


def test_generate_baked_waterfall_3d_page_file_creation():
    """Validates generation of the standalone WebGL HTML page and payload constraints."""
    with tempfile.TemporaryDirectory() as tmpdir:
        out_html = Path(tmpdir) / "test_waterfall_3d.html"
        generated = generate_baked_waterfall_3d_page(out_html)

        assert generated.exists()
        assert generated == out_html

        content = generated.read_text(encoding="utf-8")
        assert "Plotly.react" in content
        assert "plotly-2.35.2.min.js" in content
        assert "Cumulative Spectral Decay" in content
        assert "waterfall-data" in content

        # Size payload constraint: strictly < 4.0 MB
        size_mb = generated.stat().st_size / (1024 * 1024)
        assert size_mb < 4.0, f"Payload {size_mb:.2f} MB exceeds 4.0 MB limit"
