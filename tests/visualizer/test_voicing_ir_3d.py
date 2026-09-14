"""
Allomorph - Voicing IR Difference 3D Waterfall Visualizer Test Suite.
Validates the Cumulative Spectral Decay (CSD) waterfall matrix generation,
3D difference impulse response topography (h_diff(t)), time-domain FIR waveform extraction,
and standalone HTML generation.
"""

import json
import tempfile
import time
from pathlib import Path

import numpy as np

from allomorph.config import VOICES
from allomorph.visualizer import (
    build_voicing_ir_diff_3d_data,
    compute_fir_csd,
    generate_voicing_ir_3d_page,
)


def test_compute_fir_csd_structure_and_decay():
    """Validates CSD matrix calculation, time decay progression, and numerical bounds."""
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


def test_build_voicing_ir_diff_3d_data_coverage_and_bounds():
    """Validates full catalog coverage, identity pair behavior, and FIR waveform integrity."""
    t0 = time.perf_counter()
    data = build_voicing_ir_diff_3d_data(num_freqs=50, num_slices=24, max_time_ms=10.0)
    duration_ms = (time.perf_counter() - t0) * 1000.0

    # 1. Performance assertion (cached should be sub-millisecond, cold < 3500 ms)
    assert duration_ms < 3500.0, f"build_voicing_ir_diff_3d_data took {duration_ms:.2f} ms"

    # 2. Top-level keys
    assert "frequencies" in data
    assert "time_ms" in data
    assert "voices" in data
    assert "responses" in data
    assert "default_source" in data
    assert "default_target" in data

    # 3. Frequency & time grid dimensions
    assert len(data["frequencies"]) == 50
    assert len(data["time_ms"]) == 24

    # 4. Voices coverage
    assert set(data["voices"].keys()) == set(VOICES.keys())

    # 5. Responses coverage & identity pair invariants
    responses = data["responses"]
    assert set(responses.keys()) == set(VOICES.keys())

    for vid in VOICES:
        assert vid in responses[vid]
        id_entry = responses[vid][vid]
        assert "magnitude_db" in id_entry
        assert "csd_matrix" in id_entry
        assert "fir_waveform" in id_entry

        # Identity pair FIR waveform must be an ideal unit delta impulse
        fir_wave = id_entry["fir_waveform"]
        assert len(fir_wave) == 128
        assert fir_wave[0] == 1.0
        assert all(x == 0.0 for x in fir_wave[1:])

        # Identity pair initial CSD slice must be flat 0.00 dB
        csd = id_entry["csd_matrix"]
        assert len(csd) == 24
        assert all(val == 0.0 for val in csd[0])
        # Later slices decay to -60 dB floor
        assert all(val == -60.0 for val in csd[-1])


def test_generate_voicing_ir_3d_page_html_integrity():
    """Validates HTML generation, WebGL 3D surface container, and canvas waveform."""
    with tempfile.TemporaryDirectory() as tmpdir:
        out_file = Path(tmpdir) / "voicing_ir_3d.html"
        res_path = generate_voicing_ir_3d_page(out_file)
        assert res_path.exists()
        content = res_path.read_text(encoding="utf-8")

        assert "<!DOCTYPE html>" in content
        assert "Plotly" in content or "plotly" in content
        assert "plot-div" in content
        assert "fir-canvas" in content
        assert "source-select" in content
        assert "target-select" in content

        # Check embedded JSON payload
        marker_start = '<script id="waterfall-data" type="application/json">'
        assert marker_start in content
        json_part = content.split(marker_start)[1].split("</script>")[0]
        parsed_data = json.loads(json_part)
        assert "frequencies" in parsed_data
        assert "responses" in parsed_data
