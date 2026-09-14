"""
Allomorph - Voicings Comparison Visualizer Test Suite.
Validates the interactive 2D voicing comparison (Source vs Target Voicing),
3-line frequency response curves (H_src, H_tgt, H_diff), identity pairing evaluation,
and standalone HTML generation.
"""

import json
import tempfile
import time
from pathlib import Path

import numpy as np
import pytest

from allomorph.config import VOICES
from allomorph.visualizer import (
    build_voicings_comparison_data,
    build_voicings_comparison_dataframe,
    compute_curve_rms_db,
    generate_voicings_page,
)


def test_build_voicings_comparison_data_performance_and_bounds():
    """Validates warm cache performance, frequency bounds, and catalog coverage."""
    # Warm up cache
    build_voicings_comparison_data(step=3)

    t0 = time.perf_counter()
    data = build_voicings_comparison_data(step=3)
    duration_ms = (time.perf_counter() - t0) * 1000.0

    # 1. Performance guardrail: building data must execute in < 150 ms warm
    assert duration_ms < 150.0, f"build_voicings_comparison_data took {duration_ms:.2f} ms"

    # 2. Key schema assertions
    assert "frequencies" in data
    assert "voices" in data
    assert "families" in data
    assert "default_source" in data
    assert "default_target" in data

    # 3. Frequencies downsampling: 200 points across 20 Hz to 20 kHz
    freqs = data["frequencies"]
    assert len(freqs) == 200
    assert freqs[0] == pytest.approx(20.0, abs=0.2)
    assert 19000.0 <= freqs[-1] <= 20000.0

    # 4. Coverage of all registered target voices
    assert set(data["voices"].keys()) == set(VOICES.keys())

    # 5. Response curves validity and bounded physical limits
    for vid, entry in data["voices"].items():
        assert "name" in entry
        assert "family" in entry
        assert "magnitude_db" in entry
        mags = entry["magnitude_db"]
        assert len(mags) == 200
        m_arr = np.array(mags, dtype=np.float64)
        assert not np.isnan(m_arr).any()
        assert not np.isinf(m_arr).any()
        assert np.max(m_arr) < 25.0, f"{vid} peak {np.max(m_arr)} exceeds 25 dB"
        assert np.min(m_arr) > -120.0, f"{vid} notch {np.min(m_arr)} below -120 dB"

        # 2-decimal rounding guardrail
        for m in mags:
            assert round(m, 2) == m, f"Unrounded float {m} in voice {vid}"


def test_voicings_comparison_identity_and_differential_math():
    """Validates that identity pairings evaluate to exact 0.00 dB, and difference equals Target - Source."""
    # 1. Identity pair: Source == Target -> H_diff == 0.00 dB everywhere
    test_identities = [
        "precision_vintage",
        "jazz_bridge_growl",
        "stingray_parallel",
        "studio_direct",
    ]
    for vid in test_identities:
        df_id = build_voicings_comparison_dataframe(vid, vid, step=1)
        s3 = df_id.filter(df_id["line_type"] == "3. Normalized Difference (Norm. Diff)")[
            "magnitude_db"
        ]
        assert (s3 == 0.0).all(), f"Identity pair {vid} -> {vid} did not evaluate to 0.00 dB"

    # 2. Transformative pair: Target,norm - Source,norm == Norm. Diff in the passband (< 3.5 kHz)
    df_diff = build_voicings_comparison_dataframe("precision_vintage", "jazz_bridge_growl", step=1)
    s1 = df_diff.filter(df_diff["line_type"] == "1. Source Voicing")["magnitude_db"].to_numpy()
    s2 = df_diff.filter(df_diff["line_type"] == "2. Target Voicing")["magnitude_db"].to_numpy()
    s3 = df_diff.filter(df_diff["line_type"] == "3. Normalized Difference (Norm. Diff)")[
        "magnitude_db"
    ].to_numpy()
    freqs = df_diff.filter(df_diff["line_type"] == "1. Source Voicing")["frequency"].to_numpy()
    idx_pass = freqs <= 3000.0

    src_rms = compute_curve_rms_db(s1)
    tgt_rms = compute_curve_rms_db(s2)
    s1_norm = s1 - src_rms
    s2_norm = s2 - tgt_rms

    assert np.allclose((s2_norm - s1_norm)[idx_pass], s3[idx_pass], atol=0.10)
    # Beyond 5 kHz, response rolls off smoothly according to input power spectrum
    # Boost is strictly soft-knee bounded at <= +7.05 dB
    assert np.max(s3) <= 7.05
    assert s3[-1] <= 0.0


def test_generate_voicings_page_html_integrity():
    """Validates HTML generation, client-side data serialization, and DOM controls."""
    with tempfile.TemporaryDirectory() as tmpdir:
        out_file = Path(tmpdir) / "voicings.html"
        res_path = generate_voicings_page(out_file)
        assert res_path.exists()
        content = res_path.read_text(encoding="utf-8")

        # HTML structure assertions
        assert "<!DOCTYPE html>" in content
        assert "Plotly" in content or "plotly" in content
        assert "source-select" in content
        assert "target-select" in content
        assert "btn-swap" in content
        assert "plot-div" in content

        # Check that JSON data payload is embedded and parseable
        marker_start = '<script id="voicings-data" type="application/json">'
        assert marker_start in content
        json_part = content.split(marker_start)[1].split("</script>")[0]
        parsed_data = json.loads(json_part)
        assert "frequencies" in parsed_data
        assert "voices" in parsed_data
        assert len(parsed_data["voices"]) == len(VOICES)
