"""
Allomorph - Baked Responses Visualizer Test Suite.
Validates the monolithic 1-block differential transfer functions (H_diff = H_tgt / H_src),
compact matrix serialization, single-file HTML generation, and portal integration.
"""

import json
import re
import tempfile
import time
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from allomorph.config import VOICES, load_all_instruments
from allomorph.visualizer import (
    build_baked_responses_data,
    build_baked_responses_dataframe,
    generate_all_charts,
    generate_baked_responses_page,
    generate_interactive_chart,
)


def test_build_baked_responses_data_structure_and_bounds():
    """Validates the compact JSON data structure, coverage of all instruments & voices, and physical bounds."""
    # Warm up cache
    build_baked_responses_data(step=3)

    t0 = time.perf_counter()
    data = build_baked_responses_data(step=3)
    duration_ms = (time.perf_counter() - t0) * 1000.0

    # 1. Performance guardrail: building data must execute in < 150 ms warm
    assert duration_ms < 150.0, f"build_baked_responses_data took {duration_ms:.2f} ms (budget: < 150 ms)"

    # 2. Key schema assertions
    assert "frequencies" in data
    assert "instruments" in data
    assert "voices" in data
    assert "responses" in data

    # 3. Frequencies downsampling: 200 points across 20 Hz to 20 kHz
    freqs = data["frequencies"]
    assert len(freqs) == 200
    assert freqs[0] == pytest.approx(20.0, abs=0.2)
    assert 19000.0 <= freqs[-1] <= 20000.0

    # 4. Coverage of all 12 playable instruments
    all_insts = load_all_instruments()
    expected_inst_ids = {iid for iid in all_insts if iid != "canonical_intermediate"}
    assert set(data["instruments"].keys()) == expected_inst_ids

    # 5. Coverage of all 23 target voices
    expected_voice_ids = {vid for vid in VOICES if vid != "00_canonical_intermediate"}
    assert set(data["voices"].keys()) == expected_voice_ids

    # 6. Response matrix integrity and physical electroacoustic bounds
    responses = data["responses"]
    total_curves = 0
    for iid in expected_inst_ids:
        assert iid in responses
        for vid in expected_voice_ids:
            assert vid in responses[iid]
            entry = responses[iid][vid]
            assert "pickup_key" in entry
            assert "pickup_name" in entry
            assert "magnitude_db" in entry

            mags = entry["magnitude_db"]
            assert len(mags) == 200
            m_arr = np.array(mags, dtype=np.float64)
            assert not np.isnan(m_arr).any()
            assert not np.isinf(m_arr).any()

            # Physical magnitude bounds [-100 dB to +30 dB] per Guardrail 5.3
            assert np.max(m_arr) < 30.0, f"{iid} -> {vid} peak {np.max(m_arr)} exceeds 30 dB"
            assert np.min(m_arr) > -100.0, f"{iid} -> {vid} notch {np.min(m_arr)} below -100 dB"

            # 2-decimal rounding guardrail
            for m in mags:
                assert round(m, 2) == m, f"Unrounded float {m} in {iid} -> {vid}"

            total_curves += 1

    assert total_curves == len(expected_inst_ids) * len(expected_voice_ids)


def test_build_baked_responses_identity_bypass():
    """Validates that identity pairings evaluate to exact 0.00 dB reflecting skip_identity=True."""
    data = build_baked_responses_data(step=3)

    identity_pairs = [
        ("34in_standard_p", "05_vintage_62_p_alnico"),
        ("34in_active_stingray", "09_stingray_mm_parallel"),
    ]

    for iid, vid in identity_pairs:
        resp = data["responses"][iid][vid]
        mags = resp["magnitude_db"]
        assert all(m == 0.0 for m in mags), (
            f"Identity pair {iid} -> {vid} must evaluate to exact 0.00 dB, got {mags[:5]}"
        )

    # 15_neutral_character must evaluate to bit-exact 0.00 dB across ALL instruments
    for iid in data["instruments"]:
        resp = data["responses"][iid]["15_neutral_character"]
        mags = resp["magnitude_db"]
        assert all(m == 0.0 for m in mags), (
            f"15_neutral_character on {iid} must evaluate to bit-exact 0.00 dB, got {mags[:5]}"
        )

    # 15b_active_character and 15c_passive_character must be transformative (NOT flat)
    for vid in ["15b_active_character", "15c_passive_character"]:
        resp = data["responses"]["30in_emg_mmtw"][vid]
        mags = resp["magnitude_db"]
        assert not all(m == 0.0 for m in mags), f"{vid} on 30in must NOT be flat 0.00 dB"
        assert max(mags) - min(mags) > 1.0, f"{vid} dynamic range must be > 1.0 dB"

    # For 34in_active_stingray: parallel must be flat 0.00 dB, but series must NOT be flat
    ray_par_mags = data["responses"]["34in_active_stingray"]["09_stingray_mm_parallel"]["magnitude_db"]
    assert all(m == 0.0 for m in ray_par_mags), "StingRay parallel must be bit-exact flat 0.00 dB"
    ray_ser_mags = data["responses"]["34in_active_stingray"]["09b_stingray_mm_series"]["magnitude_db"]
    assert not all(m == 0.0 for m in ray_ser_mags), "StingRay series must NOT be flat on parallel StingRay"
    assert max(ray_ser_mags) - min(ray_ser_mags) > 5.0, "StingRay series dynamic range must exceed 5.0 dB"


def test_build_baked_responses_dataframe_filtering():
    """Validates the Polars DataFrame interface and multi-select filtering."""
    # Filtered by 1 instrument and 2 voices
    df_filtered = build_baked_responses_dataframe(
        instrument_ids=["34in_standard_p"],
        voice_ids=["05_vintage_62_p_alnico", "02_jazz_bass_pair"],
        step=3,
    )
    assert isinstance(df_filtered, pl.DataFrame)
    assert df_filtered.height == 400  # 2 curves * 200 points
    assert set(df_filtered["instrument_id"].unique().to_list()) == {"34in_standard_p"}
    assert set(df_filtered["voice_id"].unique().to_list()) == {
        "05_vintage_62_p_alnico",
        "02_jazz_bass_pair",
    }
    assert "label" in df_filtered.columns
    assert "magnitude_db" in df_filtered.columns
    assert "pickup_name" in df_filtered.columns

    # Full unfiltered matrix
    df_all = build_baked_responses_dataframe(step=3)
    assert df_all.height == 12 * 23 * 200  # 55,200 rows


def test_generate_baked_responses_page_single_file():
    """Validates that generate_baked_responses_page generates a self-contained, valid HTML file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        out_html = Path(tmpdir) / "baked_responses.html"
        generated_path = generate_baked_responses_page(target_html=out_html)

        assert generated_path == out_html
        assert out_html.exists()
        content = out_html.read_text(encoding="utf-8")

        # 1. File size bounded under 600 KB
        size_kb = len(content.encode("utf-8")) / 1024.0
        assert 300.0 < size_kb < 600.0, f"HTML size {size_kb:.1f} KB outside expected range [300, 600] KB"

        # 2. Key UI elements
        assert "Baked Transformations" in content
        assert "chip-btn" in content
        assert "selectAllInstruments" in content
        assert "filterVoicingsByFamily" in content
        assert "vegaEmbed" in content
        assert 'id="baked-data"' in content

        # 3. Embedded JSON validity
        match = re.search(r'<script id="baked-data" type="application/json">(.*?)</script>', content, re.DOTALL)
        assert match is not None
        embedded_json = json.loads(match.group(1).strip())
        assert len(embedded_json["instruments"]) == 12
        assert len(embedded_json["voices"]) == 23

        # 4. Valid balanced script tags
        scripts = re.findall(r'<script(?:\s+type="text/javascript")?>(.*?)</script>', content, re.DOTALL)
        for s in scripts:
            clean_s = re.sub(r"'(?:\\.|[^'])*'", "''", s)
            clean_s = re.sub(r'"(?:\\.|[^"])*"', '""', clean_s)
            clean_s = re.sub(r"`(?:\\.|[^`])*`", "``", clean_s)
            clean_s = re.sub(r"//.*", "", clean_s)
            clean_s = re.sub(r"/\*.*?\*/", "", clean_s, flags=re.DOTALL)
            stack = []
            matching = {")": "(", "}": "{", "]": "["}
            for char in clean_s:
                if char in "({[":
                    stack.append(char)
                elif char in ")}]":
                    assert stack, f"Unmatched closing '{char}'"
                    top = stack.pop()
                    assert top == matching[char], f"Mismatched '{char}': expected {matching[char]}, got {top}"
            assert len(stack) == 0, f"Unclosed braces: {stack}"


def test_portal_baked_tab_integration():
    """Validates that the portal master pages include the Baked tab and #baked routing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = Path(tmpdir)
        generated = generate_all_charts(output_dir=out_dir)

        # 1. Baked HTML generated and present in returned mapping
        assert "baked" in generated
        assert (out_dir / "baked_responses.html").exists()

        # 2. Portal index includes pnav-baked and hash routing
        index_html = (out_dir / "index.html").read_text(encoding="utf-8")
        assert 'id="pnav-baked"' in index_html
        assert "Baked Responses (1-Block)" in index_html
        assert "baked_responses.html" in index_html
        assert "hash === 'baked'" in index_html


def test_interactive_chart_mode_baked():
    """Validates that generate_interactive_chart with mode='baked' delegates to generate_baked_responses_page."""
    with tempfile.TemporaryDirectory() as tmpdir:
        out_html = Path(tmpdir) / "custom_baked.html"
        result_path = generate_interactive_chart(mode="baked", out_html=out_html)
        assert result_path == out_html
        assert out_html.exists()
        assert "Baked Transformations" in out_html.read_text(encoding="utf-8")
