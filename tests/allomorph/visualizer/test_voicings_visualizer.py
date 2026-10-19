"""
Allomorph - Voicings Comparison Visualizer Test Suite.
Validates the interactive Altair voicing comparison (Source vs Target Voicing),
3-line frequency response curves (H_src, H_tgt, H_diff), identity pairing evaluation,
and standalone Altair HTML generation.
"""

import tempfile
from pathlib import Path

import numpy as np

from allomorph.config import INSTRUMENTS
from allomorph.config.instruments import resolve_target_voicing
from allomorph.visualizer import (
    build_voicings_comparison_data,
    generate_interactive_chart,
)
from allomorph.visualizer.dataframe import build_voicings_comparison_dataframe


def test_build_voicings_comparison_data_bounds_and_coverage():
    """Validates frequency bounds and catalog coverage of build_voicings_comparison_data."""
    data = build_voicings_comparison_data(step=3)
    expected_keys = {
        f"{inst_id}:{v_id}" for inst_id, inst in INSTRUMENTS.items() for v_id in inst.voicings
    }
    assert set(data["voices"].keys()) == expected_keys

    for v_meta in data["voices"].values():
        mags = np.asarray(v_meta["magnitude_db"], dtype=np.float64)
        assert not np.isnan(mags).any()
        assert not np.isinf(mags).any()
        assert np.max(mags) < 25.0
        assert np.min(mags) >= -120.0


def test_voicings_comparison_identity_and_differential_math():
    """Validates that identity pairings evaluate to exact 0.00 dB, and difference equals Target - Source."""
    test_identities = [
        "34in_standard_p:vintage_open",
        "34in_standard_jazz:bridge_growl",
        "34in_active_stingray:parallel_classic",
        "41in_upright_bass:bridge_piezo",
    ]
    for vid in test_identities:
        inst, v_cfg = resolve_target_voicing(vid)
        df_id = build_voicings_comparison_dataframe(inst, v_cfg, inst, v_cfg, step=1)
        s3 = df_id.filter(df_id["line_type"] == "3. Difference")["magnitude_db"]
        assert (s3 == 0.0).all(), f"Identity pair {vid} -> {vid} did not evaluate to 0.00 dB"

    # Transformative pair: Source + Difference == Target everywhere
    s_inst, s_v_cfg = resolve_target_voicing("34in_standard_p:vintage_open")
    t_inst, t_v_cfg = resolve_target_voicing("34in_standard_jazz:bridge_growl")
    df_diff = build_voicings_comparison_dataframe(s_inst, s_v_cfg, t_inst, t_v_cfg, step=1)
    s1 = df_diff.filter(df_diff["line_type"] == "1. Source Voicing")["magnitude_db"].to_numpy()
    s2 = df_diff.filter(df_diff["line_type"] == "2. Target Voicing")["magnitude_db"].to_numpy()
    s3 = df_diff.filter(df_diff["line_type"] == "3. Difference")["magnitude_db"].to_numpy()

    assert np.allclose(s1 + s3, s2, atol=0.05)


def test_generate_interactive_chart_voicings_html_integrity():
    """Validates Altair HTML generation for voicings mode."""
    with tempfile.TemporaryDirectory() as tmpdir:
        out_file = Path(tmpdir) / "voicings.html"
        res_path = generate_interactive_chart(out_html=out_file, mode="voicings")
        assert res_path.exists()
        content = res_path.read_text(encoding="utf-8")
        assert "Allomorph Frequency Response Suite" in content
        assert "34in_standard_p:vintage_open" in content
