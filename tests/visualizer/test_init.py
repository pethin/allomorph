"""
Tests for allomorph.visualizer package exports and visualize CLI command.
"""

from pathlib import Path
from unittest.mock import patch

import allomorph.visualizer as viz
from allomorph.cli import main as cli_main


def test_visualizer_exports() -> None:
    """Verify public exports of allomorph.visualizer."""
    expected = {
        "DOCS_DIR",
        "F_MAX",
        "F_MIN",
        "NUM_POINTS",
        "build_voice_dataframe",
        "build_voicings_comparison_data",
        "build_voicings_comparison_dataframe",
        "build_voicings_html",
        "compute_curve_rms_db",
        "generate_all_charts",
        "generate_interactive_chart",
        "generate_voicings_page",
        "log_freqs",
        "render_chart_to_file",
    }
    assert set(viz.__all__) == expected
    for name in expected:
        assert hasattr(viz, name)


def test_visualize_cli_single_instrument(tmp_path: Path) -> None:
    """Verify allomorph visualize generates interactive chart for a specific instrument."""
    out_html = tmp_path / "single_p.html"
    rc = cli_main(["visualize", "34in_standard_p", "--mode", "output", "--out", str(out_html)])
    assert rc == 0
    assert out_html.exists()
    assert out_html.stat().st_size > 1000


def test_visualize_cli_all() -> None:
    """Verify allomorph visualize all delegates to generate_all_charts."""
    with patch("allomorph.cli.visualize_cmd.generate_all_charts") as mock_gen:
        rc = cli_main(["visualize", "all", "--out", "custom_dir"])
        assert rc == 0
        mock_gen.assert_called_once_with(output_dir=Path("custom_dir"))
