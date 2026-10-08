"""
Tests for allomorph.visualizer CLI entrypoint and package exports.
"""

from pathlib import Path
from unittest.mock import patch

from allomorph.visualizer import main


def test_visualizer_main_voicings_mode(tmp_path: Path) -> None:
    """Verify main() generates voicings comparison page."""
    out_html = tmp_path / "voicings_cli.html"
    main(["--mode", "voicings", "--out", str(out_html)])
    assert out_html.exists()
    assert out_html.stat().st_size > 1000


def test_visualizer_main_single_instrument(tmp_path: Path) -> None:
    """Verify main() generates interactive chart for a specific instrument."""
    out_html = tmp_path / "single_p.html"
    main(["--instrument", "34in_standard_p", "--mode", "output", "--out", str(out_html)])
    assert out_html.exists()
    assert out_html.stat().st_size > 1000


def test_visualizer_main_all_flag():
    """Verify main() with --all delegates to generate_all_charts."""
    with patch("allomorph.visualizer.generate_all_charts") as mock_gen:
        main(["--all", "--out", "custom_dir"])
        mock_gen.assert_called_once_with(output_dir="custom_dir")


def test_visualizer_main_instrument_all():
    """Verify main() with --instrument all and non-voicings mode delegates to generate_all_charts."""
    with patch("allomorph.visualizer.generate_all_charts") as mock_gen:
        main(["--instrument", "all", "--mode", "output"])
        mock_gen.assert_called_once_with(output_dir=None)
