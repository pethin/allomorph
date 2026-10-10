"""
Tests for HTML portal generation and metadata formatting in allomorph.visualizer.portal.
"""

from pathlib import Path

import pytest

from allomorph.config.instruments import load_instrument
from allomorph.config.voices import VOICES
from allomorph.visualizer.portal import (
    append_spec_panel,
    build_portal_html,
    format_instrument_meta,
    generate_portal_pages,
)
from allomorph.visualizer.schema import PortalInstrumentMeta


def test_append_spec_panel(tmp_path: Path):
    """Validates appending an informational spec panel right before </body>."""
    html_file = tmp_path / "chart.html"
    html_file.write_text("<html><body><div>Chart Content</div></body></html>", encoding="utf-8")

    panel_html = "<div id='spec-panel'>Spec Directive Details</div>"
    append_spec_panel(html_file, panel_html)

    updated_content = html_file.read_text(encoding="utf-8")
    assert "<div id='spec-panel'>Spec Directive Details</div>\n</body>" in updated_content
    assert "Chart Content" in updated_content

    # If </body> is not present, content should remain unmodified without errors
    no_body_file = tmp_path / "no_body.html"
    no_body_file.write_text("<div>Plain Content</div>", encoding="utf-8")
    append_spec_panel(no_body_file, panel_html)
    assert no_body_file.read_text(encoding="utf-8") == "<div>Plain Content</div>"


def test_format_instrument_meta():
    """Validates formatting an InstrumentConfig into PortalInstrumentMeta."""
    inst_30 = load_instrument("30in")
    meta_30 = format_instrument_meta(inst_30)

    assert isinstance(meta_30, PortalInstrumentMeta)
    assert meta_30.id == "30in_mustang_pj"
    assert meta_30.scale_in == 30.0
    assert meta_30.scale_m == round(30.0 * 0.0254, 4)
    assert "m/s" in meta_30.speeds_str
    assert "mm" in meta_30.pickups_summary

    inst_multi = load_instrument("37in_multiscale_dingwall")
    meta_multi = format_instrument_meta(inst_multi)
    assert meta_multi.id == "37in_multiscale_dingwall"
    assert "Bridge" in meta_multi.pickups_summary or "FD3" in meta_multi.pickups_summary


def test_build_portal_html():
    """Validates construction of responsive portal HTML string."""
    html = build_portal_html(default_id="voicings", base_url_prefix="./")
    assert "<!DOCTYPE html>" in html
    assert "Allomorph | Voicing Comparisons & Frequency Response Suite" in html
    assert f"{len(VOICES)} Catalog Voicings" in html
    assert "H_src (Cyan)" in html
    assert "H_tgt (Orange)" in html
    assert "H_diff (Purple)" in html
    assert "selectPrimaryView" in html
    assert "updateView" in html

    # Custom base URL prefix
    custom_prefix_html = build_portal_html(base_url_prefix="./frequency_responses/")
    assert "./frequency_responses/" in custom_prefix_html

    # Custom instruments meta mapping
    inst_30 = load_instrument("30in")
    custom_meta = {"30in": format_instrument_meta(inst_30)}
    html_custom = build_portal_html(instruments_meta=custom_meta)
    assert "<!DOCTYPE html>" in html_custom


def test_generate_portal_pages(tmp_path: Path):
    """Validates writing index.html to target directory."""
    generate_portal_pages(output_dir=tmp_path, default_id="voicings")
    index_file = tmp_path / "index.html"
    assert index_file.exists()
    content = index_file.read_text(encoding="utf-8")
    assert "<!DOCTYPE html>" in content
    assert "Allomorph | Voicing Comparisons" in content
    assert f"{len(VOICES)} Catalog Voicings" in content


def test_generate_portal_pages_responses_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Validates writing index.html and root frequency_responses.html when output_dir == RESPONSES_DIR."""
    resp_dir = tmp_path / "frequency_responses"
    resp_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr("allomorph.visualizer.portal.RESPONSES_DIR", resp_dir)
    monkeypatch.setattr("allomorph.visualizer.portal.DOCS_DIR", tmp_path)
    generate_portal_pages(output_dir=resp_dir)
    assert (resp_dir / "index.html").exists()
    assert (tmp_path / "frequency_responses.html").exists()
