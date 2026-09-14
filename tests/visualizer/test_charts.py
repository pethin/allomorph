"""
Tests for Altair / Plotly chart generation and portal HTML synthesis in allomorph.visualizer.
"""

import re
import tempfile
from pathlib import Path

from allomorph.config import load_all_instruments
from allomorph.visualizer import (
    generate_all_charts,
    generate_interactive_chart,
    generate_voicing_ir_3d_page,
    generate_voicings_page,
)


def test_generate_all_charts():
    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = Path(tmpdir)
        generated = generate_all_charts(output_dir=out_dir)

        # 1. Check Voicings comparison and 3D Waterfall pages
        assert "voicings" in generated
        assert "voicing_ir_3d" in generated
        assert (out_dir / "voicings.html").exists()
        assert (out_dir / "voicing_ir_3d.html").exists()

        # 2. Verify legacy per-instrument forward signal flow charts are NOT generated
        all_insts = load_all_instruments()
        for inst_id in all_insts:
            assert inst_id not in generated
            assert not (out_dir / f"{inst_id}.html").exists()

        # 3. Verify obsolete Architecture C master pages are NOT generated
        assert not (out_dir / "universal_targets.html").exists()
        assert not (out_dir / "frontend_deconvolutions.html").exists()
        assert not (out_dir / "baked_responses.html").exists()
        assert not (out_dir / "baked_waterfall_3d.html").exists()

        # 4. Check index portal in output directory
        index_path = out_dir / "index.html"
        assert index_path.exists()
        portal_content = index_path.read_text(encoding="utf-8")
        assert "Allomorph Frequency Response Suite" in portal_content
        assert "iframe" in portal_content
        assert "pnav-voicings" in portal_content
        assert "pnav-waterfall3d" in portal_content
        assert "pnav-inspector" not in portal_content
        assert "tab-btn" not in portal_content


def test_generate_interactive_chart_modes():
    with tempfile.TemporaryDirectory() as tmpdir:
        # 1. Output mode
        p_out = Path(tmpdir) / "out.html"
        generate_interactive_chart(instrument="30in", out_html=str(p_out), mode="output")
        assert p_out.exists()
        assert "Output Voice Frequency Responses" in p_out.read_text(encoding="utf-8")

        # 2. Difference mode
        p_diff = Path(tmpdir) / "diff.html"
        generate_interactive_chart(instrument="30in", out_html=str(p_diff), mode="difference")
        assert p_diff.exists()
        assert "Input/Output Differential Transfer Functions" in p_diff.read_text(encoding="utf-8")

        # 3. Unified mode with radio selector
        p_unified = Path(tmpdir) / "unified.html"
        generate_interactive_chart(instrument="30in", out_html=str(p_unified), mode="unified")
        assert p_unified.exists()
        assert "Display Mode: " in p_unified.read_text(encoding="utf-8")

        # 4. Voicings mode
        p_v = Path(tmpdir) / "voicings.html"
        generate_interactive_chart(out_html=str(p_v), mode="voicings")
        assert p_v.exists()
        assert "Voicing Comparisons" in p_v.read_text(encoding="utf-8")

        # 5. Waterfall 3D mode
        p_3d = Path(tmpdir) / "3d.html"
        generate_interactive_chart(out_html=str(p_3d), mode="voicing_ir_3d")
        assert p_3d.exists()
        assert "Voicing IR Difference 3D" in p_3d.read_text(encoding="utf-8")


def test_generate_voicings_page():
    with tempfile.TemporaryDirectory() as tmpdir:
        target_html = Path(tmpdir) / "voicings.html"
        res = generate_voicings_page(target_html)
        assert res.exists()
        content = res.read_text(encoding="utf-8")
        assert "plotly" in content.lower()
        assert "source-select" in content
        assert "target-select" in content
        assert "btn-swap" in content
        assert "plot-div" in content


def test_generate_voicing_ir_3d_page():
    with tempfile.TemporaryDirectory() as tmpdir:
        target_html = Path(tmpdir) / "voicing_ir_3d.html"
        res = generate_voicing_ir_3d_page(target_html)
        assert res.exists()
        content = res.read_text(encoding="utf-8")
        assert "plotly" in content.lower()
        assert "source-select" in content
        assert "target-select" in content
        assert "plot-div" in content
        assert "fir-canvas" in content


def test_portal_html_scripts_valid():
    """Verify that all script tags in generated portals have balanced braces and valid syntax."""
    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = Path(tmpdir)
        generate_all_charts(output_dir=out_dir)

        portal_files = [
            out_dir / "index.html",
            out_dir / "voicings.html",
            out_dir / "voicing_ir_3d.html",
        ]
        for p in portal_files:
            assert p.exists()
            content = p.read_text(encoding="utf-8")
            scripts = re.findall(
                r"<script(?:\s+type=\"text/javascript\")?>(.*?)</script>", content, re.DOTALL
            )
            for s in scripts:
                clean_s = re.sub(r"//.*", "", s)
                clean_s = re.sub(r"/\*.*?\*/", "", clean_s, flags=re.DOTALL)
                clean_s = re.sub(r"'(?:\\.|[^'])*'", "''", clean_s)
                clean_s = re.sub(r'"(?:\\.|[^"])*"', '""', clean_s)
                clean_s = re.sub(r"`(?:\\.|[^`])*`", "``", clean_s)

                stack = []
                matching = {")": "(", "}": "{", "]": "["}
                for char in clean_s:
                    if char in "({[":
                        stack.append(char)
                    elif char in ")}]":
                        assert stack, f"Unmatched closing '{char}' in {p.name}"
                        top = stack.pop()
                        assert top == matching[char], (
                            f"Mismatched '{char}' in {p.name}: expected {matching[char]}, got {top}"
                        )
