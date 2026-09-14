#!/usr/bin/env python3
"""
Allomorph - Interactive Visualizer & Frequency Analyzer CLI
Uses Polars, Altair, and Plotly to model, analyze, and render interactive frequency
response curves for all Master Voices across source bass instruments.

Delegates core logic to the library package allomorph.visualizer.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from allomorph.config import load_all_instruments, load_instrument
from allomorph.visualizer import (
    DOCS_DIR,
    F_MAX,
    F_MIN,
    NUM_POINTS,
    RESPONSES_DIR,
    append_spec_panel,
    build_portal_html,
    build_voice_dataframe,
    build_voicing_ir_diff_3d_data,
    build_voicings_comparison_data,
    build_voicings_comparison_dataframe,
    compute_fir_csd,
    format_instrument_meta,
    generate_all_charts,
    generate_interactive_chart,
    generate_portal_pages,
    generate_voicing_ir_3d_page,
    generate_voicings_page,
    log_freqs,
    main,
    render_chart_to_file,
)

__all__ = [
    "DOCS_DIR",
    "F_MAX",
    "F_MIN",
    "NUM_POINTS",
    "RESPONSES_DIR",
    "append_spec_panel",
    "build_portal_html",
    "build_voice_dataframe",
    "build_voicing_ir_diff_3d_data",
    "build_voicings_comparison_data",
    "build_voicings_comparison_dataframe",
    "compute_fir_csd",
    "format_instrument_meta",
    "generate_all_charts",
    "generate_interactive_chart",
    "generate_portal_pages",
    "generate_voicing_ir_3d_page",
    "generate_voicings_page",
    "load_all_instruments",
    "load_instrument",
    "log_freqs",
    "main",
    "render_chart_to_file",
]

if __name__ == "__main__":
    main()
