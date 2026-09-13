#!/usr/bin/env python3
"""
Allomorph - Interactive Visualizer & Frequency Analyzer CLI
Uses Polars and Altair to model, analyze, and render interactive frequency
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
    build_baked_responses_data,
    build_baked_responses_dataframe,
    build_composite_instrument_dataframe,
    build_frontend_deconvolutions_dataframe,
    build_instrument_frontend_dataframe,
    build_portal_html,
    build_universal_targets_dataframe,
    build_voice_dataframe,
    compute_canonical_intermediate_response,
    format_instrument_meta,
    generate_all_charts,
    generate_baked_responses_page,
    generate_composite_instrument_chart,
    generate_frontend_deconvolutions_chart,
    generate_instrument_frontend_chart,
    generate_interactive_chart,
    generate_portal_pages,
    generate_universal_targets_chart,
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
    "build_baked_responses_data",
    "build_baked_responses_dataframe",
    "build_composite_instrument_dataframe",
    "build_frontend_deconvolutions_dataframe",
    "build_instrument_frontend_dataframe",
    "build_portal_html",
    "build_universal_targets_dataframe",
    "build_voice_dataframe",
    "compute_canonical_intermediate_response",
    "format_instrument_meta",
    "generate_all_charts",
    "generate_baked_responses_page",
    "generate_composite_instrument_chart",
    "generate_frontend_deconvolutions_chart",
    "generate_instrument_frontend_chart",
    "generate_interactive_chart",
    "generate_portal_pages",
    "generate_universal_targets_chart",
    "load_all_instruments",
    "load_instrument",
    "log_freqs",
    "main",
    "render_chart_to_file",
]

if __name__ == "__main__":
    main()
