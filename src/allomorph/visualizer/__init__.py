"""
Allomorph Visualizer Subpackage
Interactive Altair visualizations and Polars frequency response modeling.
"""

import argparse
from collections.abc import Sequence
from pathlib import Path

from allomorph.config.instruments import load_instrument
from allomorph.visualizer.charts import (
    generate_all_charts,
    generate_interactive_chart,
    generate_voicing_ir_3d_page,
    generate_voicings_page,
    render_chart_to_file,
)
from allomorph.visualizer.dataframe import (
    F_MAX,
    F_MIN,
    NUM_POINTS,
    build_voice_dataframe,
    build_voicing_ir_diff_3d_data,
    build_voicings_comparison_data,
    build_voicings_comparison_dataframe,
    compute_curve_rms_db,
    compute_fir_csd,
    log_freqs,
)
from allomorph.visualizer.portal import (
    DOCS_DIR,
    RESPONSES_DIR,
    append_spec_panel,
    build_portal_html,
    format_instrument_meta,
    generate_portal_pages,
)
from allomorph.visualizer.schema import VisualizerCliConfig

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
    "compute_curve_rms_db",
    "compute_fir_csd",
    "format_instrument_meta",
    "generate_all_charts",
    "generate_interactive_chart",
    "generate_portal_pages",
    "generate_voicing_ir_3d_page",
    "generate_voicings_page",
    "log_freqs",
    "main",
    "render_chart_to_file",
]


def main(argv: Sequence[str] | None = None) -> None:
    """CLI entrypoint for interactive frequency response visualizer."""
    parser = argparse.ArgumentParser(
        description="Generate interactive visualizations of Allomorph voicings and instruments."
    )
    parser.add_argument(
        "--instrument",
        "-i",
        default="all",
        help="Source instrument configuration (ID, alias like 30in, 32in, path to .toml, or 'all' to generate all)",
    )
    parser.add_argument(
        "--mode",
        "-m",
        choices=[
            "voicings",
            "voicing_ir_3d",
            "waterfall3d",
            "unified",
            "output",
            "difference",
        ],
        default="voicings",
        help="Chart mode: 'voicings' (interactive 3-line voicing comparison), 'voicing_ir_3d' (3D difference IR waterfall), 'unified', 'output', or 'difference'",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Build interactive charts for all configured instruments and voicings",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output HTML file or directory path (default: docs/frequency_responses/<instrument_id>.html)",
    )
    args = parser.parse_args(argv)
    cli_cfg = VisualizerCliConfig(
        instrument=args.instrument,
        mode=args.mode,
        all=args.all,
        out=args.out,
    )

    if cli_cfg.all or (isinstance(cli_cfg.instrument, str) and cli_cfg.instrument.lower() == "all"):
        generate_all_charts(output_dir=cli_cfg.out)
    elif cli_cfg.mode in ("voicings", "voicing_ir_3d", "waterfall3d"):
        generate_interactive_chart(mode=cli_cfg.mode, out_html=cli_cfg.out)
    else:
        inst = load_instrument(cli_cfg.instrument)
        generate_interactive_chart(instrument=inst, out_html=cli_cfg.out, mode=cli_cfg.mode)
        if cli_cfg.out is None or (Path(cli_cfg.out).resolve() == RESPONSES_DIR.resolve()):
            generate_portal_pages(output_dir=RESPONSES_DIR)
