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
    generate_baked_responses_page,
    generate_composite_instrument_chart,
    generate_frontend_deconvolutions_chart,
    generate_instrument_frontend_chart,
    generate_interactive_chart,
    generate_universal_targets_chart,
    render_chart_to_file,
)
from allomorph.visualizer.dataframe import (
    F_MAX,
    F_MIN,
    NUM_POINTS,
    build_baked_responses_data,
    build_baked_responses_dataframe,
    build_composite_instrument_dataframe,
    build_frontend_deconvolutions_dataframe,
    build_instrument_frontend_dataframe,
    build_universal_targets_dataframe,
    build_voice_dataframe,
    compute_canonical_acoustic_response,
    compute_canonical_intermediate_response,
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
    "build_baked_responses_data",
    "build_baked_responses_dataframe",
    "build_composite_instrument_dataframe",
    "build_frontend_deconvolutions_dataframe",
    "build_instrument_frontend_dataframe",
    "build_portal_html",
    "build_universal_targets_dataframe",
    "build_voice_dataframe",
    "compute_canonical_acoustic_response",
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
    "log_freqs",
    "main",
    "render_chart_to_file",
]


def main(argv: Sequence[str] | None = None) -> None:
    """CLI entrypoint for interactive frequency response visualizer."""
    parser = argparse.ArgumentParser(
        description="Generate interactive Altair visualization of Allomorph voices."
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
        choices=["composite", "unified", "output", "difference", "targets", "frontends", "baked"],
        default="composite",
        help="Chart mode: 'composite' (3-curve overlay), 'unified', 'output', 'difference', 'targets', 'frontends', or 'baked'",
    )
    parser.add_argument(
        "--all", action="store_true", help="Build interactive charts for all configured instruments"
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
    elif cli_cfg.mode in ("targets", "frontends", "baked"):
        generate_interactive_chart(mode=cli_cfg.mode, out_html=cli_cfg.out)
    else:
        inst = load_instrument(cli_cfg.instrument)
        generate_interactive_chart(instrument=inst, out_html=cli_cfg.out, mode=cli_cfg.mode)
        if cli_cfg.out is None or (Path(cli_cfg.out).resolve() == RESPONSES_DIR.resolve()):
            generate_portal_pages(output_dir=RESPONSES_DIR, default_id=inst.id)
