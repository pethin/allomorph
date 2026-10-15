"""
Allomorph CLI - Visualization Subcommand
Generates Altair frequency response curves, voicing comparisons, and master interactive portal.
"""

from pathlib import Path

from allomorph.cli.console import log_info, log_step, log_success
from allomorph.config.instruments import load_instrument
from allomorph.config.scales import REPO_ROOT
from allomorph.visualizer.charts import generate_all_charts, generate_interactive_chart

DOCS_DIR = REPO_ROOT / "docs"


def visualize(
    instrument: str = "all",
    *,
    mode: str = "voicings",
    out: Path | None = None,
) -> None:
    """Generates interactive Altair visualization charts."""
    inst_desc = "all instruments" if instrument == "all" else f"instrument '{instrument}'"
    log_step(
        "VIZ", f"Generating interactive Altair visualizations for {inst_desc} (mode: {mode})..."
    )

    out_dir = Path(out) if out is not None else DOCS_DIR

    if instrument.lower() == "all":
        generate_all_charts(output_dir=out_dir)
        log_info(
            f"Interactive Frequency Response Suite generated at [bold]{out_dir / 'frequency_responses.html'}[/bold]"
        )
        log_success("Visualization generation complete.")
    else:
        inst_cfg = load_instrument(instrument)
        out_html = Path(out) if out is not None and str(out).endswith(".html") else None
        target_dir = out_html.parent if out_html is not None else out_dir

        generate_interactive_chart(
            instrument=inst_cfg,
            out_html=out_html or (target_dir / f"{inst_cfg.id}.html"),
            mode=mode,
        )
        log_info(
            f"Interactive chart generated at [bold]{out_html or (target_dir / f'{inst_cfg.id}.html')}[/bold]"
        )
        log_success(f"Visualization generation complete for {inst_cfg.id}.")
