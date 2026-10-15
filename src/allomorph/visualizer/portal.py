"""
Allomorph Visualizer - Interactive HTML Portal Generation
"""

from collections.abc import Mapping
from pathlib import Path

from allomorph.config.scales import REPO_ROOT
from allomorph.config.schema import InstrumentConfig, VoiceConfig
from allomorph.visualizer.schema import PortalInstrumentMeta

DOCS_DIR = REPO_ROOT / "docs"
RESPONSES_DIR = DOCS_DIR


def append_spec_panel(html_path: Path, panel_html: str) -> None:
    """Appends an informational HTML spec/directive panel before </body>."""
    content = html_path.read_text(encoding="utf-8")
    if "</body>" in content:
        content = content.replace("</body>", f"{panel_html}\n</body>")
        html_path.write_text(content, encoding="utf-8")


def format_instrument_meta(inst: InstrumentConfig) -> PortalInstrumentMeta:
    """Formats an InstrumentConfig into metadata suitable for the portal."""
    inst_id = inst.id
    inst_name = inst.name
    scale_in = inst.scale_length_in or 34.0
    scale_m = inst.scale_length_m or (scale_in * 0.0254)
    speeds = inst.string_wave_speeds or []
    speeds_str = ", ".join(f"{float(s):.1f} m/s" for s in speeds) if speeds else "N/A"

    parts = []
    for pid, pcfg in inst.pickups.items():
        if pcfg.type == "composite":
            continue
        pname = pcfg.name or pid
        pos_m = pcfg.position_from_bridge_m
        if pos_m:
            pos_mm = float(pos_m) * 1000.0
            parts.append(f"{pname} (@ {pos_mm:.1f}mm)")
        else:
            parts.append(pname)
    pickups_summary = " | ".join(parts) if parts else "Standard Pickups"

    return PortalInstrumentMeta(
        id=inst_id,
        name=inst_name,
        scale_in=scale_in,
        scale_m=round(scale_m, 4),
        speeds_str=speeds_str,
        pickups_summary=pickups_summary,
    )


def build_portal_html(
    instruments_meta: Mapping[str, PortalInstrumentMeta] | None = None,
    default_id: str = "voicings",
    base_url_prefix: str = "./",
    voices: Mapping[str, VoiceConfig] | None = None,
    instruments: Mapping[str, InstrumentConfig] | None = None,
) -> str:
    """Constructs the flat, self-contained interactive Frequency Response Suite HTML string."""
    from allomorph.visualizer.charts import build_voicings_html

    _ = (instruments_meta, default_id, base_url_prefix)
    return build_voicings_html(voices=voices, instruments=instruments)


def generate_portal_pages(
    output_dir: str | Path | None = None,
    default_id: str | None = None,
) -> Path:
    """Builds the flat, self-contained interactive portal at <output_dir>/frequency_responses.html."""
    from allomorph.visualizer.charts import generate_voicings_page

    _ = default_id
    out_dir = Path(output_dir) if output_dir else DOCS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    target_path = out_dir if out_dir.suffix == ".html" else (out_dir / "frequency_responses.html")
    return generate_voicings_page(target_path)
