"""
Allomorph Visualizer - Pydantic Configuration Schemas.

Defines schemas for interactive visualizer portals, chart exports, and quick comparison chips.
"""

import tomllib
from pathlib import Path
from typing import Literal

from pydantic import Field

from allomorph.base import AllomorphBaseModel

__all__ = [
    "PortalInstrumentMeta",
    "VisualizerChipConfig",
    "VisualizerCliConfig",
    "VisualizerConfig",
    "load_visualizer_config",
]

REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG_DIR = REPO_ROOT / "config"
VISUALIZER_CONFIG_PATH = CONFIG_DIR / "visualizer.toml"


class PortalInstrumentMeta(AllomorphBaseModel):
    """Metadata describing an instrument formatted for the interactive visualization portal."""

    id: str
    name: str
    scale_in: float
    scale_m: float
    speeds_str: str
    pickups_summary: str


class VisualizerCliConfig(AllomorphBaseModel):
    """Validated command-line configuration for interactive Altair visualizer."""

    instrument: str = "all"
    mode: Literal[
        "voicings",
        "unified",
        "output",
        "difference",
    ] = "voicings"
    all: bool = False
    out: Path | str | None = None


class VisualizerChipConfig(AllomorphBaseModel):
    """Configuration for a quick comparison chip button in the visualizer UI."""

    label: str
    source: str
    target: str


class VisualizerConfig(AllomorphBaseModel):
    """Declarative visualizer suite configuration loaded from config/visualizer.toml."""

    default_source: str = "precision_vintage"
    default_target: str = "jazz_bridge_growl"
    additional_source_voicings: list[str] = Field(default_factory=list)
    additional_target_voicings: list[str] = Field(default_factory=list)
    chips: list[VisualizerChipConfig] = Field(default_factory=list)


def load_visualizer_config(config_path: Path | str | None = None) -> VisualizerConfig:
    """Loads and validates VisualizerConfig from TOML."""
    p = Path(config_path) if config_path is not None else VISUALIZER_CONFIG_PATH
    if not p.is_file():
        return VisualizerConfig()
    with open(p, "rb") as f:
        data = tomllib.load(f)
    return VisualizerConfig.model_validate(data)
