"""
Allomorph Visualizer - Pydantic Configuration Schemas.

Defines schemas for interactive visualizer comparisons and quick comparison chips.
"""

import tomllib
from pathlib import Path

from pydantic import Field

from allomorph.base import AllomorphBaseModel

__all__ = [
    "VisualizerChipConfig",
    "VisualizerConfig",
    "load_visualizer_config",
]

REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG_DIR = REPO_ROOT / "config"
VISUALIZER_CONFIG_PATH = CONFIG_DIR / "visualizer.toml"


class VisualizerChipConfig(AllomorphBaseModel):
    """Configuration for a quick comparison chip button in the visualizer UI."""

    label: str
    source: str
    target: str


class VisualizerConfig(AllomorphBaseModel):
    """Declarative visualizer suite configuration loaded from config/visualizer.toml."""

    default_source: str = "34in_standard_p:vintage_open"
    default_target: str = "34in_standard_jazz:bridge_growl"
    chips: list[VisualizerChipConfig] = Field(default_factory=list)


def load_visualizer_config(config_path: Path | str | None = None) -> VisualizerConfig:
    """Loads and validates VisualizerConfig from TOML."""
    p = Path(config_path) if config_path is not None else VISUALIZER_CONFIG_PATH
    if not p.is_file():
        return VisualizerConfig()
    with open(p, "rb") as f:
        data = tomllib.load(f)
    return VisualizerConfig.model_validate(data)
