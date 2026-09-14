"""
Allomorph Visualizer - Pydantic Configuration Schemas.

Defines schemas for interactive visualizer portals and chart exports.
"""

from pathlib import Path
from typing import Literal

from allomorph.base import AllomorphBaseModel

__all__ = [
    "PortalInstrumentMeta",
    "VisualizerCliConfig",
]


class PortalInstrumentMeta(AllomorphBaseModel):
    """Metadata describing an instrument formatted for the interactive visualization portal."""

    id: str
    name: str
    scale_in: float
    scale_m: float
    speeds_str: str
    pickups_summary: str
    default_pickup: str


class VisualizerCliConfig(AllomorphBaseModel):
    """Validated command-line configuration for interactive Altair visualizer."""

    instrument: str = "all"
    mode: Literal[
        "voicings",
        "voicing_ir_3d",
        "waterfall3d",
        "unified",
        "output",
        "difference",
    ] = "voicings"
    all: bool = False
    out: Path | str | None = None
