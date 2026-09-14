"""
Allomorph Pipeline - Pydantic Configuration Schemas.

Defines schemas for pipeline CLI arguments, Tone3000 storefront pack listings,
and Neural Amp Modeler (.nam) Architecture 2 export container metadata.
"""

from collections.abc import Callable
from pathlib import Path
from typing import Literal

from pydantic import ConfigDict, Field

from allomorph.base import AllomorphBaseModel
from allomorph.circuit.schema import CircuitConfig
from allomorph.config.schema import VoiceCoilConfig, VoicePickupConfig

__all__ = [
    "ArtworkPackConfig",
    "NamExportMetadata",
    "NamSourceInstrumentMeta",
    "NamSourcePickupMeta",
    "NamTargetVoiceMeta",
    "NamTrainingConfig",
    "NamTrainingMetadata",
    "PipelineCliConfig",
    "Tone3000PackListing",
]


class PipelineCliConfig(AllomorphBaseModel):
    """Validation schema for Allomorph pipeline command-line arguments."""

    instrument: str = "all"
    stage: Literal["all", "viz", "sim", "pack", "train"] = "all"
    pickup: str | None = None
    voice: str = "all"
    train: bool = False
    vol_pos: float | None = Field(default=None, ge=0.0, le=1.0)
    tone_pos: float | None = Field(default=None, ge=0.0, le=1.0)
    blend_pos: float | None = Field(default=None, ge=0.0, le=1.0)
    pot_taper: Literal["audio", "linear", "reverse_audio", "mn_blend"] | None = None
    cable_pf: float = Field(default=750.0, ge=0.0, le=20000.0)
    normalize: Literal["auto", "rms", "peak", "none"] = "auto"
    target_dbfs: float | None = None
    out_dir: str | None = None
    input_wav: str | None = None
    output_wav: str | None = None
    export_json: str | None = None
    html: str | None = None
    backend: str = "native"
    version_tag: str | None = "auto"
    no_manifest: bool = False


class Tone3000PackListing(AllomorphBaseModel):
    """Declarative validation schema for Tone3000 storefront pack descriptions and metadata."""

    edition: str
    description: str = Field(..., max_length=10000)
    pickup_tags: list[str] = Field(default_factory=list)
    voicings: list[str] = Field(..., min_length=18, max_length=32)


class NamSourcePickupMeta(AllomorphBaseModel):
    """Source pickup geometry and physical configuration metadata."""

    id: str = ""
    name: str
    position_from_bridge_m: float = 0.0
    position_from_bridge_mm: float = 0.0
    aperture_width_in: float = 0.75
    coil_spacing_in: float = 0.0
    type: str = "single_coil"


class NamTrainingMetadata(AllomorphBaseModel):
    """Training metrics and hyperparameters for Neural Amp Modeler (.nam) models."""

    model_config = ConfigDict(extra="allow", validate_assignment=True)

    esr: float | None = None
    validation_esr: float | None = None
    epochs: int | None = None
    batch_size: int | None = None
    lr: float | None = None
    model_type: str | None = None
    architecture: str | None = None
    weights: str | None = None


class NamSourceInstrumentMeta(AllomorphBaseModel):
    """Metadata describing the physical source instrument embedded in Architecture 2 models."""

    id: str
    name: str
    scale_length_in: float
    scale_length_m: float | None = None
    string_wave_speeds: list[float] = Field(default_factory=list)
    pickup: NamSourcePickupMeta


class NamTargetVoiceMeta(AllomorphBaseModel):
    """Metadata describing the target acoustic voicing and circuit embedded in Architecture 2 models."""

    id: str
    name: str
    topology: str = ""
    resonant_frequency_hz: float = 0.0
    q_factor: float = 0.0
    target_position_34_m: float = 0.0
    effective_position_m: float = 0.0
    pickups: list[VoicePickupConfig] = Field(default_factory=list)
    coils: list[VoiceCoilConfig] = Field(default_factory=list)
    circuit: CircuitConfig | None = None


class NamExportMetadata(AllomorphBaseModel):
    """Container metadata exported with Neural Amp Modeler (.nam) models."""

    training: NamTrainingMetadata
    license: str
    copyright: str
    author: str
    version: str = Field(default="v2.1.1")
    dsp_version: int = Field(default=2)
    instrument_version: int = Field(default=1)
    voice_version: int = Field(default=1)
    allomorph_version: str = Field(default="0.2.0")
    git_commit: str | None = None
    generated_at: str | None = None
    source_instrument: NamSourceInstrumentMeta
    target_voice: NamTargetVoiceMeta


class NamTrainingConfig(AllomorphBaseModel):
    """Training hyperparameters and execution flags for NAM Architecture 2 local training."""

    instrument: str = "all"
    voice: str = "all"
    input_wav: Path | str | None = None
    output_wav: Path | str | None = None
    models_dir: Path | str = Field(default=Path("models"))
    epochs: int = Field(
        default=400,
        gt=0,
        description="Maximum number of training epochs (default: 400 for Architecture 2 studio reference)",
    )
    goal_esr: float | None = Field(
        default=0.0080,
        ge=0.0,
        description="Goal validation ESR for early stopping (default: 0.0080 for Architecture 2 studio reference)",
    )
    no_goal_esr: bool = False
    batch_size: int = Field(default=32, gt=0)
    show_plot: bool = False
    save_plot: bool = False
    basename: str | None = None
    fast_dev_run: bool = False
    gui: bool = False
    a2_lite_only: bool = Field(
        default=False,
        description="Train A2-Lite channels_8 only instead of the full slimmable container (default: False)",
    )
    version_tag: str | None = "auto"
    no_manifest: bool = False


class ArtworkPackConfig(AllomorphBaseModel):
    """Validated CAD artwork and typography configuration for Tone3000 storefront pack."""

    accent: str = Field(..., pattern=r"^#[0-9a-fA-F]{6}$")
    title: str
    desc_line1: str = ""
    desc_line2: str = ""
    scale: str
    badge2: str
    badge3: str = "FOR NAM &amp; ANAGRAM"
    voicing_count: int = 22
    num_strings: int = 4
    hero_scale: float = 1.30
    extra_bg: str = ""
    content: Callable[[str], str]
