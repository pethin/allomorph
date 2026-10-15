"""
Allomorph Pipeline - Pydantic Configuration Schemas.

Defines schemas for pipeline CLI arguments, Tone3000 storefront pack listings,
and Neural Amp Modeler (.nam) Architecture 2 export container metadata.
"""

from collections.abc import Callable

from pydantic import ConfigDict, Field

from allomorph.base import AllomorphBaseModel
from allomorph.config.schema import (
    CoilConfig,
    PackBundleConfig,
    TonePackConfig,
)

__all__ = [
    "ArtworkPackConfig",
    "NamExportMetadata",
    "NamSourceInstrumentMeta",
    "NamSourceVoicingMeta",
    "NamTargetVoicingMeta",
    "NamTrainingMetadata",
    "PackBundleConfig",
    "Tone3000PackListing",
    "TonePackConfig",
]


class Tone3000PackListing(AllomorphBaseModel):
    """Declarative validation schema for Tone3000 storefront pack descriptions and metadata."""

    edition: str
    description: str = Field(..., max_length=10000)
    pickup_tags: list[str] = Field(default_factory=list)
    voicings: list[str] = Field(..., min_length=15, max_length=32)


class NamSourceVoicingMeta(AllomorphBaseModel):
    """Source voicing configuration, harness, and physical sensing geometry metadata."""

    id: str = ""
    name: str
    tone_name: str | None = None
    harness: str = ""
    position_name: str | None = None
    position_from_bridge_m: float = 0.0
    position_from_bridge_mm: float = 0.0
    effective_position_m: float = 0.0
    effective_position_mm: float = 0.0
    aperture_width_in: float = 0.75
    coil_spacing_in: float = 0.0
    type: str = "single_coil"
    controls: dict[str, float] = Field(default_factory=dict)
    switches: dict[str, str] = Field(default_factory=dict)


class NamTrainingMetadata(AllomorphBaseModel):
    """Training metrics and hyperparameters for Neural Amp Modeler (.nam) models."""

    model_config = ConfigDict(extra="allow", validate_assignment=True)

    validation_spectral_loss: float | None = None
    validation_spectral_loss_a2_full: float | None = None
    validation_spectral_loss_a2_lite: float | None = None
    mrstft_loss: float | None = None
    mrstft_loss_ch3: float | None = None
    esr_pre_loss: float | None = None
    consecutive_gates_met: int | None = None
    epochs_trained: int | None = None
    stop_reason: str | None = None
    batch_size: int | None = None
    precision: str | None = None
    num_workers: int | None = None
    device_name: str | None = None
    epochs: int | None = None
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
    voicing: NamSourceVoicingMeta


class NamTargetVoicingMeta(AllomorphBaseModel):
    """Metadata describing the target acoustic voicing and circuit embedded in Architecture 2 models."""

    id: str
    name: str
    tone_name: str | None = None
    sensor_type: str = "magnetic"
    resonant_frequency_hz: float | None = None
    q_factor: float | None = None
    effective_position_m: float = 0.0
    effective_position_mm: float = 0.0
    coils: list[CoilConfig] = Field(default_factory=list)


class NamExportMetadata(AllomorphBaseModel):
    """Container metadata exported with Neural Amp Modeler (.nam) models."""

    model_config = ConfigDict(extra="allow", validate_assignment=True)

    training: NamTrainingMetadata
    license: str
    copyright: str
    author: str
    version: str = Field(default="v7.1.1")
    dsp_version: int = Field(default=7)
    instrument_version: int = Field(default=1)
    voice_version: int = Field(default=1)
    allomorph_version: str = Field(default="0.5.0")
    git_commit: str | None = None
    generated_at: str | None = None
    source_instrument: NamSourceInstrumentMeta
    target_voicing: NamTargetVoicingMeta


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
