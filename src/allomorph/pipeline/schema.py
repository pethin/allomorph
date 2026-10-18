"""
Allomorph Pipeline - Pydantic Configuration Schemas.

Defines schemas for pipeline CLI arguments, Tone3000 storefront pack listings,
and Neural Amp Modeler (.nam) Architecture 2 export container metadata.
"""

from collections.abc import Callable

from pydantic import ConfigDict, Field

from allomorph.base import AllomorphBaseModel
from allomorph.config.schema import (
    PackBundleConfig,
    TonePackConfig,
)
from allomorph.version import (
    ALLOMORPH_VERSION,
    DEFAULT_INST_VERSION,
    DEFAULT_VOICE_VERSION,
    DSP_GENERATION,
    resolve_tri_part_version,
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
    voicings: list[str] = Field(..., min_length=15, max_length=40)


class NamSourceVoicingMeta(AllomorphBaseModel):
    """Minimal electroacoustic voicing metadata embedded in Architecture 2 (.nam) models."""

    id: str = ""
    name: str
    tone_name: str | None = None
    sensor_type: str = "magnetic"
    magnet_type: str | None = None
    resonant_frequency_hz: float | None = None
    q_factor: float | None = None
    effective_position_m: float = 0.0
    alpha: float | None = None
    vsat: float | None = None


class NamTrainingMetadata(AllomorphBaseModel):
    """Training metrics and hyperparameters for Neural Amp Modeler (.nam) models."""

    model_config = ConfigDict(extra="allow", validate_assignment=True)

    validation_spectral_loss: float | None = None
    validation_spectral_loss_a2_full: float | None = None
    validation_spectral_loss_a2_lite: float | None = None
    validation_esr: float | None = None
    validation_esr_a2_full: float | None = None
    validation_esr_a2_lite: float | None = None
    validation_esr_ch8: float | None = None
    validation_esr_ch3: float | None = None
    mrstft_loss: float | None = None
    mrstft_loss_ch3: float | None = None
    esr_pre_loss: float | None = None
    consecutive_gates_met: int | None = None
    epochs_trained: int | None = None
    spike_restarts: int | None = None
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


class NamTargetVoicingMeta(NamSourceVoicingMeta):
    """Metadata describing the target acoustic voicing and circuit embedded in Architecture 2 models."""

    id: str


class NamExportMetadata(AllomorphBaseModel):
    """Container metadata exported with Neural Amp Modeler (.nam) models."""

    model_config = ConfigDict(extra="allow", validate_assignment=True)

    training: NamTrainingMetadata
    license: str
    copyright: str
    author: str
    version: str = Field(default_factory=resolve_tri_part_version)
    dsp_version: int = Field(default=DSP_GENERATION)
    instrument_version: int = Field(default=DEFAULT_INST_VERSION)
    voice_version: int = Field(default=DEFAULT_VOICE_VERSION)
    allomorph_version: str = Field(default=ALLOMORPH_VERSION)
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
