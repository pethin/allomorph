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
    stage: Literal["all", "viz", "sim", "pack", "train", "audit"] = "all"
    pickup: str | None = None
    voice: str = "all"
    train: bool = False
    vol_pos: float | None = Field(default=None, ge=0.0, le=1.0)
    tone_pos: float | None = Field(default=None, ge=0.0, le=1.0)
    blend_pos: float | None = Field(default=None, ge=0.0, le=1.0)
    pot_taper: Literal["audio", "linear", "reverse_audio", "mn_blend"] | None = None
    cable_pf: float = Field(default=750.0, ge=0.0, le=20000.0)
    normalize: Literal["auto", "rms", "peak", "lufs", "none"] = "auto"
    target_dbfs: float | None = None
    out_dir: str | None = None
    input_wav: str | None = None
    output_wav: str | None = None
    export_json: str | None = None
    html: str | None = None
    backend: str = "native"
    version_tag: str | None = "auto"
    no_manifest: bool = False
    clean_audio: bool = False


class Tone3000PackListing(AllomorphBaseModel):
    """Declarative validation schema for Tone3000 storefront pack descriptions and metadata."""

    edition: str
    description: str = Field(..., max_length=10000)
    pickup_tags: list[str] = Field(default_factory=list)
    voicings: list[str] = Field(..., min_length=15, max_length=32)


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
    differential_esr: float | None = None
    mrstft_loss: float | None = None
    baseline_mrstft: float | None = None
    differential_mrstft: float | None = None
    consecutive_gates_met: int | None = None
    epochs_trained: int | None = None
    stop_reason: str | None = None
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
    reference_wav: Path | str | None = None
    models_dir: Path | str = Field(default=Path("models"))
    epochs: int = Field(
        default=400,
        gt=0,
        description="Maximum number of training epochs (default: 400 for Architecture 2 studio reference)",
    )
    min_epochs: int = Field(
        default=5,
        ge=0,
        description="Warmup epoch floor before early stopping can trigger (default: 5 covering linear warmup)",
    )
    goal_esr: float | None = Field(
        default=0.00020,
        ge=0.0,
        description="Gate 1: Goal validation Global ESR for early stopping (default: 0.00020 for Architecture 2 studio reference)",
    )
    goal_delta_esr: float | None = Field(
        default=0.020,
        ge=0.0,
        description="Gate 2: Goal validation Differential Delta ESR for nuance convergence (default: 0.020)",
    )
    goal_delta_mrstft: float | None = Field(
        default=0.50,
        ge=0.0,
        description="Gate 3: Goal validation Differential MRSTFT ratio MRSTFT / max(baseline_mrstft, 1e-6) (default: 0.50)",
    )
    max_mrstft_ceiling: float = Field(
        default=0.320,
        ge=0.0,
        description="Gate 3: Absolute ceiling on MRSTFT for Gate 3 qualification (default: 0.320)",
    )
    consecutive_patience: int = Field(
        default=3,
        ge=1,
        description="Consecutive validation epochs satisfying all 3 gates before early exit (default: 3)",
    )
    patience: int = Field(
        default=12,
        ge=0,
        description="Patience epochs for plateau early stopping (default: 12, set 0 to disable)",
    )
    min_delta: float = Field(
        default=5e-6,
        ge=0.0,
        description="Minimum loss improvement to reset plateau patience (default: 5e-6)",
    )
    pre_emph_weight: float = Field(
        default=0.25,
        ge=0.0,
        description="Pre-emphasis loss weight for equalizing high-frequency resonance (default: 0.25)",
    )
    pre_emph_coef: float = Field(
        default=0.85,
        ge=0.0,
        le=1.0,
        description="Pre-emphasis filter coefficient (default: 0.85)",
    )
    mrstft_weight: float = Field(
        default=0.0010,
        ge=0.0,
        description="Multi-Resolution STFT loss weight (default: 0.0010)",
    )
    lr_scheduler: str = Field(
        default="cosine",
        description="Learning rate scheduler schedule: 'cosine' (monotonic clamped) or 'exponential'",
    )
    eta_min: float = Field(
        default=1e-5,
        ge=0.0,
        description="Minimum learning rate floor for CosineAnnealingLR (default: 1e-5)",
    )
    lr_t_max: int = Field(
        default=35,
        ge=5,
        description="Cosine decay period (T_max in epochs, default: 35)",
    )
    no_goal_esr: bool = False
    batch_size: int = Field(default=16, gt=0)
    show_plot: bool = False
    save_plot: bool = False
    basename: str | None = None
    fast_dev_run: bool = False
    gui: bool = False
    a2_lite_only: bool = Field(
        default=True,
        description="Train A2-Lite channels_8 only instead of full slimmable container (default: True, ~35% faster)",
    )
    full_slimmable: bool = Field(
        default=False,
        description="Train full slimmable container (channels_3 + channels_8) instead of A2-Lite (default: False)",
    )
    version_tag: str | None = "auto"
    no_manifest: bool = False
    include_identity: bool = False


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
