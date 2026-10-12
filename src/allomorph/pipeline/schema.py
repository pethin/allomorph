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
from allomorph.config.schema import (
    PackBundleConfig,
    TonePackConfig,
    VoiceCoilConfig,
)
from allomorph.trainer.constants import (
    DEFAULT_ETA_MIN,
    DEFAULT_LR_SCHEDULER,
    DEFAULT_LR_T_MAX,
    DEFAULT_MAX_EPOCHS,
    DEFAULT_MIN_DELTA,
    DEFAULT_MIN_EPOCHS,
    DEFAULT_MRSTFT_WEIGHT,
    DEFAULT_MSE_WEIGHT,
    DEFAULT_PATIENCE,
    NATIVE_DEFAULT_LITE_WEIGHT,
    NATIVE_DEFAULT_LR,
    NATIVE_DEFAULT_SLICE_LEN,
    NATIVE_DEFAULT_WEIGHT_DECAY,
)

__all__ = [
    "ArtworkPackConfig",
    "NamExportMetadata",
    "NamSourceInstrumentMeta",
    "NamSourceVoicingMeta",
    "NamTargetVoicingMeta",
    "NamTrainingConfig",
    "NamTrainingMetadata",
    "PackBundleConfig",
    "PipelineCliConfig",
    "Tone3000PackListing",
    "TonePackConfig",
]


class PipelineCliConfig(AllomorphBaseModel):
    """Validation schema for Allomorph pipeline command-line arguments."""

    instrument: str = "all"
    pack: str | None = None
    stage: Literal["all", "viz", "sim", "pack", "train", "audit"] = "all"
    voice: str = "all"
    train: bool = False
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
    force: bool = False
    overwrite: bool = False
    jobs: int | None = None
    max_samples: int | None = None


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
    resonant_frequency_hz: float = 0.0
    q_factor: float = 0.0
    effective_position_m: float = 0.0
    effective_position_mm: float = 0.0
    coils: list[VoiceCoilConfig] = Field(default_factory=list)


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
    target_voicing: NamTargetVoicingMeta


class NamTrainingConfig(AllomorphBaseModel):
    """Training hyperparameters and execution flags for NAM Architecture 2 local training."""

    instrument: str = "all"
    source_voicing: str | None = Field(
        default=None,
        description="Source instrument voicing setting (default: auto-detected from instrument voicings)",
    )
    pack: str | None = Field(
        default=None,
        description="Tone3000 pack identifier to train (outputs flat .nam files to tone3000/packs/[pack]/nam/)",
    )
    voice: str = "all"
    input_wav: Path | str | None = None
    output_wav: Path | str | None = None
    reference_wav: Path | str | None = None
    models_dir: Path | str = Field(default=Path("models"))
    epochs: int = Field(
        default=DEFAULT_MAX_EPOCHS,
        gt=0,
        description=f"Maximum number of training epochs (default: {DEFAULT_MAX_EPOCHS} for Architecture 2 schedule)",
    )
    min_epochs: int = Field(
        default=DEFAULT_MIN_EPOCHS,
        ge=0,
        description=f"Warmup epoch floor covering initial exploration before early stopping can engage (default: {DEFAULT_MIN_EPOCHS})",
    )
    patience: int = Field(
        default=DEFAULT_PATIENCE,
        ge=0,
        description=f"Patience epochs for plateau early stopping (default: {DEFAULT_PATIENCE}, set 0 to disable)",
    )
    min_delta: float = Field(
        default=DEFAULT_MIN_DELTA,
        ge=0.0,
        description=f"Minimum loss improvement to reset plateau patience (default: {DEFAULT_MIN_DELTA})",
    )
    mse_weight: float = Field(
        default=DEFAULT_MSE_WEIGHT,
        ge=0.0,
        description=f"Raw MSE loss weight for low-frequency amplitude anchoring (default: {DEFAULT_MSE_WEIGHT})",
    )
    mrstft_weight: float = Field(
        default=DEFAULT_MRSTFT_WEIGHT,
        ge=0.0,
        description=f"Multi-Resolution STFT loss weight (default: {DEFAULT_MRSTFT_WEIGHT})",
    )
    lr_scheduler: str = Field(
        default=DEFAULT_LR_SCHEDULER,
        description="Learning rate scheduler schedule: 'exponential' or 'cosine'",
    )
    eta_min: float = Field(
        default=DEFAULT_ETA_MIN,
        ge=0.0,
        description=f"Minimum learning rate floor for CosineAnnealingLR (default: {DEFAULT_ETA_MIN})",
    )
    lr_t_max: int = Field(
        default=DEFAULT_LR_T_MAX,
        ge=5,
        description=f"Cosine decay period (T_max in epochs, default: {DEFAULT_LR_T_MAX})",
    )
    batch_size: int | str = Field(
        default="auto",
        description="Batch size (default: 'auto' resolving dynamically to 32 on >=12GB VRAM, 16 on 6-12GB, 8 on CPU)",
    )
    precision: str = Field(
        default="auto",
        description="PyTorch Lightning precision: 'auto' (resolving to 'bf16-mixed' on native bfloat16 GPUs), '16-mixed', or '32-true'",
    )
    num_workers: int | str = Field(
        default="auto",
        description="DataLoader num_workers: 'auto' (resolving to 0 for in-memory tensor dataset), or explicit integer",
    )
    show_plot: bool = False
    save_plot: bool = False
    basename: str | None = None
    fast_dev_run: bool = False
    gui: bool = False
    version_tag: str | None = "auto"
    no_manifest: bool = False
    include_identity: bool = False
    overwrite: bool = Field(
        default=False,
        description="Force retraining of models even if .nam files already exist",
    )
    engine: str = Field(
        default="auto",
        description="Underlying neural training engine ('auto', 'native', 'stock')",
    )
    seed: int | None = Field(
        default=None,
        description="Random seed for deterministic initialization and data shuffling",
    )
    native_lr: float = Field(
        default=NATIVE_DEFAULT_LR,
        description=f"Native PyTorch engine learning rate (default: {NATIVE_DEFAULT_LR})",
    )
    native_weight_decay: float = Field(
        default=NATIVE_DEFAULT_WEIGHT_DECAY,
        description=f"Native PyTorch engine weight decay (default: {NATIVE_DEFAULT_WEIGHT_DECAY})",
    )
    native_slice_len: int = Field(
        default=NATIVE_DEFAULT_SLICE_LEN,
        description=f"Native PyTorch engine slice window length (default: {NATIVE_DEFAULT_SLICE_LEN})",
    )
    native_lite_weight: float = Field(
        default=NATIVE_DEFAULT_LITE_WEIGHT,
        description=f"Native PyTorch engine submodel Lite weight (default: {NATIVE_DEFAULT_LITE_WEIGHT})",
    )


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
