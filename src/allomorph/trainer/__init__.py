"""
Allomorph Trainer - Universal Neural Amp Modeler (NAM) Architecture 2 Engine
Provides high-fidelity studio reference neural network training, adaptive multi-tier
early stopping, and headless execution management for Darkglass Anagram and DAW plugin targets.
"""

import argparse
from collections.abc import Sequence
from typing import Any

from allomorph.trainer.callbacks import (
    AllomorphAdaptiveStopping,
    EsrProgressCallback,
    LinearWarmupCallback,
    compute_linear_slope,
    compute_window_improvement,
    evaluate_plateau_stopping_condition,
    extract_trainer_metrics,
)
from allomorph.trainer.constants import (
    DEFAULT_ARCHITECTURE,
    DEFAULT_BATCH_SIZE,
    DEFAULT_ENGINE,
    DEFAULT_ETA_MIN,
    DEFAULT_LR_SCHEDULER,
    DEFAULT_LR_T_MAX,
    DEFAULT_MAX_EPOCHS,
    DEFAULT_MIN_DELTA,
    DEFAULT_MIN_EPOCHS,
    DEFAULT_MRSTFT_FFT_SIZES,
    DEFAULT_MRSTFT_WEIGHT,
    DEFAULT_NUM_WORKERS,
    DEFAULT_PATIENCE,
    DEFAULT_PRE_EMPH_COEF,
    DEFAULT_PRE_EMPH_WEIGHT,
    DEFAULT_PRECISION,
    DEFAULT_SEED,
    VALID_ENGINES,
    VALID_PRECISION_MODES,
)
from allomorph.trainer.core import (
    AUDIO_DIR,
    DEFAULT_INPUT_PATH,
    MODELS_DIR,
    compute_baseline_delta_ratio,
    compute_baseline_mrstft,
    configure_a2_architecture,
    detect_apple_silicon_tier,
    find_sweep_input,
    get_hardware_device_name,
    is_mlx_available,
    resolve_hardware_batch_size,
    resolve_hardware_num_workers,
    resolve_hardware_precision,
    resolve_trainer_engine,
    setup_headless_environment,
    train_voice,
    train_voices_from_config,
)

# Initialize headless runtime environment on import
setup_headless_environment()


def _parse_int_or_auto(val: str) -> int | str:
    """Parses integer or 'auto' string CLI argument."""
    val_clean = val.strip()
    if val_clean.lower() == "auto":
        return "auto"
    try:
        return int(val_clean)
    except ValueError:
        return val_clean


def add_trainer_arguments(parser: argparse.ArgumentParser) -> None:
    """Registers all NAM Architecture 2 local training arguments to an ArgumentParser."""
    existing_flags: set[str] = {opt for action in parser._actions for opt in action.option_strings}

    def _add_arg(*flags: str, **kwargs: Any) -> None:
        if any(flag in existing_flags for flag in flags):
            return
        action = parser.add_argument(*flags, **kwargs)
        existing_flags.update(action.option_strings)

    _add_arg(
        "--instrument",
        "-i",
        default="all",
        help="Source instrument configuration (ID, comma-separated list, 'all', path to .toml, or alias like 30in, 32in; default: 'all')",
    )
    _add_arg(
        "--voice",
        default="all",
        help="Target pickup voice (ID, comma-separated list, or 'all'; default: 'all')",
    )
    _add_arg(
        "--source-voicing",
        "--voicing",
        "--pickup",
        "-p",
        dest="source_voicing",
        default=None,
        help="Source instrument voicing setting (default: auto-detected from instrument voicings)",
    )
    _add_arg(
        "--input",
        "--input-wav",
        dest="input_wav",
        help="Path to dry training sweep WAV (default: auto-detect optimal_bass_dry.wav)",
    )
    _add_arg(
        "--output",
        "--output-wav",
        dest="output_wav",
        help="Path to simulated target voice wet output WAV (default: audio/wet/<instrument>/<voice>.wav)",
    )
    _add_arg(
        "--models-dir",
        default=str(MODELS_DIR),
        help="Output models directory",
    )
    _add_arg(
        "--epochs",
        type=int,
        default=DEFAULT_MAX_EPOCHS,
        help=f"Maximum number of training epochs (default: {DEFAULT_MAX_EPOCHS} for Architecture 2 schedule)",
    )
    _add_arg(
        "--min-epochs",
        "--warmup-epochs",
        dest="min_epochs",
        type=int,
        default=DEFAULT_MIN_EPOCHS,
        help=f"Minimum warmup training epochs before early stopping can trigger (default: {DEFAULT_MIN_EPOCHS})",
    )
    _add_arg(
        "--patience",
        type=int,
        default=DEFAULT_PATIENCE,
        help=f"Adaptive diminishing-returns plateau patience epochs on composite val_loss (default: {DEFAULT_PATIENCE}; set to 0 to disable)",
    )
    _add_arg(
        "--min-delta",
        type=float,
        default=DEFAULT_MIN_DELTA,
        help=f"Minimum loss improvement to reset plateau patience (default: {DEFAULT_MIN_DELTA})",
    )
    _add_arg(
        "--pre-emph-weight",
        type=float,
        default=DEFAULT_PRE_EMPH_WEIGHT,
        help=f"Pre-emphasis loss weight for equalizing high-frequency resonance (default: {DEFAULT_PRE_EMPH_WEIGHT})",
    )
    _add_arg(
        "--pre-emph-coef",
        type=float,
        default=DEFAULT_PRE_EMPH_COEF,
        help=f"Pre-emphasis filter coefficient (default: {DEFAULT_PRE_EMPH_COEF})",
    )
    _add_arg(
        "--mrstft-weight",
        type=float,
        default=DEFAULT_MRSTFT_WEIGHT,
        help=f"Multi-Resolution STFT loss weight (default: {DEFAULT_MRSTFT_WEIGHT})",
    )
    _add_arg(
        "--lr-scheduler",
        choices=["cosine", "exponential"],
        default=DEFAULT_LR_SCHEDULER,
        help=f"Learning rate scheduler type (default: '{DEFAULT_LR_SCHEDULER}')",
    )
    _add_arg(
        "--eta-min",
        type=float,
        default=DEFAULT_ETA_MIN,
        help=f"Minimum learning rate floor for cosine annealing (default: {DEFAULT_ETA_MIN})",
    )
    _add_arg(
        "--lr-t-max",
        type=int,
        default=DEFAULT_LR_T_MAX,
        help=f"Cosine annealing cycle length T_max in epochs (default: {DEFAULT_LR_T_MAX})",
    )
    _add_arg(
        "--reference",
        "--reference-wav",
        dest="reference_wav",
        default=None,
        help="Path to reference source stem for differential comparison (default: auto-detected from source voicing)",
    )
    _add_arg(
        "--batch-size",
        type=_parse_int_or_auto,
        default=DEFAULT_BATCH_SIZE,
        help="Batch size (default: 'auto' resolving dynamically to 32 on >=12GB VRAM, 16 on 6-12GB, 8 on CPU, or explicit integer)",
    )
    _add_arg(
        "--precision",
        choices=list(VALID_PRECISION_MODES),
        default=DEFAULT_PRECISION,
        help="PyTorch Lightning precision: 'auto' (resolving to 'bf16-mixed' on native bfloat16 GPUs), '16-mixed', or '32-true'",
    )
    _add_arg(
        "--num-workers",
        type=_parse_int_or_auto,
        default=DEFAULT_NUM_WORKERS,
        help="DataLoader worker count: 'auto' (resolving to 0 for in-memory tensor dataset), or explicit integer",
    )
    _add_arg(
        "--show-plot",
        action="store_true",
        help="Display matplotlib validation plot window",
    )
    _add_arg(
        "--save-plot",
        action="store_true",
        help="Save validation plot as PNG in models/",
    )
    _add_arg(
        "--basename",
        help="Explicit basename for the exported .nam model file",
    )
    _add_arg(
        "--fast-dev-run",
        action="store_true",
        help="Run 1-batch dry run for smoke testing NAM training",
    )
    _add_arg(
        "--version-tag",
        default="auto",
        help="Semantic version tag (default: 'auto' -> v[dsp].[inst].[voice], or explicit string, or 'none' to disable)",
    )
    _add_arg(
        "--no-manifest",
        action="store_true",
        help="Disable generating sidecar manifest.json",
    )
    _add_arg(
        "--pack",
        "--tone-pack",
        dest="pack",
        default=None,
        help="Tone3000 pack identifier to train (outputs flat .nam files to tone3000/packs/[pack]/nam/)",
    )
    _add_arg(
        "--force",
        "--overwrite",
        dest="overwrite",
        action="store_true",
        help="Force retraining of models even if .nam files already exist",
    )
    _add_arg(
        "--include-identity",
        action="store_true",
        help="Force training even if source and target stems are identical",
    )
    _add_arg(
        "--engine",
        choices=list(VALID_ENGINES),
        default=DEFAULT_ENGINE,
        help=f"Underlying neural network training engine ('auto', 'mlx', 'torch'; default: '{DEFAULT_ENGINE}')",
    )
    _add_arg(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Random seed for reproducible weights and shuffling (default: None)",
    )
    _add_arg(
        "--gui",
        action="store_true",
        help="Launch NAM training GUI",
    )


def build_trainer_parser() -> argparse.ArgumentParser:
    """Builds the standalone command line argument parser for NAM Architecture 2 training."""
    parser = argparse.ArgumentParser(description="Allomorph NAM Architecture 2 Local Trainer")
    add_trainer_arguments(parser)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Main CLI entrypoint for Allomorph NAM Architecture 2 Local Trainer."""
    setup_headless_environment()
    parser = build_trainer_parser()
    args = parser.parse_args(argv)

    input_val = getattr(args, "input_wav", None) or getattr(args, "input", None)
    output_val = getattr(args, "output_wav", None) or getattr(args, "output", None)
    ref_val = getattr(args, "reference_wav", None) or getattr(args, "reference", None)
    overwrite_val = getattr(args, "overwrite", False) or getattr(args, "force", False)
    from allomorph.pipeline.schema import NamTrainingConfig

    cli_cfg = NamTrainingConfig.model_validate(
        {
            "instrument": getattr(args, "instrument", "all"),
            "pack": getattr(args, "pack", None),
            "overwrite": overwrite_val,
            "voice": getattr(args, "voice", "all"),
            "input_wav": input_val,
            "output_wav": output_val,
            "reference_wav": ref_val,
            "models_dir": getattr(args, "models_dir", str(MODELS_DIR)),
            "epochs": getattr(args, "epochs", DEFAULT_MAX_EPOCHS),
            "min_epochs": getattr(args, "min_epochs", DEFAULT_MIN_EPOCHS),
            "patience": getattr(args, "patience", DEFAULT_PATIENCE),
            "min_delta": getattr(args, "min_delta", DEFAULT_MIN_DELTA),
            "pre_emph_weight": getattr(args, "pre_emph_weight", DEFAULT_PRE_EMPH_WEIGHT),
            "pre_emph_coef": getattr(args, "pre_emph_coef", DEFAULT_PRE_EMPH_COEF),
            "mrstft_weight": getattr(args, "mrstft_weight", DEFAULT_MRSTFT_WEIGHT),
            "lr_scheduler": getattr(args, "lr_scheduler", DEFAULT_LR_SCHEDULER),
            "eta_min": getattr(args, "eta_min", DEFAULT_ETA_MIN),
            "lr_t_max": getattr(args, "lr_t_max", DEFAULT_LR_T_MAX),
            "batch_size": getattr(args, "batch_size", DEFAULT_BATCH_SIZE),
            "precision": getattr(args, "precision", DEFAULT_PRECISION),
            "num_workers": getattr(args, "num_workers", DEFAULT_NUM_WORKERS),
            "show_plot": getattr(args, "show_plot", False),
            "save_plot": getattr(args, "save_plot", False),
            "basename": getattr(args, "basename", None),
            "fast_dev_run": getattr(args, "fast_dev_run", False),
            "gui": getattr(args, "gui", False),
            "version_tag": getattr(args, "version_tag", "auto"),
            "no_manifest": getattr(args, "no_manifest", False),
            "include_identity": getattr(args, "include_identity", False),
            "engine": getattr(args, "engine", DEFAULT_ENGINE),
            "seed": getattr(args, "seed", DEFAULT_SEED),
        }
    )

    if cli_cfg.gui:
        try:
            from nam.cli import nam_gui

            nam_gui()
            return 0
        except (ImportError, RuntimeError) as e:
            print(f"Error: 'neural-amp-modeler' GUI could not be loaded ({e}).")
            print(
                "Note: The desktop GUI requires system Tkinter (e.g., `sudo apt install python3-tk`)."
            )
            return 1

    ok = train_voices_from_config(cli_cfg)
    return 0 if ok else 1


__all__ = [
    "AUDIO_DIR",
    "DEFAULT_ARCHITECTURE",
    "DEFAULT_BATCH_SIZE",
    "DEFAULT_ENGINE",
    "DEFAULT_ETA_MIN",
    "DEFAULT_INPUT_PATH",
    "DEFAULT_LR_SCHEDULER",
    "DEFAULT_LR_T_MAX",
    "DEFAULT_MAX_EPOCHS",
    "DEFAULT_MIN_DELTA",
    "DEFAULT_MIN_EPOCHS",
    "DEFAULT_MRSTFT_FFT_SIZES",
    "DEFAULT_MRSTFT_WEIGHT",
    "DEFAULT_NUM_WORKERS",
    "DEFAULT_PATIENCE",
    "DEFAULT_PRECISION",
    "DEFAULT_PRE_EMPH_COEF",
    "DEFAULT_PRE_EMPH_WEIGHT",
    "DEFAULT_SEED",
    "MODELS_DIR",
    "VALID_ENGINES",
    "VALID_PRECISION_MODES",
    "AllomorphAdaptiveStopping",
    "EsrProgressCallback",
    "LinearWarmupCallback",
    "add_trainer_arguments",
    "build_trainer_parser",
    "compute_baseline_delta_ratio",
    "compute_baseline_mrstft",
    "compute_linear_slope",
    "compute_window_improvement",
    "configure_a2_architecture",
    "detect_apple_silicon_tier",
    "evaluate_plateau_stopping_condition",
    "extract_trainer_metrics",
    "find_sweep_input",
    "get_hardware_device_name",
    "is_mlx_available",
    "main",
    "resolve_hardware_batch_size",
    "resolve_hardware_num_workers",
    "resolve_hardware_precision",
    "resolve_trainer_engine",
    "setup_headless_environment",
    "train_voice",
    "train_voices_from_config",
]
