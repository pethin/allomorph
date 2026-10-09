#!/usr/bin/env python3
"""
Allomorph - Universal Neural Amp Modeler (NAM) Architecture 2 Local Trainer
Delegates core training, telemetry, and early stopping logic to the library package allomorph.trainer.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from allomorph.trainer import (
    AUDIO_DIR,
    DEFAULT_ARCHITECTURE,
    DEFAULT_BATCH_SIZE,
    DEFAULT_CONSECUTIVE_PATIENCE,
    DEFAULT_ESR_TARGET,
    DEFAULT_ESR_THRESHOLD,
    DEFAULT_ETA_MIN,
    DEFAULT_GOAL_DELTA_ESR,
    DEFAULT_GOAL_DELTA_ESR_LITE,
    DEFAULT_GOAL_DELTA_MRSTFT,
    DEFAULT_GOAL_DELTA_MRSTFT_LITE,
    DEFAULT_GOAL_ESR,
    DEFAULT_GOAL_ESR_LITE,
    DEFAULT_INPUT_PATH,
    DEFAULT_LR_SCHEDULER,
    DEFAULT_LR_T_MAX,
    DEFAULT_MAX_EPOCHS,
    DEFAULT_MAX_MRSTFT_CEILING,
    DEFAULT_MAX_MRSTFT_CEILING_LITE,
    DEFAULT_MIN_DELTA,
    DEFAULT_MIN_EPOCHS,
    DEFAULT_MRSTFT_FFT_SIZES,
    DEFAULT_MRSTFT_WEIGHT,
    DEFAULT_PATIENCE,
    DEFAULT_PRE_EMPH_COEF,
    DEFAULT_PRE_EMPH_WEIGHT,
    DEFAULT_SLOPE_TOLERANCE,
    DEFAULT_SLOPE_WINDOW,
    MODELS_DIR,
    AllomorphAdaptiveStopping,
    EsrProgressCallback,
    LinearWarmupCallback,
    add_trainer_arguments,
    build_trainer_parser,
    compute_baseline_delta_ratio,
    compute_baseline_mrstft,
    compute_linear_slope,
    configure_a2_architecture,
    find_sweep_input,
    main,
    setup_headless_environment,
    train_voice,
    train_voices_from_config,
)

__all__ = [
    "AUDIO_DIR",
    "DEFAULT_ARCHITECTURE",
    "DEFAULT_BATCH_SIZE",
    "DEFAULT_CONSECUTIVE_PATIENCE",
    "DEFAULT_ESR_TARGET",
    "DEFAULT_ESR_THRESHOLD",
    "DEFAULT_ETA_MIN",
    "DEFAULT_GOAL_DELTA_ESR",
    "DEFAULT_GOAL_DELTA_ESR_LITE",
    "DEFAULT_GOAL_DELTA_MRSTFT",
    "DEFAULT_GOAL_DELTA_MRSTFT_LITE",
    "DEFAULT_GOAL_ESR",
    "DEFAULT_GOAL_ESR_LITE",
    "DEFAULT_INPUT_PATH",
    "DEFAULT_LR_SCHEDULER",
    "DEFAULT_LR_T_MAX",
    "DEFAULT_MAX_EPOCHS",
    "DEFAULT_MAX_MRSTFT_CEILING",
    "DEFAULT_MAX_MRSTFT_CEILING_LITE",
    "DEFAULT_MIN_DELTA",
    "DEFAULT_MIN_EPOCHS",
    "DEFAULT_MRSTFT_FFT_SIZES",
    "DEFAULT_MRSTFT_WEIGHT",
    "DEFAULT_PATIENCE",
    "DEFAULT_PRE_EMPH_COEF",
    "DEFAULT_PRE_EMPH_WEIGHT",
    "DEFAULT_SLOPE_TOLERANCE",
    "DEFAULT_SLOPE_WINDOW",
    "MODELS_DIR",
    "AllomorphAdaptiveStopping",
    "EsrProgressCallback",
    "LinearWarmupCallback",
    "add_trainer_arguments",
    "build_trainer_parser",
    "compute_baseline_delta_ratio",
    "compute_baseline_mrstft",
    "compute_linear_slope",
    "configure_a2_architecture",
    "find_sweep_input",
    "main",
    "setup_headless_environment",
    "train_voice",
    "train_voices_from_config",
]

if __name__ == "__main__":
    sys.exit(main())
