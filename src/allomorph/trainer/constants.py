"""
Allomorph Trainer - Constants and Hyperparameter Defaults
Schedule-driven natural convergence standards, optimizer annealing periods, and early stopping tolerances.
"""

from typing import Literal

# ------------------------------------------------------------------------------
# Stock Engine Defaults (Standard upstream 200-epoch NAM/Lightning training)
# ------------------------------------------------------------------------------
STOCK_DEFAULT_MAX_EPOCHS: int = 200
STOCK_DEFAULT_BATCH_SIZE: str = "auto"
STOCK_DEFAULT_PRECISION: str = "auto"
STOCK_DEFAULT_NUM_WORKERS: str = "auto"
STOCK_DEFAULT_VAL_LOSS_METRIC: str = "val_loss"

# ------------------------------------------------------------------------------
# Native Engine Defaults (Custom In-VRAM PyTorch A2 Training Engine)
# ------------------------------------------------------------------------------
NATIVE_DEFAULT_MIN_EPOCHS: int = 100
NATIVE_DEFAULT_MAX_EPOCHS: int = 120
NATIVE_DEFAULT_PATIENCE: int = 15
NATIVE_DEFAULT_MIN_DELTA: float = 1.0e-6
NATIVE_DEFAULT_LR: float = 4.0e-3
NATIVE_DEFAULT_WEIGHT_DECAY: float = 3.17e-7
NATIVE_DEFAULT_SLICE_LEN: int = 8192
NATIVE_DEFAULT_MSE_WEIGHT: float = 1.00
NATIVE_DEFAULT_MRSTFT_WEIGHT: float = 0.0
NATIVE_DEFAULT_LITE_WEIGHT: float = 0.50
NATIVE_DEFAULT_LR_SCHEDULER: str = "cosine"
NATIVE_DEFAULT_BATCH_SIZE: str = "auto"
NATIVE_DEFAULT_PRECISION: str = "auto"
NATIVE_DEFAULT_VAL_INTERVAL: int = 5
NATIVE_DEFAULT_SINE_REGULARIZER: bool = True
NATIVE_DEFAULT_FIR_HP_WEIGHT: float = 2.5
NATIVE_DEFAULT_FIR_HP_FC: float = 1200.0
NATIVE_DEFAULT_SINE_FREQUENCIES: tuple[float, ...] = (
    40.0,
    80.0,
    81.4,
    110.0,
    160.0,
    162.8,
    220.0,
    300.0,
    329.6,
    330.0,
    440.0,
    500.0,
    550.0,
    586.2,
    600.0,
    650.0,
    659.2,
    700.0,
    880.0,
    1000.0,
    1100.0,
    1318.5,
    1320.0,
    1760.0,
    2000.0,
    3000.0,
    5000.0,
    7000.0,
)
NATIVE_DEFAULT_SINE_AMPLITUDES: tuple[float, ...] = (0.25, 0.10, 0.05, 0.03, 0.008)

# ------------------------------------------------------------------------------
# General Platform & Architecture Invariants (Default Native Configuration)
# ------------------------------------------------------------------------------
DEFAULT_ARCHITECTURE: str = "A2"
DEFAULT_ENGINE: Literal["auto", "native", "stock"] = "auto"
VALID_ENGINES: tuple[str, ...] = ("auto", "native", "stock")
DEFAULT_SEED: int | None = None
VALID_PRECISION_MODES: tuple[str, ...] = ("auto", "bf16-mixed", "16-mixed", "32-true", "32")
DEFAULT_MRSTFT_FFT_SIZES: tuple[int, ...] = (512, 1024, 2048)

DEFAULT_MAX_EPOCHS: int = NATIVE_DEFAULT_MAX_EPOCHS
DEFAULT_MIN_EPOCHS: int = NATIVE_DEFAULT_MIN_EPOCHS
DEFAULT_PATIENCE: int = NATIVE_DEFAULT_PATIENCE
DEFAULT_MIN_DELTA: float = NATIVE_DEFAULT_MIN_DELTA
DEFAULT_MSE_WEIGHT: float = NATIVE_DEFAULT_MSE_WEIGHT
DEFAULT_MRSTFT_WEIGHT: float = NATIVE_DEFAULT_MRSTFT_WEIGHT
DEFAULT_LR_SCHEDULER: str = NATIVE_DEFAULT_LR_SCHEDULER
DEFAULT_ETA_MIN: float = 1.0e-6
DEFAULT_LR_T_MAX: int = 115
DEFAULT_BATCH_SIZE: str = "auto"
DEFAULT_PRECISION: str = "auto"
DEFAULT_NUM_WORKERS: str = "auto"
DEFAULT_VAL_LOSS_METRIC: str = "val_loss"
NAM_MODEL_VERSION: str = "0.7.0"

__all__ = [
    "DEFAULT_ARCHITECTURE",
    "DEFAULT_BATCH_SIZE",
    "DEFAULT_ENGINE",
    "DEFAULT_ETA_MIN",
    "DEFAULT_LR_SCHEDULER",
    "DEFAULT_LR_T_MAX",
    "DEFAULT_MAX_EPOCHS",
    "DEFAULT_MIN_DELTA",
    "DEFAULT_MIN_EPOCHS",
    "DEFAULT_MRSTFT_FFT_SIZES",
    "DEFAULT_MRSTFT_WEIGHT",
    "DEFAULT_MSE_WEIGHT",
    "DEFAULT_NUM_WORKERS",
    "DEFAULT_PATIENCE",
    "DEFAULT_PRECISION",
    "DEFAULT_SEED",
    "DEFAULT_VAL_LOSS_METRIC",
    "NAM_MODEL_VERSION",
    "NATIVE_DEFAULT_BATCH_SIZE",
    "NATIVE_DEFAULT_FIR_HP_FC",
    "NATIVE_DEFAULT_FIR_HP_WEIGHT",
    "NATIVE_DEFAULT_LITE_WEIGHT",
    "NATIVE_DEFAULT_LR",
    "NATIVE_DEFAULT_LR_SCHEDULER",
    "NATIVE_DEFAULT_MAX_EPOCHS",
    "NATIVE_DEFAULT_MIN_DELTA",
    "NATIVE_DEFAULT_MIN_EPOCHS",
    "NATIVE_DEFAULT_MRSTFT_WEIGHT",
    "NATIVE_DEFAULT_MSE_WEIGHT",
    "NATIVE_DEFAULT_PATIENCE",
    "NATIVE_DEFAULT_PRECISION",
    "NATIVE_DEFAULT_SINE_AMPLITUDES",
    "NATIVE_DEFAULT_SINE_FREQUENCIES",
    "NATIVE_DEFAULT_SINE_REGULARIZER",
    "NATIVE_DEFAULT_SLICE_LEN",
    "NATIVE_DEFAULT_VAL_INTERVAL",
    "NATIVE_DEFAULT_WEIGHT_DECAY",
    "STOCK_DEFAULT_BATCH_SIZE",
    "STOCK_DEFAULT_MAX_EPOCHS",
    "STOCK_DEFAULT_NUM_WORKERS",
    "STOCK_DEFAULT_PRECISION",
    "STOCK_DEFAULT_VAL_LOSS_METRIC",
    "VALID_ENGINES",
    "VALID_PRECISION_MODES",
]
