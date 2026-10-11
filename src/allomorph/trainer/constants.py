"""
Allomorph Trainer - Constants and Hyperparameter Defaults
Schedule-driven natural convergence standards, optimizer annealing periods, and early stopping tolerances.
"""

# Warmup epoch floor covering initial exploration before early stopping can engage
DEFAULT_MIN_EPOCHS: int = 35

# Architecture 2 studio schedule budget (matching Tone3000 rapid convergence)
DEFAULT_MAX_EPOCHS: int = 60

# Dynamic auto-tuning defaults for hardware execution
DEFAULT_BATCH_SIZE: str = "auto"
DEFAULT_PRECISION: str = "auto"
DEFAULT_NUM_WORKERS: str = "auto"
VALID_PRECISION_MODES: tuple[str, ...] = ("auto", "bf16-mixed", "16-mixed", "32-true", "32")

# Adaptive diminishing-returns plateau patience window in epochs
DEFAULT_PATIENCE: int = 6

# Minimum loss improvement to reset plateau patience
DEFAULT_MIN_DELTA: float = 1.0e-4

# Primary time-domain MSE loss weight (1.00 anchors sub-bass and comb notches)
DEFAULT_MSE_WEIGHT: float = 1.00

# Pre-emphasis loss weight for equalizing high-frequency resonance (0.0 eliminates sub-bass drift)
DEFAULT_PRE_EMPH_WEIGHT: float = 0.0

# Pre-emphasis filter coefficient (alpha = 0.85)
DEFAULT_PRE_EMPH_COEF: float = 0.85

# Multi-Resolution STFT loss weight for RLC resonant peak fidelity
DEFAULT_MRSTFT_WEIGHT: float = 0.0

# Standard schedule: Exponential decay for studio-grade stability
DEFAULT_LR_SCHEDULER: str = "exponential"

# Minimum learning rate floor for CosineAnnealingLR
DEFAULT_ETA_MIN: float = 5.0e-4

# Cosine decay period (T_max in epochs)
DEFAULT_LR_T_MAX: int = 45

# Multi-Band Self-Normalizing High-Pass ESR Loss parameters
DEFAULT_HP_MID_ESR_WEIGHT: float = 0.50
DEFAULT_HP_MID_CUTOFF: float = 3000.0
DEFAULT_HP_HIGH_ESR_WEIGHT: float = 0.50
DEFAULT_HP_HIGH_CUTOFF: float = 7000.0

# Default validation loss metric type
DEFAULT_VAL_LOSS_METRIC: str = "spectral"

# Architecture 2 designation
DEFAULT_ARCHITECTURE: str = "A2"

# Training engine defaults
DEFAULT_ENGINE: str = "auto"
VALID_ENGINES: tuple[str, ...] = ("auto", "torch")
DEFAULT_SEED: int | None = None

# MRSTFT loss FFT window sizes
DEFAULT_MRSTFT_FFT_SIZES: tuple[int, ...] = (512, 1024, 2048)

__all__ = [
    "DEFAULT_ARCHITECTURE",
    "DEFAULT_BATCH_SIZE",
    "DEFAULT_ENGINE",
    "DEFAULT_ETA_MIN",
    "DEFAULT_HP_HIGH_CUTOFF",
    "DEFAULT_HP_HIGH_ESR_WEIGHT",
    "DEFAULT_HP_MID_CUTOFF",
    "DEFAULT_HP_MID_ESR_WEIGHT",
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
    "DEFAULT_PRE_EMPH_COEF",
    "DEFAULT_PRE_EMPH_WEIGHT",
    "DEFAULT_SEED",
    "DEFAULT_VAL_LOSS_METRIC",
    "VALID_ENGINES",
    "VALID_PRECISION_MODES",
]

