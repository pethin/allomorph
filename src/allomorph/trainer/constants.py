"""
Allomorph Trainer - Constants and Hyperparameter Defaults
Schedule-driven natural convergence standards, optimizer annealing periods, and early stopping tolerances.
"""

# Warmup epoch floor covering initial exploration before early stopping can engage
DEFAULT_MIN_EPOCHS: int = 20

# Architecture 2 studio schedule budget (matching Cosine T_max)
DEFAULT_MAX_EPOCHS: int = 40

# Dynamic auto-tuning defaults for hardware execution
DEFAULT_BATCH_SIZE: str = "auto"
DEFAULT_PRECISION: str = "auto"
DEFAULT_NUM_WORKERS: str = "auto"
VALID_PRECISION_MODES: tuple[str, ...] = ("auto", "bf16-mixed", "16-mixed", "32-true", "32")

# Adaptive diminishing-returns plateau patience window in epochs
DEFAULT_PATIENCE: int = 8

# Minimum loss improvement to reset plateau patience
DEFAULT_MIN_DELTA: float = 1.0e-6

# Pre-emphasis loss weight for equalizing high-frequency resonance
DEFAULT_PRE_EMPH_WEIGHT: float = 0.25

# Pre-emphasis filter coefficient (alpha = 0.85)
DEFAULT_PRE_EMPH_COEF: float = 0.85

# Multi-Resolution STFT loss weight for RLC resonant peak fidelity
DEFAULT_MRSTFT_WEIGHT: float = 0.0010

# Monotonic Cosine Decay with floor clamp
DEFAULT_LR_SCHEDULER: str = "cosine"

# Minimum learning rate floor for CosineAnnealingLR
DEFAULT_ETA_MIN: float = 1e-5

# Cosine decay period (T_max in epochs)
DEFAULT_LR_T_MAX: int = 40

# Architecture 2 designation
DEFAULT_ARCHITECTURE: str = "A2"

# MRSTFT loss FFT window sizes
DEFAULT_MRSTFT_FFT_SIZES: tuple[int, ...] = (512, 1024, 2048)

__all__ = [
    "DEFAULT_ARCHITECTURE",
    "DEFAULT_BATCH_SIZE",
    "DEFAULT_ETA_MIN",
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
    "VALID_PRECISION_MODES",
]

