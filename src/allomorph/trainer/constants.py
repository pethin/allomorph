"""
Allomorph Trainer - Constants and Hyperparameter Defaults
Studio reference training standards, early stopping thresholds, and gate tolerances.
"""

# Gate 1: A2 Full Global ESR (~ -37.0 dB ESR)
DEFAULT_GOAL_ESR: float = 0.00020

# Gate 2: Nuance target <= -17.0 dB residual on pickup transformation delta
DEFAULT_GOAL_DELTA_ESR: float = 0.020

# Gate 3: Differential MRSTFT ratio (MRSTFT / max(baseline_mrstft, 1e-6))
DEFAULT_GOAL_DELTA_MRSTFT: float = 0.50

# Gate 3: Absolute ceiling on validation MRSTFT
DEFAULT_MAX_MRSTFT_CEILING: float = 0.320

# Gate 4: A2 Lite Global ESR (~ -26.0 dB ESR)
DEFAULT_GOAL_ESR_LITE: float = 0.00250

# Gate 5: A2 Lite nuance target <= -11.0 dB residual on pickup delta
DEFAULT_GOAL_DELTA_ESR_LITE: float = 0.080

# Gate 6: A2 Lite differential MRSTFT ratio
DEFAULT_GOAL_DELTA_MRSTFT_LITE: float = 0.75

# Gate 6: Absolute ceiling on A2 Lite validation MRSTFT
DEFAULT_MAX_MRSTFT_CEILING_LITE: float = 0.450

# Consecutive validation checks meeting all gates before early exit
DEFAULT_CONSECUTIVE_PATIENCE: int = 3

# Warmup epoch floor covering initial linear learning rate ramp
DEFAULT_MIN_EPOCHS: int = 5

# Architecture 2 studio reference epoch safety ceiling
DEFAULT_MAX_EPOCHS: int = 400

# Universal GPU batch size (84 updates/epoch for optimal gradient depth)
DEFAULT_BATCH_SIZE: int = 16

# Plateau patience window in epochs
DEFAULT_PATIENCE: int = 12

# Minimum loss improvement to reset plateau patience
DEFAULT_MIN_DELTA: float = 5e-6

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
DEFAULT_LR_T_MAX: int = 35

# Architecture 2 designation
DEFAULT_ARCHITECTURE: str = "A2"

# Validation ESR target synonyms
DEFAULT_ESR_TARGET: float = DEFAULT_GOAL_ESR
DEFAULT_ESR_THRESHOLD: float = DEFAULT_GOAL_ESR

# Slope window and plateau tolerance
DEFAULT_SLOPE_WINDOW: int = DEFAULT_PATIENCE
DEFAULT_SLOPE_TOLERANCE: float = DEFAULT_MIN_DELTA

# MRSTFT loss FFT window sizes
DEFAULT_MRSTFT_FFT_SIZES: tuple[int, ...] = (512, 1024, 2048)

__all__ = [
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
]
