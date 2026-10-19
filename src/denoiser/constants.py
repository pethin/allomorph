"""Constants, noise profile specifications, and naming helpers for the Allomorph Universal Bass Denoiser."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal

from allomorph.trainer.constants import (
    NATIVE_DEFAULT_EMA_DECAY,
    NATIVE_DEFAULT_HOMOTOPY_TAU_INIT,
    NATIVE_DEFAULT_LITE_WEIGHT,
    NATIVE_DEFAULT_LR,
    NATIVE_DEFAULT_MAX_EPOCHS,
    NATIVE_DEFAULT_MAX_SPIKE_RESTARTS,
    NATIVE_DEFAULT_MIN_DELTA,
    NATIVE_DEFAULT_MIN_EPOCHS,
    NATIVE_DEFAULT_MSE_WEIGHT,
    NATIVE_DEFAULT_PATIENCE,
    NATIVE_DEFAULT_SLICE_LEN,
    NATIVE_DEFAULT_SPIKE_RATIO_THRESHOLD,
    NATIVE_DEFAULT_VAL_INTERVAL,
    NATIVE_DEFAULT_WEIGHT_DECAY,
)

REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[2]
DEFAULT_DENOISER_OUTPUT_DIR: Final[Path] = REPO_ROOT / "models" / "denoiser"
DEFAULT_EXCITATION_WAV: Final[Path] = REPO_ROOT / "audio" / "input.wav"

MainsMode = Literal["both", "60", "50"]
MainsArg = Literal["both", "60", "50", "all"]
ProfileName = Literal["gentle", "balanced", "aggressive"]
ProfileArg = Literal["gentle", "balanced", "aggressive", "all"]

VALID_MAINS_MODES: Final[tuple[MainsMode, ...]] = ("both", "60", "50")
VALID_PROFILE_NAMES: Final[tuple[ProfileName, ...]] = ("gentle", "balanced", "aggressive")

MAINS_SLUG_MAP: Final[dict[MainsMode, str]] = {
    "both": "universal",
    "60": "60hz",
    "50": "50hz",
}

MAINS_DISPLAY_MAP: Final[dict[MainsMode, str]] = {
    "both": "Universal",
    "60": "60Hz",
    "50": "50Hz",
}


@dataclass(frozen=True)
class DenoiserProfileSpec:
    """Electroacoustic noise gate & spectral suppression profile specification."""

    name: ProfileName
    label: str
    threshold_dbfs: float
    noise_min_dbfs: float
    noise_max_dbfs: float
    silence_weight: float
    envelope_order: int
    ultrasonic_lpf_fc_hz: float
    description: str


DENOISER_PROFILES: Final[dict[ProfileName, DenoiserProfileSpec]] = {
    "gentle": DenoiserProfileSpec(
        name="gentle",
        label="Gentle",
        threshold_dbfs=-50.0,
        noise_min_dbfs=-68.0,
        noise_max_dbfs=-40.0,
        silence_weight=60.0,
        envelope_order=4,
        ultrasonic_lpf_fc_hz=14000.0,
        description=(
            "Ultra-transparent studio profile (-50 dBFS knee) for well-shielded instruments "
            "or exposed solo/fretless passages with long natural sustain tails."
        ),
    ),
    "balanced": DenoiserProfileSpec(
        name="balanced",
        label="Balanced",
        threshold_dbfs=-44.0,
        noise_min_dbfs=-65.0,
        noise_max_dbfs=-34.0,
        silence_weight=100.0,
        envelope_order=4,
        ultrasonic_lpf_fc_hz=14000.0,
        description=(
            "Default all-around live & studio profile (-44 dBFS knee) balancing deep 50/60 Hz "
            "buzz and SMPS switching whine suppression against natural sustain decay."
        ),
    ),
    "aggressive": DenoiserProfileSpec(
        name="aggressive",
        label="Aggressive",
        threshold_dbfs=-36.0,
        noise_min_dbfs=-60.0,
        noise_max_dbfs=-28.0,
        silence_weight=150.0,
        envelope_order=4,
        ultrasonic_lpf_fc_hz=14000.0,
        description=(
            "High-suppression stage profile (-36 dBFS knee) for unshielded vintage single-coils, "
            "dirty venue power, LED dimmers, and high-gain overdrive/fuzz chains."
        ),
    ),
}

# Denoiser Native Trainer Defaults
DENOISER_MODELED_BY: Final[str] = "Allomorph (Peter Nguyen <peter@phn.dev>)"
DENOISER_GEAR_MAKE: Final[str] = "Allomorph"
DENOISER_GEAR_TYPE: Final[str] = "pedal"
DENOISER_TONE_TYPE: Final[str] = "clean"

DENOISER_DEFAULT_MAX_EPOCHS: Final[int] = NATIVE_DEFAULT_MAX_EPOCHS
DENOISER_DEFAULT_MIN_EPOCHS: Final[int] = NATIVE_DEFAULT_MIN_EPOCHS
DENOISER_DEFAULT_PATIENCE: Final[int] = NATIVE_DEFAULT_PATIENCE
DENOISER_DEFAULT_MIN_DELTA: Final[float] = NATIVE_DEFAULT_MIN_DELTA
DENOISER_DEFAULT_LR: Final[float] = NATIVE_DEFAULT_LR
DENOISER_DEFAULT_WEIGHT_DECAY: Final[float] = NATIVE_DEFAULT_WEIGHT_DECAY
DENOISER_DEFAULT_MSE_WEIGHT: Final[float] = NATIVE_DEFAULT_MSE_WEIGHT
DENOISER_DEFAULT_LITE_WEIGHT: Final[float] = NATIVE_DEFAULT_LITE_WEIGHT
DENOISER_DEFAULT_VAL_INTERVAL: Final[int] = NATIVE_DEFAULT_VAL_INTERVAL
DENOISER_DEFAULT_EMA_DECAY: Final[float] = NATIVE_DEFAULT_EMA_DECAY
DENOISER_DEFAULT_HOMOTOPY_TAU_INIT: Final[float] = NATIVE_DEFAULT_HOMOTOPY_TAU_INIT
DENOISER_DEFAULT_SPIKE_RATIO_THRESHOLD: Final[float] = NATIVE_DEFAULT_SPIKE_RATIO_THRESHOLD
DENOISER_DEFAULT_MAX_SPIKE_RESTARTS: Final[int] = NATIVE_DEFAULT_MAX_SPIKE_RESTARTS
DENOISER_DEFAULT_SLICE_LEN: Final[int] = NATIVE_DEFAULT_SLICE_LEN

# Dual-band Blackman FIR High-Pass + Ultrasonic Low-Pass Conditioning
DENOISER_DEFAULT_FIR_HP_MID_FC: Final[float] = 1200.0
DENOISER_DEFAULT_FIR_HP_HIGH_FC: Final[float] = 5500.0
DENOISER_DEFAULT_FIR_HP_MID_WEIGHT: Final[float] = 1.5
DENOISER_DEFAULT_FIR_HP_HIGH_WEIGHT: Final[float] = 3.0
DENOISER_DEFAULT_ULTRASONIC_LPF_FC: Final[float] = 14000.0
DENOISER_FIR_NUM_TAPS: Final[int] = 255


def resolve_mains_targets(mains: str) -> tuple[MainsMode, ...]:
    """Resolves a `--mains` CLI argument (`both|60|50|all`) into canonical `MainsMode` targets."""
    cleaned = mains.strip().lower().removesuffix("hz")
    if cleaned == "all":
        return VALID_MAINS_MODES
    if cleaned in ("both", "universal"):
        return ("both",)
    if cleaned == "60":
        return ("60",)
    if cleaned == "50":
        return ("50",)
    raise ValueError(
        f"Invalid mains mode '{mains}'. Expected one of: 'both', '60', '50', or 'all'."
    )


def resolve_profile_targets(profile: str) -> tuple[ProfileName, ...]:
    """Resolves a `--profile` CLI argument (`gentle|balanced|aggressive|all`) into canonical `ProfileName` targets."""
    cleaned = profile.strip().lower()
    if cleaned == "all":
        return VALID_PROFILE_NAMES
    if cleaned == "gentle":
        return ("gentle",)
    if cleaned == "balanced":
        return ("balanced",)
    if cleaned == "aggressive":
        return ("aggressive",)
    raise ValueError(
        f"Invalid denoiser profile '{profile}'. Expected one of: 'gentle', 'balanced', 'aggressive', or 'all'."
    )


def resolve_denoiser_model_stem(mains: MainsMode, profile: ProfileName) -> str:
    """Returns the canonical filename stem (`<mains_slug>_<profile>`) for a denoiser model.

    Examples:
        - `("both", "balanced")` -> `"universal_balanced"`
        - `("60", "aggressive")` -> `"60hz_aggressive"`
        - `("50", "gentle")`     -> `"50hz_gentle"`
    """
    if mains not in MAINS_SLUG_MAP:
        raise ValueError(f"Unsupported mains mode '{mains}'.")
    if profile not in DENOISER_PROFILES:
        raise ValueError(f"Unsupported denoiser profile '{profile}'.")
    return f"{MAINS_SLUG_MAP[mains]}_{profile}"


def resolve_denoiser_display_name(mains: MainsMode, profile: ProfileName) -> str:
    """Returns the human-readable model display name embedded in `.nam` `metadata['name']`.

    Because denoiser models are published in their own dedicated Tone3000 pack,
    the display name omits any redundant `'Denoiser: '` prefix.

    Examples:
        - `("both", "balanced")` -> `"Universal Balanced"`
        - `("60", "aggressive")` -> `"60Hz Aggressive"`
        - `("50", "gentle")`     -> `"50Hz Gentle"`
    """
    if mains not in MAINS_DISPLAY_MAP:
        raise ValueError(f"Unsupported mains mode '{mains}'.")
    if profile not in DENOISER_PROFILES:
        raise ValueError(f"Unsupported denoiser profile '{profile}'.")
    return f"{MAINS_DISPLAY_MAP[mains]} {DENOISER_PROFILES[profile].label}"


def resolve_denoiser_gear_model(mains: MainsMode, profile: ProfileName) -> str:
    """Returns the `.nam` `gear_model` string for a denoiser configuration."""
    display = resolve_denoiser_display_name(mains, profile)
    return f"Universal Bass Denoiser ({display})"
