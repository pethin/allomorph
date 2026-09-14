"""
Allomorph Circuit - Audio Utilities & Calibration Discovery
Provides calibration audio discovery for optimal dry input excitation signals.
"""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]


def find_default_input_audio(version_tag: str | None = None) -> Path | None:
    """Finds or ensures the default input dry audio (optimal_bass_dry_v{dsp}.wav)."""
    from allomorph.dsp import OPTIMAL_DRY_PATH, ensure_optimal_dry_wav
    from allomorph.naming import get_optimal_dry_path

    p_versioned = get_optimal_dry_path(version_tag=version_tag)
    if p_versioned.exists():
        return p_versioned
    if OPTIMAL_DRY_PATH.exists():
        return OPTIMAL_DRY_PATH
    for candidate in ["input.wav"]:
        p = REPO_ROOT / candidate
        if p.exists():
            return p
    return ensure_optimal_dry_wav(version_tag=version_tag)
