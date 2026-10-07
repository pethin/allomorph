"""
Allomorph Circuit - Audio Utilities & Calibration Discovery
Provides calibration audio discovery for optimal dry input excitation signals.
"""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]


def find_default_input_audio(version_tag: str | None = None) -> Path:
    """Finds or ensures the default input dry string excitation audio (audio/input.wav)."""
    from allomorph.dsp import DEFAULT_INPUT_PATH, ensure_input_audio_wav
    from allomorph.naming import get_default_input_path

    for candidate in [REPO_ROOT / "input.wav", DEFAULT_INPUT_PATH, get_default_input_path()]:
        if candidate.exists():
            return candidate
    return ensure_input_audio_wav()
