"""
Allomorph - Tone3000 & Anagram Display Naming Authority.
Provides zero-scroll pedalboard display name formatting, Tone3000 filename generation,
and path-safe token sanitization.
"""

from pathlib import Path


def format_tri_part_tag(dsp_gen: int, inst_ver: int, voice_ver: int) -> str:
    """Formats a three-part version tag string e.g. 'v7.1.1'."""
    return f"v{dsp_gen}.{inst_ver}.{voice_ver}"


def sanitize_tone_name_token(token: str) -> str:
    """Sanitizes filesystem path separators ('/' and '\\') with Unicode Division Slash ('\u2215')
    and arrow symbols ('->', '-->') with Unicode Rightwards Arrow ('\u2192')
    to maintain a flat, legal filesystem structure across Windows, macOS, and Linux.
    Idempotent: sanitize(sanitize(x)) == sanitize(x).
    """
    return (
        token.strip()
        .replace("-->", "\u2192")
        .replace("->", "\u2192")
        .replace("/", "\u2215")
        .replace("\\", "\u2215")
    )


def compose_zero_scroll_model_name(
    tone_name: str,
    position_tag: str | None = None,
    version_tag: str | None = None,
    max_len: int | None = None,
) -> str:
    """Composes a zero-scroll pedalboard / Tone3000 model display name.

    Format: `Tone Name [Position] v7.1.1` or `Tone Name v7.1.1`.
    Enforces maximum character ceiling (default 34 chars for unversioned, 64 chars for versioned).
    """
    clean_tone = sanitize_tone_name_token(tone_name)
    if position_tag and position_tag.strip():
        clean_pos = sanitize_tone_name_token(position_tag)
        name = f"{clean_tone} [{clean_pos}]"
    else:
        name = clean_tone

    if version_tag:
        clean_ver = sanitize_tone_name_token(version_tag)
        if clean_ver:
            name = f"{name} {clean_ver}"

    effective_max = max_len if max_len is not None else (64 if version_tag else 34)

    if len(name) > effective_max:
        raise ValueError(
            f"T3K pack filename '{name}' exceeds {effective_max} characters ({len(name)} chars). "
            f"Tone name '{clean_tone}' or pickup position '{position_tag}' must be shortened."
        )
    return name


def get_t3k_basename(
    tone_name: str,
    position_name: str | None = None,
    version_tag: str | None = None,
    max_length: int | None = None,
) -> str:
    """Generates a Tone3000 pack model basename.

    Format:
      - Multi-pickup instruments: `Tone Name [Pickup Position]` or `Tone Name [Pickup Position] v7.1.1`
      - Single-pickup instruments: `Tone Name` or `Tone Name v7.1.1`

    Enforces that the filename (excluding the `.nam` extension) does not exceed
    `max_length` (strict Tone3000 upload ceiling is 64 chars; unversioned legacy default is 34).
    Raises diagnostic ValueError if exceeded.
    Sanitizes filesystem path separators ('/' and '\\') with Unicode Division Slash ('\u2215')
    to maintain a flat directory structure.
    """
    return compose_zero_scroll_model_name(
        tone_name=tone_name,
        position_tag=position_name,
        version_tag=version_tag,
        max_len=max_length,
    )


def get_default_input_path(audio_dir: Path | str | None = None) -> Path:
    """Returns the Path to the default dry string excitation audio file (audio/input.wav)."""
    from allomorph.config.scales import REPO_ROOT

    base_dir = Path(audio_dir) if audio_dir is not None else REPO_ROOT / "audio"
    return base_dir / "input.wav"
