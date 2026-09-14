"""
Allomorph - Target Voice and Instrument Naming Authority.
Provides pedalboard display slugs, Tone3000 filename generation,
and strict multi-pickup/single-pickup formatting constraints.
"""

import difflib
from collections.abc import Sequence
from pathlib import Path

from allomorph.config.instruments import INSTRUMENT_ALIASES, INSTRUMENTS, load_instrument
from allomorph.config.voices import VOICES

VOICE_CONCISE_SLUGS: dict[str, str] = {
    "precision_vintage": "p_vintage",
    "precision_mids": "p_mids",
    "precision_warm": "p_warm",
    "precision_active": "p_active",
    "precision_dub": "p_dub",
    "jazz_pair_open": "jazz_pair",
    "jazz_pair_mids": "jazz_mids",
    "jazz_pair_active": "jazz_active",
    "jazz_bridge_growl": "jazz_growl",
    "jazz_bridge_open": "jazz_bridge",
    "jazz_neck_warm": "jazz_neck",
    "stingray_parallel": "stingray_par",
    "stingray_series": "stingray_ser",
    "stingray_active": "stingray_act",
    "dingwall_bridge": "dingwall_brg",
    "dingwall_middle": "dingwall_mid",
    "dingwall_parallel": "dingwall_par",
    "rickenbacker_clank": "rick_clank",
    "rickenbacker_open": "rick_open",
    "pj_passive": "pj_passive",
    "pj_active": "pj_active",
    "p_mm_parallel": "pmm_parallel",
    "p_mm_series": "pmm_series",
    "mudbucker_deep": "mudbucker",
    "upright_acoustic": "upright",
    "studio_direct": "studio_di",
    "studio_active": "studio_act",
    "studio_passive": "studio_pas",
}


def get_t3k_basename(
    tone_name: str,
    position_name: str | None = None,
    version_tag: str | None = None,
    max_length: int | None = None,
    preserve_aperture: bool = False,
) -> str:
    """Generates a Tone3000 pack model basename.

    Format:
      - Multi-pickup instruments: `Tone Name [Pickup Position]` or `Tone Name [Pickup Position] v2.1.1`
      - Single-pickup instruments or preserve_aperture tones: `Tone Name` or `Tone Name v2.1.1`

    Enforces that the filename (excluding the `.nam` extension) does not exceed
    `max_length` (strict Tone3000 upload ceiling is 64 chars; unversioned legacy default is 34).
    Raises diagnostic ValueError if exceeded.
    Sanitizes filesystem path separators ('/' and '\\') with Unicode Division Slash ('\u2215')
    to maintain a flat directory structure.
    """
    clean_tone = tone_name.strip()
    is_character_tone = preserve_aperture or clean_tone in (
        "Studio Active",
        "Studio Direct",
        "Studio Passive",
    )
    if position_name and position_name.strip() and not is_character_tone:
        clean_pos = position_name.strip()
        name = f"{clean_tone} [{clean_pos}]"
    else:
        name = clean_tone

    if version_tag:
        clean_ver = version_tag.strip()
        if clean_ver:
            name = f"{name} {clean_ver}"

    name = name.replace("/", "\u2215").replace("\\", "\u2215")

    effective_max = max_length if max_length is not None else (64 if version_tag else 34)

    if len(name) > effective_max:
        raise ValueError(
            f"T3K pack filename '{name}' exceeds {effective_max} characters ({len(name)} chars). "
            f"Tone name '{clean_tone}' or pickup position '{position_name}' must be shortened."
        )
    return name


def resolve_voices(voice_arg: str | Sequence[str] | None) -> list[str]:
    """
    Parses a voice argument into a list of valid target voice IDs.
    Supports:
      - 'all' -> all configured voices in VOICES
      - Comma-separated list: '01_jazz_bass_pair,03_modern_p_ceramic'
      - Single voice ID: '03_modern_p_ceramic'
      - Partial / prefix matching: '01', '03'
    """
    if not voice_arg or not str(voice_arg).strip() or str(voice_arg).strip().lower() == "all":
        return list(VOICES.keys())

    tokens = [t.strip() for t in str(voice_arg).split(",") if t.strip()]
    resolved: list[str] = []
    for token in tokens:
        if token in VOICES:
            if token not in resolved:
                resolved.append(token)
        else:
            matches = [vid for vid in VOICES if vid.startswith(token) or token in vid]
            if matches:
                for m in matches:
                    if m not in resolved:
                        resolved.append(m)
            else:
                candidates = list(VOICES.keys())
                close = difflib.get_close_matches(token, candidates, n=3, cutoff=0.4)
                hint = f" Did you mean: {', '.join(close)}?" if close else ""
                raise ValueError(
                    f"Unknown target voice identifier '{token}'.{hint}\n"
                    f"Available target voices ({len(candidates)}): {', '.join(candidates)}"
                )
    return resolved


def resolve_instruments(instrument_arg: str | Sequence[str] | None) -> list[str]:
    """
    Parses an instrument argument into a list of valid source instrument IDs.
    Supports:
      - 'all' -> all configured playable source instruments in INSTRUMENTS
      - Comma-separated list: '30in,32in_fretless,34in_standard_p'
      - Single instrument ID or alias: '30in', 'fretless', 'jazz'
      - Partial / alias matching
    """
    all_playable: list[str] = [str(iid) for iid in sorted(INSTRUMENTS.keys())]
    if (
        not instrument_arg
        or not str(instrument_arg).strip()
        or str(instrument_arg).strip().lower() == "all"
    ):
        return all_playable

    tokens = [t.strip() for t in str(instrument_arg).split(",") if t.strip()]
    resolved: list[str] = []
    for token in tokens:
        if token.lower() == "all":
            for iid in all_playable:
                if iid not in resolved:
                    resolved.append(iid)
            continue
        try:
            cfg = load_instrument(token)
            iid = cfg.id
            if iid not in resolved:
                resolved.append(iid)
        except FileNotFoundError, KeyError:
            matches = [iid for iid in all_playable if iid.startswith(token) or token in iid]
            if matches:
                for m in matches:
                    if m not in resolved:
                        resolved.append(m)
            else:
                all_choices = sorted(set(all_playable + list(INSTRUMENT_ALIASES.keys())))
                close = difflib.get_close_matches(token, all_choices, n=3, cutoff=0.4)
                hint = f" Did you mean: {', '.join(close)}?" if close else ""
                raise ValueError(
                    f"Unknown source instrument identifier '{token}'.{hint}\n"
                    f"Available instruments ({len(all_playable)}): {', '.join(all_playable)}"
                )
    return resolved


def get_optimal_dry_basename(version_tag: str | None = None) -> str:
    """Generates the filename basename for the synthetic optimal bass dry calibration file.

    Default version tag is single-part DSP generation 'v{dsp}' (e.g. 'v2'),
    as the unvoiced synthetic excitation signal depends strictly on the core DSP physics generation.
    Example: 'optimal_bass_dry_v2'
    """
    from allomorph.version import DSP_GENERATION

    v_tag = version_tag or f"v{DSP_GENERATION}"
    return f"optimal_bass_dry_{v_tag}"


def get_optimal_dry_path(
    version_tag: str | None = None, audio_dir: Path | str | None = None
) -> Path:
    """Returns the absolute or relative Path to the versioned optimal bass dry WAV.

    Example: audio/canonical/optimal_bass_dry_v2.wav
    """
    from allomorph.config.scales import REPO_ROOT

    base_dir = Path(audio_dir) if audio_dir is not None else REPO_ROOT / "audio"
    return base_dir / "canonical" / f"{get_optimal_dry_basename(version_tag)}.wav"


def get_instrument_pickup_basename(inst_id: str, pickup: str | None = None) -> str:
    """Generates the filename basename for an instrument pickup's wet audio stem.

    Example: 'split_p' or 'bridge'
    """
    if pickup:
        return f"{pickup}"
    return f"{inst_id}"
