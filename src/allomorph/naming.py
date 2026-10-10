"""
Allomorph - Target Voice and Instrument Naming Authority.
Provides pedalboard display slugs, Tone3000 filename generation,
and strict multi-pickup/single-pickup formatting constraints.
"""

import difflib
from collections.abc import Sequence
from pathlib import Path

from allomorph.config.instruments import (
    INSTRUMENT_ALIASES,
    INSTRUMENTS,
    PACKS_CONFIG_DIR,
    TONE_PACKS,
    load_instrument,
)
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
    "upright_piezo": "upright",
    "soapbar_pair": "soapbar_par",
    "soapbar_neck": "soapbar_neck",
    "soapbar_bridge": "soapbar_brg",
    "emg_soapbar_parallel": "emg_par",
    "emg_soapbar_neck": "emg_neck",
    "emg_soapbar_bridge": "emg_brg",
}


def format_tri_part_tag(dsp_gen: int, inst_ver: int, voice_ver: int) -> str:
    """Formats a three-part version tag string e.g. 'v6.1.1'."""
    return f"v{dsp_gen}.{inst_ver}.{voice_ver}"


def sanitize_tone_name_token(token: str) -> str:
    """Sanitizes filesystem path separators ('/' and '\\') with Unicode Division Slash ('\u2215')

    to maintain a flat directory structure. Idempotent: sanitize(sanitize(x)) == sanitize(x).
    """
    return token.strip().replace("/", "\u2215").replace("\\", "\u2215")


def compose_zero_scroll_model_name(
    tone_name: str,
    position_tag: str | None = None,
    version_tag: str | None = None,
    max_len: int | None = None,
) -> str:
    """Composes a zero-scroll pedalboard / Tone3000 model display name.

    Format: `Tone Name [Position] v2.1.1` or `Tone Name v2.1.1`.
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
    pos = None if preserve_aperture else position_name
    return compose_zero_scroll_model_name(
        tone_name=tone_name,
        position_tag=pos,
        version_tag=version_tag,
        max_len=max_length,
    )


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
        except (FileNotFoundError, KeyError):
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


def resolve_packs(
    pack_arg: str | Sequence[str] | None = None,
    instrument_arg: str | Sequence[str] | None = None,
) -> list[str]:
    """
    Parses pack and instrument arguments into a list of valid tone pack IDs.

    When all packs are selected (either pack_arg is 'all' or pack_arg is None and
    instrument_arg is None or 'all'), returns only the tone packs that actually
    exist in config/packs/ (TONE_PACKS).

    Supports:
      - 'all' -> all configured tone packs in TONE_PACKS
      - Comma-separated list: '34in_standard_p,34in_standard_jazz'
      - Single pack ID or alias: '34in_active_stingray', 'stingray', '30in'
      - Explicit instrument argument if pack_arg is None and instrument_arg is not 'all'
    """
    all_available: list[str] = [str(pid) for pid in sorted(TONE_PACKS.keys())]

    # Check if pack_arg was explicitly provided
    if pack_arg is not None:
        if isinstance(pack_arg, str):
            raw_tokens = [t.strip() for t in pack_arg.split(",") if t.strip()]
        else:
            raw_tokens = [str(t).strip() for t in pack_arg if str(t).strip()]

        if raw_tokens:
            if any(t.lower() == "all" for t in raw_tokens):
                return all_available

            resolved: list[str] = []
            for token in raw_tokens:
                # 0. Check if token is a direct path to a pack directory or file
                if Path(token).is_dir() or Path(token).is_file():
                    if token not in resolved:
                        resolved.append(token)
                    continue

                # 1. Exact match in TONE_PACKS
                if token in TONE_PACKS:
                    if token not in resolved:
                        resolved.append(token)
                    continue

                # 2. Check alias or instrument ID
                matched_id: str | None = None
                if token in INSTRUMENT_ALIASES:
                    target_inst = INSTRUMENT_ALIASES[token]
                    if target_inst in TONE_PACKS:
                        matched_id = target_inst
                else:
                    try:
                        cfg = load_instrument(token)
                        if cfg.id in TONE_PACKS:
                            matched_id = cfg.id
                    except (FileNotFoundError, KeyError):
                        pass

                if matched_id is not None:
                    if matched_id not in resolved:
                        resolved.append(matched_id)
                    continue

                # 3. If it's a known instrument but has no pack, raise diagnostic FileNotFoundError
                try:
                    cfg = load_instrument(token)
                    pack_path = PACKS_CONFIG_DIR / f"{cfg.id}.toml"
                    raise FileNotFoundError(
                        f"Tone pack configuration not found for instrument '{cfg.id}'. "
                        f"Expected explicit pack configuration at '{pack_path}'. "
                        f"All tone packs must be explicitly defined in config/packs/."
                    )
                except (FileNotFoundError, KeyError) as e:
                    if "Tone pack configuration not found" in str(e):
                        raise
                    # 4. Unknown pack identifier: suggest close matches
                    close = difflib.get_close_matches(token, all_available, n=3, cutoff=0.4)
                    hint = f" Did you mean: {', '.join(close)}?" if close else ""
                    raise FileNotFoundError(
                        f"Unknown tone pack identifier '{token}'.{hint}\n"
                        f"Available tone packs ({len(all_available)}): {', '.join(all_available)}"
                    )
            return resolved

    # pack_arg was not provided (is None or empty)
    # Check if all packs / all instruments are selected
    if (
        not instrument_arg
        or not str(instrument_arg).strip()
        or str(instrument_arg).strip().lower() == "all"
    ):
        return all_available

    # A specific instrument (or list of instruments) was requested
    inst_tokens = resolve_instruments(instrument_arg)
    return inst_tokens


def get_default_input_path(audio_dir: Path | str | None = None) -> Path:
    """Returns the Path to the default dry string excitation audio file (audio/input.wav)."""
    from allomorph.config.scales import REPO_ROOT

    base_dir = Path(audio_dir) if audio_dir is not None else REPO_ROOT / "audio"
    return base_dir / "input.wav"


def get_instrument_pickup_basename(inst_id: str, pickup: str | None = None) -> str:
    """Generates the filename basename for an instrument pickup's wet audio stem.

    Example: 'split_p' or 'bridge'
    """
    if pickup:
        return f"{pickup}"
    return f"{inst_id}"
