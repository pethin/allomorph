"""
Allomorph - Naming Conventions & CLI Resolution Engine
Provides pedalboard display slugs, dynamic tier prefixes, baked filename generation,
and tolerant CLI argument parsing for instruments and voices.
"""

import difflib
from collections.abc import Sequence
from pathlib import Path

from allomorph.base import AllomorphBaseModel
from allomorph.config.instruments import INSTRUMENT_ALIASES, INSTRUMENTS, load_instrument
from allomorph.config.voices import VOICES


class TierSpec(AllomorphBaseModel):
    """Specification of an Allomorph processing/export tier."""

    name: str
    folder_name: str
    prefix: str
    description: str = ""


TIER_SPECS: dict[str, TierSpec] = {
    "dynamic": TierSpec(
        name="dynamic",
        folder_name="00_dynamic",
        prefix="dyn_",
        description="Dynamic non-linear saturation model",
    ),
    "clean": TierSpec(
        name="clean",
        folder_name="01_studio_clean",
        prefix="cln_",
        description="Studio clean linear model",
    ),
    "standard": TierSpec(
        name="standard",
        folder_name="02_standard_dynamic",
        prefix="std_",
        description="Standard dynamic model",
    ),
    "hotrod": TierSpec(
        name="hotrod",
        folder_name="03_hot_rod",
        prefix="hot_",
        description="Hot-rod high-gain model",
    ),
}
TIER_SPECS["dyn"] = TIER_SPECS["dynamic"]
TIER_SPECS["std"] = TIER_SPECS["standard"]


def get_tier_spec(tier: str | None = None) -> TierSpec:
    """Resolves a processing/export tier specification with alias normalization."""
    key = (tier or "dynamic").lower().strip()
    if key not in TIER_SPECS:
        raise KeyError(f"Unknown tier '{tier}'. Available tiers: {list(TIER_SPECS.keys())}")
    return TIER_SPECS[key]


VOICE_CONCISE_SLUGS: dict[str, str] = {
    "00_canonical_intermediate": "00_canonical",
    "01_modern_jazz_active": "01_jazz_act",
    "02_jazz_bass_pair": "02_jazz_pair",
    "02b_jazz_bass_pair_22nf": "02b_j_22nf",
    "02c_jazz_bridge_growl_bias": "02c_jaco_growl",
    "03_jazz_bridge_60s": "03_jazz_bridge",
    "04_modern_p_ceramic": "04_modern_p",
    "05_vintage_62_p_alnico": "05_vintage_p",
    "05b_vintage_62_p_22nf": "05b_p_22nf",
    "05c_vintage_62_p_47nf": "05c_p_47nf",
    "05d_vintage_50s_p_100nf": "05d_p_100nf",
    "07_modern_pj_active": "07_pj_act",
    "08_vintage_pj_passive": "08_pj_pass",
    "09_stingray_mm_parallel": "09_stingray",
    "09b_stingray_mm_series": "09b_mm_series",
    "10_rickenbacker_bridge_hpf": "10_rick_hpf",
    "11_modern_pmm_active": "11_pmm_act",
    "11b_pmm_hybrid_series": "11b_pmm_series",
    "12_mudbucker_ultra_series": "12_mudbucker",
    "13_dingwall_multiscale_bridge": "13_dingwall",
    "14_upright_bridge_transducer": "14_upright",
    "15_neutral_character": "15_neutral",
    "15b_active_character": "15b_active",
    "15c_passive_character": "15c_passive",
}


def get_baked_basename(
    voice_id: str,
    tier: str = "dynamic",
    pickup: str = "auto",
    version_tag: str | None = None,
    preserve_aperture: bool = False,
) -> str:
    """Generates a concise, distinct model/wav basename for baked voice transformations.

    Format:
      - Default auto-routed pickup or preserve_aperture tones: '{tier_prefix}{voice_slug}' (e.g. 'dyn_04_modern_p', 'dyn_15b_active')
      - With version tag: '{tier_prefix}{voice_slug}_{version_tag}' (e.g. 'dyn_04_modern_p_v2.1.1')
      - Explicit non-auto pickup override: '{tier_prefix}{voice_slug}_{pickup}_{version_tag}'
    """
    tier_spec = get_tier_spec(tier)
    prefix = tier_spec.prefix
    slug = VOICE_CONCISE_SLUGS.get(voice_id, voice_id)

    vcfg = VOICES.get(voice_id)
    is_character_tone = (
        preserve_aperture
        or (vcfg is not None and vcfg.preserve_aperture)
        or voice_id in ("15_neutral_character", "15b_active_character", "15c_passive_character")
    )

    ver_suffix = f"_{version_tag}" if version_tag else ""
    if pickup and pickup != "auto" and not is_character_tone:
        return f"{prefix}{slug}_{pickup}{ver_suffix}"
    return f"{prefix}{slug}{ver_suffix}"


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
        "Active Character",
        "Neutral Character",
        "Passive Character",
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
      - 'all' -> all configured playable source instruments in INSTRUMENTS (excluding canonical_intermediate)
      - Comma-separated list: '30in,32in_fretless,34in_standard_p'
      - Single instrument ID or alias: '30in', 'fretless', 'jazz'
      - Partial / alias matching
    """
    all_playable: list[str] = [
        str(iid) for iid in sorted(INSTRUMENTS.keys()) if str(iid) != "canonical_intermediate"
    ]
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
            if iid != "canonical_intermediate" and iid not in resolved:
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


def get_canonical_sweep_basename(version_tag: str | None = None) -> str:
    """Generates the filename basename for the Canonical Intermediate calibration sweep.

    Default version tag is single-part DSP generation 'v{dsp}' (e.g. 'v2'),
    as the universal canonical datum depends strictly on the core DSP physics generation.
    Example: 'canonical_sweep_v2'
    """
    from allomorph.version import DSP_GENERATION

    v_tag = version_tag or f"v{DSP_GENERATION}"
    return f"canonical_sweep_{v_tag}"


def get_canonical_sweep_path(
    version_tag: str | None = None, audio_dir: Path | str | None = None
) -> Path:
    """Returns the absolute or relative Path to the versioned canonical sweep WAV.

    Example: audio/canonical/canonical_sweep_v2.wav
    """
    from allomorph.config.scales import REPO_ROOT

    base_dir = Path(audio_dir) if audio_dir is not None else REPO_ROOT / "audio"
    return base_dir / "canonical" / f"{get_canonical_sweep_basename(version_tag)}.wav"


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


def get_instrument_dry_basename(inst_id: str, pickup: str | None = None) -> str:
    """Generates the filename basename for an instrument pickup's dedicated baked dry audio.

    Located in the pickup's dedicated 'dry/' subdirectory to keep it isolated from wet stems.
    Example: 'dry_34in_standard_p_split_p' or 'dry_34in_standard_jazz_bridge'
    """
    if pickup:
        return f"dry_{inst_id}_{pickup}"
    return f"dry_{inst_id}"


def get_instrument_dry_path(
    inst_id: str,
    pickup: str | None = None,
    audio_dir: Path | str | None = None,
) -> Path:
    """Returns the primary Path to the instrument pickup's dedicated baked dry WAV in its dry/ subdirectory.

    Example: audio/baked/34in_standard_p/split_p/dry/dry_34in_standard_p_split_p.wav
    """
    from allomorph.config.scales import REPO_ROOT

    base_dir = Path(audio_dir) if audio_dir is not None else REPO_ROOT / "audio"
    if pickup:
        return base_dir / "baked" / inst_id / pickup / "dry" / f"{get_instrument_dry_basename(inst_id, pickup)}.wav"
    return base_dir / "baked" / inst_id / "dry" / f"{get_instrument_dry_basename(inst_id)}.wav"

