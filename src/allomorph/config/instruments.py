"""
Source instrument configuration loading, alias resolution, and pickup lookup.
"""

import tomllib
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
CONFIG_DIR = REPO_ROOT / "config"
INSTRUMENTS_DIR = CONFIG_DIR / "instruments"

INSTRUMENT_ALIASES: dict[str, str] = {
    "30in": "30in_emg_mmtw",
    "30in_mm": "30in_emg_mmtw",
    "30in_mmtw": "30in_emg_mmtw",
    "30in_emg_mm": "30in_emg_mmtw",
    "30in_emg_mmtw": "30in_emg_mmtw",
    "32in": "32in_custom_pmm",
    "32in_fretless": "32in_fretless_pmm",
    "fretless": "32in_fretless_pmm",
    "34in": "34in_standard_p",
    "standard_p": "34in_standard_p",
    "34in_standard_jazz": "34in_standard_jazz",
    "standard_jazz": "34in_standard_jazz",
    "34in_standard_pj": "34in_standard_pj",
    "standard_pj": "34in_standard_pj",
    "34in_pj": "34in_standard_pj",
    "pj": "34in_standard_pj",
    "34in_active_stingray": "34in_active_stingray",
    "active_stingray": "34in_active_stingray",
    "stingray": "34in_active_stingray",
    "ray": "34in_active_stingray",
    "34in_active_pmm": "34in_active_pmm",
    "active_pmm": "34in_active_pmm",
    "pmm": "34in_active_pmm",
    "sandberg": "34in_active_pmm",
    "lakland": "34in_active_pmm",
    "34in_preamp_soapbar": "34in_preamp_soapbar",
    "preamp_soapbar": "34in_preamp_soapbar",
    "preamp_soapbar_5string": "34in_preamp_soapbar",
    "ibanez_sr": "34in_preamp_soapbar",
    "ibanez_sr505": "34in_preamp_soapbar",
    "sr505": "34in_preamp_soapbar",
    "sr505e": "34in_preamp_soapbar",
    "yamaha_trbx": "34in_preamp_soapbar",
    "yamaha_trbx505": "34in_preamp_soapbar",
    "trbx": "34in_preamp_soapbar",
    "trbx505": "34in_preamp_soapbar",
    "sire_f10": "34in_preamp_soapbar",
    "sire_f10_5": "34in_preamp_soapbar",
    "sire_m7": "34in_preamp_soapbar",
    "sire_m7_5": "34in_preamp_soapbar",
    "sire": "34in_preamp_soapbar",
    "marcus_miller_f10": "34in_preamp_soapbar",
    "stingray_hh": "34in_preamp_soapbar",
    "stingray5_hh": "34in_preamp_soapbar",
    "ray34hh": "34in_preamp_soapbar",
    "ray35hh": "34in_preamp_soapbar",
    "34in_active_emg": "34in_active_emg",
    "active_emg": "34in_active_emg",
    "emg40": "34in_active_emg",
    "emg40dc": "34in_active_emg",
    "emg40cs": "34in_active_emg",
    "34in_emg": "34in_active_emg",
    "spector5": "34in_active_emg",
    "spector_euro5": "34in_active_emg",
    "34in_5string_emg": "34in_active_emg",
    "30in_mustang_pj": "30in_mustang_pj",
    "mustang_pj": "30in_mustang_pj",
    "30in_mustang": "30in_mustang_pj",
    "mustang": "30in_mustang_pj",
    "30in_standard_mustang": "30in_mustang_pj",
    "standard_mustang": "30in_mustang_pj",
    "37in_multiscale_dingwall": "37in_multiscale_dingwall",
    "multiscale_dingwall": "37in_multiscale_dingwall",
    "dingwall": "37in_multiscale_dingwall",
    "combustion": "37in_multiscale_dingwall",
    "ng": "37in_multiscale_dingwall",
    "dingwall_ng": "37in_multiscale_dingwall",
    "ng2": "37in_multiscale_dingwall",
    "ng3": "37in_multiscale_dingwall",
    "37in_dingwall_ng": "37in_multiscale_dingwall",
    "34in_dingwall_sp1": "34in_dingwall_sp1",
    "35in_dingwall_sp1": "34in_dingwall_sp1",
    "dingwall_sp1": "34in_dingwall_sp1",
    "sp1": "34in_dingwall_sp1",
    "super_p": "34in_dingwall_sp1",
    "dingwall_super_p": "34in_dingwall_sp1",
    "33in_rickenbacker_4003": "33in_rickenbacker_4003",
    "rickenbacker": "33in_rickenbacker_4003",
    "rickenbacker_4003": "33in_rickenbacker_4003",
    "rick": "33in_rickenbacker_4003",
    "4003": "33in_rickenbacker_4003",
    "33in": "33in_rickenbacker_4003",
    "30in_gibson_eb0": "30in_gibson_eb0",
    "gibson_eb0": "30in_gibson_eb0",
    "eb0": "30in_gibson_eb0",
    "mudbucker": "30in_gibson_eb0",
    "30in_eb0": "30in_gibson_eb0",
    "41in_upright_bass": "41in_upright_bass",
    "upright": "41in_upright_bass",
    "upright_bass": "41in_upright_bass",
    "double_bass": "41in_upright_bass",
    "acoustic_upright": "41in_upright_bass",
    "41in": "41in_upright_bass",
    "studio_direct": "studio_direct",
    "studio": "studio_direct",
    "direct": "studio_direct",
    "di": "studio_direct",
}


from allomorph.config.schema import InstrumentConfig, PickupConfig, VoicingConfig


def load_instrument(
    identifier_or_path: str | Path | InstrumentConfig,
) -> InstrumentConfig:
    """
    Loads and validates an instrument configuration from a file path, known ID, shorthand alias, or InstrumentConfig.
    Aliases: '30in' -> '30in_emg_mmtw', '32in' -> '32in_custom_pmm', '34in' -> '34in_standard_p'.
    """
    if isinstance(identifier_or_path, InstrumentConfig):
        return identifier_or_path

    raw = str(identifier_or_path).strip()
    key = INSTRUMENT_ALIASES.get(raw, raw)

    path = Path(key)
    if not path.exists():
        if (INSTRUMENTS_DIR / f"{key}.toml").exists():
            path = INSTRUMENTS_DIR / f"{key}.toml"
        elif (INSTRUMENTS_DIR / key).exists():
            path = INSTRUMENTS_DIR / key
        elif (CONFIG_DIR / f"{key}.toml").exists():
            path = CONFIG_DIR / f"{key}.toml"
        else:
            raise FileNotFoundError(
                f"Instrument configuration not found: '{identifier_or_path}' (searched in {INSTRUMENTS_DIR})"
            )

    with open(path, "rb") as f:
        data = tomllib.load(f)
    return InstrumentConfig.model_validate(data)


def load_all_instruments(instruments_dir: str | Path | None = None) -> dict[str, InstrumentConfig]:
    """Loads all instrument definitions found in instruments_dir into validated InstrumentConfig models."""
    idir = Path(instruments_dir) if instruments_dir else INSTRUMENTS_DIR
    instruments: dict[str, InstrumentConfig] = {}
    if idir.exists():
        for p in sorted(idir.glob("*.toml")):
            with open(p, "rb") as f:
                cfg = tomllib.load(f)
                inst = InstrumentConfig.model_validate(cfg)
                instruments[inst.id] = inst
    return instruments


INSTRUMENTS: dict[str, InstrumentConfig] = load_all_instruments()

VOICE_AFFINITIES: dict[str, str] = {
    "precision_vintage": "neck",
    "precision_mids": "neck",
    "precision_warm": "neck",
    "precision_active": "neck",
    "pj_neck_active": "neck",
    "precision_dub": "neck",
    "jazz_pair_open": "parallel",
    "jazz_pair_mids": "parallel",
    "jazz_pair_active": "parallel",
    "jazz_bridge_growl": "bridge",
    "jazz_bridge_open": "bridge",
    "jazz_neck_warm": "neck",
    "stingray_parallel": "bridge",
    "stingray_series": "bridge",
    "stingray_active": "bridge",
    "dingwall_bridge": "bridge",
    "dingwall_middle": "neck",
    "dingwall_parallel": "parallel",
    "rickenbacker_clank": "bridge",
    "rickenbacker_open": "bridge",
    "pj_passive": "parallel",
    "pj_active": "parallel",
    "p_mm_parallel": "parallel",
    "p_mm_series": "parallel",
    "mudbucker_deep": "neck",
    "upright_acoustic": "neck",
    "studio_direct": "direct",
    "studio_active": "direct",
    "studio_passive": "direct",
}


def resolve_target_affinity(target: str | VoicingConfig | Any) -> str | None:
    """Resolves physical position affinity ('neck', 'bridge', 'parallel', 'direct') for a target."""
    if hasattr(target, "affinity") and getattr(target, "affinity", None):
        return str(target.affinity)
    target_str = str(target).strip()
    if target_str in ("neck", "bridge", "parallel", "direct"):
        return target_str
    if target_str in VOICE_AFFINITIES:
        return VOICE_AFFINITIES[target_str]
    if ":" in target_str:
        inst_id, voice_id = target_str.split(":", 1)
        if inst_id in INSTRUMENTS:
            inst = INSTRUMENTS[inst_id]
            if voice_id in inst.voicings:
                return inst.voicings[voice_id].affinity
    for inst in INSTRUMENTS.values():
        if target_str in inst.voicings:
            return inst.voicings[target_str].affinity
    return None


def get_source_pickup(
    instrument: InstrumentConfig, voice_id: str | VoicingConfig | Any
) -> PickupConfig:
    """
    Determines which pickup on the source instrument should be used for the target voice.
    Checks explicit pickup_mapping, falls back to affinity mapping, default_pickup, or raises diagnostic error.
    """
    inst = instrument
    pickups = inst.pickups
    if not pickups:
        raise ValueError(f"Instrument '{inst.id}' has no pickups defined.")

    # 1. Single pickup bass: zero ambiguity
    if len(pickups) == 1:
        p_key = next(iter(pickups.keys()))
        p_raw = pickups[p_key]
        p = p_raw.model_copy(deep=True)
        p.id = p_key
        return p

    target_key = voice_id.id if hasattr(voice_id, "id") and voice_id.id else str(voice_id)
    mapping = inst.pickup_mapping

    # 2. Explicit voice/target mapping
    if target_key in mapping and mapping[target_key] in pickups:
        p_raw = pickups[mapping[target_key]]
        p = p_raw.model_copy(deep=True)
        p.id = mapping[target_key]
        return p

    # 3. Physical position affinity mapping
    affinity = resolve_target_affinity(voice_id)
    if affinity and affinity in mapping and mapping[affinity] in pickups:
        p_raw = pickups[mapping[affinity]]
        p = p_raw.model_copy(deep=True)
        p.id = mapping[affinity]
        return p

    # 4. Default pickup declared on instrument
    default_key = inst.default_pickup
    if default_key:
        if default_key in pickups:
            p_raw = pickups[default_key]
            p = p_raw.model_copy(deep=True)
            p.id = default_key
            return p
        raise KeyError(
            f"Instrument '{inst.id}' default_pickup '{default_key}' "
            f"not found in pickups: {list(pickups.keys())}"
        )

    raise ValueError(
        f"Instrument '{inst.id}' defines no 'default_pickup' "
        f"and has no pickup_mapping for voice '{target_key}' (affinity: '{affinity}'). "
        f"Available pickups: {list(pickups.keys())}"
    )


STANDARD_CATALOG_TARGETS: list[tuple[str, str]] = [
    ("34in_standard_p", "vintage_open"),
    ("34in_standard_p", "vintage_mids"),
    ("34in_standard_p", "vintage_warm"),
    ("34in_standard_pj", "neck_active"),
    ("34in_standard_p", "slab_dub"),
    ("34in_standard_jazz", "pair_open"),
    ("34in_standard_jazz", "pair_mids"),
    ("34in_standard_jazz", "pair_active"),
    ("34in_standard_jazz", "bridge_growl"),
    ("34in_standard_jazz", "bridge_open"),
    ("34in_standard_jazz", "neck_warm"),
    ("34in_active_stingray", "parallel_classic"),
    ("34in_active_stingray", "series_punch"),
    ("34in_active_stingray", "active_slap"),
    ("37in_multiscale_dingwall", "bridge_clank"),
    ("37in_multiscale_dingwall", "middle_growl"),
    ("37in_multiscale_dingwall", "pair_parallel"),
    ("33in_rickenbacker_4003", "bridge_clank"),
    ("33in_rickenbacker_4003", "bridge_open"),
    ("34in_standard_pj", "pair_parallel"),
    ("34in_standard_pj", "pair_active"),
    ("34in_active_pmm", "pmm_parallel"),
    ("34in_active_pmm", "pmm_series"),
    ("30in_gibson_eb0", "neck_deep"),
    ("41in_upright_bass", "bridge_piezo_acoustic"),
    ("studio_direct", "clean_di"),
    ("studio_direct", "active_di"),
    ("studio_direct", "passive_load"),
]


@dataclass
class TargetVoicingRef:
    instrument_id: str
    voicing_id: str
    instrument: InstrumentConfig
    voicing: VoicingConfig
    source_pickup_key: str


@dataclass
class PickupBundle:
    bundle_name: str
    pickup_key: str
    pickup: PickupConfig
    targets: list[TargetVoicingRef] = field(default_factory=list)


def partition_instrument_bundles(
    source_inst: InstrumentConfig | str | Path,
    catalog_targets: Sequence[tuple[str, str]] | None = None,
) -> dict[str, PickupBundle]:
    """Partitions catalog target voicings into pickup-specific upload bundles for a source instrument."""
    inst = load_instrument(source_inst)
    targets_to_partition = (
        catalog_targets if catalog_targets is not None else STANDARD_CATALOG_TARGETS
    )

    bundles: dict[str, PickupBundle] = {}

    # Initialize a bundle for each defined pickup on the source instrument
    for p_key, p_cfg in inst.pickups.items():
        b_name = p_cfg.bundle_name or p_key
        if b_name not in bundles:
            p_copy = p_cfg.model_copy(deep=True)
            p_copy.id = p_key
            bundles[b_name] = PickupBundle(
                bundle_name=b_name,
                pickup_key=p_key,
                pickup=p_copy,
                targets=[],
            )

    # Route each target to the appropriate bundle
    for target_inst_id, target_vid in targets_to_partition:
        target_inst = (
            INSTRUMENTS[target_inst_id]
            if target_inst_id in INSTRUMENTS
            else load_instrument(target_inst_id)
        )
        if target_vid not in target_inst.voicings:
            raise KeyError(
                f"Target voicing '{target_vid}' not found on instrument '{target_inst_id}'. "
                f"Available voicings: {list(target_inst.voicings.keys())}"
            )
        target_voicing = target_inst.voicings[target_vid]

        src_pickup = get_source_pickup(inst, target_voicing)
        src_pickup_key = src_pickup.id or "default"
        b_name = src_pickup.bundle_name or src_pickup_key

        if b_name not in bundles:
            bundles[b_name] = PickupBundle(
                bundle_name=b_name,
                pickup_key=src_pickup_key,
                pickup=src_pickup,
                targets=[],
            )

        bundles[b_name].targets.append(
            TargetVoicingRef(
                instrument_id=target_inst_id,
                voicing_id=target_vid,
                instrument=target_inst,
                voicing=target_voicing,
                source_pickup_key=src_pickup_key,
            )
        )

    return bundles
