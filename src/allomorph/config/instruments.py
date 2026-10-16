"""
Source instrument configuration loading, tone pack partitioning, and voicing resolution.
"""

import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from allomorph.config.resolvers import (
    find_voicing_by_slug_or_id,
    parse_target_voicing_token,
)
from allomorph.config.schema import InstrumentConfig, TonePackConfig, VoicingConfig

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
CONFIG_DIR = REPO_ROOT / "config"
INSTRUMENTS_DIR = CONFIG_DIR / "instruments"
PACKS_CONFIG_DIR = CONFIG_DIR / "packs"


def load_instrument(
    identifier_or_path: str | Path | InstrumentConfig,
) -> InstrumentConfig:
    """
    Loads and validates an instrument configuration from a file path, canonical ID, or InstrumentConfig.
    """
    if isinstance(identifier_or_path, InstrumentConfig):
        return identifier_or_path

    key = str(identifier_or_path).strip()

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


@dataclass
class TargetVoicingRef:
    instrument_id: str
    voicing_id: str
    instrument: InstrumentConfig
    voicing: VoicingConfig
    custom_name: str | None = None


@dataclass
class VoicingBundle:
    bundle_name: str
    source_voicing: str
    position_name: str | None = None
    targets: list[TargetVoicingRef] = field(default_factory=list)
    source: str = ""


def is_identity_voicing(
    source_inst: InstrumentConfig,
    source_voicing: VoicingConfig | str,
    target_inst: InstrumentConfig,
    target_voicing: VoicingConfig | str,
) -> bool:
    """Determines if target voicing is an exact identity mapping of source voicing."""
    if source_inst.id != target_inst.id:
        return False
    if isinstance(source_voicing, str) and ":" in source_voicing:
        _, source_voicing = source_voicing.split(":", 1)
    if isinstance(target_voicing, str) and ":" in target_voicing:
        _, target_voicing = target_voicing.split(":", 1)
    src_v = (
        source_voicing
        if isinstance(source_voicing, VoicingConfig)
        else source_inst.voicings.get(source_voicing)
    )
    tgt_v = (
        target_voicing
        if isinstance(target_voicing, VoicingConfig)
        else target_inst.voicings.get(target_voicing)
    )
    if src_v is not None and tgt_v is not None:
        src_body = src_v.body_type_override or source_inst.body_type
        tgt_body = tgt_v.body_type_override or target_inst.body_type
        if (
            src_v.harness == tgt_v.harness
            and src_v.controls == tgt_v.controls
            and src_v.switches == tgt_v.switches
            and src_v.switch == tgt_v.switch
            and src_v.components == tgt_v.components
            and src_v.gain_db == tgt_v.gain_db
            and src_v.string_preset_override == tgt_v.string_preset_override
            and src_body == tgt_body
        ):
            return True
    elif isinstance(source_voicing, str) and isinstance(target_voicing, str):
        return source_voicing == target_voicing
    return False


def load_tone_pack(pack_id_or_path: str | Path | TonePackConfig) -> TonePackConfig:
    """Loads a declarative TonePackConfig from config/packs/<pack_id>.toml or a direct path."""
    if isinstance(pack_id_or_path, TonePackConfig):
        return pack_id_or_path
    key = str(pack_id_or_path).strip()
    p = Path(pack_id_or_path)
    if not p.is_file():
        if (PACKS_CONFIG_DIR / f"{key}.toml").is_file():
            p = PACKS_CONFIG_DIR / f"{key}.toml"
        elif (PACKS_CONFIG_DIR / key).is_file():
            p = PACKS_CONFIG_DIR / key
    if not p.is_file():
        raise FileNotFoundError(
            f"Tone pack configuration not found: '{pack_id_or_path}' (searched in {PACKS_CONFIG_DIR})"
        )
    with open(p, "rb") as f:
        data = tomllib.load(f)
    return TonePackConfig.model_validate(data)


def load_all_tone_packs(packs_dir: str | Path | None = None) -> dict[str, TonePackConfig]:
    """Loads all tone pack definitions found in packs_dir into validated TonePackConfig models."""
    pdir = Path(packs_dir) if packs_dir else PACKS_CONFIG_DIR
    packs: dict[str, TonePackConfig] = {}
    if pdir.exists():
        for p in sorted(pdir.glob("*.toml")):
            with open(p, "rb") as f:
                cfg = tomllib.load(f)
                pack = TonePackConfig.model_validate(cfg)
                packs[pack.id] = pack
    return packs


TONE_PACKS: dict[str, TonePackConfig] = load_all_tone_packs()


def partition_instrument_bundles(
    source_inst: InstrumentConfig | str | Path,
) -> dict[str, VoicingBundle]:
    """Partitions catalog target voicings into upload bundles for a source instrument."""
    inst = load_instrument(source_inst)

    if inst.id in TONE_PACKS:
        pack_cfg = TONE_PACKS[inst.id]
    else:
        pack_path = PACKS_CONFIG_DIR / f"{inst.id}.toml"
        if not pack_path.is_file():
            raise FileNotFoundError(
                f"Tone pack configuration not found for instrument '{inst.id}'. "
                f"Expected explicit pack configuration at '{pack_path}'. "
                f"All tone packs must be explicitly defined in config/packs/."
            )
        pack_cfg = load_tone_pack(pack_path)
    pack_bundles: dict[str, VoicingBundle] = {}
    for b_cfg in pack_cfg.bundles:
        src_inst_id, src_vid = b_cfg.source.split(":", 1)
        if src_inst_id != inst.id:
            raise ValueError(
                f"Bundle '{b_cfg.name}' source '{b_cfg.source}' instrument '{src_inst_id}' "
                f"does not match pack instrument '{inst.id}'."
            )
        if src_vid not in inst.voicings:
            raise KeyError(
                f"Bundle '{b_cfg.name}' source voicing '{src_vid}' not found on instrument '{inst.id}'. "
                f"Available voicings: {list(inst.voicings.keys())}"
            )
        src_v = inst.voicings[src_vid]
        pos_label = (
            b_cfg.name.capitalize()
            if b_cfg.name.lower() in ("parallel", "series", "split", "neck", "bridge")
            else (src_v.tone_name or b_cfg.name)
        )

        bundle = VoicingBundle(
            bundle_name=b_cfg.name,
            source_voicing=src_vid,
            position_name=pos_label,
            targets=[],
            source=b_cfg.source,
        )
        pack_bundles[b_cfg.name] = bundle

    for b_cfg in pack_cfg.bundles:
        bundle = pack_bundles[b_cfg.name]
        for target_entry in b_cfg.targets:
            target_token = target_entry if isinstance(target_entry, str) else target_entry.target
            custom_name = None if isinstance(target_entry, str) else target_entry.name

            tgt_inst_id, tgt_vid = parse_target_voicing_token(target_token, default_inst_id=inst.id)

            tgt_inst = (
                INSTRUMENTS[tgt_inst_id]
                if tgt_inst_id in INSTRUMENTS
                else load_instrument(tgt_inst_id)
            )
            tgt_v = find_voicing_by_slug_or_id(tgt_inst, tgt_vid)
            if tgt_v is None:
                raise KeyError(
                    f"Target voicing '{tgt_vid}' not found on instrument '{tgt_inst_id}'. "
                    f"Available voicings: {list(tgt_inst.voicings.keys())}"
                )

            if is_identity_voicing(inst, b_cfg.source_voicing, tgt_inst, tgt_v):
                continue

            bundle.targets.append(
                TargetVoicingRef(
                    instrument_id=tgt_inst_id,
                    voicing_id=tgt_v.id or tgt_vid,
                    instrument=tgt_inst,
                    voicing=tgt_v,
                    custom_name=custom_name,
                )
            )
    return pack_bundles


def resolve_target_voicing(
    voicing: VoicingConfig | str,
    instrument: InstrumentConfig | str | Path | None = None,
) -> tuple[InstrumentConfig, VoicingConfig]:
    """Resolves an instrument and voicing strictly from '<instrument_id>:<voicing_id>' or '<voicing_id>' with explicit instrument."""
    if isinstance(voicing, VoicingConfig):
        inst = (
            load_instrument(instrument)
            if instrument is not None and not isinstance(instrument, InstrumentConfig)
            else (instrument if isinstance(instrument, InstrumentConfig) else None)
        )
        if inst is None:
            raise ValueError("Must provide instrument when voicing is a VoicingConfig")
        return inst, voicing

    raw_voicing = str(voicing).strip()

    # Canonical reference "<instrument_id>:<voicing_id>" always takes precedence
    if ":" in raw_voicing:
        inst_part, voicing_part = raw_voicing.split(":", 1)
        inst = INSTRUMENTS[inst_part] if inst_part in INSTRUMENTS else load_instrument(inst_part)
        found_v = find_voicing_by_slug_or_id(inst, voicing_part)
        if found_v is not None:
            return inst, found_v
        if voicing_part in inst.pickups:
            raise KeyError(
                f"Pickup '{voicing_part}' cannot be used as a target voicing. All voicings must be "
                f"explicitly defined under [voicings]. Available voicings on '{inst.id}': {list(inst.voicings.keys())}"
            )
        raise KeyError(
            f"Voicing '{voicing_part}' not found on instrument '{inst.id}'. "
            f"Available voicings: {list(inst.voicings.keys())}"
        )

    if instrument is not None:
        inst = (
            load_instrument(instrument)
            if not isinstance(instrument, InstrumentConfig)
            else instrument
        )
        found_v = find_voicing_by_slug_or_id(inst, raw_voicing)
        if found_v is not None:
            return inst, found_v

        if raw_voicing in inst.pickups:
            raise KeyError(
                f"Pickup '{raw_voicing}' cannot be used as a target voicing. All voicings must be "
                f"explicitly defined under [voicings]. Available voicings on '{inst.id}': {list(inst.voicings.keys())}"
            )

        raise KeyError(
            f"Voicing or pickup '{voicing}' not found on instrument '{inst.id}'. "
            f"Available voicings: {list(inst.voicings.keys())}, pickups: {list(inst.pickups.keys())}"
        )

    raise KeyError(
        f"Target voicing '{voicing}' must be in canonical '<instrument_id>:<voicing_id>' format when no instrument is provided."
    )
