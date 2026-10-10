"""
Source instrument configuration loading, alias resolution, and pickup lookup.
"""

import tomllib
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

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
}

from allomorph.config.schema import InstrumentConfig, TonePackConfig, VoicingConfig

PACKS_CONFIG_DIR = CONFIG_DIR / "packs"


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


STANDARD_CATALOG_TARGETS: list[tuple[str, str]] = [
    ("34in_standard_p", "vintage_open"),
    ("34in_standard_p", "vintage_mids"),
    ("34in_standard_p", "vintage_warm"),
    ("34in_standard_p", "modern_active"),
    ("34in_standard_p", "slab_dub"),
    ("34in_standard_jazz", "pair_open"),
    ("34in_standard_jazz", "pair_mids"),
    ("34in_standard_jazz", "pair_active"),
    ("34in_standard_jazz", "bridge_growl"),
    ("34in_standard_jazz", "bridge_open"),
    ("34in_standard_jazz", "neck_warm"),
    ("34in_active_stingray", "parallel_classic"),
    ("34in_active_stingray", "series_punch"),
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
    ("41in_upright_bass", "bridge_piezo"),
]

# Additional source instrument voicings available in the visualizer suite
# (for comparing physical source bass configurations against target voicings)
SOURCE_CATALOG_VOICINGS: list[tuple[str, str]] = [
    ("34in_preamp_soapbar", "pair_parallel"),
    ("34in_preamp_soapbar", "neck_solo"),
    ("34in_preamp_soapbar", "bridge_solo"),
    ("34in_active_emg", "pair_parallel"),
    ("34in_active_emg", "neck_solo"),
    ("34in_active_emg", "bridge_solo"),
]


@dataclass
class TargetVoicingRef:
    instrument_id: str
    voicing_id: str
    instrument: InstrumentConfig
    voicing: VoicingConfig


@dataclass
class VoicingBundle:
    bundle_name: str
    source_voicing: str
    position_name: str | None = None
    targets: list[TargetVoicingRef] = field(default_factory=list)


def is_identity_voicing(
    source_inst: InstrumentConfig,
    source_voicing: VoicingConfig | str,
    target_inst: InstrumentConfig,
    target_voicing: VoicingConfig | str,
) -> bool:
    """Determines if target voicing is an exact identity mapping of source voicing."""
    if source_inst.id != target_inst.id:
        return False
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
        if (
            src_v.harness == tgt_v.harness
            and src_v.controls == tgt_v.controls
            and src_v.switches == tgt_v.switches
            and src_v.switch == tgt_v.switch
            and src_v.components == tgt_v.components
            and src_v.gain_db == tgt_v.gain_db
            and src_v.string_preset_override == tgt_v.string_preset_override
        ):
            return True
    elif isinstance(source_voicing, str) and isinstance(target_voicing, str):
        return source_voicing == target_voicing
    return False


def load_tone_pack(pack_id_or_path: str | Path | TonePackConfig) -> TonePackConfig:
    """Loads a declarative TonePackConfig from config/packs/<pack_id>.toml or a direct path."""
    if isinstance(pack_id_or_path, TonePackConfig):
        return pack_id_or_path
    p = Path(pack_id_or_path)
    if not p.is_file():
        if (PACKS_CONFIG_DIR / f"{pack_id_or_path}.toml").is_file():
            p = PACKS_CONFIG_DIR / f"{pack_id_or_path}.toml"
        elif (PACKS_CONFIG_DIR / pack_id_or_path).is_file():
            p = PACKS_CONFIG_DIR / pack_id_or_path
    if not p.is_file():
        raise FileNotFoundError(
            f"Tone pack configuration not found: '{pack_id_or_path}' (searched in {PACKS_CONFIG_DIR})"
        )
    with open(p, "rb") as f:
        data = tomllib.load(f)
    return TonePackConfig.model_validate(data)


def partition_instrument_bundles(
    source_inst: InstrumentConfig | str | Path,
    catalog_targets: Sequence[tuple[str, str]] | None = None,
) -> dict[str, VoicingBundle]:
    """Partitions catalog target voicings into upload bundles for a source instrument."""
    inst = load_instrument(source_inst)

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
        src_vid = b_cfg.source_voicing
        if src_vid not in inst.voicings:
            raise KeyError(
                f"Bundle '{b_cfg.name}' source_voicing '{src_vid}' not found on instrument '{inst.id}'. "
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
        )
        pack_bundles[b_cfg.name] = bundle

    from allomorph.config.voices import VOICES

    if catalog_targets is None:
        for b_cfg in pack_cfg.bundles:
            bundle = pack_bundles[b_cfg.name]
            for target_token in b_cfg.targets:
                if ":" in target_token:
                    tgt_inst_id, tgt_vid = target_token.split(":", 1)
                elif target_token in VOICES:
                    v_obj = VOICES[target_token]
                    tgt_inst_id = v_obj.instrument_id or inst.id
                    tgt_vid = v_obj.id
                else:
                    tgt_inst_id = inst.id
                    tgt_vid = target_token

                tgt_inst = (
                    INSTRUMENTS[tgt_inst_id]
                    if tgt_inst_id in INSTRUMENTS
                    else load_instrument(tgt_inst_id)
                )
                tgt_v = tgt_inst.voicings.get(tgt_vid)
                if tgt_v is None:
                    # Look up by tone slug
                    for v_cand in tgt_inst.voicings.values():
                        cand_slug = (
                            v_cand.tone_name.lower()
                            .replace(" ", "_")
                            .replace("∕", "_")
                            .replace("/", "_")
                            if v_cand.tone_name
                            else v_cand.id
                        )
                        if cand_slug == tgt_vid:
                            tgt_v = v_cand
                            break
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
                    )
                )
        return pack_bundles
    else:
        for target_inst_id, target_vid in catalog_targets:
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

            assigned_b_name = next(iter(pack_bundles.keys()))
            found = False
            for b_cfg in pack_cfg.bundles:
                if target_vid in b_cfg.targets or any(target_vid in t for t in b_cfg.targets):
                    assigned_b_name = b_cfg.name
                    found = True
                    break
            if not found and len(pack_bundles) > 1:
                for bn in pack_bundles:
                    if bn in target_vid.lower():
                        assigned_b_name = bn
                        break

            target_bundle = pack_bundles[assigned_b_name]
            if is_identity_voicing(inst, target_bundle.source_voicing, target_inst, target_voicing):
                continue

            target_bundle.targets.append(
                TargetVoicingRef(
                    instrument_id=target_inst_id,
                    voicing_id=target_vid,
                    instrument=target_inst,
                    voicing=target_voicing,
                )
            )
        return pack_bundles


def resolve_target_voicing(
    voicing: VoicingConfig | str,
    instrument: InstrumentConfig | str | Path | None = None,
) -> tuple[InstrumentConfig, VoicingConfig]:
    """Resolves an instrument and voicing from either a native voicing ID or a universal voice slug."""
    if isinstance(voicing, VoicingConfig):
        inst = (
            load_instrument(instrument)
            if instrument is not None and not isinstance(instrument, InstrumentConfig)
            else (instrument if isinstance(instrument, InstrumentConfig) else None)
        )
        if inst is None:
            raise ValueError("Must provide instrument when voicing is a VoicingConfig")
        return inst, voicing

    clean_voicing = str(voicing).lower().replace(" ", "_").replace("∕", "_").replace("/", "_")

    # If canonical reference "instrument:voicing" is supplied without explicit instrument, parse it
    if ":" in str(voicing) and instrument is None:
        inst_part, voicing_part = str(voicing).split(":", 1)
        inst = load_instrument(inst_part)
        clean_v_part = voicing_part.lower().replace(" ", "_").replace("∕", "_").replace("/", "_")
        if voicing_part in inst.voicings:
            return inst, inst.voicings[voicing_part]
        for v in inst.voicings.values():
            slug = (
                v.tone_name.lower().replace(" ", "_").replace("∕", "_").replace("/", "_")
                if v.tone_name
                else (v.id or "")
            )
            if clean_v_part in (v.id, slug):
                return inst, v

    # If instrument was provided, check its native voicings first
    if instrument is not None:
        inst = (
            load_instrument(instrument)
            if not isinstance(instrument, InstrumentConfig)
            else instrument
        )
        if voicing in inst.voicings:
            return inst, inst.voicings[voicing]
        for v in inst.voicings.values():
            slug = (
                v.tone_name.lower().replace(" ", "_").replace("∕", "_").replace("/", "_")
                if v.tone_name
                else (v.id or "")
            )
            if clean_voicing in (v.id, slug):
                return inst, v

        # Check if voicing matches a declared pickup on the instrument - reject with fail-fast KeyError
        if clean_voicing in inst.pickups or str(voicing) in inst.pickups:
            p_key = clean_voicing if clean_voicing in inst.pickups else str(voicing)
            raise KeyError(
                f"Pickup '{p_key}' cannot be used as a target voicing. All voicings must be "
                f"explicitly defined under [voicings]. Available voicings on '{inst.id}': {list(inst.voicings.keys())}"
            )

    # Search STANDARD_CATALOG_TARGETS and SOURCE_CATALOG_VOICINGS across all instruments
    for iid, vid in list(STANDARD_CATALOG_TARGETS) + list(SOURCE_CATALOG_VOICINGS):
        t_inst = load_instrument(iid)
        if vid in t_inst.voicings:
            t_v = t_inst.voicings[vid]
            slug = (
                t_v.tone_name.lower().replace(" ", "_").replace("∕", "_").replace("/", "_")
                if t_v.tone_name
                else vid
            )
            if clean_voicing in (vid, slug, (t_v.id or "").lower()):
                return t_inst, t_v

    # Fallback error reporting
    if instrument is not None:
        inst = (
            load_instrument(instrument)
            if not isinstance(instrument, InstrumentConfig)
            else instrument
        )
        raise KeyError(
            f"Voicing or pickup '{voicing}' not found on instrument '{inst.id}'. "
            f"Available voicings: {list(inst.voicings.keys())}, pickups: {list(inst.pickups.keys())}"
        )

    raise KeyError(
        f"Target voicing '{voicing}' could not be resolved from any instrument in catalog."
    )
