"""
Allomorph - Target Voice Configuration Registry.
Constructs first-principles target VoiceConfig models dynamically from the
Unified Instrument Catalog in config/instruments/*.toml.
"""

from pathlib import Path
from typing import Any, ClassVar, override

import numpy as np

from allomorph.config.resolvers import (
    build_voice_coils_for_pickup,
    compound_multi_pickup_coils,
    infer_pickup_resonance_and_q,
)
from allomorph.config.schema import (
    InstrumentConfig,
    VoiceConfig,
    VoicePickupConfig,
    VoicingConfig,
)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
CONFIG_DIR = REPO_ROOT / "config"
INSTRUMENTS_DIR = CONFIG_DIR / "instruments"


class VoiceRegistry(dict[str, VoiceConfig]):
    """
    Dictionary registry mapping target voice identifiers (slugs and canonical references)
    to validated VoiceConfig models.
    """

    ALIASES: ClassVar[dict[str, str]] = {}

    @override
    def __getitem__(self, key: str) -> VoiceConfig:
        if super().__contains__(key):
            return super().__getitem__(key)
        if key in self.ALIASES:
            return super().__getitem__(self.ALIASES[key])
        return super().__getitem__(key)

    @override
    def get(self, key: str, default: Any = None) -> Any:
        if super().__contains__(key):
            return super().get(key, default)
        if key in self.ALIASES:
            return super().get(self.ALIASES[key], default)
        return default

    @override
    def __contains__(self, key: object) -> bool:
        return super().__contains__(key) or (
            isinstance(key, str) and key in self.ALIASES and super().__contains__(self.ALIASES[key])
        )


def resolve_voicing_active_pickups(
    instrument: InstrumentConfig,
    voicing: VoicingConfig,
) -> list[str]:
    """
    Resolves the active physical pickup IDs for an instrument voicing by evaluating
    electrical transmission through the Modified Nodal Analysis (MNA) harness.
    """
    all_pickups = list(instrument.pickups.keys())
    if len(all_pickups) <= 1:
        return all_pickups

    h_id = voicing.harness
    harnesses = getattr(instrument, "harnesses", {})
    if not harnesses or h_id not in harnesses:
        return all_pickups

    from allomorph.circuit.solver import solve_mna_harness
    from allomorph.config.preamps import PREAMPS

    curves = solve_mna_harness(
        instrument,
        harnesses[h_id],
        voicing,
        freqs=np.array([200.0, 1000.0], dtype=np.float64),
        preamps=PREAMPS,
    )
    max_trans = (
        max(float(np.max(np.abs(curves[p]))) for p in all_pickups if p in curves)
        if any(p in curves for p in all_pickups)
        else 1.0
    )
    thresh = max(0.10 * max_trans, 1e-4)
    active = [p for p in all_pickups if p in curves and np.max(np.abs(curves[p])) >= thresh]
    return active if active else all_pickups


def resolve_voicing_active_coils(
    instrument: InstrumentConfig,
    voicing: VoicingConfig,
    pickup_key: str,
) -> list[str]:
    """
    Resolves active coil IDs for a specific pickup by evaluating electrical transmission
    through the Modified Nodal Analysis (MNA) harness.
    """
    p_cfg = instrument.pickups.get(pickup_key)
    if not p_cfg or not p_cfg.coils:
        return []
    if len(p_cfg.coils) <= 1:
        return [p_cfg.coils[0].id or pickup_key]

    h_id = voicing.harness
    harnesses = getattr(instrument, "harnesses", {})
    if not harnesses or h_id not in harnesses:
        return [c.id for c in p_cfg.coils if c.id]

    from allomorph.circuit.solver import solve_mna_harness
    from allomorph.config.preamps import PREAMPS

    curves = solve_mna_harness(
        instrument,
        harnesses[h_id],
        voicing,
        freqs=np.array([200.0, 1000.0], dtype=np.float64),
        preamps=PREAMPS,
    )
    all_c_keys = [
        k for c in p_cfg.coils if c.id for k in [f"{pickup_key}.{c.id}", f"{p_cfg.id}.{c.id}", c.id]
    ]
    max_c_trans = (
        max(float(np.max(np.abs(curves[k]))) for k in all_c_keys if k in curves)
        if any(k in curves for k in all_c_keys)
        else 1.0
    )
    thresh = max(0.10 * max_c_trans, 1e-4)
    active_coils: list[str] = []
    for c in p_cfg.coils:
        if not c.id:
            continue
        c_keys = [f"{pickup_key}.{c.id}", f"{p_cfg.id}.{c.id}", c.id]
        if any(k in curves and np.max(np.abs(curves[k])) >= thresh for k in c_keys):
            active_coils.append(c.id)

    return active_coils if active_coils else [c.id for c in p_cfg.coils if c.id]


def voicing_to_voice_config(
    instrument: InstrumentConfig,
    voicing: VoicingConfig,
    slug: str | None = None,
) -> VoiceConfig:
    """Dynamically converts an instrument voicing setting into a validated VoiceConfig model."""
    target_slug = slug or (
        voicing.tone_name.lower().replace(" ", "_").replace("∕", "_").replace("/", "_")
        if voicing.tone_name
        else (voicing.id or "voice")
    )

    # 1. Determine active pickups
    active_pickup_keys = resolve_voicing_active_pickups(instrument, voicing)

    for k in active_pickup_keys:
        if k not in instrument.pickups:
            raise KeyError(
                f"Pickup '{k}' not found on instrument '{instrument.id}'. "
                f"Available pickups: {list(instrument.pickups.keys())}"
            )

    voice_pickups: list[VoicePickupConfig] = []
    if len(active_pickup_keys) >= 2:
        for p_key in active_pickup_keys:
            p = instrument.pickups[p_key]
            active_c_ids = resolve_voicing_active_coils(instrument, voicing, p_key)
            sub_coils = build_voice_coils_for_pickup(p, active_c_ids, instrument=instrument)
            fr_val, q_val = infer_pickup_resonance_and_q(
                p, pickup_key=p_key, harness_str=voicing.harness or ""
            )

            voice_pickups.append(
                VoicePickupConfig(
                    name=p.name,
                    type=f"{p.type.replace('_', ' ').title()} ({instrument.name})",
                    magnet_type=p.magnet_type or "alnico_v",
                    alpha=getattr(p, "alpha", None) or 0.25,
                    fr=fr_val,
                    Q=q_val,
                    weight=1.0 / len(active_pickup_keys),
                    polarity=1.0,
                    coils=sub_coils,
                )
            )

    if voice_pickups:
        coils = compound_multi_pickup_coils(voice_pickups)
    else:
        primary_key = (
            active_pickup_keys[0] if active_pickup_keys else next(iter(instrument.pickups.keys()))
        )
        pickup = instrument.pickups[primary_key]
        active_c_ids = resolve_voicing_active_coils(instrument, voicing, primary_key)
        coils = build_voice_coils_for_pickup(pickup, active_c_ids, instrument=instrument)

    primary_pickup = (
        instrument.pickups[active_pickup_keys[0]]
        if active_pickup_keys
        else next(iter(instrument.pickups.values()))
    )
    primary_key_val = active_pickup_keys[0] if active_pickup_keys else ""
    fr, Q = infer_pickup_resonance_and_q(
        primary_pickup,
        pickup_key=primary_key_val,
        harness_str=voicing.harness or "",
    )

    if voice_pickups:
        active_mags = list(dict.fromkeys(p.magnet_type for p in voice_pickups if p.magnet_type))
        if len(active_mags) == 1:
            magnet_type = active_mags[0]
        elif len(active_mags) > 1:
            magnet_type = " / ".join(active_mags)
        else:
            magnet_type = primary_pickup.magnet_type or "alnico_v"
    else:
        magnet_type = primary_pickup.magnet_type or "alnico_v"

    if instrument.is_multiscale:
        scale = (
            "multiscale_super"
            if (instrument.scale_max_in is not None and instrument.scale_max_in <= 35.5)
            else "multiscale"
        )
    elif instrument.scale_length_in is not None and instrument.scale_length_in >= 40.0:
        scale = "upright"
    elif instrument.scale_length_in is not None and instrument.scale_length_in <= 31.0:
        scale = "30in"
    elif instrument.scale_length_in is not None and instrument.scale_length_in <= 33.0:
        scale = "32in"
    else:
        scale = "34in"

    target_string = (
        voicing.string_preset_override or instrument.strings.preset or "roundwound_nickel_standard"
    )
    gain_db = voicing.gain_db if voicing.gain_db is not None else 0.0
    alpha = primary_pickup.alpha or 0.25
    vsat = primary_pickup.vsat or 1.0

    return VoiceConfig(
        id=target_slug,
        name=voicing.tone_name or voicing.name,
        tone_name=voicing.tone_name,
        description=f"{voicing.name} ({instrument.name})",
        magnet_type=magnet_type,
        alpha=alpha,
        vsat=vsat,
        fr=fr,
        Q=Q,
        gain_db=gain_db,
        scale=scale,
        target_string=target_string,
        preserve_aperture=voicing.preserve_aperture,
        sensor_type=voicing.sensor_type,
        coils=coils,
        pickups=voice_pickups if len(voice_pickups) >= 2 else [],
        instrument_id=instrument.id,
    )


def load_voice_config(identifier_or_path: str | Path | VoiceConfig) -> VoiceConfig:
    """Loads and validates a single target voice configuration into a VoiceConfig model."""
    if isinstance(identifier_or_path, VoiceConfig):
        return identifier_or_path

    raw = str(identifier_or_path).strip()
    if raw in VOICES:
        return VOICES[raw]

    if ":" in raw:
        inst_id, voicing_id = raw.split(":", 1)
        from allomorph.config.instruments import load_instrument

        inst = load_instrument(inst_id)
        if voicing_id in inst.voicings:
            return voicing_to_voice_config(inst, inst.voicings[voicing_id], slug=raw)

    raise FileNotFoundError(f"Target voice not found: '{identifier_or_path}'")


def load_voices_config(instruments_path: str | Path | None = None) -> VoiceRegistry:
    """
    Constructs the target voice registry dynamically from instrument catalog configurations.
    """
    from allomorph.config.instruments import (
        SOURCE_CATALOG_VOICINGS,
        STANDARD_CATALOG_TARGETS,
        load_all_instruments,
    )

    instruments = load_all_instruments(instruments_path or INSTRUMENTS_DIR)
    voices_dict: dict[str, VoiceConfig] = {}
    aliases: dict[str, str] = {}

    all_catalog_entries = list(STANDARD_CATALOG_TARGETS) + list(SOURCE_CATALOG_VOICINGS)
    for inst_id, voicing_id in all_catalog_entries:
        if inst_id not in instruments:
            continue
        inst = instruments[inst_id]
        if voicing_id not in inst.voicings:
            continue
        voicing = inst.voicings[voicing_id]
        slug = (
            voicing.tone_name.lower().replace(" ", "_").replace("∕", "_").replace("/", "_")
            if voicing.tone_name
            else voicing_id
        )
        vcfg = voicing_to_voice_config(inst, voicing, slug=slug)
        voices_dict[slug] = vcfg

        # Map canonical reference and voicing ID as aliases
        canonical_ref = f"{inst_id}:{voicing_id}"
        aliases[canonical_ref] = slug
        if voicing_id not in voices_dict and voicing_id not in aliases:
            aliases[voicing_id] = slug

    common_aliases = {
        "pj_pair_active": "pj_active",
        "pj_pair_open": "pj_passive",
        "p_active": "precision_active",
        "j_active": "jazz_pair_active",
    }
    for alias_key, target_key in common_aliases.items():
        if target_key in voices_dict and alias_key not in voices_dict:
            aliases[alias_key] = target_key

    registry = VoiceRegistry(voices_dict)
    registry.ALIASES.update(aliases)
    return registry


VOICES: VoiceRegistry = load_voices_config()
