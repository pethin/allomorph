"""
Allomorph - Target Voice Configuration Registry (v0.3.0 / DSP Gen 3).
Constructs first-principles target VoiceConfig models dynamically from the
Unified Instrument Catalog in config/instruments/*.toml.
"""

from pathlib import Path
from typing import Any, ClassVar, override

from allomorph.circuit.schema import CircuitConfig
from allomorph.config.geometry import resolve_pickup_coils
from allomorph.config.schema import (
    InstrumentConfig,
    VoiceCoilConfig,
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
    pickup = instrument.pickups[voicing.pickup]

    voice_pickups: list[VoicePickupConfig] = []
    if pickup.components:
        for comp in pickup.components:
            if comp.pickup and comp.pickup in instrument.pickups:
                sub_p = instrument.pickups[comp.pickup]
                sub_coils = [
                    VoiceCoilConfig(
                        position_from_bridge_m=c.position_from_bridge_m,
                        aperture_width_in=c.aperture_width_in,
                        weight=c.weight,
                        polarity=c.polarity,
                        strings=c.strings,
                        pole_type=c.pole_type,
                    )
                    for c in resolve_pickup_coils(sub_p, instrument=instrument)
                ]
                voice_pickups.append(
                    VoicePickupConfig(
                        name=sub_p.name,
                        type=sub_p.type,
                        magnet_type=sub_p.magnet_type or pickup.magnet_type or "alnico_v",
                        alpha=getattr(sub_p, "alpha", None) or 0.25,
                        fr=sub_p.resonant_frequency_hz or 3500.0,
                        Q=sub_p.q_factor or 1.5,
                        weight=comp.weight,
                        polarity=comp.polarity,
                        coils=sub_coils,
                    )
                )

    if voice_pickups:
        coils = [
            VoiceCoilConfig(
                position_from_bridge_m=c.position_from_bridge_m,
                aperture_width_in=c.aperture_width_in,
                weight=c.weight * p.weight,
                polarity=c.polarity * p.polarity,
                strings=c.strings,
                pole_type=c.pole_type,
            )
            for p in voice_pickups
            for c in p.coils
        ]
    else:
        coils = [
            VoiceCoilConfig(
                position_from_bridge_m=c.position_from_bridge_m,
                aperture_width_in=c.aperture_width_in,
                weight=c.weight,
                polarity=c.polarity,
                strings=c.strings,
                pole_type=c.pole_type,
            )
            for c in resolve_pickup_coils(pickup, instrument=instrument)
        ]

    cdata: dict[str, Any]
    if voicing.circuit is not None:
        cdata = voicing.circuit.model_dump(exclude_unset=True)
    elif pickup.circuit is not None:
        cdata = pickup.circuit.model_dump(exclude_unset=True)
    else:
        cdata = {}
    if voicing.vol_pos is not None:
        cdata["vol_pos"] = voicing.vol_pos
    if voicing.tone_pos is not None:
        cdata["tone_pos"] = voicing.tone_pos
    if voicing.tone_cap_f is not None:
        cdata["Ctone"] = voicing.tone_cap_f
    if voicing.Rtone is not None:
        cdata["Rtone"] = voicing.Rtone
    if voicing.preamp_preset is not None:
        cdata["preamp"] = voicing.preamp_preset

    circuit = CircuitConfig.model_validate(cdata) if cdata else None

    fr = (
        voicing.resonant_frequency_hz
        or pickup.resonant_frequency_hz
        or (20000.0 if voicing.sensor_type == "direct" else 3500.0)
    )
    Q = voicing.q_factor or pickup.q_factor or (0.707 if voicing.sensor_type == "direct" else 1.5)
    magnet_type = voicing.magnet_type or pickup.magnet_type or "alnico_v"
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
    alpha = voicing.alpha or pickup.alpha or 0.25
    vsat = voicing.vsat or pickup.vsat or (pickup.circuit.vsat if pickup.circuit else None) or 1.0

    is_series = (
        voicing.affinity == "series"
        or "series" in voicing.pickup
        or "series" in target_slug
        or (circuit is not None and getattr(circuit, "topology", None) == "series")
    )
    is_parallel = (
        voicing.affinity == "parallel"
        or "parallel" in voicing.pickup
        or len(voice_pickups) >= 2
        or (circuit is not None and getattr(circuit, "topology", None) == "parallel")
    )
    blend_mode = "series" if is_series else ("parallel" if is_parallel else "single")

    return VoiceConfig(
        id=target_slug,
        name=voicing.tone_name or voicing.name,
        tone_name=voicing.tone_name,
        topology=f"{pickup.type.replace('_', ' ').title()} ({instrument.name})",
        description=f"{voicing.name} ({instrument.name})",
        blend_mode=blend_mode,
        magnet_type=magnet_type,
        alpha=alpha,
        vsat=vsat,
        fr=fr,
        Q=Q,
        gain_db=gain_db,
        scale=scale,
        target_string=target_string,
        preserve_aperture=voicing.preserve_aperture or (voicing.sensor_type == "direct"),
        sensor_type=voicing.sensor_type,
        coils=coils,
        pickups=voice_pickups if len(voice_pickups) >= 2 else [],
        circuit=circuit,
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
        STANDARD_CATALOG_TARGETS,
        load_all_instruments,
    )

    instruments = load_all_instruments(instruments_path or INSTRUMENTS_DIR)
    voices_dict: dict[str, VoiceConfig] = {}
    aliases: dict[str, str] = {}

    for inst_id, voicing_id in STANDARD_CATALOG_TARGETS:
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

    registry = VoiceRegistry(voices_dict)
    registry.ALIASES.update(aliases)
    return registry


VOICES: VoiceRegistry = load_voices_config()
