"""Deterministic resolvers and decomposition primitives for instrument voicings and pickups.

Provides pure, testable units for pickup resonance inference, coil resolution
and compounding, voicing token parsing, slug matching, and bundle assignment.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

from allomorph.config.geometry import resolve_pickup_coils
from allomorph.config.schema import (
    InstrumentConfig,
    PackBundleConfig,
    PickupConfig,
    VoiceCoilConfig,
    VoicePickupConfig,
    VoicingConfig,
)

if TYPE_CHECKING:
    from allomorph.config.instruments import VoicingBundle


def infer_pickup_resonance_and_q(
    pickup: PickupConfig,
    pickup_key: str = "",
    harness_str: str = "",
) -> tuple[float, float]:
    """Infers resonant frequency (Hz) and Q factor for a pickup based on configuration and position.

    Explicit pickup values take priority. Otherwise, values are inferred from pickup type,
    name/key (e.g. bridge vs neck), and active/passive harness characteristics.
    """
    is_passive = "passive" in (harness_str or "").lower()
    is_active = (
        "active" in (harness_str or "").lower()
        or getattr(pickup, "has_internal_buffer", False)
        or getattr(pickup, "type", "") == "active"
    )
    p_name_lower = (pickup.name or "").lower()
    p_key_lower = pickup_key.lower()

    # Resonant frequency fr
    if pickup.resonant_frequency_hz is not None:
        fr = float(pickup.resonant_frequency_hz)
    elif pickup.type == "split_coil":
        fr = 2800.0 if is_passive else 4800.0
    elif "bridge" in p_key_lower or "bridge" in p_name_lower:
        fr = 3200.0 if is_passive else 4600.0
    elif "neck" in p_key_lower or "neck" in p_name_lower:
        fr = 3600.0 if is_passive else 5200.0
    else:
        fr = 5200.0 if is_active else 3500.0

    # Q factor
    if pickup.q_factor is not None:
        q = float(pickup.q_factor)
    elif "bridge" in p_key_lower or "bridge" in p_name_lower:
        q = 1.6 if is_passive else 1.8
    elif pickup.type == "split_coil":
        q = 1.4 if is_passive else 1.7
    else:
        q = 1.5 if is_passive else 1.7

    return fr, q


def build_voice_coils_for_pickup(
    pickup: PickupConfig,
    active_coil_ids: Sequence[str] | None = None,
    instrument: InstrumentConfig | None = None,
) -> list[VoiceCoilConfig]:
    """Resolves individual physical coils for a pickup filtered by active coil IDs."""
    raw_coils = resolve_pickup_coils(pickup, instrument=instrument)
    if active_coil_ids:
        raw_coils = [c for c in raw_coils if c.id in active_coil_ids or not c.id]

    return [
        VoiceCoilConfig(
            position_from_bridge_m=c.position_from_bridge_m,
            aperture_width_in=c.aperture_width_in,
            weight=c.weight,
            polarity=c.polarity,
            strings=c.strings,
            pole_type=c.pole_type,
        )
        for c in raw_coils
    ]


def compound_multi_pickup_coils(
    voice_pickups: Sequence[VoicePickupConfig],
) -> list[VoiceCoilConfig]:
    """Compounds coils from multiple voice pickups by scaling weight and polarity by parent pickup."""
    return [
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


def parse_target_voicing_token(
    target_token: str,
    default_inst_id: str,
) -> tuple[str, str]:
    """Parses a target voicing token into (instrument_id, voicing_id).

    Supports canonical 'instrument_id:voicing_id', universal catalog voice slugs,
    or local voicing IDs defaulting to default_inst_id.
    """
    if ":" in target_token:
        tgt_inst_id, tgt_vid = target_token.split(":", 1)
        return tgt_inst_id, tgt_vid

    from allomorph.config.voices import VOICES

    if target_token in VOICES:
        v_obj = VOICES[target_token]
        tgt_inst_id = v_obj.instrument_id or default_inst_id
        tgt_vid = v_obj.id
        return tgt_inst_id, tgt_vid

    return default_inst_id, target_token


def find_voicing_by_slug_or_id(
    instrument: InstrumentConfig,
    voicing_id_or_slug: str,
) -> VoicingConfig | None:
    """Finds a voicing on an instrument by direct ID or normalized tone name slug."""
    if voicing_id_or_slug in instrument.voicings:
        return instrument.voicings[voicing_id_or_slug]

    clean_target = voicing_id_or_slug.lower().replace(" ", "_").replace("∕", "_").replace("/", "_")
    for v in instrument.voicings.values():
        if v.id == voicing_id_or_slug or v.id == clean_target:
            return v
        if v.tone_name:
            slug = v.tone_name.lower().replace(" ", "_").replace("∕", "_").replace("/", "_")
            if slug == clean_target or slug == voicing_id_or_slug:
                return v

    return None


def assign_target_to_bundle(
    target_vid: str,
    pack_bundles: Mapping[str, VoicingBundle],
    bundle_configs: Sequence[PackBundleConfig],
) -> str:
    """Assigns a target voicing ID to an appropriate pack bundle name."""
    # 1. Check explicit bundle targets in config
    for b_cfg in bundle_configs:
        if target_vid in b_cfg.targets or any(target_vid in t for t in b_cfg.targets):
            return b_cfg.name

    # 2. Check position keyword match if multiple bundles
    if len(pack_bundles) > 1:
        for bn in pack_bundles:
            if bn in target_vid.lower():
                return bn

    # 3. Fallback to first available bundle
    return next(iter(pack_bundles.keys()))
