"""
Pickup coil geometry resolution, multi-coil component decomposition,
and magnetic pole piece geometry inference.
"""

from collections.abc import Sequence

from allomorph.config.schema import (
    CoilConfig,
    InstrumentConfig,
    PickupComponentConfig,
    PickupConfig,
    VoiceCoilConfig,
    VoiceConfig,
    VoicePickupConfig,
)


def _infer_pole_type(
    pickup_or_voice: PickupConfig | VoiceConfig | VoicePickupConfig | None = None,
    coil: CoilConfig | VoiceCoilConfig | PickupComponentConfig | None = None,
) -> str:
    """
    Infers magnetic pole geometry: 'rod' (cylindrical Alnico rod, Airy/Bessel spatial window)
    or 'blade' (continuous steel/ceramic bar blade, 1D rectangular slit).
    """
    if coil and coil.pole_type:
        return str(coil.pole_type).lower()
    if pickup_or_voice is not None:
        if isinstance(pickup_or_voice, PickupConfig) and pickup_or_voice.pole_type:
            return str(pickup_or_voice.pole_type).lower()
        mag = str(pickup_or_voice.magnet_type or "").lower()
        p_type = str(
            pickup_or_voice.sensor_type
            if isinstance(pickup_or_voice, VoiceConfig)
            else pickup_or_voice.type
        ).lower()
        p_name = str(pickup_or_voice.name).lower()
        if "blade" in p_name or "blade" in p_type or "bar" in p_name:
            return "blade"
        if "emg" in p_name or "emg" in p_type:
            return "blade"
        if "active" in mag:
            return "blade"
        if "alnico" in mag or "single_coil" in p_type or "split" in p_type:
            return "rod"
    return "rod"


def compute_coil_offset_from_spacing(
    center_m: float,
    coil_spacing_in: float,
    coil_index: int,
    num_coils: int = 2,
) -> float:
    """Computes the physical position of a coil within a multi-coil array centered at center_m.

    For a symmetric array, the center of gravity satisfies:
    sum(compute_coil_offset_from_spacing(center, d, i, N) for i in range(N)) / N == center_m
    """
    d_m = coil_spacing_in * 0.0254
    offset = (coil_index - (num_coils - 1) / 2.0) * d_m
    return center_m + offset


def assign_string_registers_to_coils(
    coils: Sequence[CoilConfig],
    pole_type: str = "rod",
    coil_geometry: str = "parallel",
) -> list[CoilConfig]:
    """Assigns string register coverage across a list of coils, ensuring full coverage without gaps."""
    updated: list[CoilConfig] = []
    n = len(coils)
    for i, c in enumerate(coils):
        c_copy = c.model_copy(deep=True)
        if not c_copy.pole_type:
            c_copy.pole_type = pole_type
        if not c_copy.strings or c_copy.strings == ["all"]:
            if coil_geometry == "split" and n == 2:
                c_copy.strings = [3, 4] if i == 0 else [1, 2]
            else:
                c_copy.strings = ["all"]
        updated.append(c_copy)
    return updated


def resolve_pickup_coils(
    pickup: PickupConfig, instrument: InstrumentConfig | None = None
) -> list[CoilConfig]:
    """
    Resolves an instrument pickup configuration into a canonical list of CoilConfig models.
    Handles:
      - Composite pickups ('components' referencing other pickups with weights)
      - Explicit coil arrays ('coils' with per-string bindings)
      - Dual-coil humbuckers with coil spacing 'd'
      - Single-coil / split-coil fallbacks
    """
    # 1. Composite blend / sum
    comp_list = getattr(pickup, "components", None) or []
    if pickup.type == "composite" or bool(comp_list):
        resolved: list[CoilConfig] = []
        pickups_map: dict[str, PickupConfig] = instrument.pickups if instrument else {}
        for comp in comp_list:
            p_ref = comp.pickup
            c_weight = float(comp.weight)
            c_pol = float(comp.polarity)
            if p_ref:
                if p_ref not in pickups_map:
                    inst_id = instrument.id if instrument else "unknown"
                    raise KeyError(
                        f"Composite pickup references non-existent pickup '{p_ref}' in instrument "
                        f"'{inst_id}'. Available pickups: {list(pickups_map.keys())}"
                    )
                sub_coils = resolve_pickup_coils(pickups_map[p_ref], instrument)
                for sc in sub_coils:
                    sc_copy = sc.model_copy(deep=True)
                    sc_copy.weight = sc.weight * c_weight
                    sc_copy.polarity = sc.polarity * c_pol
                    if not sc_copy.pole_type:
                        sc_copy.pole_type = _infer_pole_type(pickups_map[p_ref], sc)
                    resolved.append(sc_copy)
            elif comp.position_from_bridge_m is not None:
                resolved.append(
                    CoilConfig(
                        position_from_bridge_m=float(comp.position_from_bridge_m),
                        aperture_width_in=float(comp.aperture_width_in),
                        weight=c_weight,
                        polarity=c_pol,
                        strings=list(comp.strings),
                        pole_type=_infer_pole_type(pickup, comp),
                    )
                )
            else:
                inst_id = instrument.id if instrument else "unknown"
                raise ValueError(
                    f"Composite pickup component in '{inst_id}' "
                    "must specify either 'pickup' or 'position_from_bridge_m'."
                )
        if resolved:
            return resolved

    # 2. Explicit coils list
    if pickup.coils:
        return [c.model_copy(deep=True) for c in pickup.coils]

    # 3. Dual-coil humbucker via coil_spacing_in
    pos_m = float(pickup.position_from_bridge_m or 0.08)
    w_in = float(pickup.aperture_width_in)
    d_in = float(pickup.coil_spacing_in)
    p_pole = _infer_pole_type(pickup)
    default_w = 0.45 if p_pole == "rod" else 0.35
    eff_w = w_in / 2.0 if (w_in > 0.0 and w_in != 0.75) else default_w

    if d_in > 0:
        pos_neck = compute_coil_offset_from_spacing(pos_m, d_in, coil_index=0, num_coils=2)
        pos_bridge = compute_coil_offset_from_spacing(pos_m, d_in, coil_index=1, num_coils=2)
        return [
            CoilConfig(
                id="neck",
                position_from_bridge_m=pos_neck,
                aperture_width_in=eff_w,
                weight=0.5,
                polarity=1.0,
                strings=["all"],
                pole_type=p_pole,
            ),
            CoilConfig(
                id="bridge",
                position_from_bridge_m=pos_bridge,
                aperture_width_in=eff_w,
                weight=0.5,
                polarity=1.0,
                strings=["all"],
                pole_type=p_pole,
            ),
        ]

    # 4. Standard single coil
    single_w = w_in if (w_in > 0.0 and w_in != 0.75) else default_w
    return [
        CoilConfig(
            id="single",
            position_from_bridge_m=pos_m,
            aperture_width_in=single_w,
            weight=1.0,
            polarity=1.0,
            strings=["all"],
            pole_type=p_pole,
        )
    ]


def resolve_voice_pickups(voice_cfg: VoiceConfig) -> list[VoicePickupConfig]:
    """
    Resolves a target voice configuration into a canonical list of VoicePickupConfig models.
    Handles multi-pickup voices and single-pickup fallbacks.
    """
    if voice_cfg.pickups:
        return voice_cfg.pickups

    # Fallback for single-pickup voices: wrap top-level voice coils/fr/Q
    top_coils = resolve_voice_coils(voice_cfg, _from_pickups=False)
    return [
        VoicePickupConfig(
            name=voice_cfg.name,
            type=voice_cfg.sensor_type,
            magnet_type=voice_cfg.magnet_type,
            fr=voice_cfg.fr,
            Q=voice_cfg.Q,
            weight=1.0,
            polarity=1.0,
            coils=top_coils,
        )
    ]


def resolve_voice_coils(
    voice_cfg: VoiceConfig, _from_pickups: bool = True
) -> list[VoiceCoilConfig]:
    """Resolves target voice configuration into a canonical list of VoiceCoilConfig models."""

    if _from_pickups and voice_cfg.pickups:
        all_coils = []
        for p in voice_cfg.pickups:
            p_weight = float(p.weight)
            p_pol = float(p.polarity)
            for c in p.coils:
                all_coils.append(
                    VoiceCoilConfig(
                        position_from_bridge_m=c.position_from_bridge_m,
                        aperture_width_in=c.aperture_width_in,
                        weight=c.weight * p_weight,
                        polarity=c.polarity * p_pol,
                        strings=list(c.strings),
                        pole_type=c.pole_type or "rod",
                    )
                )
        if all_coils:
            return all_coils

    if voice_cfg.coils:
        return [c.model_copy(deep=True) for c in voice_cfg.coils]

    return [
        VoiceCoilConfig(
            position_from_bridge_m=0.088,
            aperture_width_in=0.75,
            weight=1.0,
            polarity=1.0,
            strings=["all"],
            pole_type=_infer_pole_type(voice_cfg),
        )
    ]


def compute_effective_position(coils: Sequence[CoilConfig | VoiceCoilConfig]) -> float:
    """Computes weighted average physical position from bridge in meters."""
    if not coils:
        return 0.08
    total_w = sum(float(c.weight) for c in coils)
    if total_w == 0:
        c0 = coils[0]
        return float(c0.position_from_bridge_m)
    return sum(float(c.position_from_bridge_m) * float(c.weight) for c in coils) / total_w
