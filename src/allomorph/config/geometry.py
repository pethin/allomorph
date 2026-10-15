"""
Pickup coil geometry resolution, multi-coil component decomposition,
and magnetic pole piece geometry inference.
"""

from collections.abc import Sequence

from allomorph.config.schema import (
    CoilConfig,
    InstrumentConfig,
    PickupConfig,
)


def _infer_pole_type(
    pickup: PickupConfig | None = None,
    coil: CoilConfig | None = None,
) -> str:
    """
    Infers magnetic pole geometry: 'rod' (cylindrical Alnico rod, Airy/Bessel spatial window)
    or 'blade' (continuous steel/ceramic bar blade, 1D rectangular slit).
    """
    if coil and coil.pole_type:
        return str(coil.pole_type).lower()
    if pickup is not None:
        if pickup.pole_type:
            return str(pickup.pole_type).lower()
        mag = str(pickup.magnet_type or "").lower()
        p_type = str(pickup.type).lower()
        p_name = str(pickup.name).lower()
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
      - Explicit coil arrays ('coils' with per-string bindings)
      - Dual-coil humbuckers with coil spacing 'd'
      - Single-coil / split-coil fallbacks
    """
    _ = instrument

    # 1. Explicit coils list
    if pickup.coils:
        return [c.model_copy(deep=True) for c in pickup.coils]

    # 2. Dual-coil humbucker via coil_spacing_in
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

    # 3. Standard single coil
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


def compute_effective_position(coils: Sequence[CoilConfig]) -> float:
    """Computes weighted average physical position from bridge in meters."""
    if not coils:
        return 0.08
    total_w = sum(float(c.weight) for c in coils)
    if total_w == 0:
        c0 = coils[0]
        return float(c0.position_from_bridge_m)
    return sum(float(c.position_from_bridge_m) * float(c.weight) for c in coils) / total_w
