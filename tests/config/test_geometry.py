"""
Tests for pickup coil geometry, electroacoustic decomposition, and coordinate resolution.
"""

import math

from allomorph.config import (
    compute_effective_position,
    load_instrument,
    resolve_pickup_coils,
    resolve_voicing_electroacoustics,
)
from allomorph.config.geometry import _infer_pole_type
from allomorph.config.schema import CoilConfig, PickupConfig


def test_resolve_pickup_coils():
    # 1. 30" MMTW (physical mmtwx dual-coil transducer)
    inst_30 = load_instrument("30in_emg_mmtw")
    coils_30 = resolve_pickup_coils(inst_30.pickups["mmtwx"], inst_30)
    assert len(coils_30) == 2
    assert all("all" in c.strings for c in coils_30)
    assert math.isclose(coils_30[0].position_from_bridge_m, 0.08893, abs_tol=1e-4)
    assert math.isclose(coils_30[1].position_from_bridge_m, 0.06607, abs_tol=1e-4)

    # 2. 34" P (split coils with E/A and D/G binding)
    inst_p = load_instrument("34in_standard_p")
    coils_p = resolve_pickup_coils(inst_p.pickups["split_p"], inst_p)
    assert len(coils_p) == 2
    assert coils_p[0].strings == [3, 4]
    assert coils_p[1].strings == [1, 2]
    assert math.isclose(coils_p[0].position_from_bridge_m, 0.1390, abs_tol=1e-4)
    assert math.isclose(coils_p[1].position_from_bridge_m, 0.1110, abs_tol=1e-4)

    # 3. 34" Jazz (physical neck and bridge pickups)
    inst_j = load_instrument("34in_standard_jazz")
    coils_neck = resolve_pickup_coils(inst_j.pickups["neck"], inst_j)
    coils_bridge = resolve_pickup_coils(inst_j.pickups["bridge"], inst_j)
    assert len(coils_neck) == 1
    assert math.isclose(coils_neck[0].position_from_bridge_m, 0.1556, abs_tol=1e-4)
    assert len(coils_bridge) == 1
    assert math.isclose(coils_bridge[0].position_from_bridge_m, 0.0635, abs_tol=1e-4)


def test_resolve_voicing_electroacoustics_pickups_and_coils():
    # 1. Multi-pickup: 34in_standard_jazz:pair_active
    inst_j = load_instrument("34in_standard_jazz")
    ea_act = resolve_voicing_electroacoustics(inst_j, inst_j.voicings["pair_active"])
    assert len(ea_act.coils) == 2
    assert ea_act.coils[0].position_from_bridge_m == 0.1556
    assert ea_act.coils[1].position_from_bridge_m == 0.0635
    assert ea_act.coils[0].strings == ["all"]
    eff01 = compute_effective_position(ea_act.coils)
    assert 0.10 < eff01 < 0.12  # Average of 0.1556 and 0.0635 is 0.10955

    # 2. Multi-pickup: 34in_standard_jazz:pair_open
    ea_jazz = resolve_voicing_electroacoustics(inst_j, inst_j.voicings["pair_open"])
    assert len(ea_jazz.coils) == 2

    # 3. Multi-pickup: 34in_standard_pj:pair_active (Ultra II 3-band active P/J)
    inst_pj = load_instrument("34in_standard_pj")
    ea_pj_act = resolve_voicing_electroacoustics(inst_pj, inst_pj.voicings["pair_active"])
    assert len(ea_pj_act.coils) == 3
    assert ea_pj_act.coils[0].strings == [3, 4]
    assert ea_pj_act.coils[1].strings == [1, 2]
    assert ea_pj_act.coils[2].strings == ["all"]

    # 4. Single-pickup: 34in_standard_pj:neck_active
    ea_p_act = resolve_voicing_electroacoustics(inst_pj, inst_pj.voicings["neck_active"])
    assert len(ea_p_act.coils) == 2
    assert ea_p_act.coils[0].strings == [3, 4]
    assert ea_p_act.coils[1].strings == [1, 2]

    # 5. Multi-pickup: 34in_active_pmm:pmm_series & pmm_parallel
    inst_pmm = load_instrument("34in_active_pmm")
    ea_pmm_ser = resolve_voicing_electroacoustics(inst_pmm, inst_pmm.voicings["pmm_series"])
    assert len(ea_pmm_ser.coils) == 4
    assert ea_pmm_ser.coils[0].strings == [3, 4]
    assert ea_pmm_ser.coils[1].strings == [1, 2]
    assert ea_pmm_ser.coils[2].strings == ["all"]
    assert ea_pmm_ser.coils[3].strings == ["all"]


def test_infer_pole_type_and_geometry_branches():
    import pytest
    from pydantic import ValidationError

    # 1. From coil pole_type
    c_rod = CoilConfig(position_from_bridge_m=0.08, pole_type="rod")
    assert _infer_pole_type(coil=c_rod) == "rod"
    c_blade = CoilConfig(position_from_bridge_m=0.08, pole_type="blade")
    assert _infer_pole_type(coil=c_blade) == "blade"

    # 2. From pickup pole_type when coil.pole_type is None
    p_blade = PickupConfig(name="Blade Pickup", type="single", pole_type="blade")
    assert _infer_pole_type(p_blade) == "blade"

    # 3. Default to "rod" when neither coil nor pickup specifies pole_type (no name substring heuristics)
    p_default = PickupConfig(name="Bar Magnet EMG-40HZ", type="humbucker")
    assert _infer_pole_type(p_default) == "rod"

    # 4. Invalid pole_type or magnet_type is rejected fail-fast by Pydantic Literal validation
    with pytest.raises(ValidationError):
        PickupConfig.model_validate({"name": "Bad Pole", "pole_type": "hex_screw"})
    with pytest.raises(ValidationError):
        PickupConfig.model_validate({"name": "Bad Magnet", "magnet_type": "piezo"})


def test_resolve_pickup_coils_explicit_and_single_fallback():
    # 1. Explicit coils list on PickupConfig
    explicit = PickupConfig(
        name="Explicit Coils",
        type="custom",
        coils=[
            CoilConfig(
                position_from_bridge_m=0.12,
                aperture_width_in=0.6,
                weight=1.0,
                polarity=1.0,
                strings=["all"],
            )
        ],
    )
    res_exp = resolve_pickup_coils(explicit)
    assert len(res_exp) == 1
    assert res_exp[0].position_from_bridge_m == 0.12

    # 2. Standard single coil fallback
    sc = PickupConfig(
        name="Single Coil",
        type="single",
        position_from_bridge_m=0.08,
        aperture_width_in=0.75,
    )
    sc_coils = resolve_pickup_coils(sc)
    assert len(sc_coils) == 1
    assert sc_coils[0].position_from_bridge_m == 0.08


def test_compute_effective_position_edge_cases():
    # 1. Empty coils list returns default 0.08m
    assert compute_effective_position([]) == 0.08

    # 2. Total weight sum == 0 returns position of first coil
    c0 = CoilConfig(position_from_bridge_m=0.15, weight=1.0)
    c1 = CoilConfig(position_from_bridge_m=0.05, weight=1.0)
    object.__setattr__(c0, "weight", 0.0)
    object.__setattr__(c1, "weight", 0.0)
    assert compute_effective_position([c0, c1]) == 0.15
