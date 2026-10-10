"""
Tests for pickup and target voice coil geometry and coordinate resolution.
"""

import math

import pytest

from allomorph.config import (
    VOICES,
    compute_effective_position,
    load_instrument,
    resolve_pickup_coils,
    resolve_voice_coils,
    resolve_voice_pickups,
)
from allomorph.config.schema import VoiceCoilConfig, VoiceConfig


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


def test_resolve_voice_pickups():
    # 1. Multi-pickup: jazz_pair_active
    p_act = resolve_voice_pickups(VOICES["jazz_pair_active"])
    assert len(p_act) == 2
    assert p_act[0].fr == 5200.0
    assert p_act[0].Q == 1.7
    assert p_act[1].fr == 4600.0
    assert p_act[1].Q == 1.8

    # 2. Multi-pickup: jazz_pair_open
    p_jazz = resolve_voice_pickups(VOICES["jazz_pair_open"])
    assert len(p_jazz) == 2
    assert p_jazz[0].fr == 3600.0
    assert p_jazz[0].Q == 1.5
    assert p_jazz[1].fr == 3200.0
    assert p_jazz[1].Q == 1.6

    # 3. Multi-pickup: pj_active
    p_pj_act = resolve_voice_pickups(VOICES["pj_active"])
    assert len(p_pj_act) == 2
    assert p_pj_act[0].fr == 4800.0
    assert p_pj_act[0].Q == 1.7
    assert p_pj_act[0].weight == 0.5
    assert len(p_pj_act[0].coils) == 2  # P-split E/A + D/G
    assert p_pj_act[1].fr == 4600.0
    assert p_pj_act[1].Q == 1.8
    assert p_pj_act[1].weight == 0.5
    assert len(p_pj_act[1].coils) == 1  # J-bridge

    # 4. Multi-pickup: pj_passive
    p_pj_pas = resolve_voice_pickups(VOICES["pj_passive"])
    assert len(p_pj_pas) == 2
    assert p_pj_pas[0].fr == 2800.0
    assert p_pj_pas[0].Q == 1.4
    assert p_pj_pas[1].fr == 3200.0
    assert p_pj_pas[1].Q == 1.6

    # 5. Multi-pickup: p_mm_series & p_mm_parallel
    p_pmm = resolve_voice_pickups(VOICES["p_mm_series"])
    assert len(p_pmm) == 2
    assert p_pmm[0].fr == 3200.0
    assert p_pmm[1].fr == 3500.0

    p_pmm_act = resolve_voice_pickups(VOICES["p_mm_parallel"])
    assert len(p_pmm_act) == 2
    assert p_pmm_act[0].fr == 3200.0
    assert p_pmm_act[1].fr == 3500.0

    # 6. Single-pickup voice auto-wrapping: precision_active
    p_p = resolve_voice_pickups(VOICES["precision_active"])
    assert len(p_p) == 1
    assert p_p[0].fr == 4800.0
    assert p_p[0].Q == 1.70
    assert p_p[0].weight == 1.0
    assert len(p_p[0].coils) == 2


def test_resolve_voice_coils():
    # 01 Modern active jazz should have 2 coils, each with strings=["all"]
    c01 = resolve_voice_coils(VOICES["jazz_pair_active"])
    assert len(c01) == 2
    assert c01[0].position_from_bridge_m == 0.1556
    assert c01[1].position_from_bridge_m == 0.0635
    assert c01[0].strings == ["all"]

    # 04 Modern active split P ceramic should have 2 split coils with specific string bindings
    c04 = resolve_voice_coils(VOICES["precision_active"])
    assert len(c04) == 2
    assert c04[0].strings == [3, 4]
    assert c04[0].position_from_bridge_m == 0.1390
    assert c04[1].strings == [1, 2]
    assert c04[1].position_from_bridge_m == 0.1110

    # 07 Modern PJ active should have 3 coils (split P + J bridge)
    c07 = resolve_voice_coils(VOICES["pj_active"])
    assert len(c07) == 3
    assert c07[0].strings == [3, 4]
    assert c07[1].strings == [1, 2]
    assert c07[2].strings == ["all"]

    # 11 P/MM hybrid should have 4 coils (split P + MM humbucker pair)
    c11 = resolve_voice_coils(VOICES["p_mm_series"])
    assert len(c11) == 4
    assert c11[0].strings == [3, 4]
    assert c11[1].strings == [1, 2]
    assert c11[2].strings == ["all"]
    assert c11[3].strings == ["all"]

    c11_act = resolve_voice_coils(VOICES["p_mm_parallel"])
    assert len(c11_act) == 4
    assert c11_act[0].strings == [3, 4]
    assert c11_act[1].strings == [1, 2]
    assert c11_act[2].strings == ["all"]
    assert c11_act[3].strings == ["all"]

    # Check effective positions are calculated correctly
    eff01 = compute_effective_position(c01)
    assert 0.10 < eff01 < 0.12  # Average of 0.1556 and 0.0635 is 0.10955

    # Custom voice coil resolution test
    custom_voice = VoiceConfig(
        id="custom_voice",
        name="Custom Voice",
        description="Test custom voice",
        fr=3000.0,
        Q=1.5,
        coils=[
            VoiceCoilConfig(
                position_from_bridge_m=0.066,
                aperture_width_in=0.75,
                weight=1.0,
                polarity=1.0,
                strings=["all"],
            )
        ],
    )
    c_custom = resolve_voice_coils(custom_voice)
    assert len(c_custom) == 1
    assert c_custom[0].strings == ["all"]


def test_resolve_pickup_coils_strict_errors():
    """Verify that composite pickups with invalid references raise KeyError/ValueError."""
    import pytest

    from allomorph.config.schema import PickupComponentConfig, PickupConfig

    inst = load_instrument("34in_standard_p")

    # 1. References non-existent pickup
    bad_composite = PickupConfig(
        name="Bad Composite",
        type="composite",
        components=[PickupComponentConfig(pickup="non_existent_pickup")],
    )
    with pytest.raises(KeyError, match="references non-existent pickup 'non_existent_pickup'"):
        resolve_pickup_coils(bad_composite, inst)

    # 2. Component missing both 'pickup' and 'position_from_bridge_m'
    empty_component = PickupConfig(
        name="Empty Component",
        type="composite",
        components=[PickupComponentConfig(weight=1.0)],
    )
    with pytest.raises(
        ValueError, match="must specify either 'pickup' or 'position_from_bridge_m'"
    ):
        resolve_pickup_coils(empty_component, inst)


def test_infer_pole_type_and_geometry_branches():
    from allomorph.config.geometry import _infer_pole_type
    from allomorph.config.schema import CoilConfig, PickupConfig, VoiceConfig

    # 1. From coil pole_type
    c_rod = CoilConfig(position_from_bridge_m=0.08, pole_type="rod")
    assert _infer_pole_type(coil=c_rod) == "rod"

    # 2. From pickup pole_type
    p_blade = PickupConfig(name="Blade Pickup", type="single", pole_type="blade")
    assert _infer_pole_type(p_blade) == "blade"

    # 3. From name/type containing 'blade' or 'bar'
    p_bar = PickupConfig(name="Bar Magnet Pickup", type="single")
    assert _infer_pole_type(p_bar) == "blade"

    # 4. From EMG in name or type
    p_emg = PickupConfig(name="EMG-40HZ", type="humbucker")
    assert _infer_pole_type(p_emg) == "blade"

    # 5. From active in magnet_type
    p_act = PickupConfig(name="Active Sensor", type="single", magnet_type="active")
    assert _infer_pole_type(p_act) == "blade"

    # 6. From alnico or single_coil
    p_aln = PickupConfig(name="Vintage Single", type="single_coil", magnet_type="alnico_v")
    assert _infer_pole_type(p_aln) == "rod"

    # 7. VoiceConfig inspection
    v_blade = VoiceConfig(
        id="v_b",
        name="Blade Voice",
        description="Test blade voice",
        fr=3000.0,
        Q=1.5,
    )
    assert _infer_pole_type(v_blade) == "blade"


def test_resolve_pickup_coils_composite_direct_and_humbuckers():
    from allomorph.config.schema import CoilConfig, PickupComponentConfig, PickupConfig

    # 1. Composite with explicit position_from_bridge_m component
    comp_direct = PickupConfig(
        name="Direct Composite",
        type="composite",
        components=[
            PickupComponentConfig(
                position_from_bridge_m=0.10,
                aperture_width_in=0.75,
                weight=1.0,
                polarity=1.0,
                strings=["all"],
            )
        ],
    )
    coils = resolve_pickup_coils(comp_direct)
    assert len(coils) == 1
    assert coils[0].position_from_bridge_m == 0.10

    # 2. Explicit coils list on PickupConfig
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

    # 3. Dual coil humbucker via coil_spacing_in > 0
    hb = PickupConfig(
        name="Humbucker",
        type="humbucker",
        position_from_bridge_m=0.08,
        aperture_width_in=1.5,
        coil_spacing_in=0.75,
    )
    hb_coils = resolve_pickup_coils(hb)
    assert len(hb_coils) == 2
    d_m = 0.75 * 0.0254
    assert hb_coils[0].position_from_bridge_m == pytest.approx(0.08 - d_m / 2.0)
    assert hb_coils[1].position_from_bridge_m == pytest.approx(0.08 + d_m / 2.0)

    # 4. Standard single coil fallback
    sc = PickupConfig(
        name="Single Coil",
        type="single",
        position_from_bridge_m=0.08,
        aperture_width_in=0.75,
        coil_spacing_in=0.0,
    )
    sc_coils = resolve_pickup_coils(sc)
    assert len(sc_coils) == 1
    assert sc_coils[0].position_from_bridge_m == 0.08


def test_resolve_voice_coils_empty_fallback():
    from allomorph.config.schema import VoiceConfig

    # Voice with no pickups and no coils falls back to default 0.088m coil
    v_empty = VoiceConfig(
        id="empty_voice",
        name="Empty Voice",
        description="Test empty voice",
        fr=3000.0,
        Q=1.5,
    )
    coils = resolve_voice_coils(v_empty)
    assert len(coils) == 1
    assert coils[0].position_from_bridge_m == 0.088


def test_compute_effective_position_edge_cases():
    from allomorph.config.schema import CoilConfig

    # 1. Empty coils list returns default 0.08m
    assert compute_effective_position([]) == 0.08

    # 2. Total weight sum == 0 returns position of first coil
    c0 = CoilConfig(position_from_bridge_m=0.15, weight=1.0)
    c1 = CoilConfig(position_from_bridge_m=0.05, weight=1.0)
    object.__setattr__(c0, "weight", 0.0)
    object.__setattr__(c1, "weight", 0.0)
    assert compute_effective_position([c0, c1]) == 0.15
