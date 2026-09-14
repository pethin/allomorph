"""
Tests for pickup and target voice coil geometry and coordinate resolution.
"""

import math

from allomorph.circuit.schema import CircuitConfig
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
    # 1. 30" MMTW (dual and single coil)
    inst_30 = load_instrument("30in")
    coils_30_dual = resolve_pickup_coils(inst_30.pickups["mmtw_dual"], inst_30)
    assert len(coils_30_dual) == 2
    assert all("all" in c.strings for c in coils_30_dual)

    coils_30_single = resolve_pickup_coils(inst_30.pickups["mmtw_single"], inst_30)
    assert len(coils_30_single) == 1
    assert math.isclose(coils_30_single[0].position_from_bridge_m, 0.06607, abs_tol=1e-4)

    # 2. 34" P (split coils with E/A and D/G binding)
    inst_p = load_instrument("34in_standard_p")
    coils_p = resolve_pickup_coils(inst_p.pickups["split_p"], inst_p)
    assert len(coils_p) == 2
    assert coils_p[0].strings == [3, 4]
    assert coils_p[1].strings == [1, 2]
    assert math.isclose(coils_p[0].position_from_bridge_m, 0.1390, abs_tol=1e-4)
    assert math.isclose(coils_p[1].position_from_bridge_m, 0.1110, abs_tol=1e-4)

    # 3. 34" Jazz (composite pair_parallel)
    inst_j = load_instrument("34in_standard_jazz")
    coils_j = resolve_pickup_coils(inst_j.pickups["pair_parallel"], inst_j)
    assert len(coils_j) == 2
    assert math.isclose(coils_j[0].position_from_bridge_m, 0.1556, abs_tol=1e-4)
    assert math.isclose(coils_j[1].position_from_bridge_m, 0.0635, abs_tol=1e-4)
    assert coils_j[0].weight == 0.5
    assert coils_j[1].weight == 0.5


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
        topology="single",
        description="Test custom voice",
        fr=3000.0,
        Q=1.5,
        circuit=CircuitConfig(topology="single", L=3.0, Rdc=6000.0, Reddy=2000.0, Ccoil=50e-12),
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
