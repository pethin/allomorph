"""
Tests for source instrument pickup switch position routing and voice mapping.
"""

import math

from allomorph.circuit.schema import CircuitConfig
from allomorph.config import (
    VOICES,
    get_source_pickup,
    load_instrument,
)


def test_30in_mmtw_routing():
    inst = load_instrument("30in")
    assert inst.id == "30in_emg_mmtw"
    assert "mmtw_dual" in inst.pickups
    assert "mmtw_single" in inst.pickups

    # 60s Jazz Bridge should route to single-coil mode
    j_pickup = get_source_pickup(inst, "jazz_bridge_open")
    assert j_pickup.name == "EMG MMTW Single-Coil (Bridge Coil)"
    assert j_pickup.position_from_bridge_m is not None
    assert math.isclose(j_pickup.position_from_bridge_m, 0.06607, abs_tol=1e-4)

    # StingRay should route to dual-coil mode
    mm_pickup = get_source_pickup(inst, "stingray_parallel")
    assert mm_pickup.name == "EMG MMTW Dual-Coil (Centerline)"
    assert mm_pickup.position_from_bridge_m is not None
    assert math.isclose(mm_pickup.position_from_bridge_m, 0.0775, abs_tol=1e-4)
    assert mm_pickup.coil_spacing_in is not None
    assert math.isclose(mm_pickup.coil_spacing_in, 0.90, abs_tol=1e-4)


def test_30in_mm_legacy_routing():
    inst = load_instrument("30in_emg_mm")
    assert inst.id == "30in_emg_mmtw"
    pickup = get_source_pickup(inst, "stingray_parallel")
    assert pickup.name == "EMG MMTW Dual-Coil (Centerline)"
    assert pickup.position_from_bridge_m is not None
    assert math.isclose(pickup.position_from_bridge_m, 0.0775, abs_tol=1e-4)
    assert pickup.coil_spacing_in is not None
    assert math.isclose(pickup.coil_spacing_in, 0.90, abs_tol=1e-4)


def test_32in_custom_pmm_routing():
    inst = load_instrument("32in")
    assert inst.id == "32in_custom_pmm"

    # P-Bass voices route to neck PX split-coil
    p_pickup = get_source_pickup(inst, "precision_active")
    assert p_pickup.name == "Reverse EMG PX Split-Coil (Neck)"
    assert p_pickup.position_from_bridge_m is not None
    assert math.isclose(p_pickup.position_from_bridge_m, 0.1228, abs_tol=1e-4)

    # 60s Jazz Bridge routes to bridge single coil
    j_pickup = get_source_pickup(inst, "jazz_bridge_open")
    assert j_pickup.name == "EMG MMTWX Single-Coil (Bridge)"
    assert j_pickup.position_from_bridge_m is not None
    assert math.isclose(j_pickup.position_from_bridge_m, 0.0508, abs_tol=1e-4)

    # StingRay MM routes to dual coil centerline
    mm_pickup = get_source_pickup(inst, "stingray_parallel")
    assert mm_pickup.name == "EMG MMTWX Dual-Coil (Centerline)"
    assert mm_pickup.position_from_bridge_m is not None
    assert math.isclose(mm_pickup.position_from_bridge_m, 0.0622, abs_tol=1e-4)

    # P/J hybrid routes to parallel P/J pair (Reverse PX + MMTWX single-coil)
    pj_pickup = get_source_pickup(inst, "pj_active")
    assert pj_pickup.name == "EMG PX + MMTWX Single Parallel (P/J Mode)"
    assert pj_pickup.type == "composite"
    assert pj_pickup.components is not None
    assert len(pj_pickup.components) == 2
    assert pj_pickup.components[0].pickup == "px"
    assert pj_pickup.components[1].pickup == "mmtwx_single"
    assert pj_pickup.position_from_bridge_m is not None
    assert math.isclose(pj_pickup.position_from_bridge_m, 0.0868, abs_tol=1e-4)

    # P/MM series & active voices route to physical parallel center detent blend
    pmm_pickup = get_source_pickup(inst, "p_mm_series")
    assert pmm_pickup.name == "EMG PX + MMTWX Parallel (P/MM Mode)"
    assert pmm_pickup.position_from_bridge_m is not None
    assert math.isclose(pmm_pickup.position_from_bridge_m, 0.0868, abs_tol=1e-4)

    pmm_act_pickup = get_source_pickup(inst, "p_mm_parallel")
    assert pmm_act_pickup.name == "EMG PX + MMTWX Parallel (P/MM Mode)"
    assert pmm_act_pickup.position_from_bridge_m is not None
    assert math.isclose(pmm_act_pickup.position_from_bridge_m, 0.0868, abs_tol=1e-4)

    # Mudbucker routes to neck PX
    mud_pickup = get_source_pickup(inst, "mudbucker_deep")
    assert mud_pickup.name == "Reverse EMG PX Split-Coil (Neck)"
    assert mud_pickup.position_from_bridge_m is not None
    assert math.isclose(mud_pickup.position_from_bridge_m, 0.1228, abs_tol=1e-4)


def test_32in_fretless_pmm_routing():
    inst = load_instrument("32in_fretless")
    assert inst.id == "32in_fretless_pmm"
    assert inst.scale_length_in == 32.0
    assert inst.default_pickup == "pcsx"

    # Alias check
    inst_alias = load_instrument("fretless")
    assert inst_alias.id == "32in_fretless_pmm"

    # PCSX reverse split placement: 266mm from 12th fret = 140.4mm from bridge
    pcsx = inst.pickups["pcsx"]
    assert pcsx.position_from_bridge_m is not None
    assert math.isclose(pcsx.position_from_bridge_m, 0.1404, abs_tol=1e-4)
    assert pcsx.resonant_frequency_hz == 2610.0
    assert pcsx.q_factor == 1.35
    assert len(pcsx.coils) == 2
    assert pcsx.coils[0].strings == [1, 2]
    assert math.isclose(pcsx.coils[0].position_from_bridge_m, 0.1544, abs_tol=1e-4)
    assert pcsx.coils[1].strings == [3, 4]
    assert math.isclose(pcsx.coils[1].position_from_bridge_m, 0.1264, abs_tol=1e-4)

    # Upright voice routes to solo reverse PCSX neck split-coil (100%)
    up_pickup = get_source_pickup(inst, "upright_acoustic")
    assert up_pickup.name == "Reverse EMG PCSX Split-Coil (Neck)"
    assert up_pickup.type == "split_coil"


def test_34in_standard_pj_routing():
    inst = load_instrument("34in_standard_pj")
    assert inst.id == "34in_standard_pj"
    assert inst.scale_length_in == 34.0
    assert inst.electronics == "passive"
    assert "p" in inst.pickups
    assert "j" in inst.pickups
    assert "pair_parallel" in inst.pickups
    assert inst.default_pickup == "pair_parallel"

    # Split-P pickup
    p_pickup = inst.pickups["p"]
    assert p_pickup.type == "split_coil"
    assert p_pickup.position_from_bridge_m is not None
    assert math.isclose(p_pickup.position_from_bridge_m, 0.1250, abs_tol=1e-4)
    assert isinstance(p_pickup.circuit, CircuitConfig)
    assert p_pickup.circuit.topology == "single"
    assert len(p_pickup.coils) == 2

    # Jazz Bridge pickup
    j_pickup = inst.pickups["j"]
    assert j_pickup.type == "single_coil"
    assert j_pickup.position_from_bridge_m is not None
    assert math.isclose(j_pickup.position_from_bridge_m, 0.0635, abs_tol=1e-4)
    assert isinstance(j_pickup.circuit, CircuitConfig)
    assert j_pickup.circuit.topology == "single"
    assert len(j_pickup.coils) == 1

    # Parallel composite pair
    pair = inst.pickups["pair_parallel"]
    assert pair.type == "composite"
    assert isinstance(pair.circuit, CircuitConfig)
    assert pair.circuit.topology == "parallel"
    assert pair.components is not None
    assert len(pair.components) == 2
    assert pair.components[0].pickup == "p"
    assert pair.components[1].pickup == "j"

    # Shorthand aliases check
    assert load_instrument("standard_pj").id == "34in_standard_pj"
    assert load_instrument("pj").id == "34in_standard_pj"
    assert load_instrument("34in_pj").id == "34in_standard_pj"

    # Routing checks: P voices
    assert get_source_pickup(inst, "precision_active").id == "p"
    assert get_source_pickup(inst, "precision_vintage").id == "p"
    assert get_source_pickup(inst, "precision_warm").id == "p"
    assert get_source_pickup(inst, "mudbucker_deep").id == "p"
    assert get_source_pickup(inst, "upright_acoustic").id == "p"

    # Routing checks: Bridge voices
    assert get_source_pickup(inst, "jazz_bridge_open").id == "j"
    assert get_source_pickup(inst, "stingray_parallel").id == "j"
    assert get_source_pickup(inst, "rickenbacker_clank").id == "j"
    assert get_source_pickup(inst, "dingwall_bridge").id == "j"

    # Routing checks: Parallel pair voices
    assert get_source_pickup(inst, "jazz_pair_active").id == "pair_parallel"
    assert get_source_pickup(inst, "jazz_pair_open").id == "pair_parallel"
    assert get_source_pickup(inst, "pj_active").id == "pair_parallel"
    assert get_source_pickup(inst, "pj_passive").id == "pair_parallel"
    assert get_source_pickup(inst, "p_mm_series").id == "pair_parallel"
    assert get_source_pickup(inst, "p_mm_parallel").id == "pair_parallel"


def test_34in_active_stingray_routing():
    inst = load_instrument("34in_active_stingray")
    assert inst.id == "34in_active_stingray"
    assert inst.scale_length_in == 34.0
    assert inst.electronics == "active"
    assert "mm_parallel" in inst.pickups
    assert inst.default_pickup == "mm_parallel"

    # MM pickup definition
    mm = inst.pickups["mm_parallel"]
    assert mm.type == "dual_coil_parallel"
    assert mm.position_from_bridge_m is not None
    assert math.isclose(mm.position_from_bridge_m, 0.0660, abs_tol=1e-4)
    assert mm.resonant_frequency_hz == 4200.0
    assert mm.q_factor == 1.60
    assert len(mm.coils) == 2

    # Shorthand aliases check
    assert load_instrument("active_stingray").id == "34in_active_stingray"
    assert load_instrument("stingray").id == "34in_active_stingray"
    assert load_instrument("ray").id == "34in_active_stingray"

    # All target voices map to mm_parallel
    for vid in VOICES:
        pickup = get_source_pickup(inst, vid)
        assert pickup.id == "mm_parallel"


def test_34in_preamp_soapbar_routing():
    inst = load_instrument("34in_preamp_soapbar")
    assert inst.id == "34in_preamp_soapbar"
    assert inst.scale_length_in == 34.0
    assert inst.electronics == "active"
    assert "neck" in inst.pickups
    assert "bridge" in inst.pickups
    assert "pair_parallel" in inst.pickups
    assert inst.default_pickup == "pair_parallel"

    # Neck soapbar
    neck = inst.pickups["neck"]
    assert neck.position_from_bridge_m is not None
    assert math.isclose(neck.position_from_bridge_m, 0.1350, abs_tol=1e-4)
    assert neck.resonant_frequency_hz == 3800.0
    assert neck.q_factor == 1.45
    assert len(neck.coils) == 2
    assert neck.circuit is not None
    assert neck.circuit.L == 3.8
    assert neck.circuit.Rdc == 8600.0
    assert neck.circuit.active is True

    # Bridge soapbar
    bridge = inst.pickups["bridge"]
    assert bridge.position_from_bridge_m is not None
    assert math.isclose(bridge.position_from_bridge_m, 0.0550, abs_tol=1e-4)
    assert bridge.resonant_frequency_hz == 4100.0
    assert bridge.q_factor == 1.40
    assert len(bridge.coils) == 2
    assert bridge.circuit is not None
    assert bridge.circuit.L == 4.2
    assert bridge.circuit.Rdc == 9400.0
    assert bridge.circuit.active is True

    # Composite pair
    pair = inst.pickups["pair_parallel"]
    assert pair.type == "composite"
    assert pair.components is not None
    assert len(pair.components) == 2
    assert pair.components[0].pickup == "neck"
    assert pair.components[1].pickup == "bridge"
    assert pair.circuit is not None
    assert pair.circuit.topology == "parallel"
    assert pair.circuit.active is True

    # Shorthand aliases check
    assert load_instrument("preamp_soapbar").id == "34in_preamp_soapbar"
    assert load_instrument("ibanez_sr").id == "34in_preamp_soapbar"
    assert load_instrument("yamaha_trbx").id == "34in_preamp_soapbar"
    assert load_instrument("trbx").id == "34in_preamp_soapbar"

    # Removed backward compatibility aliases must raise FileNotFoundError
    import pytest

    with pytest.raises(FileNotFoundError):
        load_instrument("active_soapbar")
    with pytest.raises(FileNotFoundError):
        load_instrument("soapbar")

    # P voices route to neck soapbar
    assert get_source_pickup(inst, "precision_active").id == "neck"
    assert get_source_pickup(inst, "precision_vintage").id == "neck"
    assert get_source_pickup(inst, "mudbucker_deep").id == "neck"
    assert get_source_pickup(inst, "upright_acoustic").id == "neck"

    # Bridge voices route to bridge soapbar
    assert get_source_pickup(inst, "jazz_bridge_open").id == "bridge"
    assert get_source_pickup(inst, "stingray_parallel").id == "bridge"
    assert get_source_pickup(inst, "rickenbacker_clank").id == "bridge"
    assert get_source_pickup(inst, "dingwall_bridge").id == "bridge"

    # Parallel voices route to pair_parallel
    assert get_source_pickup(inst, "jazz_pair_active").id == "pair_parallel"
    assert get_source_pickup(inst, "jazz_pair_open").id == "pair_parallel"
    assert get_source_pickup(inst, "pj_active").id == "pair_parallel"
    assert get_source_pickup(inst, "pj_passive").id == "pair_parallel"
    assert get_source_pickup(inst, "p_mm_series").id == "pair_parallel"
    assert get_source_pickup(inst, "p_mm_parallel").id == "pair_parallel"


def test_30in_mustang_pj_routing():
    inst = load_instrument("30in_mustang_pj")
    assert inst.id == "30in_mustang_pj"
    assert inst.scale_length_in == 30.0
    assert inst.electronics == "passive"
    assert "p" in inst.pickups
    assert "j" in inst.pickups
    assert "pair_parallel" in inst.pickups
    assert inst.default_pickup == "pair_parallel"

    p = inst.pickups["p"]
    assert p.type == "split_coil"
    assert p.position_from_bridge_m is not None
    assert math.isclose(p.position_from_bridge_m, 0.1550, abs_tol=1e-4)
    assert p.resonant_frequency_hz == 2800.0
    assert len(p.coils) == 2

    j = inst.pickups["j"]
    assert j.type == "single_coil"
    assert j.position_from_bridge_m is not None
    assert math.isclose(j.position_from_bridge_m, 0.0550, abs_tol=1e-4)
    assert j.resonant_frequency_hz == 3200.0

    # Aliases
    assert load_instrument("mustang").id == "30in_mustang_pj"
    assert load_instrument("30in_mustang").id == "30in_mustang_pj"
    assert load_instrument("mustang_pj").id == "30in_mustang_pj"

    # Smart voice routing
    assert get_source_pickup(inst, "precision_vintage").id == "p"
    assert get_source_pickup(inst, "jazz_bridge_open").id == "j"
    assert get_source_pickup(inst, "jazz_pair_open").id == "pair_parallel"


def test_37in_multiscale_dingwall_routing():
    inst = load_instrument("37in_multiscale_dingwall")
    assert inst.id == "37in_multiscale_dingwall"
    assert inst.scale_length_in == 37.0
    assert inst.electronics == "active"
    assert len(inst.string_wave_speeds) == 5
    assert inst.strings is not None
    assert inst.strings.gauge == "45-130"
    assert "bridge" in inst.pickups
    assert "middle" in inst.pickups
    assert "pair_parallel" in inst.pickups

    # Bridge pickup (1:1 with Voice 13)
    b = inst.pickups["bridge"]
    assert b.position_from_bridge_m is not None
    assert math.isclose(b.position_from_bridge_m, 0.0480, abs_tol=1e-4)
    assert b.resonant_frequency_hz == 3400.0

    # Middle pickup
    mid = inst.pickups["middle"]
    assert mid.position_from_bridge_m is not None
    assert math.isclose(mid.position_from_bridge_m, 0.0960, abs_tol=1e-4)

    # Aliases
    assert load_instrument("dingwall").id == "37in_multiscale_dingwall"
    assert load_instrument("combustion").id == "37in_multiscale_dingwall"
    assert load_instrument("ng").id == "37in_multiscale_dingwall"
    assert load_instrument("dingwall_ng").id == "37in_multiscale_dingwall"
    assert load_instrument("ng2").id == "37in_multiscale_dingwall"
    assert load_instrument("ng3").id == "37in_multiscale_dingwall"
    assert load_instrument("37in_dingwall_ng").id == "37in_multiscale_dingwall"
    assert inst.is_multiscale is True
    assert inst.scale_min_in == 34.0
    assert inst.scale_max_in == 37.0

    # Voice 13 maps to bridge
    assert get_source_pickup(inst, "dingwall_bridge").id == "bridge"
    assert get_source_pickup(inst, "precision_vintage").id == "middle"
    assert get_source_pickup(inst, "jazz_pair_active").id == "pair_parallel"


def test_34in_dingwall_sp1_routing():
    inst = load_instrument("34in_dingwall_sp1")
    assert inst.id == "34in_dingwall_sp1"
    assert inst.scale_length_in == 35.0
    assert inst.is_multiscale is True
    assert inst.scale_min_in == 32.0
    assert inst.scale_max_in == 35.0
    assert inst.electronics == "passive"
    assert len(inst.string_wave_speeds) == 5
    assert inst.strings is not None
    assert inst.strings.gauge == "45-130"
    assert "p" in inst.pickups
    assert "bridge" in inst.pickups
    assert "pair_parallel" in inst.pickups

    # Dual-P split coil
    p = inst.pickups["p"]
    assert p.type == "split_coil"
    assert p.position_from_bridge_m is not None
    assert math.isclose(p.position_from_bridge_m, 0.1250, abs_tol=1e-4)
    assert len(p.coils) == 2
    assert p.coils[0].strings == [3, 4, 5]
    assert p.coils[1].strings == [1, 2]

    # FD3n Bridge
    b = inst.pickups["bridge"]
    assert b.position_from_bridge_m is not None
    assert math.isclose(b.position_from_bridge_m, 0.0580, abs_tol=1e-4)

    # Aliases
    assert load_instrument("sp1").id == "34in_dingwall_sp1"
    assert load_instrument("dingwall_sp1").id == "34in_dingwall_sp1"
    assert load_instrument("35in_dingwall_sp1").id == "34in_dingwall_sp1"
    assert load_instrument("super_p").id == "34in_dingwall_sp1"
    assert load_instrument("dingwall_super_p").id == "34in_dingwall_sp1"

    # Routing
    assert get_source_pickup(inst, "precision_vintage").id == "p"
    assert get_source_pickup(inst, "jazz_bridge_open").id == "bridge"
    assert get_source_pickup(inst, "pj_passive").id == "pair_parallel"
