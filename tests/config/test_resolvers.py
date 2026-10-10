"""Deterministic unit tests for config/resolvers.py decomposition primitives."""

from __future__ import annotations

import pytest

from allomorph.config.instruments import VoicingBundle
from allomorph.config.resolvers import (
    assign_target_to_bundle,
    build_voice_coils_for_pickup,
    compound_multi_pickup_coils,
    find_voicing_by_slug_or_id,
    infer_pickup_resonance_and_q,
    parse_target_voicing_token,
)
from allomorph.config.schema import (
    CoilConfig,
    PackBundleConfig,
    PickupConfig,
    VoiceCoilConfig,
    VoicePickupConfig,
    VoicingConfig,
)


def test_infer_pickup_resonance_and_q_explicit():
    pickup = PickupConfig(
        id="custom",
        name="Custom Pickup",
        type="single_coil",
        resonant_frequency_hz=3450.0,
        q_factor=1.95,
    )
    fr, q = infer_pickup_resonance_and_q(pickup, pickup_key="bridge", harness_str="passive_500k")
    assert fr == 3450.0
    assert q == 1.95


def test_infer_pickup_resonance_and_q_split_coil():
    pickup = PickupConfig(
        id="p_split",
        name="Precision Split",
        type="split_coil",
    )
    # Passive
    fr_p, q_p = infer_pickup_resonance_and_q(pickup, pickup_key="neck", harness_str="passive_250k")
    assert fr_p == 2800.0
    assert q_p == 1.4

    # Active
    fr_a, q_a = infer_pickup_resonance_and_q(pickup, pickup_key="neck", harness_str="active_preamp")
    assert fr_a == 4800.0
    assert q_a == 1.7


def test_infer_pickup_resonance_and_q_bridge_and_neck():
    p_bridge = PickupConfig(id="j_b", name="Jazz Bridge", type="single_coil")
    p_neck = PickupConfig(id="j_n", name="Jazz Neck", type="single_coil")

    # Bridge passive vs active
    fr_bp, q_bp = infer_pickup_resonance_and_q(p_bridge, pickup_key="bridge", harness_str="passive")
    assert fr_bp == 3200.0
    assert q_bp == 1.6

    fr_ba, q_ba = infer_pickup_resonance_and_q(p_bridge, pickup_key="bridge", harness_str="active")
    assert fr_ba == 4600.0
    assert q_ba == 1.8

    # Neck passive vs active
    fr_np, q_np = infer_pickup_resonance_and_q(p_neck, pickup_key="neck", harness_str="passive")
    assert fr_np == 3600.0
    assert q_np == 1.5

    fr_na, q_na = infer_pickup_resonance_and_q(p_neck, pickup_key="neck", harness_str="active")
    assert fr_na == 5200.0
    assert q_na == 1.7


def test_infer_pickup_resonance_and_q_fallback():
    p_gen = PickupConfig(id="gen", name="Generic Pickup", type="humbucker")

    fr_p, q_p = infer_pickup_resonance_and_q(p_gen, pickup_key="middle", harness_str="passive")
    assert fr_p == 3500.0
    assert q_p == 1.5

    fr_a, q_a = infer_pickup_resonance_and_q(p_gen, pickup_key="middle", harness_str="active")
    assert fr_a == 5200.0
    assert q_a == 1.7


def test_build_voice_coils_for_pickup():
    coils = [
        CoilConfig(
            id="c1", position_from_bridge_m=0.080, aperture_width_in=0.75, weight=1.0, polarity=1.0
        ),
        CoilConfig(
            id="c2", position_from_bridge_m=0.060, aperture_width_in=0.75, weight=1.0, polarity=-1.0
        ),
    ]
    pickup = PickupConfig(id="dual", name="Dual Coil", type="humbucker", coils=coils)

    # All coils
    v_coils = build_voice_coils_for_pickup(pickup)
    assert len(v_coils) == 2
    assert v_coils[0].polarity == 1.0
    assert v_coils[1].polarity == -1.0

    # Filtered by active coil ID
    v_filtered = build_voice_coils_for_pickup(pickup, active_coil_ids=["c1"])
    assert len(v_filtered) == 1
    assert v_filtered[0].position_from_bridge_m == pytest.approx(0.080)


def test_compound_multi_pickup_coils():
    p1_coils = [
        VoiceCoilConfig(
            position_from_bridge_m=0.100, aperture_width_in=0.75, weight=1.0, polarity=1.0
        ),
    ]
    p2_coils = [
        VoiceCoilConfig(
            position_from_bridge_m=0.050, aperture_width_in=0.75, weight=0.8, polarity=-1.0
        ),
    ]
    p1 = VoicePickupConfig(
        name="Neck", type="Single", fr=3600.0, Q=1.5, weight=0.5, polarity=1.0, coils=p1_coils
    )
    p2 = VoicePickupConfig(
        name="Bridge", type="Single", fr=3200.0, Q=1.6, weight=0.5, polarity=-1.0, coils=p2_coils
    )

    compounded = compound_multi_pickup_coils([p1, p2])
    assert len(compounded) == 2
    assert compounded[0].weight == 0.5
    assert compounded[0].polarity == 1.0

    assert compounded[1].weight == pytest.approx(0.4)
    assert compounded[1].polarity == 1.0  # -1.0 * -1.0


def test_parse_target_voicing_token():
    # Colon syntax
    inst_id, vid = parse_target_voicing_token(
        "34in_active_stingray:stingray_open", default_inst_id="my_inst"
    )
    assert inst_id == "34in_active_stingray"
    assert vid == "stingray_open"

    # Fallback to default
    inst_id2, vid2 = parse_target_voicing_token("custom_tone", default_inst_id="my_inst")
    assert inst_id2 == "my_inst"
    assert vid2 == "custom_tone"


def test_find_voicing_by_slug_or_id():
    from tests.conftest import make_generic_instrument_config

    inst = make_generic_instrument_config()
    v1 = VoicingConfig(
        id="v1", name="Vintage Split", tone_name="Precision Vintage", harness="passive"
    )
    v2 = VoicingConfig(
        id="v2", name="Bridge Growl", tone_name="Jazz Bridge Growl", harness="passive"
    )
    inst.voicings["v1"] = v1
    inst.voicings["v2"] = v2

    # Match by exact ID
    assert find_voicing_by_slug_or_id(inst, "v1") is v1

    # Match by normalized tone name slug
    assert find_voicing_by_slug_or_id(inst, "precision_vintage") is v1
    assert find_voicing_by_slug_or_id(inst, "jazz_bridge_growl") is v2

    # Non-existent returns None
    assert find_voicing_by_slug_or_id(inst, "non_existent") is None


def test_assign_target_to_bundle():
    b_neck = VoicingBundle(
        bundle_name="neck", source_voicing="v_neck", position_name="Neck", targets=[]
    )
    b_bridge = VoicingBundle(
        bundle_name="bridge", source_voicing="v_bridge", position_name="Bridge", targets=[]
    )
    pack_bundles = {"neck": b_neck, "bridge": b_bridge}

    b_configs = [
        PackBundleConfig(name="neck", source_voicing="v_neck", targets=["target_a"]),
        PackBundleConfig(name="bridge", source_voicing="v_bridge", targets=["target_b"]),
    ]

    # Explicit config target
    assert assign_target_to_bundle("target_a", pack_bundles, b_configs) == "neck"
    assert assign_target_to_bundle("target_b", pack_bundles, b_configs) == "bridge"

    # Position keyword fallback
    assert assign_target_to_bundle("custom_bridge_lead", pack_bundles, b_configs) == "bridge"

    # Default fallback
    assert assign_target_to_bundle("mystery_voicing", pack_bundles, b_configs) == "neck"
