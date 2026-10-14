"""Deterministic unit tests for config/resolvers.py decomposition primitives."""

from __future__ import annotations

import numpy as np
import pytest

from allomorph.config.instruments import VoicingBundle
from allomorph.config.resolvers import (
    assign_target_to_bundle,
    build_voice_coils_for_pickup,
    compound_multi_pickup_coils,
    determine_pickup_resonance_and_q,
    find_voicing_by_slug_or_id,
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
from allomorph.dsp.constants import FREQS


def test_determine_pickup_resonance_and_q_synthetic_rlc():
    """Verify sub-bin interpolation and analytical Q extraction on synthetic 2nd-order RLC."""
    f = np.logspace(np.log10(20), np.log10(20000), 1000)
    w = 2.0 * np.pi * f
    w0 = 2.0 * np.pi * 3500.0
    q_true = 1.8
    # Theoretical peak for 2nd-order lowpass: f_peak = f0 * sqrt(1 - 1/(2*Q^2)) = 3218.6 Hz
    h = w0**2 / (-(w**2) + 1j * w * w0 / q_true + w0**2)
    fr, q = determine_pickup_resonance_and_q(np.abs(h), f)
    assert fr is not None and 3210.0 <= fr <= 3225.0
    assert q is not None and 1.4 <= q <= 1.6


def test_determine_pickup_resonance_and_q_flat_response():
    """Verify flat response returns (None, None)."""
    f = np.asarray(FREQS, dtype=np.float64)
    h_flat = np.ones_like(f)
    fr, q = determine_pickup_resonance_and_q(h_flat, f)
    assert fr is None
    assert q is None


def test_determine_pickup_resonance_and_q_below_threshold():
    """Verify bumps below +0.20 dB threshold are rejected as flat."""
    f = np.asarray(FREQS, dtype=np.float64)
    h_small_bump = np.ones_like(f)
    mid_idx = len(f) // 2
    # Add +0.10 dB bump (approx 1.0116 lin)
    h_small_bump[mid_idx] = 1.0116
    fr, q = determine_pickup_resonance_and_q(h_small_bump, f)
    assert fr is None
    assert q is None


def test_determine_pickup_resonance_and_q_monotonic_cutoff():
    """Verify monotonic 1st-order lowpass (boundary peak at DC) returns (None, None)."""
    f = np.asarray(FREQS, dtype=np.float64)
    h_lp = 1.0 / (1.0 + (f / 1000.0) ** 2)
    fr, q = determine_pickup_resonance_and_q(h_lp, f)
    assert fr is None
    assert q is None


def test_determine_pickup_resonance_and_q_none_and_empty():
    """Verify None and empty inputs return (None, None) gracefully."""
    assert determine_pickup_resonance_and_q(None) == (None, None)
    assert determine_pickup_resonance_and_q([]) == (None, None)
    assert determine_pickup_resonance_and_q([1.0, 2.0]) == (None, None)


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
        PackBundleConfig(name="neck", source="inst_test:v_neck", targets=["inst_target:target_a"]),
        PackBundleConfig(
            name="bridge", source="inst_test:v_bridge", targets=["inst_target:target_b"]
        ),
    ]

    # Explicit config target
    assert assign_target_to_bundle("inst_target:target_a", pack_bundles, b_configs) == "neck"
    assert assign_target_to_bundle("target_a", pack_bundles, b_configs) == "neck"
    assert assign_target_to_bundle("inst_target:target_b", pack_bundles, b_configs) == "bridge"
    assert assign_target_to_bundle("target_b", pack_bundles, b_configs) == "bridge"

    # Position keyword fallback
    assert assign_target_to_bundle("custom_bridge_lead", pack_bundles, b_configs) == "bridge"

    # Default fallback
    assert assign_target_to_bundle("mystery_voicing", pack_bundles, b_configs) == "neck"
