"""
Tests for Instrument Configuration, Routing, and Pickup Bundles in allomorph.config.instruments.
"""

import pytest

from allomorph.config.instruments import (
    get_source_pickup,
    is_identity_voicing,
    load_all_instruments,
    load_instrument,
)
from allomorph.config.schema import InstrumentConfig


def test_get_source_pickup_invalid_mapping_raises_key_error():
    """Bug 7 regression: pickup_mapping referencing non-existent pickup must raise KeyError, never silently fall back."""
    inst = load_instrument("34in_standard_jazz")
    broken_inst = inst.model_copy(deep=True)
    broken_inst.pickup_mapping = {"broken_voice": "non_existent_pickup"}

    with pytest.raises(KeyError, match="non_existent_pickup"):
        get_source_pickup(broken_inst, "broken_voice")


def test_is_identity_voicing_tone_cap_mismatch():
    """Bug 8 regression: target with explicit tone cap must not be considered identity to a pickup with no tone cap."""
    inst = load_instrument("30in_emg_mmtw")
    tgt_voicing = inst.voicings["dual_mode"].model_copy(deep=True)
    tgt_voicing.id = "dual_mode_custom"
    tgt_voicing.tone_cap_f = 47e-9  # 47 nF
    # EMG MMTW pickup has Ctone=None; an explicit tone cap override is NOT an identity match
    assert is_identity_voicing(inst, "mmtw_dual", inst, tgt_voicing) is False


def test_load_instrument_aliases():
    """Verify standard aliases resolve correctly to canonical instruments."""
    aliases = ["30in", "32in", "34in", "dingwall", "upright", "rick", "eb0"]
    for alias in aliases:
        cfg = load_instrument(alias)
        assert isinstance(cfg, InstrumentConfig)
        assert cfg.id is not None
        assert cfg.name is not None


def test_load_instrument_by_path():
    """Verify loading directly from a Path object."""
    from allomorph.config.instruments import INSTRUMENTS_DIR

    p = INSTRUMENTS_DIR / "34in_standard_p.toml"
    cfg = load_instrument(p)
    assert cfg.id == "34in_standard_p"


def test_load_instrument_not_found():
    """Verify unknown instrument identifier raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError, match="Instrument configuration not found"):
        load_instrument("non_existent_bass_12345")


def test_load_instrument_already_config():
    """Verify passing an InstrumentConfig instance returns it unchanged."""
    inst = load_instrument("34in_standard_p")
    same = load_instrument(inst)
    assert same is inst


def test_load_all_instruments():
    """Verify load_all_instruments loads all configuration files in directory."""
    all_insts = load_all_instruments()
    assert len(all_insts) >= 8
    assert "34in_standard_p" in all_insts
    assert "34in_standard_jazz" in all_insts
    assert "30in_emg_mmtw" in all_insts


def test_resolve_target_affinity():
    """Verify target affinity resolution classifies correctly into neck, bridge, parallel, direct."""
    from allomorph.config.instruments import resolve_target_affinity

    # Bridge affinities
    assert resolve_target_affinity("bridge_open") == "bridge"
    assert resolve_target_affinity("bridge_growl") == "bridge"

    # Neck affinities
    assert resolve_target_affinity("neck_warm") == "neck"

    # Parallel / Pair affinities
    assert resolve_target_affinity("pair_open") == "parallel"

    # Direct / acoustic affinities
    assert resolve_target_affinity("bridge_piezo_acoustic") in ("direct", "neck")


def test_partition_instrument_bundles_single_pickup():
    """Verify bundle partitioning for a single-pickup instrument."""
    from allomorph.config.instruments import partition_instrument_bundles

    inst = load_instrument("30in_emg_mmtw")
    bundles = partition_instrument_bundles(inst)
    assert len(bundles) >= 1
    # Check that each bundle has valid targets and dry stem mappings
    for b_name, bundle in bundles.items():
        assert bundle.bundle_name == b_name
        assert bundle.pickup_key in inst.pickups
        assert len(bundle.targets) > 0


def test_partition_instrument_bundles_multi_pickup():
    """Verify bundle partitioning for a multi-pickup instrument (Jazz Bass)."""
    from allomorph.config.instruments import partition_instrument_bundles

    inst = load_instrument("34in_standard_jazz")
    bundles = partition_instrument_bundles(inst)
    # Jazz bass should have multiple pickup bundles (neck, bridge, pair)
    assert len(bundles) >= 2
    for b_name, bundle in bundles.items():
        assert bundle.bundle_name == b_name
        assert bundle.pickup_key in inst.pickups


def test_get_source_pickup_with_explicit_mapping():
    """Verify get_source_pickup honors pickup_mapping table."""
    inst = load_instrument("34in_standard_jazz")
    assert inst.pickup_mapping is not None
    # For a mapped voice, it must resolve to the mapped pickup
    for voice_id, pkey in inst.pickup_mapping.items():
        pickup = get_source_pickup(inst, voice_id)
        assert pickup.id == pkey


def test_is_identity_voicing_exact_and_different():
    """Verify is_identity_voicing returns True for exact match and False for differences."""
    inst_p = load_instrument("34in_standard_p")
    inst_j = load_instrument("34in_standard_jazz")

    # Cross-instrument is always False
    assert is_identity_voicing(inst_p, "split_coil", inst_j, inst_j.voicings["pair_open"]) is False

    # Within instrument: check voice that matches pickup
    p_voice = inst_p.voicings.get("vintage_open")
    if p_voice is not None:
        assert is_identity_voicing(inst_p, "split_p", inst_p, p_voice) is True


def test_resolve_target_affinity_composite_and_fallbacks():
    """Verify resolve_target_affinity on direct tags, composite colon notation, and objects."""
    from allomorph.config.instruments import resolve_target_affinity

    # 1. Direct positions
    for pos in ("neck", "bridge", "parallel", "direct"):
        assert resolve_target_affinity(pos) == pos

    # 2. Composite notation inst_id:voice_id
    assert resolve_target_affinity("34in_standard_p:vintage_open") == "neck"

    # 3. Object with .affinity attribute
    class DummyVoicing:
        affinity = "bridge"

    assert resolve_target_affinity(DummyVoicing()) == "bridge"

    # 4. Unknown string returns None
    assert resolve_target_affinity("completely_unknown_string_123") is None


def test_get_source_pickup_edge_cases():
    """Verify get_source_pickup default_pickup, empty pickups, and affinity error handling."""
    inst = load_instrument("34in_standard_p")

    # 1. Empty pickups raises ValueError
    empty_inst = inst.model_copy(deep=True)
    empty_inst.pickups = {}
    with pytest.raises(ValueError, match="has no pickups defined"):
        get_source_pickup(empty_inst, "vintage_open")

    # 2. Multi-pickup with valid default_pickup fallback
    multi = load_instrument("34in_standard_jazz").model_copy(deep=True)
    multi.pickup_mapping = {}  # Clear explicit and affinity mapping
    multi.default_pickup = "neck"
    p = get_source_pickup(multi, "unmapped_voice")
    assert p.id == "neck"

    # 3. Multi-pickup with non-existent default_pickup raises KeyError
    object.__setattr__(multi, "default_pickup", "non_existent_default")
    with pytest.raises(KeyError, match="default_pickup 'non_existent_default' not found"):
        get_source_pickup(multi, "unmapped_voice")

    # 4. Multi-pickup with no default and no mapping raises ValueError
    multi.default_pickup = None
    with pytest.raises(ValueError, match="defines no 'default_pickup'"):
        get_source_pickup(multi, "unmapped_voice")

    # 5. Affinity mapping to non-existent pickup raises KeyError
    multi.pickup_mapping = {"neck": "broken_pickup_id"}
    with pytest.raises(KeyError, match="broken_pickup_id"):
        get_source_pickup(multi, "neck_warm")


def test_is_identity_voicing_control_deviations():
    """Verify is_identity_voicing detects control deviations (vol, tone, blend, EQ, strings)."""
    inst = load_instrument("34in_standard_p")
    base_voice = inst.voicings["vintage_open"]

    # Nominal check
    assert is_identity_voicing(inst, "split_p", inst, base_voice) is True

    # Volume deviation
    v_vol = base_voice.model_copy(deep=True)
    v_vol.id = "v_vol"
    v_vol.vol_pos = 0.8
    assert is_identity_voicing(inst, "split_p", inst, v_vol) is False

    # Tone deviation
    v_tone = base_voice.model_copy(deep=True)
    v_tone.id = "v_tone"
    v_tone.tone_pos = 0.5
    assert is_identity_voicing(inst, "split_p", inst, v_tone) is False

    # Blend deviation
    v_blend = base_voice.model_copy(deep=True)
    v_blend.id = "v_blend"
    v_blend.blend_pos = 0.2
    assert is_identity_voicing(inst, "split_p", inst, v_blend) is False

    # Gain deviation
    v_gain = base_voice.model_copy(deep=True)
    v_gain.id = "v_gain"
    v_gain.gain_db = 3.0
    assert is_identity_voicing(inst, "split_p", inst, v_gain) is False

    # Preamp preset mismatch
    v_pre = base_voice.model_copy(deep=True)
    v_pre.id = "v_pre"
    v_pre.preamp_preset = "sadowsky_2band"
    assert is_identity_voicing(inst, "split_p", inst, v_pre) is False

    # String preset mismatch
    v_str = base_voice.model_copy(deep=True)
    v_str.id = "v_str"
    v_str.string_preset_override = "flatwound_classic_deep"
    assert is_identity_voicing(inst, "split_p", inst, v_str) is False
