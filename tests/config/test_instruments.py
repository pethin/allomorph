"""
Tests for Instrument Configuration, Routing, and Pickup Bundles in allomorph.config.instruments.
"""

import pytest

from allomorph.config.instruments import (
    is_identity_voicing,
    load_all_instruments,
    load_instrument,
)
from allomorph.config.schema import InstrumentConfig


def test_is_identity_voicing_tone_cap_mismatch():
    """Bug 8 regression: target with explicit tone cap must not be considered identity to a pickup with no tone cap."""
    inst = load_instrument("30in_emg_mmtw")
    tgt_voicing = inst.voicings["dual_mode"].model_copy(deep=True)
    tgt_voicing.id = "dual_mode_custom"
    tgt_voicing.components["controls.tone.cap"] = 47e-9  # 47 nF
    # EMG MMTW pickup has Ctone=None; an explicit tone cap override is NOT an identity match
    assert is_identity_voicing(inst, "mmtwx", inst, tgt_voicing) is False


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




def test_partition_instrument_bundles_single_pickup():
    """Verify bundle partitioning for a single-pickup instrument."""
    from allomorph.config.instruments import partition_instrument_bundles

    inst = load_instrument("34in_standard_p")
    bundles = partition_instrument_bundles(inst)
    assert len(bundles) >= 1
    # Check that each bundle has valid targets and dry stem mappings
    for b_name, bundle in bundles.items():
        assert bundle.bundle_name == b_name
        assert bundle.source_voicing in inst.voicings
        assert len(bundle.targets) > 0


def test_partition_instrument_bundles_undeclared_pack_raises():
    """Verify calling partition_instrument_bundles on undeclared pack raises FileNotFoundError."""
    import pytest

    from allomorph.config.instruments import partition_instrument_bundles

    inst = load_instrument("30in_emg_mmtw")
    with pytest.raises(FileNotFoundError, match="Tone pack configuration not found"):
        partition_instrument_bundles(inst)


def test_partition_instrument_bundles_multi_pickup():
    """Verify bundle partitioning for a multi-pickup instrument (Jazz Bass)."""
    from allomorph.config.instruments import partition_instrument_bundles

    inst = load_instrument("34in_standard_jazz")
    bundles = partition_instrument_bundles(inst)
    # Jazz bass should have multiple pickup bundles (neck, bridge, pair)
    assert len(bundles) >= 2
    for b_name, bundle in bundles.items():
        assert bundle.bundle_name == b_name
        assert bundle.source_voicing in inst.voicings


def test_is_identity_voicing_exact_and_different():
    """Verify is_identity_voicing returns True for exact match and False for differences."""
    inst_p = load_instrument("34in_standard_p")
    inst_j = load_instrument("34in_standard_jazz")

    # Cross-instrument is always False
    assert is_identity_voicing(inst_p, "vintage_open", inst_j, inst_j.voicings["pair_open"]) is False

    # Within instrument: check voice that matches
    p_voice = inst_p.voicings.get("vintage_open")
    if p_voice is not None:
        assert is_identity_voicing(inst_p, "vintage_open", inst_p, p_voice) is True


def test_is_identity_voicing_control_deviations():
    """Verify is_identity_voicing detects control deviations (vol, tone, blend, EQ, strings)."""
    inst = load_instrument("34in_standard_p")
    base_voice = inst.voicings["vintage_open"]

    # Nominal check
    assert is_identity_voicing(inst, "vintage_open", inst, base_voice) is True

    # Volume deviation
    v_vol = base_voice.model_copy(deep=True)
    v_vol.id = "v_vol"
    v_vol.controls["vol"] = 0.8
    assert is_identity_voicing(inst, "vintage_open", inst, v_vol) is False

    # Tone deviation
    v_tone = base_voice.model_copy(deep=True)
    v_tone.id = "v_tone"
    v_tone.controls["tone"] = 0.5
    assert is_identity_voicing(inst, "vintage_open", inst, v_tone) is False

    # Blend deviation
    v_blend = base_voice.model_copy(deep=True)
    v_blend.id = "v_blend"
    v_blend.controls["blend"] = 0.2
    assert is_identity_voicing(inst, "vintage_open", inst, v_blend) is False

    # Gain deviation
    v_gain = base_voice.model_copy(deep=True)
    v_gain.id = "v_gain"
    v_gain.gain_db = 3.0
    assert is_identity_voicing(inst, "vintage_open", inst, v_gain) is False

    # Preamp preset / harness mismatch
    v_pre = base_voice.model_copy(deep=True)
    v_pre.id = "v_pre"
    v_pre.harness = "active"
    assert is_identity_voicing(inst, "vintage_open", inst, v_pre) is False

    # String preset mismatch
    v_str = base_voice.model_copy(deep=True)
    v_str.id = "v_str"
    v_str.string_preset_override = "flatwound_classic_deep"
    assert is_identity_voicing(inst, "vintage_open", inst, v_str) is False


def test_is_identity_voicing_str_inputs():
    """Verify is_identity_voicing with string inputs."""
    inst = load_instrument("34in_standard_p")
    assert is_identity_voicing(inst, "vintage_open", inst, "vintage_open") is True
    assert is_identity_voicing(inst, "vintage_open", inst, "vintage_warm") is False


def test_load_tone_pack_and_errors():
    """Verify loading tone pack by id, instance, or raising FileNotFoundError."""
    from allomorph.config.instruments import load_tone_pack
    from allomorph.config.schema import TonePackConfig

    tp = load_tone_pack("34in_standard_p")
    assert isinstance(tp, TonePackConfig)
    assert load_tone_pack(tp) is tp

    with pytest.raises(FileNotFoundError, match="Tone pack configuration not found"):
        load_tone_pack("nonexistent_tone_pack_999")


def test_partition_instrument_bundles_target_variants():
    """Verify partition_instrument_bundles with catalog_targets and KeyError."""
    from allomorph.config.instruments import partition_instrument_bundles

    inst = load_instrument("34in_standard_p")
    targets = [
        ("34in_standard_jazz", "pair_open"),
        ("34in_standard_jazz", "bridge_growl"),
    ]
    bundles = partition_instrument_bundles(inst, catalog_targets=targets)
    assert "split" in bundles
    assert len(bundles["split"].targets) == 2

    # Target voicing not found raises KeyError
    bad_targets = [("34in_standard_jazz", "nonexistent_voicing_123")]
    with pytest.raises(KeyError, match="Target voicing 'nonexistent_voicing_123' not found"):
        partition_instrument_bundles(inst, catalog_targets=bad_targets)


def test_load_instrument_filename_directly():
    """Verify loading instrument using just filename."""
    cfg = load_instrument("34in_standard_p.toml")
    assert cfg.id == "34in_standard_p"
