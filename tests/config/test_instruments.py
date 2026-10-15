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


def test_load_instrument_canonical_ids():
    """Verify canonical instrument IDs resolve and non-canonical aliases raise FileNotFoundError."""
    canonical_ids = ["30in_mustang_pj", "34in_standard_p", "37in_multiscale_dingwall"]
    for cid in canonical_ids:
        cfg = load_instrument(cid)
        assert isinstance(cfg, InstrumentConfig)
        assert cfg.id == cid
        assert cfg.name is not None

    with pytest.raises(FileNotFoundError):
        load_instrument("mustang")


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
    assert (
        is_identity_voicing(inst_p, "vintage_open", inst_j, inst_j.voicings["pair_open"]) is False
    )

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


def test_load_all_tone_packs_catalog():
    """Verify load_all_tone_packs loads all defined tone packs in config/packs/."""
    from allomorph.config.instruments import TONE_PACKS, load_all_tone_packs

    packs = load_all_tone_packs()
    assert len(packs) == 7
    assert set(packs.keys()) == set(TONE_PACKS.keys())
    assert "30in_mustang_pj" in packs
    assert "34in_standard_p" in packs
    for pid, pcfg in packs.items():
        assert pcfg.id == pid
        assert len(pcfg.bundles) > 0


def test_partition_instrument_bundles_explicit_tone_pack():
    """Verify partition_instrument_bundles resolves explicit tone pack bundles."""
    from allomorph.config.instruments import partition_instrument_bundles

    inst = load_instrument("34in_standard_p")
    bundles = partition_instrument_bundles(inst)
    assert "split" in bundles
    assert len(bundles["split"].targets) > 0


def test_load_instrument_filename_directly():
    """Verify loading instrument using just filename."""
    cfg = load_instrument("34in_standard_p.toml")
    assert cfg.id == "34in_standard_p"


def test_3coil_pmm_compound_response():
    """Verify that 3-coil P/MM blend evaluates 3 distinct physical coil positions."""
    from allomorph.config.geometry import resolve_pickup_coils

    inst = load_instrument("32in_custom_pmm")
    coils = [c for p in inst.pickups.values() for c in resolve_pickup_coils(p, inst)]

    # Should have 4 coil records (PX D/G, PX E/A, MMTWX neck, MMTWX bridge)
    assert len(coils) == 4

    # Check that strings 1/2 (D/G) see 3 active coils and strings 3/4 (E/A) see 3 active coils
    dg_coils = [c for c in coils if "all" in c.strings or 1 in c.strings or "D" in c.strings]
    assert len(dg_coils) == 3

    ea_coils = [c for c in coils if "all" in c.strings or 4 in c.strings or "E" in c.strings]
    assert len(ea_coils) == 3

    # Positions should match blueprint: PX DG 136.8mm, MMTWX neck 73.7mm, MMTWX bridge 50.8mm
    positions = sorted([round(c.position_from_bridge_m * 1000, 1) for c in dg_coils])
    assert positions == [50.8, 73.7, 136.8]


def test_resolve_target_voicing_errors():
    """Verify diagnostic exceptions in resolve_target_voicing."""
    from allomorph.config.instruments import resolve_target_voicing

    inst = load_instrument("34in_standard_p")
    voicing = inst.voicings["vintage_open"]

    # Passing VoicingConfig without instrument raises ValueError
    with pytest.raises(ValueError, match="Must provide instrument"):
        resolve_target_voicing(voicing, instrument=None)

    # Passing pickup name as voicing raises KeyError per Guardrail 5.3.4
    with pytest.raises(KeyError, match="cannot be used as a target voicing"):
        resolve_target_voicing("split_p", instrument=inst)

    # Missing voicing raises KeyError
    with pytest.raises(KeyError, match="must be in canonical"):
        resolve_target_voicing("non_existent_voicing_xyz")


def test_load_all_tone_packs():
    """Verify loading all 7 tone packs, checking storefront descriptions and bundle partitioning."""
    from allomorph.config.instruments import (
        load_tone_pack,
        partition_instrument_bundles,
    )
    from allomorph.config.schema import TonePackConfig

    pack_ids = [
        "34in_standard_p",
        "34in_standard_jazz",
        "34in_standard_pj",
        "30in_mustang_pj",
        "34in_preamp_soapbar",
        "34in_active_stingray",
        "34in_active_emg",
    ]

    for pid in pack_ids:
        pack = load_tone_pack(pid)
        assert isinstance(pack, TonePackConfig)
        assert pack.id == pid
        assert len(pack.instruments) >= 1
        assert len(pack.storefront_description) > 0
        full_desc = pack.full_storefront_description
        assert "LICENSE & DISCLAIMER" in full_desc
        assert len(full_desc) <= 10000

        # Verify bundle partitioning
        bundles = partition_instrument_bundles(pid)
        assert len(bundles) >= 1
        for b in bundles.values():
            assert len(b.targets) >= 1


def test_ultra2_pj_and_pure_passive_p_configurations():
    """Verify Fender Ultra II PJ active harness, voicings, and pure passive P bass."""
    from allomorph.config.instruments import resolve_target_voicing

    # 1. 34in_standard_p is strictly passive
    inst_p = load_instrument("34in_standard_p")
    assert "active" not in inst_p.harnesses
    assert "passive" in inst_p.harnesses
    assert "modern_active" not in inst_p.voicings
    assert set(inst_p.voicings.keys()) == {
        "vintage_open",
        "vintage_mids",
        "vintage_warm",
        "slab_dub",
    }

    # 2. 34in_standard_pj has Ultra II pickups and active harness
    inst_pj = load_instrument("34in_standard_pj")
    assert "ultra2_split_p" in inst_pj.pickups
    assert "ultra2_bridge_j" in inst_pj.pickups
    assert "active" in inst_pj.harnesses

    h_active = inst_pj.harnesses["active"]
    assert h_active.preamp == "fender_ultra2_3band"
    assert "blend" in h_active.controls
    assert "tone" in h_active.controls
    assert "vol" in h_active.controls
    assert "mid_freq" in h_active.switches
    assert h_active.switches["mid_freq"].default == "325hz"

    # Active voicings
    assert "neck_active" in inst_pj.voicings
    v_neck = inst_pj.voicings["neck_active"]
    assert v_neck.tone_name == "Precision Active"
    assert v_neck.controls.get("blend") == 0.0

    assert "pair_active" in inst_pj.voicings
    v_pair = inst_pj.voicings["pair_active"]
    assert v_pair.tone_name == "PJ Active"
    assert v_pair.controls.get("blend") == 0.5

    # 3. Target resolution
    resolved_inst, resolved_v = resolve_target_voicing("34in_standard_pj:neck_active")
    assert resolved_inst.id == "34in_standard_pj"
    assert resolved_v.id == "neck_active"
    assert resolved_v.tone_name == "Precision Active"

    # 4. Old modern_active cleanly removed (no backward compat)
    with pytest.raises(KeyError):
        resolve_target_voicing("34in_standard_p:modern_active")
