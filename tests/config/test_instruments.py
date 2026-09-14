"""
Tests for source instrument configuration loaders, validation, and electrical parameters.
"""

import math
import tempfile
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest

from allomorph.config import (
    STRINGS,
    VOICES,
    InstrumentConfig,
    PickupConfig,
    get_instrument_string,
    get_source_pickup,
    load_all_instruments,
    load_instrument,
)
from allomorph.config.instruments import INSTRUMENT_ALIASES
from allomorph.dsp import NUM_TAPS
from allomorph.naming import resolve_instruments, resolve_voices
from allomorph.physics import compute_voice_prefilter_firs


def test_load_all_default_instruments():
    instruments = load_all_instruments()
    expected_ids = [
        "30in_emg_mmtw",
        "32in_custom_pmm",
        "32in_fretless_pmm",
        "34in_standard_p",
        "34in_standard_jazz",
        "34in_standard_pj",
        "34in_active_pmm",
        "34in_active_stingray",
        "34in_preamp_soapbar",
        "34in_active_emg",
        "30in_mustang_pj",
        "37in_multiscale_dingwall",
        "34in_dingwall_sp1",
        "33in_rickenbacker_4003",
        "30in_gibson_eb0",
        "41in_upright_bass",
        "studio_direct",
    ]
    for iid in expected_ids:
        assert iid in instruments, f"Default instrument '{iid}' not found"

    for iid, cfg in instruments.items():
        assert isinstance(cfg, InstrumentConfig), f"Instrument {iid} is not an InstrumentConfig"
        assert cfg.id == iid
        assert cfg.name
        assert cfg.scale_length_in is not None
        assert cfg.scale_length_in > 0
        assert len(cfg.string_wave_speeds) in [4, 5]
        assert cfg.default_pickup in cfg.pickups
        assert cfg.version >= 1
        for v_name, vcfg in cfg.voicings.items():
            assert vcfg.version >= 1, f"Voicing {iid}:{v_name} missing version"

        for p_name, pcfg in cfg.pickups.items():
            assert isinstance(pcfg, PickupConfig), f"Pickup {p_name} is not a PickupConfig"
            assert pcfg.name
            assert pcfg.position_from_bridge_m is not None and pcfg.position_from_bridge_m > 0
            assert pcfg.aperture_width_in > 0
            assert pcfg.coil_spacing_in >= 0


def test_load_custom_user_bass_toml():
    """Verify that any future bass or external user bass can be loaded from an arbitrary TOML file."""
    custom_toml = """
id = "custom_35in_soapbar"
name = '35" Custom 5-String Soapbar'
scale_length_in = 35.0
scale_length_m = 0.889
string_wave_speeds = [73.2, 97.8, 130.5, 174.2]
default_pickup = "bridge_soapbar"

[pickups.neck_soapbar]
name = "Neck Dual Soapbar"
position_from_bridge_m = 0.1400
aperture_width_in = 1.35
coil_spacing_in = 0.65

[pickups.bridge_soapbar]
name = "Bridge Dual Soapbar"
position_from_bridge_m = 0.0550
aperture_width_in = 1.35
coil_spacing_in = 0.65

[pickup_mapping]
"precision_active" = "neck_soapbar"
"stingray_parallel" = "bridge_soapbar"
"""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_file = Path(tmpdir) / "my_custom_bass.toml"
        tmp_file.write_text(custom_toml, encoding="utf-8")

        inst = load_instrument(tmp_file)
        assert inst.id == "custom_35in_soapbar"
        assert inst.scale_length_in == 35.0

        p_pick = get_source_pickup(inst, "precision_active")
        assert p_pick.name == "Neck Dual Soapbar"

        # Compute prefilter FIR using the custom user bass
        firs = compute_voice_prefilter_firs(
            "precision_active", instrument=tmp_file, num_taps=NUM_TAPS, normalize=True
        )
        assert len(firs) == 1
        fir = firs[0]
        assert len(fir) == NUM_TAPS
        max_peak = max(abs(x) for x in fir)
        assert math.isclose(max_peak, 0.99, rel_tol=1e-3)


def test_standard_instruments_electrical_parameters():
    """Verify that standard instrument pickups specify resonant_frequency_hz and q_factor."""
    inst_p = load_instrument("34in_standard_p")
    p_pickup = inst_p.pickups["split_p"]
    assert p_pickup.resonant_frequency_hz == 2800.0
    assert p_pickup.q_factor == 1.40

    inst_j = load_instrument("34in_standard_jazz")
    j_neck = inst_j.pickups["neck"]
    assert j_neck.resonant_frequency_hz == 3600.0
    assert j_neck.q_factor == 1.50

    j_bridge = inst_j.pickups["bridge"]
    assert j_bridge.resonant_frequency_hz == 3200.0
    assert j_bridge.q_factor == 1.60

    j_pair = inst_j.pickups["pair_parallel"]
    assert j_pair.resonant_frequency_hz == 3900.0
    assert j_pair.q_factor == 1.30

    inst_pj = load_instrument("34in_standard_pj")
    pj_p = inst_pj.pickups["p"]
    assert pj_p.resonant_frequency_hz == 2800.0
    assert pj_p.q_factor == 1.40

    pj_j = inst_pj.pickups["j"]
    assert pj_j.resonant_frequency_hz == 3200.0
    assert pj_j.q_factor == 1.60

    pj_pair = inst_pj.pickups["pair_parallel"]
    assert pj_pair.resonant_frequency_hz == 2800.0
    assert pj_pair.q_factor == 1.20


def test_32in_pj_blend_parallel_definition():
    """Verify that both 32in P+TWX instruments define the parallel P/J mode (single-coil TWX)."""
    for iid in ["32in_custom_pmm", "32in_fretless_pmm"]:
        inst = load_instrument(iid)
        assert "pj_blend_parallel" in inst.pickups
        pj = inst.pickups["pj_blend_parallel"]
        assert pj.type == "composite"
        assert pj.components is not None
        assert len(pj.components) == 2
        assert pj.components[1].pickup == "mmtwx_single"
        assert pj.resonant_frequency_hz is not None and pj.resonant_frequency_hz > 0
        assert pj.q_factor is not None and pj.q_factor > 0
        assert inst.pickup_mapping["pj_active"] == "pj_blend_parallel"
        assert inst.pickup_mapping["pj_passive"] == "pj_blend_parallel"


def test_all_instruments_have_valid_string_presets():
    """Verify that every default instrument declares a string preset that exists in STRINGS catalog."""
    instruments = load_all_instruments()
    for iid, cfg in instruments.items():
        if cfg.strings and cfg.strings.preset:
            preset = cfg.strings.preset
            assert preset in STRINGS, (
                f"Instrument '{iid}' declares unknown string preset '{preset}'"
            )


def test_active_identity_differential_flatness():
    """Verify that active source instruments matching their target voice evaluate to 0.00 dB flat."""
    import numpy as np

    from allomorph.visualizer import build_voice_dataframe

    # 1. 37" Multi-Scale Dingwall -> Voice 13 Dingwall Bridge (< 0.05 dB flat)
    df_ding = build_voice_dataframe(
        "dingwall_bridge",
        VOICES["dingwall_bridge"],
        instrument="37in_multiscale_dingwall",
        mode="difference",
    )
    mags_ding = df_ding["magnitude_db"].to_numpy()
    assert np.all(np.abs(mags_ding) < 0.05), (
        f"Dingwall identity differential not flat: max abs={np.max(np.abs(mags_ding))}"
    )

    # 2. 34" Active StingRay -> Voice 09 Music Man MM (< 0.05 dB flat)
    df_ray = build_voice_dataframe(
        "stingray_parallel",
        VOICES["stingray_parallel"],
        instrument="34in_active_stingray",
        mode="difference",
    )
    mags_ray = df_ray["magnitude_db"].to_numpy()
    assert np.all(np.abs(mags_ray) < 0.05), (
        f"StingRay identity differential not flat: max abs={np.max(np.abs(mags_ray))}"
    )


def test_small_sample_delay_inter_pickup_coherence_decay():
    """Verify that dual-pickup configurations with small inter-pickup sample delay (<= 5 samples) apply coherence decay."""
    import numpy as np

    from allomorph.visualizer import build_voice_dataframe

    # 34" Preamp Soapbar Bass playing 01 Modern Active Jazz Pair has delta_samples = 5.
    # Must apply spatial coherence decay without plunging into unphysical -40 dB razor notches.
    df = build_voice_dataframe(
        "jazz_pair_active",
        VOICES["jazz_pair_active"],
        instrument="34in_preamp_soapbar",
        mode="difference",
    )
    mags = df["magnitude_db"].to_numpy()
    min_db = np.min(mags)
    assert min_db > -20.0, f"Soapbar on Jazz Pair has unregularized comb notch: min={min_db} dB"
    # Mid-scoop depth relative to fundamental passband must be smooth and authentic (8 dB to 16 dB depth)
    scoop_depth = mags[0] - min_db
    assert 8.0 <= scoop_depth <= 16.0, (
        f"Expected smooth authentic acoustic mid-scoop depth, got {scoop_depth} dB"
    )


def test_resolve_instruments():
    """Verify resolve_instruments handles 'all', defaults, comma lists, aliases, and unknown tokens."""
    # 1. 'all' returns all 17 playable instruments
    all_insts = resolve_instruments("all")
    assert len(all_insts) == 17
    expected_17 = {
        "30in_emg_mmtw",
        "30in_mustang_pj",
        "32in_custom_pmm",
        "32in_fretless_pmm",
        "34in_active_pmm",
        "34in_active_stingray",
        "34in_dingwall_sp1",
        "34in_active_emg",
        "34in_preamp_soapbar",
        "34in_standard_jazz",
        "34in_standard_p",
        "34in_standard_pj",
        "37in_multiscale_dingwall",
        "33in_rickenbacker_4003",
        "30in_gibson_eb0",
        "41in_upright_bass",
        "studio_direct",
    }
    assert set(all_insts) == expected_17

    # 2. None, empty string, or whitespace defaults to all playable
    assert resolve_instruments(None) == all_insts
    assert resolve_instruments("") == all_insts
    assert resolve_instruments("   ") == all_insts

    # 3. Comma-separated list with exact IDs and aliases
    res = resolve_instruments("30in, fretless, 34in_standard_p")
    assert res == ["30in_emg_mmtw", "32in_fretless_pmm", "34in_standard_p"]

    # 4. Aliases
    assert resolve_instruments("mustang") == ["30in_mustang_pj"]
    assert resolve_instruments("dingwall") == ["37in_multiscale_dingwall"]
    assert resolve_instruments("preamp_soapbar") == ["34in_preamp_soapbar"]
    assert resolve_instruments("ibanez_sr") == ["34in_preamp_soapbar"]
    assert resolve_instruments("yamaha_trbx") == ["34in_preamp_soapbar"]
    assert resolve_instruments("trbx") == ["34in_preamp_soapbar"]
    assert resolve_instruments("active_emg") == ["34in_active_emg"]
    assert resolve_instruments("emg40") == ["34in_active_emg"]
    assert resolve_instruments("spector5") == ["34in_active_emg"]
    assert resolve_instruments("ray") == ["34in_active_stingray"]

    # 5. Unknown and removed backward compatibility tokens raise diagnostic ValueError (Guardrail 5.3.5)
    for removed_token in [
        "34in_active_soapbar",
        "active_soapbar",
        "34in_emg_soapbar",
        "emg_soapbar",
    ]:
        with pytest.raises(ValueError, match="Unknown source instrument identifier"):
            resolve_instruments(removed_token)

    # Generic substring 'soapbar' matches only preamp soapbar now
    assert resolve_instruments("soapbar") == ["34in_preamp_soapbar"]
    with pytest.raises(ValueError, match="Unknown source instrument identifier 'nonexistent_bass'"):
        resolve_instruments("nonexistent_bass, 30in")

    # 6. Entirely unknown token raises diagnostic ValueError with close match hints
    with pytest.raises(
        ValueError, match="Unknown source instrument identifier 'completely_bogus_token'"
    ):
        resolve_instruments("completely_bogus_token")


def test_instrument_aliases_typing_and_coverage():
    """Verify INSTRUMENT_ALIASES is a dict[str, str] mapping valid aliases to canonical instrument IDs."""
    assert isinstance(INSTRUMENT_ALIASES, dict)
    all_insts = set(load_all_instruments().keys())
    for alias, canonical in INSTRUMENT_ALIASES.items():
        assert isinstance(alias, str)
        assert isinstance(canonical, str)
        assert canonical in all_insts, f"Alias '{alias}' maps to unknown instrument '{canonical}'"


def test_resolve_voices():
    """Verify voice resolution with shorthand, aliases, and comma separation."""
    all_voices = resolve_voices("all")
    assert len(all_voices) == len(VOICES)
    assert all_voices == list(VOICES.keys())

    # Single voice
    assert resolve_voices("precision_active") == ["precision_active"]
    # Shorthand prefix / slug matching
    assert resolve_voices("upright") == ["upright_acoustic"]
    assert resolve_voices("mudbucker") == ["mudbucker_deep"]
    assert resolve_voices("jazz_pair_mids") == ["jazz_pair_mids"]
    assert resolve_voices("studio_act") == ["studio_active"]
    assert resolve_voices("studio") == [
        "studio_direct",
        "studio_active",
        "studio_passive",
    ]
    assert resolve_voices("precision") == [
        "precision_vintage",
        "precision_mids",
        "precision_warm",
        "precision_active",
        "precision_dub",
    ]
    assert resolve_voices("stingray") == [
        "stingray_parallel",
        "stingray_series",
        "stingray_active",
    ]
    assert resolve_voices("pj") == ["pj_passive", "pj_active"]
    assert resolve_voices("p_mm") == ["p_mm_parallel", "p_mm_series"]

    # Unknown voice raises diagnostic ValueError (Guardrail 5.3.5)
    with pytest.raises(ValueError, match="Unknown target voice identifier 'nonexistent_voice'"):
        resolve_voices("nonexistent_voice")


def test_get_source_pickup_strict_errors():
    """Verify that get_source_pickup raises clear configuration errors instead of silent fallbacks."""
    import pytest

    # 1. Empty pickups dictionary
    with pytest.raises(ValueError, match="has no pickups defined"):
        get_source_pickup(InstrumentConfig(id="broken_bass", pickups={}), "precision_active")

    # 2. No default_pickup and no mapping on multi-pickup instrument
    no_default = InstrumentConfig(
        id="no_default_bass",
        pickups={"neck": PickupConfig(name="Neck"), "bridge": PickupConfig(name="Bridge")},
    )
    with pytest.raises(ValueError, match="defines no 'default_pickup' and has no pickup_mapping"):
        get_source_pickup(no_default, "precision_active")

    # 3. default_pickup specifies a non-existent pickup key
    with pytest.raises(KeyError, match="default_pickup 'non_existent' not found in pickups"):
        InstrumentConfig(
            id="bad_default_bass",
            default_pickup="non_existent",
            pickups={"neck": PickupConfig(name="Neck")},
        )


def test_string_preset_strict_errors():
    """Verify that invalid string presets raise KeyError instead of silent fallbacks."""
    import pytest

    from allomorph.config.schema import InstrumentStringsConfig
    from allomorph.config.strings import get_instrument_string, get_voice_string

    with pytest.raises(KeyError, match="String preset 'imaginary_flats' not found"):
        get_instrument_string(
            InstrumentConfig(strings=InstrumentStringsConfig(preset="imaginary_flats"))
        )

    voice = VOICES["precision_active"].model_copy(update={"target_string": "unknown_target_wire"})
    with pytest.raises(KeyError, match="String preset 'unknown_target_wire' not found"):
        get_voice_string(voice)


def test_scale_resolution_strict_errors():
    """Verify that invalid scale parameters raise ValueError/TypeError instead of silent 34in fallback."""
    import pytest

    from allomorph.config.scales import resolve_scale_range

    # 1. Valid None returns standard 34" baseline
    assert resolve_scale_range(None) == (0.8636, 0.8636)

    # 2. Unknown scale name
    with pytest.raises(
        ValueError, match="Unknown scale or instrument identifier '99in_super_bass'"
    ):
        resolve_scale_range("99in_super_bass")

    # 3. Object missing scale keys
    with pytest.raises(ValueError, match="no valid scale specification"):
        resolve_scale_range(
            InstrumentConfig.model_construct(scale_length_in=None, scale_length_m=None)
        )

    # 4. Invalid types
    with pytest.raises(TypeError, match="Cannot resolve scale range"):
        resolve_scale_range(cast(Any, object()))


def test_34in_active_emg_configuration():
    """Verify 34in_active_emg physical geometry, 5-string wave speeds, EMG active parameters, and deconvolution."""
    from allomorph.visualizer import build_voice_dataframe

    inst = load_instrument("34in_active_emg")
    assert inst.id == "34in_active_emg"
    assert inst.scale_length_in == 34.0
    assert inst.scale_length_m == pytest.approx(0.8636)
    assert inst.electronics == "active"
    assert inst.string_wave_speeds == [53.31, 71.16, 95.00, 126.81, 169.27]

    # Explicit alias resolution
    assert load_instrument("active_emg").id == "34in_active_emg"
    assert load_instrument("34in_emg").id == "34in_active_emg"
    assert load_instrument("emg40").id == "34in_active_emg"

    # Deprecated / removed aliases must raise FileNotFoundError
    for removed in ["34in_emg_soapbar", "emg_soapbar"]:
        with pytest.raises(FileNotFoundError):
            load_instrument(removed)

    # Verify physical strings resolution
    s_cfg = get_instrument_string(inst)
    assert s_cfg.preset == "roundwound_nickel_5string"
    assert s_cfg.tension_lbs == 195.0
    assert s_cfg.core == "hex"

    # Verify pickups
    assert "neck" in inst.pickups
    assert "bridge" in inst.pickups
    assert "pair_parallel" in inst.pickups

    neck = inst.pickups["neck"]
    assert neck.resonant_frequency_hz == 4150.0
    assert neck.q_factor == 1.40
    assert neck.magnet_type == "ceramic"
    assert neck.pole_type == "blade"
    assert neck.position_from_bridge_m == 0.1350

    bridge = inst.pickups["bridge"]
    assert bridge.resonant_frequency_hz == 4150.0
    assert bridge.q_factor == 1.40
    assert bridge.magnet_type == "ceramic"
    assert bridge.pole_type == "blade"
    assert bridge.position_from_bridge_m == 0.0550

    pair = inst.pickups["pair_parallel"]
    assert pair.resonant_frequency_hz == 4150.0
    assert pair.q_factor == 1.35
    assert pair.position_from_bridge_m == 0.0950

    # Verify embedded active circuits
    assert neck.circuit is not None
    assert neck.circuit.active is True
    assert neck.circuit.Rvol == pytest.approx(25000.0)
    assert bridge.circuit is not None
    assert bridge.circuit.active is True
    assert bridge.circuit.Rvol == pytest.approx(25000.0)
    assert pair.circuit is not None
    assert pair.circuit.active is True
    assert pair.circuit.Rvol == pytest.approx(25000.0)

    # Verify smart voice mapping coverage
    for vid in VOICES:
        p = get_source_pickup(inst, vid)
        assert p.id in ["neck", "bridge", "pair_parallel"]

    # Verify differential deconvolution curve generation
    for test_vid in ["jazz_pair_active", "precision_vintage", "stingray_parallel"]:
        df = build_voice_dataframe(test_vid, VOICES[test_vid], instrument=inst, mode="difference")
        assert df is not None and len(df) > 0
        mags = df["magnitude_db"].to_numpy()
        assert not np.any(np.isnan(mags))
        assert np.min(mags) > -30.0
        assert np.max(mags) < 30.0


def test_34in_preamp_soapbar_configuration():
    """Verify 34in_preamp_soapbar passive pickup RLC circuits, active buffer isolation, and backward compatibility aliases."""
    from allomorph.circuit import load_circuit
    from allomorph.visualizer import build_voice_dataframe

    inst = load_instrument("34in_preamp_soapbar")
    assert inst.id == "34in_preamp_soapbar"
    assert inst.scale_length_in == 34.0
    assert inst.scale_length_m == pytest.approx(0.8636)
    assert inst.electronics == "active"
    assert len(inst.string_wave_speeds) == 5
    assert inst.string_wave_speeds[0] == pytest.approx(53.31)
    assert inst.strings.preset == "roundwound_nickel_5string"

    # Verify pickups and embedded passive circuits
    assert "neck" in inst.pickups
    assert "bridge" in inst.pickups
    assert "pair_parallel" in inst.pickups

    neck = inst.pickups["neck"]
    assert neck.pole_type == "rod"
    assert neck.magnet_type == "alnico_v"
    assert neck.resonant_frequency_hz == 3800.0
    assert neck.circuit is not None
    assert neck.circuit.active is True
    assert neck.circuit.preamp == "none"
    neck_circ = load_circuit(neck.circuit)
    assert neck_circ.L == pytest.approx(3.8)
    assert neck_circ.Rdc == pytest.approx(8600.0)
    assert neck_circ.Reddy == pytest.approx(95000.0)

    bridge = inst.pickups["bridge"]
    assert bridge.pole_type == "rod"
    assert bridge.magnet_type == "alnico_v"
    assert bridge.resonant_frequency_hz == 4100.0
    assert bridge.circuit is not None
    assert bridge.circuit.active is True
    assert bridge.circuit.preamp == "none"
    bridge_circ = load_circuit(bridge.circuit)
    assert bridge_circ.L == pytest.approx(4.2)
    assert bridge_circ.Rdc == pytest.approx(9400.0)
    assert bridge_circ.Reddy == pytest.approx(90000.0)

    pair = inst.pickups["pair_parallel"]
    assert pair.pole_type == "rod"
    assert pair.magnet_type == "alnico_v"
    assert pair.circuit is not None
    assert pair.circuit.active is True
    assert pair.circuit.preamp == "none"
    assert pair.circuit.Rvol == pytest.approx(500000.0)

    # Explicit aliases
    for alias in [
        "34in_preamp_soapbar",
        "preamp_soapbar",
        "preamp_soapbar_5string",
        "ibanez_sr",
        "ibanez_sr505",
        "sr505",
        "sr505e",
        "yamaha_trbx",
        "yamaha_trbx505",
        "trbx",
        "trbx505",
        "sire_f10",
        "sire_f10_5",
        "sire_m7",
        "sire_m7_5",
        "sire",
        "marcus_miller_f10",
        "stingray_hh",
        "stingray5_hh",
        "ray34hh",
        "ray35hh",
    ]:
        aliased_inst = load_instrument(alias)
        assert aliased_inst.id == "34in_preamp_soapbar"

    # Deprecated / removed aliases must raise FileNotFoundError (no silent backward compatibility fallback)
    for deprecated in ["34in_active_soapbar", "active_soapbar", "soapbar"]:
        with pytest.raises(FileNotFoundError):
            load_instrument(deprecated)

    # Verify differential deconvolution curves
    for test_vid in ["jazz_pair_active", "precision_vintage", "stingray_parallel"]:
        df = build_voice_dataframe(test_vid, VOICES[test_vid], instrument=inst, mode="difference")
        assert df is not None and len(df) > 0
        mags = df["magnitude_db"].to_numpy()
        assert not np.any(np.isnan(mags))
        assert np.min(mags) > -30.0
        assert np.max(mags) < 30.0


def test_calibrated_reference_instruments():
    """Verify calibrated reference instruments: Rick 4003, Dingwall Multiscale, Dingwall SP1, and Mustang PJ."""
    # 1. Rickenbacker 4003
    rick = load_instrument("33in_rickenbacker_4003")
    assert rick.string_wave_speeds == [69.60, 92.90, 124.01, 165.53]
    b = rick.pickups["bridge"]
    b_hpf = rick.pickups["bridge_hpf"]
    assert b.position_from_bridge_m == pytest.approx(0.0570)
    assert b.magnet_type == "ceramic"
    assert b_hpf.position_from_bridge_m == pytest.approx(0.0570)
    assert b_hpf.aperture_width_in == pytest.approx(0.75)
    assert b_hpf.magnet_type == "ceramic"
    assert b_hpf.circuit is not None
    assert b_hpf.circuit.Crick == pytest.approx(4.7e-9)

    # 2. Dingwall 37" Multi-Scale
    ding = load_instrument("37in_multiscale_dingwall")
    assert "pair_series" in ding.pickups
    p_ser = ding.pickups["pair_series"]
    assert p_ser.type == "composite"
    assert p_ser.circuit is not None
    assert p_ser.circuit.topology == "series"
    assert p_ser.circuit.neck is not None and p_ser.circuit.neck.L == pytest.approx(2.3)
    assert p_ser.circuit.bridge is not None and p_ser.circuit.bridge.L == pytest.approx(2.3)
    p_par = ding.pickups["pair_parallel"]
    assert p_par.circuit is not None
    assert p_par.circuit.neck is not None and p_par.circuit.neck.L == pytest.approx(2.3)
    assert p_par.circuit.bridge is not None and p_par.circuit.bridge.L == pytest.approx(2.3)
    assert ding.voicings["pair_series"].pickup == "pair_series"
    assert ding.voicings["pair_series"].gain_db == pytest.approx(5.6)

    # 3. Dingwall SP1 32"-35" Super Multi-Scale
    sp1 = load_instrument("34in_dingwall_sp1")
    sp1_par = sp1.pickups["pair_parallel"]
    assert sp1_par.circuit is not None
    assert sp1_par.circuit.bridge is not None
    assert sp1_par.circuit.bridge.L == pytest.approx(2.3)
    assert sp1_par.circuit.bridge.Rdc == pytest.approx(4400.0)

    # 4. Mustang PJ 30" Short Scale
    mustang = load_instrument("30in_mustang_pj")
    mustang_par = mustang.pickups["pair_parallel"]
    assert mustang_par.circuit is not None
    assert mustang_par.circuit.Rbot == pytest.approx(250000.0)
