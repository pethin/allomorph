"""
Tests for circuit parser, netlist validation, and audio discovery utilities.
"""

import tempfile
from pathlib import Path

import pytest

from allomorph.circuit import (
    AUDIO_DIR,
    CircuitModel,
    find_default_input_audio,
    load_circuit,
    parse_spice_val,
    simulate_voice,
)
from allomorph.circuit.parser import MAGNET_PROPERTIES
from allomorph.circuit.schema import CircuitConfig, MagnetPropertiesConfig
from allomorph.config import VOICES, load_instrument


def test_parse_spice_val():
    assert parse_spice_val("4.8") == pytest.approx(4.8)
    assert parse_spice_val("9.5k") == pytest.approx(9500.0)
    assert parse_spice_val("110k") == pytest.approx(110000.0)
    assert parse_spice_val("80p") == pytest.approx(80e-12)
    assert parse_spice_val("47n") == pytest.approx(47e-9)
    assert parse_spice_val("20u") == pytest.approx(20e-6)
    assert parse_spice_val("1Meg") == pytest.approx(1e6)
    assert parse_spice_val("10") == pytest.approx(10.0)


def test_load_all_voice_circuits():
    for vid, cfg in VOICES.items():
        assert cfg.circuit is not None, f"Voice {vid} missing [circuit] configuration"
        model = load_circuit(cfg.circuit)

        assert model.L > 0
        assert model.Rdc > 0
        assert model.Reddy > 0
        assert model.Ccoil > 0
        assert model.Ccable > 0

        if model.has_active_buffer:
            assert model.R_out > 0
            assert model.R_preamp_in > 0
            assert model.C_preamp_in > 0
        else:
            assert model.Rtop > 0
            assert model.Rbot > 0

        if vid in [
            "jazz_pair_active",
            "jazz_pair_open",
            "jazz_pair_mids",
            "jazz_bridge_growl",
            "pj_active",
            "pj_passive",
            "p_mm_parallel",
            "stingray_parallel",
            "dingwall_parallel",
            "soapbar_pair",
            "active_emg_pair",
        ]:
            assert model.topology == "parallel"
            assert model.L_b > 0
            assert model.Rdc_b > 0
        elif vid in ["p_mm_series", "stingray_series", "dingwall_series"]:
            assert model.topology == "series"
            assert model.L_b > 0
            assert model.Rdc_b > 0
        else:
            assert model.topology == "single"


def test_circuit_from_dict_and_shorthand():
    """Verify CircuitModel.from_dict parses shorthand strings and nested tables properly."""
    cfg = {
        "topology": "parallel",
        "neck": {
            "L": "3.2",
            "Rdc": "7.2k",
            "Reddy": "135k",
            "Ccoil": "70p",
            "vsat": 0.45,
        },
        "bridge": {
            "L": "3.6",
            "Rdc": "7.8k",
            "Reddy": "125k",
            "Ccoil": "70p",
            "vsat": 0.55,
        },
        "Rvol": "500k",
        "Rtone": "250k",
        "Ctone": "47n",
        "Ccable": "750p",
        "active": True,
        "preamp": "sadowsky_2band",
    }
    m = CircuitModel.from_dict(cfg)
    assert m.topology == "parallel"
    assert m.L == pytest.approx(3.2)
    assert m.Rdc == pytest.approx(7200.0)
    assert m.Reddy == pytest.approx(135000.0)
    assert m.Ccoil == pytest.approx(70e-12)
    assert m.vsat_n == pytest.approx(0.45)
    assert m.L_b == pytest.approx(3.6)
    assert m.Rdc_b == pytest.approx(7800.0)
    assert m.Reddy_b == pytest.approx(125000.0)
    assert m.Ccoil_b == pytest.approx(70e-12)
    assert m.vsat_b == pytest.approx(0.55)
    assert m.Rbot_default == pytest.approx(500000.0)
    assert m.Rtone == pytest.approx(250000.0)
    assert m.Ctone == pytest.approx(47e-9)
    assert m.Ccable == pytest.approx(750e-12)
    assert m.has_active_buffer is True
    assert m.preamp_type == "sadowsky_2band"

    # Direct validation into CircuitConfig model and from_circuit_config
    c_cfg = CircuitConfig.model_validate(cfg)
    m_from_cfg = CircuitModel.from_circuit_config(c_cfg)
    assert m_from_cfg.topology == "parallel"
    assert m_from_cfg.L == pytest.approx(3.2)
    assert m_from_cfg.L_b == pytest.approx(3.6)
    assert m_from_cfg.preamp_type == "sadowsky_2band"
    assert m.Ccoil_b == pytest.approx(70e-12)
    assert m.vsat_b == pytest.approx(0.55)
    assert m.Rtop == pytest.approx(10.0)
    assert m.Rbot == pytest.approx(500000.0)
    assert m.Rtone == pytest.approx(250000.0)
    assert m.Ctone == pytest.approx(47e-9)
    assert m.Ccable == pytest.approx(750e-12)
    assert m.has_active_buffer is True
    assert m.preamp_type == "sadowsky_2band"


def test_default_output_directories():
    """Verify that default outputs are stored in audio/<inst_id>/."""
    inst_cfg = load_instrument("30in")
    inst_id = inst_cfg.id
    assert inst_id == "30in_emg_mmtw"
    inst_audio_dir = AUDIO_DIR / inst_id
    assert inst_audio_dir.parent == AUDIO_DIR


def test_sweep_audio_auto_detection():
    """Verify that simulate_voice automatically finds optimal_bass_dry.wav even if given None or missing input path."""
    sweep = find_default_input_audio()
    assert sweep is not None
    assert sweep.exists()
    assert sweep.name == "input.wav" and sweep.suffix == ".wav"

    with tempfile.TemporaryDirectory() as tmpdir:
        out_wav = Path(tmpdir) / "auto_sweep_out.wav"
        # Test with input_wav=None
        res = simulate_voice(
            "precision_active",
            input_wav=None,
            output_wav=out_wav,
            instrument="30in",
            max_samples=4800,
        )
        assert res is True
        assert out_wav.exists() and out_wav.stat().st_size > 1000

        # Test with input_wav pointing to missing file (fallback behavior to input.wav)
        out_wav_fallback = Path(tmpdir) / "fallback_sweep_out.wav"
        res_fallback = simulate_voice(
            "precision_active",
            input_wav="missing_sweep.wav",
            output_wav=out_wav_fallback,
            instrument="30in",
            max_samples=4800,
        )
        assert res_fallback is True
        assert out_wav_fallback.exists() and out_wav_fallback.stat().st_size > 1000


def test_magnet_properties_configuration():
    """Verify MAGNET_PROPERTIES defines strongly typed MagnetPropertiesConfig for all magnet types."""
    required_types = [
        "alnico_v",
        "alnico_ii",
        "alnico_iii",
        "ceramic",
        "ceramic_steel",
        "ceramic_and_steel",
        "ceramic_alnico_hybrid",
        "hybrid",
        "neodymium",
        "piezo",
        "active",
        "ideal",
        "ideal_passive",
        "linear",
    ]
    for mag in required_types:
        assert mag in MAGNET_PROPERTIES
        props = MAGNET_PROPERTIES[mag]
        assert isinstance(props, MagnetPropertiesConfig)

    cs = MAGNET_PROPERTIES["ceramic_steel"]
    assert cs.k_eddy == pytest.approx(0.060)
    assert cs.vsat == pytest.approx(1.28)
    assert cs.alpha == pytest.approx(0.15)

    # Specific property checks
    a3 = MAGNET_PROPERTIES["alnico_iii"]
    assert a3.k_body == pytest.approx(0.09)
    assert a3.alpha == pytest.approx(0.30)
    assert a3.k_core == pytest.approx(0.09)
    assert a3.vsat == pytest.approx(0.96)

    neo = MAGNET_PROPERTIES["neodymium"]
    assert neo.k_body == pytest.approx(0.02)
    assert neo.vsat == pytest.approx(1.80)

    act = MAGNET_PROPERTIES["active"]
    assert act.k_body == 0.0

    ideal = MAGNET_PROPERTIES["ideal"]
    assert ideal.alpha == 0.0
    assert ideal.alpha3 == 0.0
    assert ideal.k_sag == 0.0
    assert ideal.k_eddy == 0.0
    assert ideal.k_core == 0.0
    assert ideal.vsat == 20.0


def test_load_circuit_formats_and_errors():
    from allomorph.circuit.parser import load_circuit, parse_netlist
    from allomorph.config.schema import PickupConfig

    # 1. From existing CircuitModel (copy)
    c_base = load_circuit("precision_vintage")
    c_copy = load_circuit(c_base)
    assert c_copy is not c_base
    assert c_copy.L == c_base.L

    # 2. From instrument ID string
    c_inst = load_circuit("34in_standard_p")
    assert c_inst.L > 0

    # 3. Pickup without circuit raises ValueError
    p_no_circ = PickupConfig(name="No Circuit", type="single", circuit=None)
    with pytest.raises(ValueError, match="has no embedded circuit configuration"):
        load_circuit(p_no_circ)

    # 4. Deprecated .cir files raise ValueError
    with pytest.raises(ValueError, match="Legacy SPICE ASCII netlists .* are deprecated"):
        load_circuit("legacy_circuit.cir")

    # 5. Non-existent circuit string raises ValueError
    with pytest.raises(ValueError, match="Could not load circuit from"):
        load_circuit("completely_unknown_circuit_source_123")

    # 6. parse_netlist delegates directly to load_circuit
    assert parse_netlist("precision_vintage").L == c_base.L


def test_circuit_config_detailed_properties():
    """Verify parser correctly maps all optional circuit properties and controls."""
    cfg = {
        "topology": "parallel",
        "L": 3.0,
        "L_core": 0.5,
        "R_core": 500.0,
        "Rdc": 6000.0,
        "Reddy": 150000.0,
        "Ccoil": 60e-12,
        "L_b": 3.5,
        "L_core_b": 0.6,
        "R_core_b": 600.0,
        "Rdc_b": 7000.0,
        "Reddy_b": 140000.0,
        "Ccoil_b": 70e-12,
        "Rvol": 250000.0,
        "Rtone": 250000.0,
        "Ctone": 47e-9,
        "series_hpf_cap_nf": 3.3,
        "Ctb": 1e-9,
        "Rtb_par": 150000.0,
        "Rtb_ser": 1000.0,
        "Rpot_n": 250000.0,
        "Rpot_b": 250000.0,
        "active": True,
        "preamp_gain": 2.0,
        "R_preamp_in": 1e6,
        "C_preamp_in": 10e-12,
        "R_out": 100.0,
        "no_eq": True,
        "tan_delta": 0.02,
        "tan_delta_coil": 0.015,
        "Ranagram": 1e6,
        "Canagram": 22e-12,
        "alpha_dielectric_tone": 0.98,
        "alpha_dielectric_cable": 0.99,
        "chi_mu": 0.2,
        "chi_mu_b": 0.25,
        "omega_mu": 20000.0,
        "k_dist": 0.1,
        "k_dist_b": 0.12,
        "omega_dist": 15000.0,
        "k_skin": 0.05,
        "f_skin": 5000.0,
        "k_skin_b": 0.06,
        "f_skin_b": 5500.0,
        "vol_pos": 0.8,
        "tone_pos": 0.5,
        "blend_pos": 0.3,
        "pot_taper": "linear",
    }
    m = CircuitModel.from_dict(cfg)
    assert m.L_core == pytest.approx(0.5)
    assert m.R_core == pytest.approx(500.0)
    assert m.L_core_b == pytest.approx(0.6)
    assert m.R_core_b == pytest.approx(600.0)
    assert m.Crick == pytest.approx(3.3e-9)
    assert m.Ctb == pytest.approx(1e-9)
    assert m.Rtb_par == pytest.approx(150000.0)
    assert m.Rtb_ser == pytest.approx(1000.0)
    assert m.Rpot_n_default == pytest.approx(250000.0)
    assert m.Rpot_b_default == pytest.approx(250000.0)
    assert m.preamp_gain == pytest.approx(2.0)
    assert m.R_preamp_in == pytest.approx(1e6)
    assert m.C_preamp_in == pytest.approx(10e-12)
    assert m.R_out == pytest.approx(100.0)
    assert m.no_eq is True
    assert m.tan_delta == pytest.approx(0.02)
    assert m.tan_delta_coil == pytest.approx(0.015)
    assert m.Ranagram == pytest.approx(1e6)
    assert m.Canagram == pytest.approx(22e-12)
    assert m.alpha_dielectric_tone == pytest.approx(0.98)
    assert m.alpha_dielectric_cable == pytest.approx(0.99)
    assert m.chi_mu == pytest.approx(0.2)
    assert m.k_skin == pytest.approx(0.05)
    assert m.pot_taper == "linear"
