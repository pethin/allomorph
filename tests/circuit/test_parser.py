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
            "dingwall_parallel",
        ]:
            assert model.topology == "parallel"
            assert model.L_b > 0
            assert model.Rdc_b > 0
        elif vid in ["p_mm_series", "p_mm_series"]:
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
    assert sweep.name.startswith("optimal_bass_dry") and sweep.suffix == ".wav"

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

        # Test with input_wav pointing to missing file (fallback behavior to optimal_bass_dry.wav)
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
        "ceramic_alnico_hybrid",
        "hybrid",
        "neodymium",
        "piezo",
        "active",
        "ideal",
        "ideal_passive",
        "canonical_ideal",
        "linear",
    ]
    for mag in required_types:
        assert mag in MAGNET_PROPERTIES
        props = MAGNET_PROPERTIES[mag]
        assert isinstance(props, MagnetPropertiesConfig)

    # Specific property checks
    a3 = MAGNET_PROPERTIES["alnico_iii"]
    assert a3.k_body == pytest.approx(0.09)
    assert a3.alpha == pytest.approx(0.30)
    assert a3.k_core == pytest.approx(0.09)
    assert a3.vsat == pytest.approx(0.48)

    neo = MAGNET_PROPERTIES["neodymium"]
    assert neo.k_body == pytest.approx(0.02)
    assert neo.vsat == pytest.approx(0.90)

    act = MAGNET_PROPERTIES["active"]
    assert act.k_body == 0.0

    ideal = MAGNET_PROPERTIES["ideal"]
    assert ideal.alpha == 0.0
    assert ideal.alpha3 == 0.0
    assert ideal.k_sag == 0.0
    assert ideal.k_eddy == 0.0
    assert ideal.k_core == 0.0
    assert ideal.vsat == 10.0
