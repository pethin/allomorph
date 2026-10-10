"""
Tests for circuit parser, netlist validation, and audio discovery utilities.
"""

import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from allomorph.config.schema import InstrumentConfig

from allomorph.circuit import (
    AUDIO_DIR,
    eval_pot_taper,
    find_default_input_audio,
    parse_spice_val,
    simulate_instrument_voicing,
)
from allomorph.circuit.parser import MAGNET_PROPERTIES
from allomorph.circuit.schema import MagnetPropertiesConfig


def test_parse_spice_val():
    assert parse_spice_val("4.8") == pytest.approx(4.8)
    assert parse_spice_val("9.5k") == pytest.approx(9500.0)
    assert parse_spice_val("110k") == pytest.approx(110000.0)
    assert parse_spice_val("80p") == pytest.approx(80e-12)
    assert parse_spice_val("47n") == pytest.approx(47e-9)
    assert parse_spice_val("20u") == pytest.approx(20e-6)
    assert parse_spice_val("1Meg") == pytest.approx(1e6)
    assert parse_spice_val("10") == pytest.approx(10.0)


def test_default_output_directories(generic_instrument_config: InstrumentConfig):
    """Verify that default outputs are stored in audio/<inst_id>/."""
    inst_id = generic_instrument_config.id
    inst_audio_dir = AUDIO_DIR / inst_id
    assert inst_audio_dir.parent == AUDIO_DIR


def test_sweep_audio_auto_detection(generic_instrument_config: InstrumentConfig):
    """Verify that simulate_instrument_voicing automatically finds default input audio if given None or missing input path."""
    sweep = find_default_input_audio()
    assert sweep is not None
    assert sweep.exists()
    assert sweep.name == "input.wav" and sweep.suffix == ".wav"

    with tempfile.TemporaryDirectory() as tmpdir:
        out_wav = Path(tmpdir) / "auto_sweep_out.wav"
        # Test with input_wav=None
        res = simulate_instrument_voicing(
            instrument=generic_instrument_config,
            voicing=generic_instrument_config.voicings["generic_voice"],
            input_wav=None,
            output_wav=out_wav,
            max_samples=4800,
        )
        assert res.exists() and res.stat().st_size > 1000

        # Test with input_wav pointing to missing file (fallback behavior to input.wav)
        out_wav_fallback = Path(tmpdir) / "fallback_sweep_out.wav"
        res_fallback = simulate_instrument_voicing(
            instrument=generic_instrument_config,
            voicing=generic_instrument_config.voicings["generic_voice"],
            input_wav="missing_sweep.wav",
            output_wav=out_wav_fallback,
            max_samples=4800,
        )
        assert res_fallback.exists() and res_fallback.stat().st_size > 1000


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


def test_eval_pot_taper_pot_curves():
    """Verify eval_pot_taper calculates accurate continuous pot resistance fractions."""
    # CTS audio 10%
    assert eval_pot_taper(0.0, "audio") == 0.0
    assert eval_pot_taper(1.0, "audio") == 1.0
    assert eval_pot_taper(0.5, "audio") == pytest.approx(0.10, abs=1e-3)

    # Bourns audio 15%
    assert eval_pot_taper(0.5, "audio15") == pytest.approx(0.15, abs=1e-3)

    # Linear
    assert eval_pot_taper(0.5, "linear") == pytest.approx(0.50)
