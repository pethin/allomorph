"""
Tests for target voices catalog, electrical parameters, SPICE netlist existence, and transparency.
"""

import numpy as np

from allomorph.circuit.solver import solve_mna_harness
from allomorph.config import VOICES
from allomorph.config.instruments import load_instrument
from allomorph.config.schema import VoiceConfig
from allomorph.dsp import FREQS


def test_voice_parameter_validity():
    """Verify that all target voices are validated VoiceConfig models satisfying physical bounds."""
    for vid, cfg in VOICES.items():
        assert isinstance(cfg, VoiceConfig), f"{vid} is not a VoiceConfig instance"
        assert cfg.name and cfg.description
        assert cfg.fr > 0
        is_active = bool(cfg.instrument_id and "active" in cfg.instrument_id) or cfg.sensor_type == "direct"
        max_fr = 20000.0 if (cfg.preserve_aperture or is_active) else 6000.0
        assert 200.0 <= cfg.fr <= max_fr, f"{vid} fr outside audible musical range: {cfg.fr}"
        assert cfg.Q > 0
        assert len(cfg.coils) >= 1
        for c in cfg.coils:
            assert c.position_from_bridge_m > 0
            assert c.aperture_width_in > 0
            assert c.weight > 0
            assert len(c.strings) >= 1

        if cfg.pickups:
            assert len(cfg.pickups) >= 2, f"{vid} pickups list has fewer than 2 pickups"
            for p in cfg.pickups:
                assert p.name
                assert p.fr > 0
                assert p.Q > 0
                assert p.weight > 0
                assert len(p.coils) >= 1


def test_voice_netlist_existence():
    """Verify that every target voice defines a valid instrument harness and resolves in MNA."""
    for vid, cfg in VOICES.items():
        assert cfg.instrument_id is not None, f"{vid} missing instrument_id"
        inst = load_instrument(cfg.instrument_id)
        assert inst is not None, f"Instrument '{cfg.instrument_id}' could not be loaded for {vid}"
        assert inst.voicings, f"Instrument '{cfg.instrument_id}' has no voicings"
        v_match = None
        for v in inst.voicings.values():
            if (v.tone_name and v.tone_name == cfg.tone_name) or v.id == vid or v.name == cfg.name:
                v_match = v
                break
        if v_match is None:
            v_match = next(iter(inst.voicings.values()))
        assert v_match.harness in inst.harnesses, f"Harness '{v_match.harness}' missing from {inst.id}"
        harness = inst.harnesses[v_match.harness]
        curves = solve_mna_harness(inst, harness, v_match, freqs=FREQS)
        assert len(curves) >= 1
        for c in curves.values():
            assert len(c) == len(FREQS)
            assert np.all(np.isfinite(c))


def test_voices_have_no_hardcoded_source_datums():
    # Target voices should be purely decoupled from source instrument geometries
    for vid, cfg in VOICES.items():
        assert not hasattr(cfg, "src_32"), f"{vid} contains deprecated hardcoded 'src_32' datum"
        assert not hasattr(cfg, "src_30"), f"{vid} contains deprecated hardcoded 'src_30' datum"


def test_multi_magnet_target_voicing_composite_formatting():
    """Verify that multi-pickup voicings with distinct magnets format composite magnet_type cleanly."""
    from allomorph.config.voices import voicing_to_voice_config

    inst = load_instrument("32in_custom_pmm")
    v_blend = voicing_to_voice_config(inst, inst.voicings["blend_parallel"])
    assert v_blend.magnet_type == "ceramic / ceramic_steel"
    assert v_blend.pickups is not None
    assert len(v_blend.pickups) == 2
    assert v_blend.pickups[0].magnet_type == "ceramic"
    assert v_blend.pickups[1].magnet_type == "ceramic_steel"

    # Single pickup voicing preserves that pickup's magnet
    v_solo = voicing_to_voice_config(inst, inst.voicings["px_solo"])
    assert v_solo.magnet_type == "ceramic"


def test_voice_aliases_resolution():
    """Verify that voice aliases resolve cleanly to valid target VoiceConfigs."""
    assert "precision_active" in VOICES
    assert VOICES["precision_active"].id == "precision_active"
    assert "p_active" in VOICES
    assert VOICES["p_active"].id == "precision_active"
    assert "j_active" in VOICES
    assert VOICES["j_active"].id == "jazz_pair_active"
    assert "pj_pair_active" in VOICES
    assert VOICES["pj_pair_active"].id == "pj_active"
