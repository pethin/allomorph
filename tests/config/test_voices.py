"""
Tests for target voices catalog, electrical parameters, SPICE netlist existence, and transparency.
"""

from allomorph.circuit import load_circuit
from allomorph.circuit.schema import CircuitConfig
from allomorph.config import VOICES
from allomorph.config.schema import VoiceConfig


def test_voice_parameter_validity():
    """Verify that all target voices are validated VoiceConfig models satisfying physical bounds."""
    for vid, cfg in VOICES.items():
        assert isinstance(cfg, VoiceConfig), f"{vid} is not a VoiceConfig instance"
        assert cfg.name and cfg.topology and cfg.description
        assert cfg.fr > 0
        max_fr = 20000.0 if cfg.preserve_aperture else 6000.0
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
    """Verify that every target voice defines a valid CircuitConfig model that parses into a CircuitModel."""
    for vid, cfg in VOICES.items():
        assert isinstance(cfg.circuit, CircuitConfig), f"{vid} missing CircuitConfig"
        m = load_circuit(cfg.circuit)
        assert m.L > 0


def test_voices_have_no_hardcoded_source_datums():
    # Target voices should be purely decoupled from source instrument geometries
    for vid, cfg in VOICES.items():
        assert not hasattr(cfg, "src_32"), f"{vid} contains deprecated hardcoded 'src_32' datum"
        assert not hasattr(cfg, "src_30"), f"{vid} contains deprecated hardcoded 'src_30' datum"
