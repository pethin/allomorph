"""
Tests for target voices catalog, electrical parameters, SPICE netlist existence, and transparency.
"""

import numpy as np

from allomorph.circuit import compute_circuit_transfer_functions, load_circuit
from allomorph.circuit.schema import CircuitConfig
from allomorph.config import VOICES
from allomorph.config.schema import VoiceConfig
from allomorph.dsp import FREQS, NUM_TAPS
from allomorph.physics import compute_voice_prefilter_firs
from allomorph.visualizer import build_voice_dataframe


def test_voice_parameter_validity():
    """Verify that all target voices are validated VoiceConfig models satisfying physical bounds."""
    for vid, cfg in VOICES.items():
        assert isinstance(cfg, VoiceConfig), f"{vid} is not a VoiceConfig instance"
        assert cfg.name and cfg.topology and cfg.description
        assert cfg.fr > 0
        max_fr = 20000.0 if cfg.preserve_aperture or vid == "studio_direct" else 6000.0
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


def test_studio_direct_properties():
    """Validates that studio_direct preserves aperture and acts as a transparent studio DI."""
    # 1. Prefilter FIR on standard bass preserves physical aperture (unit impulse)
    firs = compute_voice_prefilter_firs("studio_direct", instrument="34in_standard_p")
    assert len(firs) == 1
    fir = np.array(firs[0])
    assert len(fir) == NUM_TAPS
    assert fir[0] == 1.0
    assert np.all(fir[1:] == 0.0)

    # 2. Circuit transfer function must be identically 1.0 across all frequencies (no_eq buffer)
    cfg = VOICES["studio_direct"]
    assert cfg.preserve_aperture is True
    model = load_circuit(cfg.circuit)
    assert getattr(model, "no_eq", False) is True
    curves = compute_circuit_transfer_functions(model, freqs=FREQS)
    assert len(curves) == 1
    assert np.all(np.array(curves[0]) == 1.0)

    # 3. Output mode dataframe must be bit-exact 0.00 dB (flat studio DI target)
    df_out = build_voice_dataframe(
        "studio_direct", cfg, instrument="34in_standard_p", mode="output"
    )
    mags_out = df_out["magnitude_db"].to_numpy()
    assert np.all(mags_out == 0.0)


def test_studio_active_buffer_properties():
    """Validates that studio_active preserves aperture and acts as an active buffer."""
    # 1. Prefilter FIR preserves physical aperture (unit impulse)
    firs = compute_voice_prefilter_firs("studio_active", instrument="34in_standard_p")
    assert len(firs) == 1
    fir = np.array(firs[0])
    assert fir[0] == 1.0
    assert np.all(fir[1:] == 0.0)

    # 2. Netlist models active buffer with flat contour
    cfg = VOICES["studio_active"]
    assert cfg.preserve_aperture is True
    model = load_circuit(cfg.circuit)
    assert model.has_active_buffer is True
    assert model.preamp_type == "none"
    assert model.R_preamp_in >= 1.0e6


def test_studio_passive_properties():
    """Validates that studio_passive preserves aperture and models passive RLC loading."""
    firs = compute_voice_prefilter_firs("studio_passive", instrument="34in_standard_p")
    assert len(firs) == 1
    fir = np.array(firs[0])
    assert fir[0] == 1.0
    assert np.all(fir[1:] == 0.0)

    cfg = VOICES["studio_passive"]
    assert cfg.preserve_aperture is True
    model = load_circuit(cfg.circuit)
    assert model.has_active_buffer is False
    assert model.L == 4.2
    assert model.Rdc == 8500.0
    assert model.Rbot == 250000.0
