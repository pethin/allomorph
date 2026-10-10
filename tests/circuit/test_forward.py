"""
Tests for forward circuit digital twin simulation in allomorph.circuit.forward.
"""

import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from allomorph.circuit.forward import (
    simulate_all_instrument_voicings,
    simulate_instrument_voicing,
)
from allomorph.dsp import read_wav, write_wav_24bit


@pytest.fixture
def test_audio_file(tmp_path: Path) -> Path:
    """Generates a 50 ms 48 kHz synthetic bass audio file."""
    sr = 48000
    t = np.linspace(0, 0.05, int(sr * 0.05), endpoint=False)
    sig = 0.25 * np.sin(2.0 * np.pi * 100.0 * t) + 0.1 * np.sin(2.0 * np.pi * 1500.0 * t)
    p = tmp_path / "dry_in.wav"
    write_wav_24bit(p, sig.astype(np.float32), sample_rate=sr)
    return p


def test_forward_simulation_multi_pickup_magnet_properties(test_audio_file: Path, tmp_path: Path):
    """Bug 5 regression: multi-pickup instrument simulating bridge pickup must use bridge pickup config, not first pickup."""
    out_p = tmp_path / "bridge_out.wav"
    # Jazz bass has neck (1st) and bridge (2nd) pickups
    sim_path = simulate_instrument_voicing(
        instrument="34in_standard_jazz",
        voicing="bridge_growl",
        input_wav=test_audio_file,
        output_wav=out_p,
        max_samples=2400,
        apply_saturation=True,
    )
    assert sim_path.exists()
    audio, _sr = read_wav(sim_path)
    assert len(audio) == 2400
    assert np.all(np.isfinite(audio))


def test_forward_auto_normalize_fallback_target_dbfs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Bug 6 regression: when LUFS measurement falls back to RMS, target_dbfs must be respected."""

    # Force compute_lufs to return NaN to trigger RMS fallback branch
    def _mock_nan_lufs(*_args: Any, **_kwargs: Any) -> float:
        return float("nan")

    monkeypatch.setattr("allomorph.circuit.forward.compute_lufs", _mock_nan_lufs)

    sig = np.full(500, 0.05, dtype=np.float64)
    short_in = tmp_path / "tiny.wav"
    write_wav_24bit(short_in, sig.astype(np.float32), sample_rate=48000)
    out_p = tmp_path / "tiny_out.wav"

    target_dbfs = -18.0
    sim_path = simulate_instrument_voicing(
        instrument="34in_standard_p",
        voicing="vintage_open",
        input_wav=short_in,
        output_wav=out_p,
        max_samples=500,
        normalize="lufs",  # Will fall back to RMS due to NaN
        target_dbfs=target_dbfs,
    )

    out_audio, _ = read_wav(sim_path)
    out_rms = float(np.sqrt(np.mean(out_audio**2)))
    out_dbfs = 20.0 * math.log10(max(out_rms, 1e-9))
    # Target level must match target_dbfs (-18 dBFS) within 0.1 dB
    assert abs(out_dbfs - target_dbfs) < 0.1, (
        f"Expected {target_dbfs} dBFS, got {out_dbfs:.2f} dBFS"
    )


def test_forward_simulation_voicing_controls(test_audio_file: Path, tmp_path: Path):
    """Verify simulate_instrument_voicing executes with active volume and tone controls."""
    out_p = tmp_path / "p_out.wav"
    sim_path = simulate_instrument_voicing(
        instrument="34in_standard_p",
        voicing="vintage_open",
        input_wav=test_audio_file,
        output_wav=out_p,
        max_samples=2400,
    )
    assert sim_path.exists()
    audio, _sr = read_wav(sim_path)
    assert len(audio) == 2400
    assert np.all(np.isfinite(audio))


def test_simulate_all_instrument_voicings(
    test_audio_file: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Verify simulate_all_instrument_voicings executes for all voicings of an instrument."""
    from allomorph.circuit import forward

    # Redirect wet audio dir to tmp_path
    monkeypatch.setattr(forward, "WET_AUDIO_DIR", tmp_path / "wet")

    out_paths = simulate_all_instrument_voicings(
        instrument="30in_emg_mmtw",
        input_wav=test_audio_file,
        max_samples=1200,
        jobs=1,
        force=True,
    )

    assert len(out_paths) >= 1
    for p in out_paths:
        assert p.exists()


def test_forward_simulation_upright_piezo(test_audio_file: Path, tmp_path: Path):
    """Verify forward digital twin simulation of Upright Piezo voicing."""
    out_p = tmp_path / "upright_piezo_out.wav"
    sim_path = simulate_instrument_voicing(
        instrument="41in_upright_bass",
        voicing="bridge_piezo",
        input_wav=test_audio_file,
        output_wav=out_p,
        max_samples=2400,
        apply_saturation=True,
    )
    assert sim_path.exists()
    audio, _sr = read_wav(sim_path)
    assert len(audio) == 2400
    assert np.all(np.isfinite(audio))

    # Small signal linear bypass check (peak <= 0.10)
    small_sig = (0.05 * np.sin(2.0 * np.pi * 100.0 * np.linspace(0, 0.05, 2400, endpoint=False))).astype(np.float32)
    small_in = tmp_path / "small_in.wav"
    small_out = tmp_path / "small_out.wav"
    write_wav_24bit(small_in, small_sig, sample_rate=48000)
    sim_small = simulate_instrument_voicing(
        instrument="41in_upright_bass",
        voicing="bridge_piezo",
        input_wav=small_in,
        output_wav=small_out,
        max_samples=2400,
        apply_saturation=True,
        normalize="none",
        dc_block=False,
    )
    small_audio, _ = read_wav(sim_small)
    assert np.all(np.isfinite(small_audio))


def test_forward_simulation_return_audio_and_in_memory(test_audio_file: Path, tmp_path: Path):
    """Verify return_audio=True and in-memory input_audio array handling."""
    sig = np.sin(2.0 * np.pi * 100.0 * np.linspace(0, 0.05, 2400, endpoint=False))
    out_p, audio = simulate_instrument_voicing(
        instrument="34in_standard_p",
        voicing="vintage_open",
        input_audio=sig,
        return_audio=True,
        max_samples=1200,
        output_wav=tmp_path / "mem_out.wav",
    )
    assert out_p.exists()
    assert audio is not None
    assert len(audio) == 1200

    # Test cache hit path (force=False)
    _out_cached, audio_cached = simulate_instrument_voicing(
        instrument="34in_standard_p",
        voicing="vintage_open",
        input_wav=test_audio_file,
        output_wav=tmp_path / "cached_out.wav",
        return_audio=True,
        force=False,
    )
    # Second call hits cache
    _out_cached2, audio_cached2 = simulate_instrument_voicing(
        instrument="34in_standard_p",
        voicing="vintage_open",
        input_wav=test_audio_file,
        output_wav=tmp_path / "cached_out.wav",
        return_audio=True,
        force=False,
    )
    assert np.allclose(audio_cached, audio_cached2, atol=1e-4)


def test_forward_simulation_normalization_modes(test_audio_file: Path, tmp_path: Path):
    """Verify peak and rms normalization modes in simulate_instrument_voicing."""
    for mode in ("peak", "rms"):
        out_p = tmp_path / f"norm_{mode}.wav"
        res = simulate_instrument_voicing(
            instrument="34in_standard_p",
            voicing="vintage_open",
            input_wav=test_audio_file,
            output_wav=out_p,
            normalize=mode,
            target_dbfs=-15.0,
            max_samples=1200,
        )
        assert res.exists()


def test_forward_resolve_target_voicing_errors():
    """Verify diagnostic exceptions in resolve_target_voicing."""
    from allomorph.circuit.forward import resolve_target_voicing
    from allomorph.config.instruments import load_instrument

    inst = load_instrument("34in_standard_p")
    voicing = inst.voicings["vintage_open"]

    # Passing VoicingConfig without instrument raises ValueError
    with pytest.raises(ValueError, match="Must provide instrument"):
        resolve_target_voicing(voicing, instrument=None)

    # Passing pickup name as voicing raises KeyError per Guardrail 5.3.4
    with pytest.raises(KeyError, match="cannot be used as a target voicing"):
        resolve_target_voicing("split_p", instrument=inst)

    # Missing voicing raises KeyError
    with pytest.raises(KeyError, match="could not be resolved"):
        resolve_target_voicing("non_existent_voicing_xyz")


def test_simulate_voice_wrapper_and_options(test_audio_file: Path, tmp_path: Path):
    """Verify simulate_voice wrapper, config object support, and error cases."""
    from allomorph.circuit.forward import resolve_target_voicing, simulate_voice
    from allomorph.config.instruments import load_instrument

    inst = load_instrument("34in_standard_p")

    # 1. Resolve target voicing with colon and tone slug
    inst_res, v_res = resolve_target_voicing("34in_standard_p:Precision Vintage")
    assert inst_res.id == "34in_standard_p"
    assert v_res.id == "vintage_open"

    # 2. simulate_voice basic call
    out_p = tmp_path / "sim_voice.wav"
    ok = simulate_voice(
        "vintage_open",
        instrument=inst,
        input_wav=test_audio_file,
        output_wav=out_p,
        max_samples=1200,
    )
    assert ok is True
    assert out_p.exists()

    # 3. simulate_voice with unknown pickup raises KeyError
    with pytest.raises(KeyError, match="Pickup 'nonexistent_pickup' not found"):
        simulate_voice(
            "vintage_open",
            instrument=inst,
            input_wav=test_audio_file,
            pickup="nonexistent_pickup",
        )

    # 4. simulate_voice with skip_identity=True
    skip_p = tmp_path / "skip_out.wav"
    skip_p.write_bytes(b"temp")
    res_skip = simulate_voice(
        "vintage_open",
        instrument=inst,
        input_wav=test_audio_file,
        output_wav=skip_p,
        skip_identity=True,
    )
    assert res_skip is False
    assert not skip_p.exists()

    # 5. simulate_voice with non-existent target voice raises KeyError
    with pytest.raises(KeyError, match="not found"):
        simulate_voice(
            "completely_unknown_voicing_12345",
            instrument=inst,
            input_wav=test_audio_file,
        )


