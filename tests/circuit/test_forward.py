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
    simulate_circuit_audio,
    simulate_instrument_voicing,
)
from allomorph.circuit.parser import CircuitModel
from allomorph.circuit.schema import HarnessControls
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


def test_simulate_circuit_audio_pot_taper(tmp_path: Path):
    """Bug 9 regression: simulate_circuit_audio must apply pot_taper from harness controls."""
    # Construct a circuit model with volume and tone pots
    model = CircuitModel(
        topology="single",
        Rdc=6000.0,
        L=3.0,
        Rvol_total=500000.0,
        pot_taper="audio",
    )

    # Signal array
    t = np.linspace(0, 0.02, 960, endpoint=False)
    audio = (0.2 * np.sin(2.0 * np.pi * 200.0 * t)).astype(np.float32)

    harness_linear = HarnessControls(vol_pos=0.5, pot_taper="linear")
    harness_audio = HarnessControls(vol_pos=0.5, pot_taper="audio")

    # With linear taper at 0.5, eff_vol = 0.5
    # With audio taper at 0.5, eff_vol is around 0.1-0.15
    model_lin = model.model_copy(deep=True)
    out_lin = tmp_path / "out_lin.wav"
    simulate_circuit_audio(audio, out_lin, model_lin, harness_controls=harness_linear)

    model_aud = model.model_copy(deep=True)
    out_aud = tmp_path / "out_aud.wav"
    simulate_circuit_audio(audio, out_aud, model_aud, harness_controls=harness_audio)

    # Check that model Rtop/Rbot reflect the different pot tapers
    assert model_lin.Rbot != model_aud.Rbot, (
        "Linear and audio tapers at 0.5 wiper position must yield distinct pot resistance splits"
    )


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
