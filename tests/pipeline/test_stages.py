"""
Tests for pipeline execution stages in allomorph.pipeline.stages.
"""

import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from allomorph.dsp import write_wav_24bit
from allomorph.pipeline.stages import run_circuit_simulation, run_training, run_visualization


@pytest.fixture
def mini_dry_audio(tmp_path: Path) -> Path:
    sr = 48000
    t = np.linspace(0, 0.05, int(sr * 0.05), endpoint=False)
    audio = 0.2 * np.sin(2.0 * np.pi * 100.0 * t)
    p = tmp_path / "test_dry.wav"
    write_wav_24bit(p, audio.astype(np.float32), sample_rate=sr)
    return p


def test_run_visualization_mocked(monkeypatch: pytest.MonkeyPatch):
    """Verify run_visualization constructs proper CLI invocation."""
    calls = []

    def _mock_run(cmd: Any, **kwargs: Any) -> subprocess.CompletedProcess[str]:
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, returncode=0)

    monkeypatch.setattr(subprocess, "run", _mock_run)

    run_visualization(instrument="all")
    assert len(calls) == 1
    assert "--all" in calls[0]

    run_visualization(instrument="30in")
    assert len(calls) == 2
    assert "--instrument" in calls[1]
    assert "30in" in calls[1]


def test_run_circuit_simulation_invalid_backend():
    """Verify invalid backend raises ValueError."""
    with pytest.raises(ValueError, match="Unsupported backend"):
        run_circuit_simulation("vintage_open", backend="spice3")


def test_run_circuit_simulation_success(mini_dry_audio: Path, tmp_path: Path):
    """Verify run_circuit_simulation succeeds with native engine on valid inputs."""
    out_p = tmp_path / "out_sim.wav"
    ok = run_circuit_simulation(
        voice="vintage_open",
        instrument="34in_standard_p",
        input_wav=mini_dry_audio,
        output_wav=out_p,
        max_samples=1200,
    )
    assert ok is True
    assert out_p.exists()


def test_run_circuit_simulation_error_handling(tmp_path: Path):
    """Verify run_circuit_simulation catches errors and returns False."""
    out_p = tmp_path / "out_err.wav"
    ok = run_circuit_simulation(
        voice="non_existent_voice",
        instrument="34in_standard_p",
        input_wav="/non/existent/path.wav",
        output_wav=out_p,
    )
    assert ok is False


def test_run_training_mocked(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Verify run_training constructs proper in-process train_voice invocation."""
    calls: list[dict[str, Any]] = []

    def _mock_train_voice(**kwargs: Any) -> bool:
        calls.append(kwargs)
        return True

    import allomorph.trainer

    monkeypatch.setattr(allomorph.trainer, "train_voice", _mock_train_voice)

    ok = run_training(
        instrument="34in_standard_p",
        voice="vintage_open",
        input_wav=tmp_path / "input.wav",
        output_wav=tmp_path / "output.wav",
        epochs=10,
        min_epochs=5,
        patience=3,
        min_delta=1e-5,
        fast_dev_run=True,
        no_manifest=True,
    )

    assert ok is True
    assert len(calls) == 1
    call = calls[0]
    assert call["instrument"] == "34in_standard_p"
    assert call["voice"] == "vintage_open"
    assert call["epochs"] == 10
    assert call["min_epochs"] == 5
    assert call["patience"] == 3
    assert call["min_delta"] == 1e-5
    assert call["fast_dev_run"] is True
    assert call["no_manifest"] is True
