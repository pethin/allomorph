"""
Tests for concurrent batch circuit simulation in allomorph.pipeline.batch.
"""

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from allomorph.dsp import write_wav_24bit
from allomorph.pipeline.batch import _run_circuit_simulation_task, run_spice_batch


@pytest.fixture
def short_dry_audio(tmp_path: Path) -> Path:
    """Creates a 50 ms dry audio test file."""
    sr = 48000
    t = np.linspace(0, 0.05, int(sr * 0.05), endpoint=False)
    audio = 0.2 * np.sin(2.0 * np.pi * 100.0 * t)
    p = tmp_path / "test_dry.wav"
    write_wav_24bit(p, audio.astype(np.float32), sample_rate=sr)
    return p


def test_run_spice_batch_invalid_backend():
    """Verify non-native backend raises ValueError."""
    with pytest.raises(ValueError, match="Unsupported backend"):
        run_spice_batch(backend="ltspice")


def test_run_circuit_simulation_task(short_dry_audio: Path, tmp_path: Path):
    """Verify top-level task runner completes successfully."""
    out_wav = tmp_path / "out_task.wav"
    task_args = ("vintage_open", "34in_standard_p", short_dry_audio, 1200, out_wav)
    voice, success = _run_circuit_simulation_task(task_args)
    assert voice == "vintage_open"
    assert success is True
    assert out_wav.exists()


def test_run_spice_batch_sequential(short_dry_audio: Path, tmp_path: Path):
    """Verify sequential batch simulation with 1 voice."""
    ok = run_spice_batch(
        voices=["vintage_open"],
        instrument="34in_standard_p",
        input_wav=short_dry_audio,
        max_samples=1200,
        output_dir=tmp_path,
        jobs=1,
    )
    assert ok is True
    assert (tmp_path / "out_vintage_open.wav").exists()


def test_run_spice_batch_parallel(short_dry_audio: Path, tmp_path: Path):
    """Verify parallel batch simulation with 2 voices."""
    ok = run_spice_batch(
        voices=["vintage_open", "vintage_mids"],
        instrument="34in_standard_p",
        input_wav=short_dry_audio,
        max_samples=1200,
        output_dir=tmp_path,
        jobs=2,
    )
    assert ok is True
    assert (tmp_path / "out_vintage_open.wav").exists()
    assert (tmp_path / "out_vintage_mids.wav").exists()


def _failing_task_for_test(task_args: Any) -> None:
    raise KeyError("simulated_missing_magnet_property")


def test_run_spice_batch_worker_exception_handling(
    short_dry_audio: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Bug 11 regression: worker raising KeyError must be caught gracefully and not crash process pool."""
    monkeypatch.setattr(
        "allomorph.pipeline.batch._run_circuit_simulation_task", _failing_task_for_test
    )

    ok = run_spice_batch(
        voices=["vintage_open", "vintage_mids"],
        instrument="34in_standard_p",
        input_wav=short_dry_audio,
        max_samples=1200,
        output_dir=tmp_path,
        jobs=2,
    )
    assert ok is False


def test_simulate_all_instrument_voicings(
    short_dry_audio: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Verify simulate_all_instrument_voicings executes for all voicings of an instrument."""
    from allomorph.circuit import forward
    from allomorph.pipeline.batch import simulate_all_instrument_voicings

    # Redirect wet audio dir to tmp_path
    monkeypatch.setattr(forward, "WET_AUDIO_DIR", tmp_path / "wet")

    out_paths = simulate_all_instrument_voicings(
        instrument="30in_emg_mmtw",
        input_wav=short_dry_audio,
        max_samples=1200,
        jobs=1,
        force=True,
    )

    assert len(out_paths) >= 1
    for p in out_paths:
        assert p.exists()


def test_simulate_voice_wrapper_and_options(short_dry_audio: Path, tmp_path: Path):
    """Verify simulate_voice wrapper, config object support, and error cases."""
    from allomorph.config.instruments import load_instrument, resolve_target_voicing
    from allomorph.pipeline.batch import simulate_voice

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
        input_wav=short_dry_audio,
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
            input_wav=short_dry_audio,
            pickup="nonexistent_pickup",
        )

    # 4. simulate_voice with skip_identity=True
    skip_p = tmp_path / "skip_out.wav"
    skip_p.write_bytes(b"temp")
    res_skip = simulate_voice(
        "vintage_open",
        instrument=inst,
        input_wav=short_dry_audio,
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
            input_wav=short_dry_audio,
        )
