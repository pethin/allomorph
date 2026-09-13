"""
Unit tests for Allomorph --stage bake identity model skipping.
Verifies that bit-for-bit identity voices (H_diff == 1.0) are omitted from
baked exports when skip_identity=True, while transformative voices are produced.
"""

from pathlib import Path

import pytest

from allomorph.circuit.schema import SimulationConfig
from allomorph.circuit.simulation import simulate_voice


def test_simulation_config_skip_identity_field():
    """Verify SimulationConfig has skip_identity field defaulting to False."""
    cfg = SimulationConfig()
    assert cfg.skip_identity is False

    cfg_skip = SimulationConfig(skip_identity=True)
    assert cfg_skip.skip_identity is True


def test_simulate_voice_skips_identity_when_flag_set(tmp_path: Path):
    """Verify simulate_voice returns False and removes output file when skip_identity=True on identity pair."""
    out_wav = tmp_path / "stingray_parallel.wav"
    out_wav.write_bytes(b"placeholder")

    cfg = SimulationConfig(
        instrument="34in_active_stingray",
        output_wav=out_wav,
        skip_identity=True,
        max_samples=2400,
    )

    success = simulate_voice("09_stingray_mm_parallel", config=cfg)
    assert success is False
    assert not out_wav.exists()


def test_simulate_voice_produces_file_for_transformative_voice(tmp_path: Path):
    """Verify simulate_voice produces a file and returns True for transformative voices even with skip_identity=True."""
    out_wav = tmp_path / "vintage_p.wav"

    cfg = SimulationConfig(
        instrument="34in_active_stingray",
        output_wav=out_wav,
        skip_identity=True,
        max_samples=2400,
    )

    success = simulate_voice("05_vintage_62_p_alnico", config=cfg)
    assert success is True
    assert out_wav.exists()
    assert out_wav.stat().st_size > 0


def test_pipeline_cli_bake_skips_identity(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Verify that running CLI with --stage bake on an identity voice does not output a WAV file."""
    import allomorph.pipeline.cli
    from allomorph.naming import get_baked_basename
    from allomorph.pipeline.cli import main

    test_audio_dir = tmp_path / "audio"
    test_models_dir = tmp_path / "models"
    monkeypatch.setattr(allomorph.pipeline.cli, "AUDIO_DIR", test_audio_dir)
    monkeypatch.setattr(allomorph.pipeline.cli, "MODELS_DIR", test_models_dir)

    test_argv = [
        "--stage",
        "bake",
        "--instrument",
        "34in_active_stingray",
        "--voice",
        "09_stingray_mm_parallel",
        "--max-samples",
        "2400",
    ]
    main(test_argv)

    baked_wav = (
        test_audio_dir
        / "baked"
        / "34in_active_stingray"
        / f"{get_baked_basename('09_stingray_mm_parallel', 'dynamic')}.wav"
    )
    assert not baked_wav.exists(), f"Identity voice should not be output: {baked_wav}"
    # Verify the test output directory was properly created in tmp_path without touching repo audio
    assert (test_audio_dir / "baked" / "34in_active_stingray").exists()


def test_simulate_voice_skips_neutral_character_identity(tmp_path: Path):
    """Verify simulate_voice skips 15_neutral_character when skip_identity=True."""
    for inst in ["34in_standard_p", "30in_emg_mmtw", "34in_active_stingray"]:
        out_wav = tmp_path / f"neutral_{inst}.wav"
        cfg = SimulationConfig(
            instrument=inst,
            output_wav=out_wav,
            skip_identity=True,
            max_samples=2400,
        )
        success = simulate_voice("15_neutral_character", config=cfg)
        assert success is False, f"15_neutral_character on {inst} must be skipped as identity"
        assert not out_wav.exists()


def test_simulate_voice_produces_active_and_passive_character(tmp_path: Path):
    """Verify simulate_voice produces 15b_active_character and 15c_passive_character when skip_identity=True."""
    for vid in ["15b_active_character", "15c_passive_character"]:
        for inst in ["34in_standard_p", "30in_emg_mmtw"]:
            out_wav = tmp_path / f"{vid}_{inst}.wav"
            cfg = SimulationConfig(
                instrument=inst,
                output_wav=out_wav,
                skip_identity=True,
                max_samples=2400,
            )
            success = simulate_voice(vid, config=cfg)
            assert success is True, f"{vid} on {inst} must NOT be skipped as identity"
            assert out_wav.exists()
            assert out_wav.stat().st_size > 0





