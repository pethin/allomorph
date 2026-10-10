"""
Tests for pipeline CLI in allomorph.pipeline.cli.
"""

from pathlib import Path
from typing import Any

import pytest

from allomorph.pipeline.cli import main


def test_pipeline_cli_pack_input_wav_propagation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Bug 3 regression: verify custom --input audio is propagated to export_tone_pack in pack stage."""
    calls = []

    def _mock_export_tone_pack(instrument: Any, **kwargs: Any) -> Path:
        calls.append((instrument, kwargs))
        return tmp_path

    monkeypatch.setattr("allomorph.pipeline.pack.export_tone_pack", _mock_export_tone_pack)

    custom_input = tmp_path / "custom_dry.wav"
    custom_input.write_bytes(b"dummy wav data")

    main(argv=["--stage", "pack", "--instrument", "30in", "--input", str(custom_input), "--force"])

    assert len(calls) == 1
    _inst, kwargs = calls[0]
    assert kwargs.get("input_wav") == str(custom_input)
    assert kwargs.get("overwrite") is True


def test_pipeline_cli_all_input_wav_propagation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Bug 3 regression: verify custom --input audio is propagated to export_tone_pack in all stage."""
    sim_calls = []
    pack_calls = []
    viz_calls = []

    def _mock_sim(*args: Any, **kwargs: Any) -> list[Any]:
        sim_calls.append((args, kwargs))
        empty_list: list[Any] = []
        return empty_list

    def _mock_pack(*args: Any, **kwargs: Any) -> Path:
        pack_calls.append((args, kwargs))
        return tmp_path

    def _mock_viz(*args: Any, **kwargs: Any) -> None:
        viz_calls.append((args, kwargs))

    monkeypatch.setattr("allomorph.circuit.forward.simulate_all_instrument_voicings", _mock_sim)
    monkeypatch.setattr("allomorph.pipeline.pack.export_tone_pack", _mock_pack)
    monkeypatch.setattr("allomorph.pipeline.cli.run_visualization", _mock_viz)

    custom_input = tmp_path / "custom_dry.wav"
    custom_input.write_bytes(b"dummy wav data")

    main(argv=["--stage", "all", "--instrument", "30in", "--input", str(custom_input)])

    assert len(sim_calls) == 1
    assert len(pack_calls) == 1
    assert len(viz_calls) == 1

    _, p_kwargs = pack_calls[0]
    assert p_kwargs.get("input_wav") == str(custom_input)


def test_pipeline_cli_viz_stage(monkeypatch: pytest.MonkeyPatch):
    """Verify --stage viz calls run_visualization."""
    viz_calls = []

    def _mock_viz(*args: Any, **kwargs: Any) -> None:
        viz_calls.append((args, kwargs))

    monkeypatch.setattr("allomorph.pipeline.cli.run_visualization", _mock_viz)
    main(argv=["--stage", "viz", "--instrument", "34in_standard_p"])
    assert len(viz_calls) == 1
    assert viz_calls[0][1].get("instrument") == "34in_standard_p"


def test_pipeline_cli_train_stage(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Verify --stage train calls train_voices_from_config with proper configuration."""
    train_calls: list[Any] = []

    def _mock_train(cfg: Any) -> bool:
        train_calls.append(cfg)
        return True

    monkeypatch.setattr("allomorph.trainer.train_voices_from_config", _mock_train)

    custom_input = tmp_path / "custom_input.wav"
    custom_input.write_bytes(b"wav")

    main(
        argv=[
            "--stage",
            "train",
            "--instrument",
            "34in_standard_p",
            "--voice",
            "vintage_open",
            "--input",
            str(custom_input),
            "--fast-dev-run",
        ]
    )

    assert len(train_calls) == 1
    cfg = train_calls[0]
    assert cfg.voice == "vintage_open"
    assert cfg.fast_dev_run is True
    assert cfg.input_wav == str(custom_input)


def test_pipeline_cli_list_flags(capsys: pytest.CaptureFixture[str]):
    """Verify --list-instruments and --list-voices output."""
    main(argv=["--list-instruments"])
    captured_inst = capsys.readouterr().out
    assert "Available Allomorph Source Instruments:" in captured_inst

    main(argv=["--list-voices"])
    captured_voice = capsys.readouterr().out
    assert "Available Allomorph Target Pickup Voices" in captured_voice


def test_pipeline_cli_audit_stage(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    """Verify --stage audit calls audit_wet_audio_catalog and prints telemetry."""
    from allomorph.circuit.audit import AudioAuditReport

    mock_report = AudioAuditReport(
        total_files=2,
        valid_files=2,
        clipped_files=0,
        mean_lufs=-20.5,
        min_lufs=-21.0,
        max_lufs=-20.0,
        mean_true_peak_dbfs=-0.5,
        max_true_peak_dbfs=-0.1,
        records=[],
    )

    def _mock_audit(**_kwargs: Any) -> AudioAuditReport:
        return mock_report

    monkeypatch.setattr("allomorph.circuit.audit.audit_wet_audio_catalog", _mock_audit)
    main(argv=["--stage", "audit"])
    out = capsys.readouterr().out
    assert "AUDIO TELEMETRY REPORT" in out
    assert "Total Files:        2" in out


def test_pipeline_cli_sim_stage(monkeypatch: pytest.MonkeyPatch):
    """Verify --stage sim calls simulate_all_instrument_voicings."""
    sim_calls = []

    def _mock_sim(inst: Any, **kwargs: Any) -> None:
        sim_calls.append((inst, kwargs))

    monkeypatch.setattr("allomorph.circuit.forward.simulate_all_instrument_voicings", _mock_sim)
    main(argv=["--stage", "sim", "--instrument", "30in", "--max-samples", "2400"])
    assert len(sim_calls) == 1
    assert sim_calls[0][0] == "30in_emg_mmtw"
    assert sim_calls[0][1]["max_samples"] == 2400


def test_allomorph_cli_root_entrypoint(monkeypatch: pytest.MonkeyPatch):
    """Verify allomorph.cli.main invokes pipeline CLI main."""
    import allomorph.cli

    cli_calls = []

    def _mock_pipeline_main(argv: list[str] | None = None) -> None:
        cli_calls.append(argv)

    monkeypatch.setattr("allomorph.cli.pipeline_main", _mock_pipeline_main)
    allomorph.cli.main(["--stage", "viz"])
    assert len(cli_calls) == 1
    assert cli_calls[0] == ["--stage", "viz"]


def test_pipeline_cli_list_commands(capsys: pytest.CaptureFixture[str]):
    """Verify --list-instruments and --list-voices execute and format output."""
    main(argv=["--list-instruments"])
    out = capsys.readouterr().out
    assert "Available Allomorph Source Instruments:" in out
    assert "30in_emg_mmtw" in out

    main(argv=["--list-voices"])
    out_voices = capsys.readouterr().out
    assert "Available Allomorph Target Pickup Voices" in out_voices
    assert "precision_vintage" in out_voices


def test_pipeline_cli_argument_errors():
    """Verify validation of jobs and max-samples arguments."""
    with pytest.raises(SystemExit):
        main(argv=["--jobs", "0"])

    with pytest.raises(SystemExit):
        main(argv=["--max-samples", "0"])


def test_pipeline_cli_clean_audio(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Verify --clean-audio clears existing files and directories."""
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    (audio_dir / "test_file.wav").write_bytes(b"dummy")
    sub_dir = audio_dir / "subdir"
    sub_dir.mkdir()
    (sub_dir / "sub_file.wav").write_bytes(b"dummy2")

    def _dummy_viz(*args: object, **kwargs: object) -> None:
        pass

    monkeypatch.setattr("allomorph.circuit.forward.AUDIO_DIR", audio_dir)
    monkeypatch.setattr("allomorph.pipeline.cli.run_visualization", _dummy_viz)

    main(argv=["--clean-audio", "--stage", "viz"])
    assert not (audio_dir / "test_file.wav").exists()
    assert not sub_dir.exists()

