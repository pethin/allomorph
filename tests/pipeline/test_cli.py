"""
Tests for modern Cyclopts CLI in allomorph.cli.
"""

from pathlib import Path
from typing import Any

import pytest

from allomorph.cli import main


def test_cli_main_help() -> None:
    code = main(["--help"])
    assert code == 0


def test_cli_visualize(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []

    def _mock_gen_all(output_dir: Any = None) -> None:
        calls.append(output_dir)

    monkeypatch.setattr("allomorph.cli.visualize_cmd.generate_all_charts", _mock_gen_all)
    code = main(["visualize"])
    assert code == 0
    assert len(calls) == 1

    # Test alias 'viz'
    code = main(["viz"])
    assert code == 0
    assert len(calls) == 2


def test_cli_visualize_specific_instrument(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []

    def _mock_gen_interactive(**kwargs: Any) -> None:
        calls.append(kwargs)

    def _mock_gen_portal(**kwargs: Any) -> None:
        pass

    monkeypatch.setattr(
        "allomorph.cli.visualize_cmd.generate_interactive_chart", _mock_gen_interactive
    )
    monkeypatch.setattr("allomorph.cli.visualize_cmd.generate_portal_pages", _mock_gen_portal)

    code = main(["viz", "34in_standard_p"])
    assert code == 0
    assert len(calls) == 1
    assert calls[0]["instrument"].id == "34in_standard_p"


def test_cli_stems_generate(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []

    def _mock_sim(**kwargs: Any) -> Path:
        calls.append(kwargs)
        return Path("audio/wet/dummy.wav")

    monkeypatch.setattr("allomorph.cli.stems_cmd.simulate_instrument_voicing", _mock_sim)

    code = main(["stems", "generate", "instrument:34in_standard_p:precision_vintage", "--force"])
    assert code == 0
    assert len(calls) == 1
    assert calls[0]["voicing"].tone_name == "Precision Vintage"
    assert calls[0]["force"] is True


def test_cli_stems_prune_dry_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("allomorph.cli.stems_cmd.WET_AUDIO_DIR", tmp_path)
    inst_dir = tmp_path / "34in_standard_p"
    inst_dir.mkdir(parents=True)
    orphan_file = inst_dir / "nonexistent_tone.wav"
    orphan_file.write_bytes(b"dummy")

    code = main(["stems", "prune", "--dry-run"])
    assert code == 0
    assert orphan_file.exists()


def test_cli_stems_prune_actual(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("allomorph.cli.stems_cmd.WET_AUDIO_DIR", tmp_path)
    inst_dir = tmp_path / "34in_standard_p"
    inst_dir.mkdir(parents=True)
    orphan_file = inst_dir / "nonexistent_tone.wav"
    orphan_file.write_bytes(b"dummy")

    code = main(["stems", "prune"])
    assert code == 0
    assert not orphan_file.exists()


def test_cli_pack_export(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []

    def _mock_export(instrument: Any, **kwargs: Any) -> Path:
        calls.append((instrument, kwargs))
        return Path("tone3000/packs/34in_standard_p")

    monkeypatch.setattr("allomorph.cli.pack_cmd.export_tone_pack", _mock_export)

    code = main(["pack", "pack:34in_standard_p", "--force"])
    assert code == 0
    assert len(calls) == 1
    assert calls[0][0].id == "34in_standard_p"
    assert calls[0][1]["overwrite"] is True


def test_cli_audit(monkeypatch: pytest.MonkeyPatch) -> None:
    from allomorph.circuit.diagnostics import AudioAuditReport

    def _mock_audit(**kwargs: Any) -> AudioAuditReport:
        return AudioAuditReport(
            total_files=5,
            valid_files=5,
            clipped_files=0,
            mean_lufs=-20.5,
            min_lufs=-20.6,
            max_lufs=-20.4,
            mean_true_peak_dbfs=-0.5,
            max_true_peak_dbfs=-0.2,
            records=[],
        )

    monkeypatch.setattr("allomorph.cli.audit_cmd.audit_wet_audio_catalog", _mock_audit)

    code = main(["audit"])
    assert code == 0


def test_cli_list() -> None:
    assert main(["list"]) == 0
    assert main(["list", "instruments"]) == 0
    assert main(["list", "voices"]) == 0
    assert main(["list", "packs"]) == 0


def test_cli_train_strict_validation_stock_with_native_option() -> None:
    code = main(["train", "--engine", "stock", "--slice-len", "4096"])
    assert code == 1


def test_cli_train_strict_validation_native_with_stock_option() -> None:
    code = main(["train", "--engine", "native", "--val-loss-metric", "val_loss"])
    assert code == 1


def test_cli_train_pack_dispatch(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []

    def _mock_train_pack(pack: Any, **kwargs: Any) -> Path:
        calls.append((pack, kwargs))
        return Path("tone3000/packs/34in_standard_p/nam")

    monkeypatch.setattr("allomorph.cli.train_cmd.train_tone_pack", _mock_train_pack)

    code = main(["train", "pack:34in_standard_p", "--epochs", "50", "--force", "--no-prune"])
    assert code == 0
    assert len(calls) == 1
    assert calls[0][0] == "34in_standard_p"
    assert calls[0][1]["epochs"] == 50
    assert calls[0][1]["overwrite"] is True
