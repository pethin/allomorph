"""
Tests for modern Cyclopts CLI in allomorph.cli.
"""

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from allomorph.cli import main
from allomorph.dsp.io import write_wav_24bit
from allomorph.version import DSP_GENERATION


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

    monkeypatch.setattr(
        "allomorph.cli.visualize_cmd.generate_interactive_chart", _mock_gen_interactive
    )

    code = main(["viz", "34in_standard_p"])
    assert code == 0
    assert len(calls) == 1
    assert calls[0]["instrument"].id == "34in_standard_p"


def test_cli_stems_generate(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []

    def _mock_sim(**kwargs: Any) -> Path:
        calls.append(kwargs)
        return Path("audio/stems/dummy.wav")

    monkeypatch.setattr("allomorph.cli.stems_cmd.simulate_instrument_voicing", _mock_sim)

    code = main(["stems", "generate", "instrument:34in_standard_p:vintage_open", "--force"])
    assert code == 0
    assert len(calls) == 1
    assert calls[0]["voicing"].tone_name == "Precision Vintage"
    assert calls[0]["force"] is True


def test_cli_stems_generate_auto_creates_dry_audio_when_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify stems generate automatically invokes ensure_input_audio_wav when dry audio is absent."""
    missing_dry = tmp_path / "audio" / "input.wav"
    ensure_calls: list[dict[str, Any]] = []
    sim_calls: list[dict[str, Any]] = []

    def _mock_ensure(output_path: Any = None, **kwargs: Any) -> Path:
        ensure_calls.append({"path": output_path, **kwargs})
        p = Path(output_path) if output_path else missing_dry
        p.parent.mkdir(parents=True, exist_ok=True)
        write_wav_24bit(p, np.zeros(100, dtype=np.float32), 48000)
        return p

    def _mock_sim(**kwargs: Any) -> Path:
        sim_calls.append(kwargs)
        return Path("audio/stems/dummy.wav")

    monkeypatch.setattr("allomorph.naming.get_default_input_path", lambda: missing_dry)
    monkeypatch.setattr("allomorph.cli.stems_cmd.get_default_input_path", lambda: missing_dry)
    monkeypatch.setattr("allomorph.cli.stems_cmd.REPO_ROOT", tmp_path / "repo")
    monkeypatch.setattr("allomorph.cli.stems_cmd.STEMS_AUDIO_DIR", tmp_path / "stems")
    monkeypatch.setattr("allomorph.cli.stems_cmd.ensure_input_audio_wav", _mock_ensure)
    monkeypatch.setattr("allomorph.cli.stems_cmd.simulate_instrument_voicing", _mock_sim)

    code = main(["stems", "generate", "instrument:34in_standard_p:vintage_open"])
    assert code == 0
    assert len(ensure_calls) == 1
    assert ensure_calls[0]["overwrite"] is True
    assert len(sim_calls) == 1
    assert sim_calls[0]["input_wav"] == missing_dry


def test_cli_stems_generate_reuses_fresh_dry_audio(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify stems generate reuses cached dry audio when present and valid without force."""
    import json

    audio_dir = tmp_path / "audio"
    audio_dir.mkdir(parents=True)
    existing_dry = audio_dir / "input.wav"
    existing_dry.write_bytes(b"RIFF" + b"\x00" * 100)

    from allomorph.version import DSP_GENERATION

    manifest_p = audio_dir / "manifest.json"
    manifest_p.write_text(
        json.dumps(
            {
                "files": {
                    "input.wav": {
                        "dry_token": "alo_dry_cached_123",
                        "size_bytes": existing_dry.stat().st_size,
                        "recipe_tag": f"synthetic_optimal_bass_dry_v{DSP_GENERATION}",
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    ensure_calls: list[Any] = []
    sim_calls: list[Any] = []

    def _mock_ensure(output_path: Any = None, **kwargs: Any) -> Path:
        ensure_calls.append(output_path)
        return Path(output_path) if output_path else existing_dry

    def _mock_sim(**kwargs: Any) -> Path:
        sim_calls.append(kwargs)
        return Path("audio/stems/dummy.wav")

    monkeypatch.setattr("allomorph.naming.get_default_input_path", lambda: existing_dry)
    monkeypatch.setattr("allomorph.cli.stems_cmd.get_default_input_path", lambda: existing_dry)
    monkeypatch.setattr("allomorph.cli.stems_cmd.REPO_ROOT", tmp_path / "repo")
    monkeypatch.setattr("allomorph.cli.stems_cmd.STEMS_AUDIO_DIR", tmp_path / "stems")
    monkeypatch.setattr("allomorph.cli.stems_cmd.ensure_input_audio_wav", _mock_ensure)
    monkeypatch.setattr("allomorph.cli.stems_cmd.simulate_instrument_voicing", _mock_sim)

    code = main(["stems", "generate", "instrument:34in_standard_p:vintage_open"])
    assert code == 0
    # ensure_input_audio_wav should NOT have been called
    assert len(ensure_calls) == 0
    assert len(sim_calls) == 1
    assert sim_calls[0]["input_wav"] == existing_dry


def test_cli_stems_generate_force_regenerates_dry_audio(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify stems generate with --force triggers overwrite re-synthesis of dry excitation."""
    import json

    audio_dir = tmp_path / "audio"
    audio_dir.mkdir(parents=True)
    existing_dry = audio_dir / "input.wav"
    existing_dry.write_bytes(b"RIFF" + b"\x00" * 100)

    manifest_p = audio_dir / "manifest.json"
    manifest_p.write_text(
        json.dumps(
            {
                "files": {
                    "input.wav": {
                        "dry_token": "alo_dry_cached_123",
                        "size_bytes": existing_dry.stat().st_size,
                        "recipe_tag": f"synthetic_optimal_bass_dry_v{DSP_GENERATION}",
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    ensure_calls: list[dict[str, Any]] = []

    def _mock_ensure(output_path: Any = None, **kwargs: Any) -> Path:
        ensure_calls.append({"path": output_path, **kwargs})
        return Path(output_path) if output_path else existing_dry

    def _mock_sim(**kwargs: Any) -> Path:
        return Path("audio/stems/dummy.wav")

    monkeypatch.setattr("allomorph.naming.get_default_input_path", lambda: existing_dry)
    monkeypatch.setattr("allomorph.cli.stems_cmd.get_default_input_path", lambda: existing_dry)
    monkeypatch.setattr("allomorph.cli.stems_cmd.REPO_ROOT", tmp_path / "repo")
    monkeypatch.setattr("allomorph.cli.stems_cmd.STEMS_AUDIO_DIR", tmp_path / "stems")
    monkeypatch.setattr("allomorph.cli.stems_cmd.ensure_input_audio_wav", _mock_ensure)
    monkeypatch.setattr("allomorph.cli.stems_cmd.simulate_instrument_voicing", _mock_sim)

    code = main(["stems", "generate", "instrument:34in_standard_p:vintage_open", "--force"])
    assert code == 0
    # ensure_input_audio_wav SHOULD have been called with overwrite=True
    assert len(ensure_calls) == 1
    assert ensure_calls[0]["overwrite"] is True


def test_cli_stems_generate_excitation_target_alone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify 'stems generate excitation' synthesizes the excitation audio without requiring instruments."""
    target_dry = tmp_path / "audio" / "input.wav"
    ensure_calls: list[dict[str, Any]] = []
    sim_calls: list[Any] = []

    def _mock_ensure(output_path: Any = None, **kwargs: Any) -> Path:
        ensure_calls.append({"path": output_path, **kwargs})
        p = Path(output_path) if output_path else target_dry
        p.parent.mkdir(parents=True, exist_ok=True)
        write_wav_24bit(p, np.zeros(100, dtype=np.float32), 48000)
        return p

    def _mock_sim(**kwargs: Any) -> Path:
        sim_calls.append(kwargs)
        return Path("audio/stems/dummy.wav")

    monkeypatch.setattr("allomorph.naming.get_default_input_path", lambda: target_dry)
    monkeypatch.setattr("allomorph.cli.stems_cmd.get_default_input_path", lambda: target_dry)
    monkeypatch.setattr("allomorph.cli.stems_cmd.REPO_ROOT", tmp_path / "repo")
    monkeypatch.setattr("allomorph.cli.stems_cmd.STEMS_AUDIO_DIR", tmp_path / "stems")
    monkeypatch.setattr("allomorph.cli.stems_cmd.ensure_input_audio_wav", _mock_ensure)
    monkeypatch.setattr("allomorph.cli.stems_cmd.simulate_instrument_voicing", _mock_sim)

    code = main(["stems", "generate", "excitation"])
    assert code == 0
    assert len(ensure_calls) == 1
    assert len(sim_calls) == 0  # No instruments simulated


def test_cli_stems_generate_excitation_target_with_instrument(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify 'stems generate excitation <instrument>' synthesizes excitation and then instrument stems."""
    target_dry = tmp_path / "audio" / "input.wav"
    ensure_calls: list[dict[str, Any]] = []
    sim_calls: list[Any] = []

    def _mock_ensure(output_path: Any = None, **kwargs: Any) -> Path:
        ensure_calls.append({"path": output_path, **kwargs})
        p = Path(output_path) if output_path else target_dry
        p.parent.mkdir(parents=True, exist_ok=True)
        write_wav_24bit(p, np.zeros(100, dtype=np.float32), 48000)
        return p

    def _mock_sim(**kwargs: Any) -> Path:
        sim_calls.append(kwargs)
        return Path("audio/stems/dummy.wav")

    monkeypatch.setattr("allomorph.naming.get_default_input_path", lambda: target_dry)
    monkeypatch.setattr("allomorph.cli.stems_cmd.get_default_input_path", lambda: target_dry)
    monkeypatch.setattr("allomorph.cli.stems_cmd.REPO_ROOT", tmp_path / "repo")
    monkeypatch.setattr("allomorph.cli.stems_cmd.STEMS_AUDIO_DIR", tmp_path / "stems")
    monkeypatch.setattr("allomorph.cli.stems_cmd.ensure_input_audio_wav", _mock_ensure)
    monkeypatch.setattr("allomorph.cli.stems_cmd.simulate_instrument_voicing", _mock_sim)

    code = main(["stems", "generate", "excitation", "instrument:34in_standard_p:vintage_open"])
    assert code == 0
    assert len(ensure_calls) == 1
    assert len(sim_calls) == 1
    assert sim_calls[0]["voicing"].tone_name == "Precision Vintage"


def test_cli_stems_prune_dry_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("allomorph.cli.stems_cmd.STEMS_AUDIO_DIR", tmp_path)
    inst_dir = tmp_path / "34in_standard_p"
    inst_dir.mkdir(parents=True)
    orphan_file = inst_dir / "nonexistent_tone.wav"
    orphan_file.write_bytes(b"dummy")

    code = main(["stems", "prune", "--dry-run"])
    assert code == 0
    assert orphan_file.exists()


def test_cli_stems_prune_actual(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("allomorph.cli.stems_cmd.STEMS_AUDIO_DIR", tmp_path)
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

    def _mock_needs_repack(*args: Any, **kwargs: Any) -> bool:
        return True

    monkeypatch.setattr("allomorph.cli.pack_cmd.export_tone_pack", _mock_export)
    monkeypatch.setattr("allomorph.cli.pack_cmd.check_pack_needs_repacking", _mock_needs_repack)

    code = main(["pack", "pack:34in_standard_p"])
    assert code == 0
    assert len(calls) == 1
    assert calls[0][0].id == "34in_standard_p"


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

    monkeypatch.setattr("allomorph.cli.audit_cmd.audit_stems_audio_catalog", _mock_audit)

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
    assert calls[0][1]["silent"] is False


def test_cli_train_standalone_dispatch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, Any]] = []

    def _mock_train_voice(*args: Any, **kwargs: Any) -> bool:
        calls.append(kwargs)
        return True

    monkeypatch.setattr("allomorph.cli.train_cmd.MODELS_DIR", tmp_path / "models")
    monkeypatch.setattr("allomorph.cli.train_cmd.train_voice", _mock_train_voice)

    code = main(
        [
            "train",
            "--source",
            "30in_emg_mmtw:dual_mode",
            "--targets",
            "34in_standard_p:vintage_open",
            "--epochs",
            "50",
        ]
    )
    assert code == 0
    assert len(calls) == 1
    assert calls[0]["voice"] == "34in_standard_p:vintage_open"
    assert calls[0]["source_voicing"] == "dual_mode"
    assert calls[0]["epochs"] == 50
    assert calls[0]["silent"] is False
    assert calls[0]["overwrite"] is False

    # Verify --silent and --force flags pass through to train_voice
    calls.clear()
    code = main(
        [
            "train",
            "--source",
            "30in_emg_mmtw:dual_mode",
            "--targets",
            "34in_standard_p:vintage_open",
            "--silent",
            "--force",
        ]
    )
    assert code == 0
    assert len(calls) == 1
    assert calls[0]["silent"] is True
    assert calls[0]["overwrite"] is True

    # Verify comma-separated targets dispatch multiple target voices
    calls.clear()
    code = main(
        [
            "train",
            "--source",
            "30in_emg_mmtw:dual_mode",
            "--targets",
            "34in_standard_p:vintage_open,30in_gibson_eb0:neck_deep",
        ]
    )
    assert code == 0
    assert len(calls) == 2
    assert calls[0]["voice"] == "34in_standard_p:vintage_open"
    assert calls[1]["voice"] == "30in_gibson_eb0:neck_deep"


def test_cli_train_standalone_source_validation() -> None:
    # Missing --source for instrument target fails fast with exit code 1
    assert main(["train", "instrument:34in_standard_p:vintage_open"]) == 1
    # Passing --source with pack target fails fast with exit code 1
    assert main(["train", "pack:34in_standard_p", "--source", "30in_emg_mmtw:dual_mode"]) == 1
    # Malformed --source without colon fails fast with exit code 1
    assert main(["train", "34in_standard_p:vintage_open", "--source", "30in_emg_mmtw"]) == 1


def test_resolve_targets_context_aware_shorthand_and_pack_voicing() -> None:
    from allomorph.cli.targets import get_available_pack_ids, resolve_targets
    from allomorph.config import TONE_PACKS

    assert set(get_available_pack_ids()) == set(TONE_PACKS.keys())

    # 1. Context-aware shorthand: instrument vs pack
    inst_targets = resolve_targets(
        ("34in_standard_p",), allow_shorthand=True, shorthand_kind="instrument"
    )
    assert len(inst_targets) == 1
    assert inst_targets[0].kind == "instrument"
    assert inst_targets[0].id == "34in_standard_p"

    pack_targets = resolve_targets(
        ("34in_standard_p",), allow_shorthand=True, shorthand_kind="pack"
    )
    assert len(pack_targets) == 1
    assert pack_targets[0].kind == "pack"
    assert pack_targets[0].id == "34in_standard_p"

    # 2. pack:<pack_id>:<tgt_inst_id>:<tgt_voicing_id>
    pv_targets = resolve_targets(("pack:34in_standard_p:34in_standard_p:slab_dub",))
    assert len(pv_targets) == 1
    assert pv_targets[0].kind == "pack"
    assert pv_targets[0].id == "34in_standard_p"
    assert pv_targets[0].sub_target == "34in_standard_p:slab_dub"


def test_prune_stale_instrument_models_three_level_hierarchy(tmp_path: Path) -> None:
    from allomorph.cli.train_cmd import prune_stale_instrument_models

    # Create 3-level hierarchy: models/<src_inst>/<src_voicing>/<tgt_inst>/<file>.nam
    src_inst_models_dir = tmp_path / "models" / "30in_emg_mmtw"
    tgt_dir = src_inst_models_dir / "dual_mode" / "34in_standard_p"
    tgt_dir.mkdir(parents=True)

    active_nam = tgt_dir / "vintage_open_v7.1.1.nam"
    stale_nam = tgt_dir / "obsolete_voicing_v0.0.0.nam"
    active_nam.write_bytes(b"{}")
    stale_nam.write_bytes(b"{}")

    pruned = prune_stale_instrument_models("30in_emg_mmtw", src_inst_models_dir)
    assert len(pruned) == 1
    assert active_nam.exists()
    assert not stale_nam.exists()
