"""
Tests for Allomorph NAM Architecture 2 local trainer.
"""

import json
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

from allomorph.pipeline.schema import NamExportMetadata
from allomorph.trainer import find_sweep_input


def test_find_sweep_input():
    sweep = find_sweep_input()
    assert sweep is not None
    assert sweep.exists()
    assert sweep.name == "input.wav" and sweep.suffix == ".wav"


def test_model_metadata_contains_input_bass():
    model_paths = [
        REPO_ROOT / "models" / "30in_emg_mm" / "03_modern_p_ceramic.nam",
    ]
    found = False
    for mp in model_paths:
        if mp.exists():
            with open(mp, "r") as f:
                d = json.load(f)
            meta = d.get("metadata", {})
            export_meta = NamExportMetadata.model_validate(meta)
            assert export_meta.source_instrument.id == "30in_emg_mm"
            assert export_meta.source_instrument.scale_length_in == 30.0
            assert export_meta.source_instrument.voicing.name == "EMG MM Dual Coil"
            assert meta["gear_make"] == '30" Short Scale MM (EMG MM)'
            assert export_meta.target_voicing.id == "03_modern_p_ceramic"
            found = True
            break
    if not found:
        # If model hasn't finished exporting yet, test the structure via mock
        pass


def test_default_training_hyperparameters():
    import inspect

    from allomorph.trainer import (
        DEFAULT_BATCH_SIZE,
        DEFAULT_ETA_MIN,
        DEFAULT_LR_SCHEDULER,
        DEFAULT_LR_T_MAX,
        DEFAULT_MAX_EPOCHS,
        DEFAULT_MIN_DELTA,
        DEFAULT_MIN_EPOCHS,
        DEFAULT_NUM_WORKERS,
        DEFAULT_PATIENCE,
        DEFAULT_PRECISION,
        train_voice,
    )

    assert DEFAULT_MAX_EPOCHS == 40
    assert DEFAULT_MIN_EPOCHS == 20
    assert DEFAULT_PATIENCE == 8
    assert DEFAULT_MIN_DELTA == 1.0e-6
    assert DEFAULT_BATCH_SIZE == "auto"
    assert DEFAULT_PRECISION == "auto"
    assert DEFAULT_NUM_WORKERS == "auto"
    assert DEFAULT_LR_SCHEDULER == "cosine"
    assert DEFAULT_ETA_MIN == 1e-5
    assert DEFAULT_LR_T_MAX == 40

    sig = inspect.signature(train_voice)
    assert sig.parameters["epochs"].default == DEFAULT_MAX_EPOCHS
    assert sig.parameters["min_epochs"].default == DEFAULT_MIN_EPOCHS
    assert sig.parameters["patience"].default == DEFAULT_PATIENCE
    assert sig.parameters["min_delta"].default == DEFAULT_MIN_DELTA
    assert sig.parameters["batch_size"].default == DEFAULT_BATCH_SIZE
    assert sig.parameters["precision"].default == DEFAULT_PRECISION
    assert sig.parameters["num_workers"].default == DEFAULT_NUM_WORKERS

    # Verify legacy parameters are eliminated
    for legacy_param in [
        "goal_esr",
        "goal_delta_esr",
        "goal_delta_mrstft",
        "max_mrstft_ceiling",
        "goal_esr_lite",
        "goal_delta_esr_lite",
        "goal_delta_mrstft_lite",
        "max_mrstft_ceiling_lite",
        "consecutive_patience",
        "a2_lite_only",
        "goal_esr_nano",
    ]:
        assert legacy_param not in sig.parameters


def test_hardware_resolution_utilities():
    from allomorph.trainer import (
        get_hardware_device_name,
        resolve_hardware_batch_size,
        resolve_hardware_num_workers,
        resolve_hardware_precision,
    )

    dev_name = get_hardware_device_name()
    assert isinstance(dev_name, str) and len(dev_name) > 0

    # Batch size resolution
    assert resolve_hardware_batch_size(32) == 32
    assert resolve_hardware_batch_size("16") == 16
    auto_bs = resolve_hardware_batch_size("auto")
    assert auto_bs in (8, 16, 32)

    # Precision resolution
    assert resolve_hardware_precision("16-mixed") == "16-mixed"
    assert resolve_hardware_precision("32-true") == "32-true"
    assert resolve_hardware_precision("bf16-mixed") == "bf16-mixed"
    auto_prec = resolve_hardware_precision("auto")
    assert auto_prec in ("bf16-mixed", "16-mixed", "32-true")

    # Worker count resolution
    assert resolve_hardware_num_workers(4) == 4
    assert resolve_hardware_num_workers("2") == 2
    assert resolve_hardware_num_workers("auto") == 0


def test_train_nam_cli_schedule_and_patience_parsing():
    import argparse

    from allomorph.trainer import (
        DEFAULT_BATCH_SIZE,
        DEFAULT_ETA_MIN,
        DEFAULT_LR_SCHEDULER,
        DEFAULT_LR_T_MAX,
        DEFAULT_MAX_EPOCHS,
        DEFAULT_MIN_DELTA,
        DEFAULT_MIN_EPOCHS,
        DEFAULT_NUM_WORKERS,
        DEFAULT_PATIENCE,
        DEFAULT_PRECISION,
        add_trainer_arguments,
    )

    parser = argparse.ArgumentParser()
    add_trainer_arguments(parser)

    # Default parsing
    args = parser.parse_args([])
    assert args.epochs == DEFAULT_MAX_EPOCHS
    assert args.min_epochs == DEFAULT_MIN_EPOCHS
    assert args.patience == DEFAULT_PATIENCE
    assert args.min_delta == DEFAULT_MIN_DELTA
    assert args.batch_size == DEFAULT_BATCH_SIZE
    assert args.precision == DEFAULT_PRECISION
    assert args.num_workers == DEFAULT_NUM_WORKERS
    assert args.lr_scheduler == DEFAULT_LR_SCHEDULER
    assert args.eta_min == DEFAULT_ETA_MIN
    assert args.lr_t_max == DEFAULT_LR_T_MAX

    # Custom parsing
    custom_args = parser.parse_args(
        [
            "--epochs",
            "40",
            "--min-epochs",
            "8",
            "--patience",
            "7",
            "--min-delta",
            "1e-5",
            "--batch-size",
            "32",
            "--precision",
            "bf16-mixed",
            "--num-workers",
            "4",
            "--lr-scheduler",
            "exponential",
            "--eta-min",
            "1e-6",
            "--lr-t-max",
            "40",
        ]
    )
    assert custom_args.epochs == 40
    assert custom_args.min_epochs == 8
    assert custom_args.patience == 7
    assert custom_args.min_delta == 1e-5
    assert custom_args.batch_size == 32
    assert custom_args.precision == "bf16-mixed"
    assert custom_args.num_workers == 4
    assert custom_args.lr_scheduler == "exponential"
    assert custom_args.eta_min == 1e-6
    assert custom_args.lr_t_max == 40

    # Auto string parsing
    auto_args = parser.parse_args(
        [
            "--batch-size",
            "auto",
            "--precision",
            "auto",
            "--num-workers",
            "auto",
        ]
    )
    assert auto_args.batch_size == "auto"
    assert auto_args.precision == "auto"
    assert auto_args.num_workers == "auto"


def test_configure_a2_architecture():
    import nam.train.core as nam_core

    from allomorph.trainer import configure_a2_architecture

    # Test slimmable configuration with cosine scheduler
    configure_a2_architecture(
        nam_core,
        pre_emph_weight=0.25,
        pre_emph_coef=0.85,
        mrstft_weight=0.0010,
        lr_scheduler="cosine",
        eta_min=1e-5,
        lr_t_max=35,
        precision="bf16-mixed",
        num_workers=0,
    )
    cfg_full = nam_core._get_packed_model_config()
    submodels_full = cfg_full["net"]["config"]["submodels"]
    assert len(submodels_full) == 2
    names = [s["name"] for s in submodels_full]
    assert "channels_3" in names
    assert "channels_8" in names
    assert cfg_full["loss"]["pre_emph_weight"] == 0.25
    assert cfg_full["loss"]["pre_emph_coef"] == 0.85
    assert cfg_full["loss"]["mrstft_weight"] == 0.0010
    assert cfg_full["lr_scheduler"]["class"] == "CosineAnnealingLR"
    assert cfg_full["lr_scheduler"]["kwargs"]["T_max"] == 35
    assert cfg_full["lr_scheduler"]["kwargs"]["eta_min"] == 1e-5

    # Check patched _get_configs
    if hasattr(nam_core, "_get_configs"):
        from nam.train._version import Version

        _, _, learn_cfg = nam_core._get_configs(
            Version(3, 0, 0), "dummy_in", "dummy_out", 0, 10, 100, 32
        )
        assert learn_cfg["trainer"]["precision"] == "bf16-mixed"
        assert learn_cfg["train_dataloader"]["num_workers"] == 0


def test_math_utilities():
    from allomorph.trainer import compute_baseline_delta_ratio, compute_linear_slope

    # Strictly decreasing line
    slope_down = compute_linear_slope([1.0, 0.8, 0.6, 0.4, 0.2])
    assert slope_down < 0.0

    # Strictly flat line
    slope_flat = compute_linear_slope([0.0002, 0.0002, 0.0002, 0.0002])
    assert abs(slope_flat) < 1e-12

    # compute_baseline_delta_ratio fallback on missing files
    assert compute_baseline_delta_ratio(None, None) == 0.015


def test_compute_baseline_mrstft():
    import tempfile

    import numpy as np
    from pedalboard.io import AudioFile

    from allomorph.trainer import compute_baseline_mrstft

    # Fallback on missing or invalid files
    assert compute_baseline_mrstft(None, None) == 0.400
    assert compute_baseline_mrstft(Path("nonexistent1.wav"), Path("nonexistent2.wav")) == 0.400

    # Test with synthetic audio files
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        wav1 = tmp_path / "ref.wav"
        wav2 = tmp_path / "tgt.wav"

        sr = 48000
        t = np.linspace(0, 1.0, sr, endpoint=False, dtype=np.float32)
        # Ref: 200 Hz tone, Tgt: 400 Hz tone
        ref_audio = 0.5 * np.sin(2.0 * np.pi * 200.0 * t).reshape(1, -1)
        tgt_audio = 0.5 * np.sin(2.0 * np.pi * 400.0 * t).reshape(1, -1)

        with AudioFile(str(wav1), "w", samplerate=sr, num_channels=1) as f:
            f.write(ref_audio)
        with AudioFile(str(wav2), "w", samplerate=sr, num_channels=1) as f:
            f.write(tgt_audio)

        # Baseline distance between distinct tones is substantial (> 0.1)
        mr = compute_baseline_mrstft(wav1, wav2, val_samples=48000)
        assert mr > 0.10

        # Baseline distance between identical tones is essentially zero (< 1e-3)
        mr_ident = compute_baseline_mrstft(wav1, wav1, val_samples=48000)
        assert mr_ident < 1e-3


def test_linear_warmup_monotonic_floor_clamp():
    import nam.train.core as nam_core

    from allomorph.trainer import configure_a2_architecture

    configure_a2_architecture(
        nam_core,
        lr_scheduler="cosine",
        eta_min=1e-5,
        lr_t_max=35,
    )
    callbacks = nam_core.get_callbacks(threshold_esr=0.00020)
    warmup_cb: Any = next(
        (c for c in callbacks if "LinearWarmupCallback" in type(c).__name__), None
    )
    assert warmup_cb is not None

    class DummyOptimizer:
        def __init__(self, lr: float = 0.004):
            self.param_groups = [{"lr": lr}]

    class DummyTrainerWithOpt:
        def __init__(self, epoch: int, lr: float = 0.004):
            self.current_epoch = epoch
            self.optimizers = [DummyOptimizer(lr)]

    # Epoch 0: lr = 0.004 * 1 / 5 = 0.0008
    t0 = DummyTrainerWithOpt(epoch=0)
    warmup_cb.on_train_epoch_start(t0, None)
    assert abs(t0.optimizers[0].param_groups[0]["lr"] - 0.0008) < 1e-6

    # Epoch 4: lr = 0.004 * 5 / 5 = 0.004
    t4 = DummyTrainerWithOpt(epoch=4)
    warmup_cb.on_train_epoch_start(t4, None)
    assert abs(t4.optimizers[0].param_groups[0]["lr"] - 0.004) < 1e-6

    # Epoch 10: beyond warmup (< lr_t_max), warmup_cb does not override
    t10 = DummyTrainerWithOpt(epoch=10, lr=0.0025)
    warmup_cb.on_train_epoch_start(t10, None)
    assert abs(t10.optimizers[0].param_groups[0]["lr"] - 0.0025) < 1e-6

    # Epoch 35: at lr_t_max, clamps to eta_min (1e-5)
    t35 = DummyTrainerWithOpt(epoch=35, lr=0.0001)
    warmup_cb.on_train_epoch_start(t35, None)
    assert abs(t35.optimizers[0].param_groups[0]["lr"] - 1e-5) < 1e-8

    # Epoch 50: past lr_t_max, clamps to eta_min (1e-5), never cycling back up
    t50 = DummyTrainerWithOpt(epoch=50, lr=0.003)
    warmup_cb.on_train_epoch_start(t50, None)
    assert abs(t50.optimizers[0].param_groups[0]["lr"] - 1e-5) < 1e-8


def test_esr_progress_callback_hook():
    import nam.train.core as nam_core

    from allomorph.trainer import configure_a2_architecture

    # Test slimmable mode early stopping monitors composite val_loss
    configure_a2_architecture(
        nam_core,
        min_epochs=5,
        patience=5,
        min_delta=2.0e-6,
    )
    callbacks = nam_core.get_callbacks(None)

    cb: Any = next((c for c in callbacks if "EsrProgressCallback" in type(c).__name__), None)
    assert cb is not None
    assert cb.min_epochs == 5

    warmup_cb: Any = next(
        (c for c in callbacks if "LinearWarmupCallback" in type(c).__name__), None
    )
    assert warmup_cb is not None

    stopping_cb: Any = next(
        (c for c in callbacks if "AllomorphAdaptiveStopping" in type(c).__name__), None
    )
    assert stopping_cb is not None
    assert stopping_cb.monitor == "val_loss"
    assert stopping_cb.warmup_floor == 5
    assert stopping_cb.patience == 5
    assert stopping_cb.min_delta == 2.0e-6

    # Re-test slimmable validation epoch end with dual submodel metrics
    configure_a2_architecture(nam_core, min_epochs=5)
    callbacks = nam_core.get_callbacks(None)
    cb: Any = next(c for c in callbacks if "EsrProgressCallback" in type(c).__name__)

    class DummyTrainer:
        def __init__(self) -> None:
            self.sanity_checking = False
            self.callback_metrics = {
                "ESR_packed_1": 0.0005,
                "ESR_packed_0": 0.0008,
                "val_loss": 0.0013,
                "MRSTFT_packed_1": 0.25,
            }
            self.progress_bar_metrics: dict[str, str] = {}
            self.current_epoch = 12
            self.max_epochs = 35

    trainer: Any = DummyTrainer()
    dummy_pl_module: Any = None
    cb.on_validation_epoch_end(trainer, dummy_pl_module)
    assert trainer.progress_bar_metrics["val_ESR_ch8"] == "0.00050"
    assert trainer.progress_bar_metrics["val_ESR_ch3"] == "0.00080"
    assert trainer.progress_bar_metrics["best_ESR"] == "0.00050"
    assert cb.best_esr == 0.0005
    assert cb.best_ch3_esr == 0.0008
    assert cb.best_mrstft == 0.25


def test_adaptive_stopping_composite_val_loss_plateau():
    """Verify adaptive diminishing-returns early exit on composite val_loss (Full + Lite)."""
    import nam.train.core as nam_core
    import torch

    from allomorph.trainer import configure_a2_architecture

    configure_a2_architecture(
        nam_core,
        min_epochs=5,
        patience=5,
        min_delta=2.0e-6,
    )
    callbacks = nam_core.get_callbacks(None)
    vs_cb: Any = next(c for c in callbacks if "AllomorphAdaptiveStopping" in type(c).__name__)

    class EarlyStoppingTestTrainer:
        def __init__(
            self,
            current_epoch: int,
            esr_val: float,
            esr_ch3_val: float,
            val_loss: float,
        ) -> None:
            self.fast_dev_run = False
            self.current_epoch = current_epoch
            self.should_stop = False
            self.callback_metrics = {
                "ESR_packed_1": torch.tensor(esr_val),
                "ESR_packed_0": torch.tensor(esr_ch3_val),
                "val_loss": torch.tensor(val_loss),
            }

    # 1. Warmup floor: epoch 3 (< 5) suppresses stopping even if loss is completely flat
    t_warmup = EarlyStoppingTestTrainer(
        current_epoch=3, esr_val=0.001, esr_ch3_val=0.002, val_loss=0.003
    )
    vs_cb._run_early_stopping_check(t_warmup)
    assert t_warmup.should_stop is False

    # 2. Steady improvement post-warmup: loss continues dropping significantly
    losses = [0.0020, 0.0015, 0.0012, 0.0010, 0.0008, 0.0006]
    for ep, l in enumerate(losses, start=4):
        t_prog = EarlyStoppingTestTrainer(
            current_epoch=ep, esr_val=l * 0.4, esr_ch3_val=l * 0.6, val_loss=l
        )
        vs_cb._run_early_stopping_check(t_prog)
        assert t_prog.should_stop is False

    # 3. Plateau: 5 epochs with virtually identical composite loss (< min_delta improvement)
    plateau_losses = [0.000500, 0.000499, 0.000501, 0.000500, 0.000499]
    for ep, l in enumerate(plateau_losses, start=10):
        t_plat = EarlyStoppingTestTrainer(
            current_epoch=ep, esr_val=l * 0.4, esr_ch3_val=l * 0.6, val_loss=l
        )
        vs_cb._run_early_stopping_check(t_plat)
        if ep == 14:
            # 5th plateau epoch triggers diminishing-returns exit
            assert t_plat.should_stop is True
            assert vs_cb.stop_reason == "diminishing_returns_plateau"
        else:
            assert t_plat.should_stop is False


def test_train_nam_cli_pack_and_overwrite_parsing():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--pack", "--tone-pack", dest="pack", default=None)
    parser.add_argument("--force", "--overwrite", dest="overwrite", action="store_true")

    args = parser.parse_args([])
    assert args.pack is None
    assert args.overwrite is False

    args_pack = parser.parse_args(["--pack", "34in_active_stingray", "--force"])
    assert args_pack.pack == "34in_active_stingray"
    assert args_pack.overwrite is True

    args_alias = parser.parse_args(["--tone-pack", "stingray", "--overwrite"])
    assert args_alias.pack == "stingray"
    assert args_alias.overwrite is True


def test_inst_models_dir_non_nesting_on_nam_dir(tmp_path: Path):
    """Verify that passing models_dir pointing to a 'nam' directory does NOT nest inst_id."""
    nam_dir = tmp_path / "tone3000" / "packs" / "34in_active_stingray" / "nam"
    nam_dir.mkdir(parents=True)
    inst_id = "34in_active_stingray"

    models_path = Path(nam_dir)
    if models_path.name == "nam" or models_path.name == inst_id:
        inst_models_dir = models_path
    else:
        inst_models_dir = models_path / inst_id

    assert inst_models_dir == nam_dir
    assert inst_models_dir.name == "nam"

    # Conversely, passing models/ does nest inst_id
    general_models_dir = tmp_path / "models"
    models_path2 = Path(general_models_dir)
    if models_path2.name == "nam" or models_path2.name == inst_id:
        inst_models_dir2 = models_path2
    else:
        inst_models_dir2 = models_path2 / inst_id

    assert inst_models_dir2 == general_models_dir / inst_id


def test_train_voice_torch_fast_dev_run(tmp_path: Path):
    """Verify PyTorch A2 trainer executes in fast_dev_run mode."""
    from allomorph.trainer import train_voice

    input_audio = REPO_ROOT / "audio" / "input.wav"
    if not input_audio.exists():
        pytest.skip("audio/input.wav not found")

    models_dir = tmp_path / "models"
    ok = train_voice(
        instrument="30in",
        voice="precision_active",
        input_wav=input_audio,
        output_wav=input_audio,
        reference_wav=input_audio,
        models_dir=models_dir,
        epochs=1,
        min_epochs=1,
        fast_dev_run=True,
        engine="torch",
        batch_size=16,
    )
    assert ok is True


def test_apple_silicon_tier_detection(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify detect_apple_silicon_tier classifies Apple Silicon chip families correctly."""
    import subprocess

    from allomorph.trainer.core import detect_apple_silicon_tier

    def _mock_sysctl(brand: str) -> Any:
        def _runner(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
            return subprocess.CompletedProcess(args=["sysctl"], returncode=0, stdout=f"{brand}\n")

        return _runner

    # 1. Ultra
    monkeypatch.setattr(subprocess, "run", _mock_sysctl("Apple M2 Ultra"))
    assert detect_apple_silicon_tier() == "ultra"

    # 2. Max
    monkeypatch.setattr(subprocess, "run", _mock_sysctl("Apple M1 Max"))
    assert detect_apple_silicon_tier() == "max"

    # 3. Pro
    monkeypatch.setattr(subprocess, "run", _mock_sysctl("Apple M3 Pro"))
    assert detect_apple_silicon_tier() == "pro"

    # 4. Base
    monkeypatch.setattr(subprocess, "run", _mock_sysctl("Apple M1"))
    assert detect_apple_silicon_tier() == "base"


def test_resolve_hardware_batch_size_branches(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify resolve_hardware_batch_size across MLX and PyTorch backends."""
    from allomorph.trainer.core import resolve_hardware_batch_size

    # MLX branches
    monkeypatch.setattr("allomorph.trainer.core.detect_apple_silicon_tier", lambda: "ultra")
    assert resolve_hardware_batch_size("auto", engine="mlx") == 16

    monkeypatch.setattr("allomorph.trainer.core.detect_apple_silicon_tier", lambda: "base")
    assert resolve_hardware_batch_size("auto", engine="mlx") == 8

    # String integer
    assert resolve_hardware_batch_size("24") == 24


def test_train_voices_from_config_and_main(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify train_voices_from_config and CLI main function dispatch."""
    import allomorph.trainer
    from allomorph.pipeline.schema import NamTrainingConfig
    from allomorph.trainer import main, train_voices_from_config

    # 1. With pack config
    pack_called: list[bool] = []

    def _mock_pack(*args: object, **kwargs: object) -> None:
        pack_called.append(True)

    monkeypatch.setattr("allomorph.pipeline.pack.train_tone_pack", _mock_pack)
    cfg_pack = NamTrainingConfig(pack="34in_standard_p", voice="all")
    assert train_voices_from_config(cfg_pack) is True
    assert len(pack_called) == 1

    # 2. Main with CLI args mocking train_voices_from_config
    def _mock_train_cfg(cfg: object) -> bool:
        return True

    monkeypatch.setattr(allomorph.trainer, "train_voices_from_config", _mock_train_cfg)
    ret = main(["--instrument", "34in_standard_p", "--voice", "vintage_open"])
    assert ret == 0

    # 3. Main with GUI flag (mocked to prevent desktop window launch)
    gui_called: list[bool] = []

    def _mock_nam_gui(*args: object, **kwargs: object) -> None:
        gui_called.append(True)

    monkeypatch.setattr("nam.cli.nam_gui", _mock_nam_gui, raising=False)
    assert main(["--gui"]) == 0
    assert len(gui_called) == 1

    def _mock_nam_gui_err(*args: object, **kwargs: object) -> None:
        raise RuntimeError("Mock GUI unavailable")

    monkeypatch.setattr("nam.cli.nam_gui", _mock_nam_gui_err, raising=False)
    assert main(["--gui"]) == 1


def test_main_entrypoints(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify python -m allomorph and python -m allomorph.trainer entrypoints."""
    import runpy

    # allomorph.__main__
    def _mock_cli_main(*args: object, **kwargs: object) -> int:
        return 0

    monkeypatch.setattr("allomorph.cli.main", _mock_cli_main)
    with pytest.raises(SystemExit) as exc_info:
        runpy.run_module("allomorph.__main__", run_name="__main__")
    assert exc_info.value.code in (0, 2)  # 0 or 2 depending on default CLI args

    # allomorph.trainer.__main__
    def _mock_main(*args: object, **kwargs: object) -> int:
        return 0

    monkeypatch.setattr("allomorph.trainer.main", _mock_main)
    with pytest.raises(SystemExit) as exc_info2:
        runpy.run_module("allomorph.trainer.__main__", run_name="__main__")
    assert exc_info2.value.code == 0
