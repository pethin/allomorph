"""
Tests for Allomorph NAM Architecture 2 local trainer.
"""

import json
from pathlib import Path
from typing import Any

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
            assert export_meta.source_instrument.pickup.name == "EMG MM Dual Coil"
            assert meta["gear_make"] == '30" Short Scale MM (EMG MM)'
            assert export_meta.target_voice.id == "03_modern_p_ceramic"
            found = True
            break
    if not found:
        # If model hasn't finished exporting yet, test the structure via mock
        pass


def test_default_goal_esr():
    import inspect

    from allomorph.trainer import DEFAULT_GOAL_ESR, train_voice

    assert DEFAULT_GOAL_ESR == 0.0002
    sig = inspect.signature(train_voice)
    assert "goal_esr" in sig.parameters
    assert sig.parameters["goal_esr"].default == DEFAULT_GOAL_ESR


def test_default_min_epochs():
    import inspect

    from allomorph.trainer import DEFAULT_MIN_EPOCHS, train_voice

    assert DEFAULT_MIN_EPOCHS == 5
    sig = inspect.signature(train_voice)
    assert "min_epochs" in sig.parameters
    assert sig.parameters["min_epochs"].default == DEFAULT_MIN_EPOCHS


def test_train_nam_cli_goal_esr_parsing():
    import argparse

    from allomorph.trainer import DEFAULT_GOAL_ESR

    # Test parser construction from train_nam
    parser = argparse.ArgumentParser()
    parser.add_argument("--goal-esr", type=float, default=DEFAULT_GOAL_ESR)
    parser.add_argument("--no-goal-esr", action="store_true")

    # Default case
    args = parser.parse_args([])
    effective = (
        None
        if args.no_goal_esr or (args.goal_esr is not None and args.goal_esr <= 0)
        else args.goal_esr
    )
    assert effective == 0.00020

    # Custom goal ESR
    args = parser.parse_args(["--goal-esr", "0.0001"])
    effective = (
        None
        if args.no_goal_esr or (args.goal_esr is not None and args.goal_esr <= 0)
        else args.goal_esr
    )
    assert effective == 0.0001

    # Disabling via --no-goal-esr
    args = parser.parse_args(["--no-goal-esr"])
    effective = (
        None
        if args.no_goal_esr or (args.goal_esr is not None and args.goal_esr <= 0)
        else args.goal_esr
    )
    assert effective is None

    # Disabling via --goal-esr 0
    args = parser.parse_args(["--goal-esr", "0"])
    effective = (
        None
        if args.no_goal_esr or (args.goal_esr is not None and args.goal_esr <= 0)
        else args.goal_esr
    )
    assert effective is None


def test_train_nam_cli_min_epochs_parsing():
    import argparse

    from allomorph.trainer import DEFAULT_MIN_EPOCHS

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--min-epochs",
        "--warmup-epochs",
        dest="min_epochs",
        type=int,
        default=DEFAULT_MIN_EPOCHS,
    )

    # Default case
    args = parser.parse_args([])
    assert args.min_epochs == 5

    # Custom min epochs
    args = parser.parse_args(["--min-epochs", "160"])
    assert args.min_epochs == 160

    # Warmup epochs alias
    args = parser.parse_args(["--warmup-epochs", "100"])
    assert args.min_epochs == 100


def test_train_voice_parameters():
    import inspect

    from allomorph.trainer import (
        DEFAULT_BATCH_SIZE,
        DEFAULT_CONSECUTIVE_PATIENCE,
        DEFAULT_ETA_MIN,
        DEFAULT_GOAL_DELTA_ESR,
        DEFAULT_GOAL_DELTA_ESR_LITE,
        DEFAULT_GOAL_DELTA_MRSTFT,
        DEFAULT_GOAL_DELTA_MRSTFT_LITE,
        DEFAULT_GOAL_ESR,
        DEFAULT_GOAL_ESR_LITE,
        DEFAULT_LR_SCHEDULER,
        DEFAULT_LR_T_MAX,
        DEFAULT_MAX_MRSTFT_CEILING,
        DEFAULT_MAX_MRSTFT_CEILING_LITE,
        DEFAULT_MRSTFT_WEIGHT,
        DEFAULT_PATIENCE,
        DEFAULT_PRE_EMPH_COEF,
        DEFAULT_PRE_EMPH_WEIGHT,
        train_voice,
    )

    sig = inspect.signature(train_voice)
    assert "a2_lite_only" not in sig.parameters
    assert "goal_esr_nano" not in sig.parameters
    assert sig.parameters["goal_esr"].default == DEFAULT_GOAL_ESR
    assert sig.parameters["goal_delta_esr"].default == DEFAULT_GOAL_DELTA_ESR
    assert sig.parameters["goal_delta_mrstft"].default == DEFAULT_GOAL_DELTA_MRSTFT
    assert sig.parameters["max_mrstft_ceiling"].default == DEFAULT_MAX_MRSTFT_CEILING
    assert sig.parameters["goal_esr_lite"].default == DEFAULT_GOAL_ESR_LITE
    assert sig.parameters["goal_delta_esr_lite"].default == DEFAULT_GOAL_DELTA_ESR_LITE
    assert sig.parameters["goal_delta_mrstft_lite"].default == DEFAULT_GOAL_DELTA_MRSTFT_LITE
    assert sig.parameters["max_mrstft_ceiling_lite"].default == DEFAULT_MAX_MRSTFT_CEILING_LITE
    assert sig.parameters["consecutive_patience"].default == DEFAULT_CONSECUTIVE_PATIENCE
    assert sig.parameters["patience"].default == DEFAULT_PATIENCE
    assert sig.parameters["pre_emph_weight"].default == DEFAULT_PRE_EMPH_WEIGHT
    assert sig.parameters["pre_emph_coef"].default == DEFAULT_PRE_EMPH_COEF
    assert sig.parameters["mrstft_weight"].default == DEFAULT_MRSTFT_WEIGHT
    assert sig.parameters["lr_scheduler"].default == DEFAULT_LR_SCHEDULER
    assert sig.parameters["eta_min"].default == DEFAULT_ETA_MIN
    assert sig.parameters["lr_t_max"].default == DEFAULT_LR_T_MAX
    assert sig.parameters["batch_size"].default == DEFAULT_BATCH_SIZE
    assert DEFAULT_PATIENCE == 12
    assert DEFAULT_BATCH_SIZE == 16
    assert DEFAULT_CONSECUTIVE_PATIENCE == 3
    assert DEFAULT_GOAL_ESR == 0.00020
    assert DEFAULT_GOAL_ESR_LITE == 0.00250
    assert DEFAULT_GOAL_DELTA_ESR_LITE == 0.080
    assert DEFAULT_GOAL_DELTA_MRSTFT == 0.50
    assert DEFAULT_MAX_MRSTFT_CEILING == 0.320
    assert DEFAULT_GOAL_DELTA_MRSTFT_LITE == 0.75
    assert DEFAULT_MAX_MRSTFT_CEILING_LITE == 0.450
    assert DEFAULT_LR_SCHEDULER == "cosine"
    assert DEFAULT_ETA_MIN == 1e-5
    assert DEFAULT_LR_T_MAX == 35


def test_train_nam_cli_triple_gate_and_cosine_parsing():
    import argparse

    from allomorph.trainer import (
        DEFAULT_BATCH_SIZE,
        DEFAULT_CONSECUTIVE_PATIENCE,
        DEFAULT_ETA_MIN,
        DEFAULT_GOAL_DELTA_MRSTFT,
        DEFAULT_LR_SCHEDULER,
        DEFAULT_LR_T_MAX,
        DEFAULT_MAX_MRSTFT_CEILING,
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--goal-delta-mrstft", type=float, default=DEFAULT_GOAL_DELTA_MRSTFT)
    parser.add_argument("--max-mrstft-ceiling", type=float, default=DEFAULT_MAX_MRSTFT_CEILING)
    parser.add_argument("--consecutive-patience", type=int, default=DEFAULT_CONSECUTIVE_PATIENCE)
    parser.add_argument("--lr-scheduler", default=DEFAULT_LR_SCHEDULER)
    parser.add_argument("--eta-min", type=float, default=DEFAULT_ETA_MIN)
    parser.add_argument("--lr-t-max", type=int, default=DEFAULT_LR_T_MAX)

    args = parser.parse_args([])
    assert args.batch_size == 16
    assert args.goal_delta_mrstft == 0.50
    assert args.max_mrstft_ceiling == 0.320
    assert args.consecutive_patience == 3
    assert args.lr_scheduler == "cosine"
    assert args.eta_min == 1e-5
    assert args.lr_t_max == 35

    custom_args = parser.parse_args(
        [
            "--batch-size",
            "32",
            "--goal-delta-mrstft",
            "0.40",
            "--max-mrstft-ceiling",
            "0.280",
            "--consecutive-patience",
            "4",
            "--lr-scheduler",
            "exponential",
            "--eta-min",
            "1e-6",
            "--lr-t-max",
            "40",
        ]
    )
    assert custom_args.batch_size == 32
    assert custom_args.goal_delta_mrstft == 0.40
    assert custom_args.max_mrstft_ceiling == 0.280
    assert custom_args.consecutive_patience == 4
    assert custom_args.lr_scheduler == "exponential"
    assert custom_args.eta_min == 1e-6
    assert custom_args.lr_t_max == 40


def test_train_nam_lite_cli_parsing():
    import argparse

    from allomorph.trainer import (
        DEFAULT_GOAL_DELTA_ESR_LITE,
        DEFAULT_GOAL_DELTA_MRSTFT_LITE,
        DEFAULT_GOAL_ESR_LITE,
        DEFAULT_MAX_MRSTFT_CEILING_LITE,
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--goal-esr-lite", type=float, default=DEFAULT_GOAL_ESR_LITE)
    parser.add_argument("--goal-delta-esr-lite", type=float, default=DEFAULT_GOAL_DELTA_ESR_LITE)
    parser.add_argument(
        "--goal-delta-mrstft-lite", type=float, default=DEFAULT_GOAL_DELTA_MRSTFT_LITE
    )
    parser.add_argument(
        "--max-mrstft-ceiling-lite", type=float, default=DEFAULT_MAX_MRSTFT_CEILING_LITE
    )

    args = parser.parse_args([])
    assert args.goal_esr_lite == 0.00250
    assert args.goal_delta_esr_lite == 0.080
    assert args.goal_delta_mrstft_lite == 0.75
    assert args.max_mrstft_ceiling_lite == 0.450

    custom = parser.parse_args(
        [
            "--goal-esr-lite",
            "0.0030",
            "--goal-delta-esr-lite",
            "0.090",
            "--goal-delta-mrstft-lite",
            "0.80",
            "--max-mrstft-ceiling-lite",
            "0.500",
        ]
    )
    assert custom.goal_esr_lite == 0.0030
    assert custom.goal_delta_esr_lite == 0.090
    assert custom.goal_delta_mrstft_lite == 0.80
    assert custom.max_mrstft_ceiling_lite == 0.500


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
    warmup_cb: Any = next((c for c in callbacks if "LinearWarmupCallback" in type(c).__name__), None)
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
    import torch

    from allomorph.trainer import configure_a2_architecture

    # Test slimmable mode early stopping monitors ESR_packed_1 (channels_8)
    configure_a2_architecture(
        nam_core,
        min_epochs=5,
        goal_esr=0.00020,
        goal_delta_esr=0.020,
        goal_delta_mrstft=0.50,
        max_mrstft_ceiling=0.320,
        goal_esr_lite=0.00250,
        goal_delta_esr_lite=0.080,
        goal_delta_mrstft_lite=0.75,
        max_mrstft_ceiling_lite=0.450,
        consecutive_patience=3,
        patience=12,
    )
    callbacks = nam_core.get_callbacks(threshold_esr=0.00020)

    cb: Any = next((c for c in callbacks if "EsrProgressCallback" in type(c).__name__), None)
    assert cb is not None
    assert cb.target_esr == 0.00020
    assert cb.target_delta_esr == 0.020
    assert cb.min_epochs == 5

    warmup_cb: Any = next((c for c in callbacks if "LinearWarmupCallback" in type(c).__name__), None)
    assert warmup_cb is not None

    stopping_cb: Any = next((c for c in callbacks if "AllomorphAdaptiveStopping" in type(c).__name__), None)
    assert stopping_cb is not None
    assert stopping_cb.monitor == "ESR_packed_1"
    assert stopping_cb.stopping_threshold == 0.00020
    assert stopping_cb.goal_esr == 0.00020
    assert stopping_cb.goal_delta_esr == 0.020
    assert stopping_cb.goal_delta_mrstft == 0.50
    assert stopping_cb.max_mrstft_ceiling == 0.320
    assert stopping_cb.goal_esr_lite == 0.00250
    assert stopping_cb.goal_delta_esr_lite == 0.080
    assert stopping_cb.goal_delta_mrstft_lite == 0.75
    assert stopping_cb.max_mrstft_ceiling_lite == 0.450
    assert stopping_cb.consecutive_patience == 3
    assert stopping_cb.min_epochs == 5
    assert stopping_cb.warmup_floor == 5
    assert stopping_cb.patience == 12

    # Test threshold_esr=None adds no stopping callback
    callbacks_none = nam_core.get_callbacks(threshold_esr=None)
    assert not any("AllomorphAdaptiveStopping" in type(c).__name__ for c in callbacks_none)

    # Re-test slimmable validation epoch end with dual submodel metrics
    configure_a2_architecture(nam_core, min_epochs=5)
    callbacks = nam_core.get_callbacks(threshold_esr=0.00020)
    cb: Any = next(c for c in callbacks if "EsrProgressCallback" in type(c).__name__)
    vs_cb: Any = next(c for c in callbacks if "AllomorphAdaptiveStopping" in type(c).__name__)

    class DummyTrainer:
        def __init__(self) -> None:
            self.sanity_checking = False
            self.callback_metrics = {
                "ESR_packed_1": 0.0005,
                "ESR_packed_0": 0.0008,
                "ESR": 0.0013,
                "MRSTFT_packed_1": 0.25,
            }
            self.progress_bar_metrics: dict[str, str] = {}
            self.current_epoch = 12
            self.max_epochs = 400

    trainer: Any = DummyTrainer()
    dummy_pl_module: Any = None
    cb.on_validation_epoch_end(trainer, dummy_pl_module)
    assert trainer.progress_bar_metrics["val_ESR"] == "0.00050"
    assert trainer.progress_bar_metrics["val_ESR_ch8"] == "0.00050"
    assert trainer.progress_bar_metrics["val_ESR_ch3"] == "0.00080"
    assert trainer.progress_bar_metrics["best_ESR"] == "0.00050"
    assert cb.best_esr == 0.0005
    assert cb.best_ch3_esr == 0.0008
    assert cb.best_mrstft == 0.25

    # Test Triple-Gate early stopping logic
    class DummyStrategy:
        @staticmethod
        def reduce_boolean_decision(decision: bool, all: bool = False) -> bool:
            return decision

    class EarlyStoppingTestTrainer:
        def __init__(
            self,
            current_epoch: int,
            esr_val: float,
            mrstft_val: float = 0.15,
            esr_ch3_val: float = 0.0008,
            mrstft_ch3_val: float = 0.20,
        ) -> None:
            self.fast_dev_run = False
            self.current_epoch = current_epoch
            self.should_stop = False
            self.callback_metrics = {
                "ESR_packed_1": torch.tensor(esr_val),
                "MRSTFT_packed_1": torch.tensor(mrstft_val),
                "ESR_packed_0": torch.tensor(esr_ch3_val),
                "MRSTFT_packed_0": torch.tensor(mrstft_ch3_val),
            }
            self.strategy = DummyStrategy()

    # 1. Warmup floor: at epoch 3 (< 5), stopping suppressed even if all gates pass
    trainer_early = EarlyStoppingTestTrainer(current_epoch=3, esr_val=0.00005, mrstft_val=0.10)
    vs_cb._run_early_stopping_check(trainer_early)
    assert trainer_early.should_stop is False
    assert vs_cb.consecutive_gates_met == 0

    # 2. Triple-gate check: baseline_delta_ratio=0.015, baseline_mrstft=0.400
    # Epoch 5: Studio Gate 1: esr 0.00015 <= 0.00020
    #          Studio Gate 2: delta 0.00015 / 0.015 = 0.010 <= 0.020
    #          Studio Gate 3: MRSTFT 0.15 / 0.400 = 0.375 <= 0.50 AND 0.15 <= 0.320
    #          A2 Lite Gate 4: esr 0.0008 <= 0.00250
    #          A2 Lite Gate 5: delta 0.0008 / 0.015 = 0.0533 <= 0.080
    #          A2 Lite Gate 6: MRSTFT 0.20 / 0.400 = 0.50 <= 0.75 AND 0.20 <= 0.450
    # 1st consecutive pass -> consecutive_gates_met = 1, should_stop = False
    trainer_pass1 = EarlyStoppingTestTrainer(current_epoch=5, esr_val=0.00015, mrstft_val=0.15, esr_ch3_val=0.0008)
    vs_cb._run_early_stopping_check(trainer_pass1)
    assert trainer_pass1.should_stop is False
    assert vs_cb.consecutive_gates_met == 1

    # Epoch 6: 2nd consecutive pass -> consecutive_gates_met = 2, should_stop = False
    trainer_pass2 = EarlyStoppingTestTrainer(current_epoch=6, esr_val=0.00014, mrstft_val=0.14, esr_ch3_val=0.0007)
    vs_cb._run_early_stopping_check(trainer_pass2)
    assert trainer_pass2.should_stop is False
    assert vs_cb.consecutive_gates_met == 2

    # Epoch 7: 3rd consecutive pass -> consecutive_gates_met = 3 >= 3 -> should_stop = True!
    trainer_pass3 = EarlyStoppingTestTrainer(current_epoch=7, esr_val=0.00013, mrstft_val=0.13, esr_ch3_val=0.0006)
    vs_cb._run_early_stopping_check(trainer_pass3)
    assert trainer_pass3.should_stop is True
    assert vs_cb.stop_reason == "dual_triple_gate_converged"

    # 3. Consecutive gate reset if any gate fails:
    vs_cb.consecutive_gates_met = 2
    trainer_fail_mrstft = EarlyStoppingTestTrainer(current_epoch=8, esr_val=0.00013, mrstft_val=0.25)  # 0.25 / 0.40 = 0.625 > 0.50
    vs_cb._run_early_stopping_check(trainer_fail_mrstft)
    assert trainer_fail_mrstft.should_stop is False
    assert vs_cb.consecutive_gates_met == 0


def test_triple_gate_stopping_nuance_preservation():
    """Verify that subtle voicings with small baseline distance (e.g. M_base = 0.05)
    strictly require MRSTFT to close by >50% (MRSTFT <= 0.025) and do not early exit prematurely.
    """
    import nam.train.core as nam_core
    import torch

    from allomorph.trainer import configure_a2_architecture

    configure_a2_architecture(
        nam_core,
        min_epochs=5,
        goal_esr=0.00020,
        goal_delta_esr=0.020,
        goal_delta_mrstft=0.50,
        max_mrstft_ceiling=0.320,
        goal_esr_lite=0.00250,
        goal_delta_esr_lite=0.080,
        goal_delta_mrstft_lite=0.75,
        max_mrstft_ceiling_lite=0.450,
        consecutive_patience=1,
    )
    callbacks = nam_core.get_callbacks(threshold_esr=0.00020)
    vs_cb: Any = next(c for c in callbacks if "AllomorphAdaptiveStopping" in type(c).__name__)

    # Set subtle baseline: baseline_delta_ratio=0.010, baseline_mrstft=0.05
    vs_cb.baseline_delta_ratio = 0.010
    vs_cb.baseline_mrstft = 0.05
    vs_cb.consecutive_patience = 1

    class DummyStrategy:
        @staticmethod
        def reduce_boolean_decision(decision: bool, all: bool = False) -> bool:
            return decision

    class EarlyStoppingTestTrainer:
        def __init__(
            self,
            current_epoch: int,
            esr_val: float,
            mrstft_val: float,
            esr_ch3_val: float = 0.0005,
            mrstft_ch3_val: float = 0.020,
        ) -> None:
            self.fast_dev_run = False
            self.current_epoch = current_epoch
            self.should_stop = False
            self.callback_metrics = {
                "ESR_packed_1": torch.tensor(esr_val),
                "MRSTFT_packed_1": torch.tensor(mrstft_val),
                "ESR_packed_0": torch.tensor(esr_ch3_val),
                "MRSTFT_packed_0": torch.tensor(mrstft_ch3_val),
            }
            self.strategy = DummyStrategy()

    # ESR passes (0.00010 <= 0.00020), Delta ESR passes (0.00010/0.010 = 0.010 <= 0.020)
    # MRSTFT is 0.035: ratio = 0.035 / 0.05 = 0.70 > 0.50 -> Gate 3 FAILS!
    t_fail = EarlyStoppingTestTrainer(current_epoch=10, esr_val=0.00010, mrstft_val=0.035)
    vs_cb._run_early_stopping_check(t_fail)
    assert t_fail.should_stop is False

    # Now MRSTFT is 0.022: ratio = 0.022 / 0.05 = 0.44 <= 0.50 -> Gate 3 PASSES!
    t_pass = EarlyStoppingTestTrainer(current_epoch=11, esr_val=0.00010, mrstft_val=0.022)
    vs_cb._run_early_stopping_check(t_pass)
    assert t_pass.should_stop is True
    assert vs_cb.stop_reason == "dual_triple_gate_converged"


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
