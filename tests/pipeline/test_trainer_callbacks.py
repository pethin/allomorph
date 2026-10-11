"""
Deterministic unit tests for Allomorph trainer early stopping callbacks and pure helper functions.
"""

from typing import Any

from allomorph.trainer.callbacks import (
    AllomorphAdaptiveStopping,
    compute_linear_slope,
    compute_pareto_quality_score,
    compute_window_improvement,
    evaluate_plateau_stopping_condition,
    extract_trainer_metrics,
)


class MockTensor:
    """Mock tensor object implementing .item() for metric extraction tests."""

    def __init__(self, val: float) -> None:
        self._val = float(val)

    def item(self) -> float:
        return self._val


def test_compute_linear_slope() -> None:
    # Insufficient points (< 2) returns 0.0
    assert compute_linear_slope([]) == 0.0
    assert compute_linear_slope([42.0]) == 0.0

    # Flat line has slope 0.0
    assert abs(compute_linear_slope([1.0, 1.0, 1.0, 1.0])) < 1e-12

    # Linear increase: y = 2*x + 1 -> points [1, 3, 5, 7] -> slope 2.0
    slope_inc = compute_linear_slope([1.0, 3.0, 5.0, 7.0])
    assert abs(slope_inc - 2.0) < 1e-12

    # Linear decrease: y = -0.5*x -> points [0, -0.5, -1.0] -> slope -0.5
    slope_dec = compute_linear_slope([0.0, -0.5, -1.0])
    assert abs(slope_dec - (-0.5)) < 1e-12


def test_extract_trainer_metrics_scalars() -> None:
    metrics = {
        "ESR_packed_1": 0.0025,
        "MRSTFT_packed_1": 0.045,
        "ESR_packed_0": 0.0080,
        "MRSTFT_packed_0": 0.082,
        "val_loss": 0.0105,
    }
    extracted = extract_trainer_metrics(metrics)
    assert extracted["esr"] == 0.0025
    assert extracted["mrstft"] == 0.045
    assert extracted["esr_ch3"] == 0.0080
    assert extracted["mrstft_ch3"] == 0.082
    assert extracted["val_loss"] == 0.0105


def test_extract_trainer_metrics_tensors() -> None:
    metrics = {
        "ESR": MockTensor(0.0018),
        "MRSTFT": MockTensor(0.035),
        "ESR_packed_0": MockTensor(0.0055),
        "MRSTFT_packed_0": MockTensor(0.070),
        "val_loss": MockTensor(0.0073),
    }
    extracted = extract_trainer_metrics(metrics)
    assert extracted["esr"] == 0.0018
    assert extracted["mrstft"] == 0.035
    assert extracted["esr_ch3"] == 0.0055
    assert extracted["mrstft_ch3"] == 0.070
    assert extracted["val_loss"] == 0.0073


def test_extract_trainer_metrics_composite_val_loss_fallback() -> None:
    # Explicit val_loss omitted; should compute esr + 0.5 * esr_ch3
    metrics = {
        "ESR": 0.0020,
        "ESR_packed_0": 0.0060,
    }
    extracted = extract_trainer_metrics(metrics)
    assert extracted["esr"] == 0.0020
    assert extracted["mrstft"] is None
    assert extracted["esr_ch3"] == 0.0060
    assert extracted["val_loss"] is not None
    assert abs(extracted["val_loss"] - 0.0050) < 1e-12

    # If esr_ch3 is absent, fallback val_loss is just esr
    metrics_solo = {"ESR": 0.0030}
    extracted_solo = extract_trainer_metrics(metrics_solo)
    assert extracted_solo["val_loss"] == 0.0030

    # Completely empty metrics dictionary returns all None
    empty_extracted = extract_trainer_metrics({})
    assert empty_extracted["esr"] is None
    assert empty_extracted["val_loss"] is None


def test_compute_window_improvement() -> None:
    # Short history returns 0.0
    assert compute_window_improvement([0.01, 0.009], patience=5) == 0.0
    assert compute_window_improvement([0.01], patience=0) == 0.0

    # Sufficient history: improvement = history[-patience] - history[-1]
    history = [0.05, 0.04, 0.03, 0.02, 0.015, 0.012, 0.010]
    # For patience=4: start of window is 0.02, end is 0.010 -> diff = 0.010
    imp = compute_window_improvement(history, patience=4)
    assert abs(imp - 0.010) < 1e-12


def test_plateau_stopping_warmup() -> None:
    # Even if slope is flat and improvement is 0, during warmup floor it must not stop
    history = [0.01, 0.01, 0.01, 0.01, 0.01]
    should_stop, reason = evaluate_plateau_stopping_condition(
        val_loss_history=history,
        val_loss_slope=0.0,
        epoch=3,
        warmup_floor=5,
        patience=3,
        min_delta=1e-5,
    )
    assert should_stop is False
    assert reason is None


def test_plateau_stopping_steep_descent() -> None:
    # History shows improvement < min_delta but slope is still steep (< -5e-7)
    history = [0.010, 0.009, 0.008, 0.007, 0.006]
    should_stop, reason = evaluate_plateau_stopping_condition(
        val_loss_history=history,
        val_loss_slope=-1e-4,  # Steep downward trajectory
        epoch=12,
        warmup_floor=5,
        patience=5,
        min_delta=1e-2,  # Improvement 0.004 is less than 0.01
    )
    assert should_stop is False
    assert reason is None


def test_plateau_stopping_large_delta() -> None:
    # Slope is flat (-1e-7 >= -5e-7), but improvement (0.005) >= min_delta (1e-4)
    history = [0.015, 0.013, 0.012, 0.011, 0.010]
    should_stop, reason = evaluate_plateau_stopping_condition(
        val_loss_history=history,
        val_loss_slope=-1e-7,
        epoch=12,
        warmup_floor=5,
        patience=5,
        min_delta=1e-4,
    )
    assert should_stop is False
    assert reason is None


def test_plateau_stopping_triggered() -> None:
    # Past warmup + patience, flat slope, negligible improvement (< min_delta)
    history = [0.005001, 0.005000, 0.005000, 0.005000, 0.005000]
    should_stop, reason = evaluate_plateau_stopping_condition(
        val_loss_history=history,
        val_loss_slope=-1e-8,
        epoch=15,
        warmup_floor=5,
        patience=5,
        min_delta=1e-5,
    )
    assert should_stop is True
    assert reason == "diminishing_returns_plateau"


def test_allomorph_adaptive_stopping_integration() -> None:
    class MockTrainer:
        def __init__(self) -> None:
            self.current_epoch: int = 0
            self.callback_metrics: dict[str, Any] = {}
            self.should_stop: bool = False

    callback = AllomorphAdaptiveStopping(
        warmup_floor=3,
        patience=3,
        min_delta=1e-5,
    )
    trainer = MockTrainer()

    # Epoch 0..2: warmup
    for ep in range(3):
        trainer.current_epoch = ep
        trainer.callback_metrics = {"val_loss": 0.010 - ep * 0.001}
        callback._run_early_stopping_check(trainer)
        assert trainer.should_stop is False
        assert "Warmup" in callback.get_status_str(ep)

    # Epoch 3..8: flat progression past warmup
    stopped = False
    for ep in range(3, 9):
        trainer.current_epoch = ep
        trainer.callback_metrics = {"val_loss": 0.0070001}
        callback._run_early_stopping_check(trainer)
        if trainer.should_stop:
            stopped = True
            assert "diminishing_returns_plateau" in callback.stop_reason
            break

    assert stopped is True


def test_extract_trainer_metrics_multiband() -> None:
    metrics = {
        "ESR_HP_high_packed_1": 0.0125,
        "ESR_HP_mid_packed_1": 0.0085,
        "mb_loss_packed_1": 0.0105,
        "mb_loss_packed_0": 0.0142,
        "val_loss": 0.0176,
    }
    extracted = extract_trainer_metrics(metrics)
    assert extracted["hp_high"] == 0.0125
    assert extracted["hp_mid"] == 0.0085
    assert extracted["mb_ch8"] == 0.0105
    assert extracted["mb_ch3"] == 0.0142
    assert extracted["val_loss"] == 0.0176


def test_high_frequency_differentiator_guard() -> None:
    class MockTrainer:
        def __init__(self) -> None:
            self.current_epoch: int = 0
            self.callback_metrics: dict[str, Any] = {}
            self.should_stop: bool = False

    callback = AllomorphAdaptiveStopping(
        warmup_floor=3,
        patience=3,
        min_delta=1e-4,
    )
    trainer = MockTrainer()

    # Epoch 0..2: warmup
    for ep in range(3):
        trainer.current_epoch = ep
        trainer.callback_metrics = {
            "val_loss": 0.020 - ep * 0.001,
            "ESR_HP_high_packed_1": 0.040 - ep * 0.005,
        }
        callback._run_early_stopping_check(trainer)
        assert trainer.should_stop is False

    # Epoch 3..6: val_loss is flat (0.0170), but hp_high is steeply descending
    # (differentiator actively forming)
    for ep in range(3, 7):
        trainer.current_epoch = ep
        trainer.callback_metrics = {
            "val_loss": 0.01700,  # Flat val_loss
            "ESR_HP_high_packed_1": 0.025 - (ep - 3) * 0.004,  # Steeply dropping hp_high
        }
        callback._run_early_stopping_check(trainer)
        # Even though patience (3) would normally expire at epoch 6, the HF guard MUST prevent stopping!
        assert trainer.should_stop is False

    # Now let hp_high also plateau for 4 epochs
    stopped = False
    for ep in range(7, 12):
        trainer.current_epoch = ep
        trainer.callback_metrics = {
            "val_loss": 0.01700,
            "ESR_HP_high_packed_1": 0.0090,  # Flat hp_high
        }
        callback._run_early_stopping_check(trainer)
        if trainer.should_stop:
            stopped = True
            break

    assert stopped is True


def test_adaptive_stopping_checkpoint_restoration() -> None:
    import torch
    from torch import nn

    class MockNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.fc = nn.Linear(2, 2, bias=False)

    class MockLightningModule:
        def __init__(self, net: MockNet) -> None:
            self._net = net

    class MockTrainer:
        def __init__(self, pl_module: MockLightningModule) -> None:
            self.current_epoch: int = 0
            self.callback_metrics: dict[str, Any] = {}
            self.should_stop: bool = False
            self.lightning_module = pl_module

    net = MockNet()
    with torch.no_grad():
        net.fc.weight.fill_(1.0)

    pl_mod = MockLightningModule(net)
    trainer = MockTrainer(pl_mod)

    callback = AllomorphAdaptiveStopping(
        warmup_floor=1,
        patience=3,
        min_delta=1e-4,
    )

    # Epoch 0: initial
    trainer.current_epoch = 0
    trainer.callback_metrics = {"val_loss": 0.020}
    callback._run_early_stopping_check(trainer)

    # Epoch 1: best epoch!
    with torch.no_grad():
        net.fc.weight.fill_(42.0)
    trainer.current_epoch = 1
    trainer.callback_metrics = {"val_loss": 0.010}  # New global best
    callback._run_early_stopping_check(trainer)
    assert callback.best_epoch == 1
    assert abs(callback.best_val_loss - 0.010) < 1e-12

    # Epoch 2..4: weights drift / overfit, loss degrades
    with torch.no_grad():
        net.fc.weight.fill_(999.0)
    for ep in range(2, 5):
        trainer.current_epoch = ep
        trainer.callback_metrics = {"val_loss": 0.015}  # Degraded loss
        callback._run_early_stopping_check(trainer)

    # Currently net has degraded weights (999.0)
    assert float(net.fc.weight[0, 0].item()) == 999.0

    # Call on_train_end: must restore best weights from epoch 1 (42.0)!
    callback.on_train_end(trainer, pl_mod)
    assert float(net.fc.weight[0, 0].item()) == 42.0


def test_compute_pareto_quality_score() -> None:
    # 1. Zero or negative min_mse_full returns unpenalized loss
    assert compute_pareto_quality_score(0.010, 0.0001, 0.0) == 0.010
    assert compute_pareto_quality_score(0.010, 0.0001, -1.0) == 0.010

    # 2. Exact match (zero drift) returns unpenalized loss
    assert compute_pareto_quality_score(0.010, 0.0001, 0.0001) == 0.010

    # 3. Drift within tolerance (5% drift with 8% tolerance) returns unpenalized loss
    score_within = compute_pareto_quality_score(
        mb_full_loss=0.010,
        mse_full=0.000105,
        min_mse_full=0.000100,
        drift_tolerance=0.08,
    )
    assert abs(score_within - 0.010) < 1e-12

    # 4. Drift exceeding tolerance (18% drift with 8% tolerance, gain 4.0):
    # excess = 0.18 - 0.08 = 0.10 -> penalty multiplier = 1.0 + 4.0 * 0.10 = 1.40
    # Q = 0.010 * 1.40 = 0.014
    score_penalized = compute_pareto_quality_score(
        mb_full_loss=0.010,
        mse_full=0.000118,
        min_mse_full=0.000100,
        drift_tolerance=0.08,
        drift_penalty_gain=4.0,
    )
    assert abs(score_penalized - 0.014) < 1e-12


def test_waveform_degradation_guard() -> None:
    # Warmup floor check: never stop during warmup
    should_stop, reason = evaluate_plateau_stopping_condition(
        val_loss_history=[0.01],
        val_loss_slope=0.0,
        epoch=2,
        warmup_floor=5,
        patience=3,
        min_delta=1e-4,
        hp_high_slope=0.0,
        mse_drift_ratio=0.50,
    )
    assert should_stop is False
    assert reason is None

    # Treble still descending: HF guard takes priority over drift
    should_stop, reason = evaluate_plateau_stopping_condition(
        val_loss_history=[0.01, 0.01, 0.01, 0.01],
        val_loss_slope=-1e-4,
        epoch=10,
        warmup_floor=5,
        patience=3,
        min_delta=1e-4,
        hp_high_slope=-1.0e-4,  # Treble actively forming
        mse_drift_ratio=0.35,  # High drift
    )
    assert should_stop is False
    assert reason is None

    # Treble flat (slope >= -5e-5) AND waveform drift > 25% -> immediate exit!
    should_stop, reason = evaluate_plateau_stopping_condition(
        val_loss_history=[0.01, 0.01, 0.01, 0.01],
        val_loss_slope=0.0,
        epoch=10,
        warmup_floor=5,
        patience=3,
        min_delta=1e-4,
        hp_high_slope=-1.0e-5,  # Treble converged/flat
        mse_drift_ratio=0.28,  # > 25% drift
    )
    assert should_stop is True
    assert reason == "waveform_mse_drift_detected"


def test_pareto_checkpoint_selection_avoids_drifted_checkpoint() -> None:
    import torch
    from torch import nn

    class MockNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.fc = nn.Linear(2, 2, bias=False)

    class MockLightningModule:
        def __init__(self, net: MockNet) -> None:
            self._net = net

    class MockTrainer:
        def __init__(self, pl_module: MockLightningModule) -> None:
            self.current_epoch: int = 0
            self.callback_metrics: dict[str, Any] = {}
            self.should_stop: bool = False
            self.lightning_module = pl_module

    net = MockNet()
    pl_mod = MockLightningModule(net)
    trainer = MockTrainer(pl_mod)

    callback = AllomorphAdaptiveStopping(
        warmup_floor=1,
        patience=4,
        min_delta=1e-4,
        drift_tolerance=0.08,
        drift_penalty_gain=4.0,
    )

    # Epoch 0: initial baseline (min MSE = 0.00010)
    with torch.no_grad():
        net.fc.weight.fill_(1.0)
    trainer.current_epoch = 0
    trainer.callback_metrics = {
        "val_loss": 0.050,
        "mb_loss_packed_1": 0.050,
        "MSE_packed_1": 0.00010,
    }
    callback._run_early_stopping_check(trainer)

    # Epoch 1: Pareto-Optimal Epoch!
    # Low MSE (0.00010), excellent balance (mb_loss = 0.020)
    with torch.no_grad():
        net.fc.weight.fill_(77.0)
    trainer.current_epoch = 1
    trainer.callback_metrics = {
        "val_loss": 0.020,
        "mb_loss_packed_1": 0.020,
        "MSE_packed_1": 0.00010,
    }
    callback._run_early_stopping_check(trainer)
    assert callback.best_epoch == 1
    assert abs(callback.best_pareto_score - 0.020) < 1e-12

    # Epoch 2: Treble overfit / Sub-bass drift!
    # Unweighted raw loss drops to 0.018 (slightly better treble),
    # BUT MSE drifts to 0.000130 (+30% drift, which shifts sub-bass by +3 dB in physical audio).
    # Pareto penalty: excess = 0.30 - 0.08 = 0.22 -> multiplier = 1 + 4.0 * 0.22 = 1.88
    # Pareto Q = 0.018 * 1.88 = 0.03384 > 0.020
    # Must NOT become the new best!
    with torch.no_grad():
        net.fc.weight.fill_(999.0)
    trainer.current_epoch = 2
    trainer.callback_metrics = {
        "val_loss": 0.018,
        "mb_loss_packed_1": 0.018,
        "MSE_packed_1": 0.000130,
    }
    callback._run_early_stopping_check(trainer)

    # Verification: best_epoch must still be Epoch 1!
    assert callback.best_epoch == 1
    assert abs(callback.best_pareto_score - 0.020) < 1e-12

    # Verify on_train_end restores the Pareto-optimal weights (77.0) and rejects the drifted weights (999.0)!
    callback.on_train_end(trainer, pl_mod)
    assert float(net.fc.weight[0, 0].item()) == 77.0
