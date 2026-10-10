"""
Deterministic unit tests for Allomorph trainer early stopping callbacks and pure helper functions.
"""

from typing import Any

from allomorph.trainer.callbacks import (
    AllomorphAdaptiveStopping,
    compute_linear_slope,
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
    # Explicit val_loss omitted; should compute esr + esr_ch3
    metrics = {
        "ESR": 0.0020,
        "ESR_packed_0": 0.0060,
    }
    extracted = extract_trainer_metrics(metrics)
    assert extracted["esr"] == 0.0020
    assert extracted["mrstft"] is None
    assert extracted["esr_ch3"] == 0.0060
    assert extracted["val_loss"] is not None
    assert abs(extracted["val_loss"] - 0.0080) < 1e-12

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
            assert callback.stop_reason == "diminishing_returns_plateau"
            break

    assert stopped is True
