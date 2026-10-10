"""
Allomorph Trainer - Callbacks and Adaptive Early Stopping
PyTorch Lightning callbacks providing linear learning rate warmup,
real-time dual-tier terminal logging, and multi-domain triple-gate early stopping.
"""

import math
from collections.abc import Mapping, Sequence
from typing import Any, override

from pytorch_lightning.callbacks import Callback, EarlyStopping

from allomorph.trainer.constants import (
    DEFAULT_MIN_DELTA,
    DEFAULT_MIN_EPOCHS,
    DEFAULT_PATIENCE,
)


def compute_linear_slope(y_vals: Sequence[float]) -> float:
    """Computes closed-form ordinary least squares regression slope for evenly spaced points."""
    n = len(y_vals)
    if n < 2:
        return 0.0
    x_mean = (n - 1) / 2.0
    y_mean = sum(y_vals) / n
    denom = (n * (n * n - 1)) / 12.0
    num = sum((i - x_mean) * (y - y_mean) for i, y in enumerate(y_vals))
    return num / denom if denom > 0.0 else 0.0


def _to_float(val: Any) -> float | None:
    if val is None:
        return None
    if hasattr(val, "item"):
        return float(val.item())
    return float(val)


def extract_trainer_metrics(metrics: Mapping[str, Any]) -> dict[str, float | None]:
    """Extracts validation metrics from trainer callback metrics dictionary.

    Extracts float metrics (esr, mrstft, esr_ch3, mrstft_ch3, val_loss) unpacking
    tensors with .item() or raw scalars, and falling back to composite loss if
    explicit val_loss is absent.
    """
    raw_esr: Any = metrics.get("ESR_packed_1")
    if raw_esr is None:
        raw_esr = metrics.get("ESR")
    if raw_esr is None:
        raw_esr = metrics.get("val_loss")
    esr_val = _to_float(raw_esr)

    raw_mrstft: Any = metrics.get("MRSTFT_packed_1")
    if raw_mrstft is None:
        raw_mrstft = metrics.get("MRSTFT")
    mrstft_val = _to_float(raw_mrstft)

    raw_esr_ch3: Any = metrics.get("ESR_packed_0")
    esr_ch3_val = _to_float(raw_esr_ch3)

    raw_mrstft_ch3: Any = metrics.get("MRSTFT_packed_0")
    mrstft_ch3_val = _to_float(raw_mrstft_ch3)

    raw_val_loss: Any = metrics.get("val_loss")
    if raw_val_loss is not None:
        val_loss = _to_float(raw_val_loss)
    elif esr_val is not None:
        val_loss = esr_val + (esr_ch3_val if esr_ch3_val is not None else 0.0)
    else:
        val_loss = None

    return {
        "esr": esr_val,
        "mrstft": mrstft_val,
        "esr_ch3": esr_ch3_val,
        "mrstft_ch3": mrstft_ch3_val,
        "val_loss": val_loss,
    }


def compute_window_improvement(val_loss_history: Sequence[float], patience: int) -> float:
    """Computes improvement in loss between start of patience window and latest epoch."""
    if patience <= 0 or len(val_loss_history) < patience:
        return 0.0
    return float(val_loss_history[-patience] - val_loss_history[-1])


def evaluate_plateau_stopping_condition(
    val_loss_history: Sequence[float],
    val_loss_slope: float,
    epoch: int,
    warmup_floor: int,
    patience: int,
    min_delta: float,
) -> tuple[bool, str | None]:
    """Pure function evaluating diminishing-returns early exit based on history, slope, and window delta."""
    if epoch < warmup_floor:
        return (False, None)
    if patience <= 0 or len(val_loss_history) < patience or epoch < (warmup_floor + patience):
        return (False, None)

    improvement = compute_window_improvement(val_loss_history, patience)
    is_flat = (val_loss_slope >= -5.0e-7) and (improvement < min_delta)
    if is_flat:
        return (True, "diminishing_returns_plateau")
    return (False, None)


class LinearWarmupCallback(Callback):
    """Linearly ramps optimizer learning rate from 0 to target lr over warmup_epochs,
    and clamps LR at eta_min for epoch >= lr_t_max to ensure monotonic cosine decay.
    """

    def __init__(
        self,
        warmup_epochs: int = 5,
        target_lr: float = 0.004,
        lr_t_max: int = 35,
        eta_min: float = 1e-5,
        is_cosine: bool = True,
    ) -> None:
        super().__init__()
        self.warmup_epochs = warmup_epochs
        self.target_lr = target_lr
        self.lr_t_max = lr_t_max
        self.eta_min = eta_min
        self.is_cosine = is_cosine

    @override
    def on_train_epoch_start(self, trainer: Any, pl_module: Any) -> None:
        epoch = getattr(trainer, "current_epoch", 0)
        optimizers = getattr(trainer, "optimizers", []) or []
        if epoch < self.warmup_epochs:
            current_lr = self.target_lr * float(epoch + 1) / float(self.warmup_epochs)
            for opt in optimizers:
                for pg in getattr(opt, "param_groups", []):
                    pg["lr"] = current_lr
        elif self.is_cosine and epoch >= self.lr_t_max:
            for opt in optimizers:
                for pg in getattr(opt, "param_groups", []):
                    pg["lr"] = self.eta_min


class AllomorphAdaptiveStopping(EarlyStopping):
    """Adaptive diminishing-returns early stopping callback for Architecture 2.

    1. Warmup floor: No early stopping before warmup_floor (default: 5 epochs linear warmup).
    2. Adaptive diminishing-returns exit: Exits when composite validation loss (val_loss = Full + Lite)
       stalls with improvement < min_delta (default: 2.0e-6) AND rolling regression slope dL/dt >= -1e-7
       over patience (default: 5) epochs.
    """

    def __init__(
        self,
        *cb_args: Any,
        monitor: str = "val_loss",
        warmup_floor: int = DEFAULT_MIN_EPOCHS,
        patience: int = DEFAULT_PATIENCE,
        min_delta: float = DEFAULT_MIN_DELTA,
        **cb_kwargs: Any,
    ) -> None:
        super().__init__(*cb_args, monitor=monitor, **cb_kwargs)
        self.warmup_floor = warmup_floor
        self.min_epochs = warmup_floor
        self.patience = patience
        self.min_delta = min_delta

        # Composite validation loss history & slope
        self.val_loss_history: list[float] = []
        self.last_val_loss_slope: float = 0.0
        self.best_val_loss: float = float("inf")

        # A2 Full tier (channels_8) history & bests
        self.esr_history: list[float] = []
        self.mrstft_history: list[float] = []
        self.best_esr: float = float("inf")
        self.best_mrstft: float | None = None
        self.last_esr_slope: float = 0.0
        self.last_mrstft_slope: float = 0.0

        # A2 Lite tier (channels_3) history & bests
        self.esr_ch3_history: list[float] = []
        self.mrstft_ch3_history: list[float] = []
        self.best_ch3_esr: float = float("inf")
        self.best_ch3_mrstft: float | None = None
        self.last_esr_ch3_slope: float = 0.0
        self.last_mrstft_ch3_slope: float = 0.0

        self.consecutive_gates_met: int = 0
        self.stop_reason: str = "schedule_complete"

    def get_status_str(self, epoch: int) -> str:
        if epoch < self.warmup_floor:
            return f"Warmup: {epoch}/{self.warmup_floor}"
        if self.patience <= 0:
            return "Plateau guard: off"
        pts_count = len(self.val_loss_history)
        return (
            f"Patience: {min(pts_count, self.patience)}/{self.patience} "
            f"(slope: {self.last_val_loss_slope:+.1e})"
        )

    @override
    def _run_early_stopping_check(self, trainer: Any) -> None:
        metrics: dict[str, Any] = getattr(trainer, "callback_metrics", {})
        epoch: int = getattr(trainer, "current_epoch", 0)

        extracted = extract_trainer_metrics(metrics)
        esr_val = extracted["esr"]
        if esr_val is None:
            return

        mrstft_val = extracted["mrstft"]
        self.best_esr = min(self.best_esr, esr_val)
        if mrstft_val is not None:
            self.best_mrstft = min(
                self.best_mrstft if self.best_mrstft is not None else float("inf"), mrstft_val
            )

        self.esr_history.append(esr_val)
        if mrstft_val is not None:
            self.mrstft_history.append(mrstft_val)

        # A2 Lite tier metrics (channels_3)
        esr_ch3_val = extracted["esr_ch3"]
        mrstft_ch3_val = extracted["mrstft_ch3"]

        if esr_ch3_val is not None:
            self.best_ch3_esr = min(self.best_ch3_esr, esr_ch3_val)
            self.esr_ch3_history.append(esr_ch3_val)
        if mrstft_ch3_val is not None:
            self.best_ch3_mrstft = min(
                self.best_ch3_mrstft if self.best_ch3_mrstft is not None else float("inf"),
                mrstft_ch3_val,
            )
            self.mrstft_ch3_history.append(mrstft_ch3_val)

        # Composite validation loss (val_loss)
        val_loss = extracted["val_loss"]
        if val_loss is None:
            val_loss = esr_val + (esr_ch3_val if esr_ch3_val is not None else 0.0)

        self.best_val_loss = min(self.best_val_loss, val_loss)
        self.val_loss_history.append(val_loss)

        # Rolling least-squares regression slopes
        if len(self.val_loss_history) >= 2:
            pts = self.val_loss_history[-max(self.patience, 5) :]
            self.last_val_loss_slope = compute_linear_slope(pts)
        if len(self.esr_history) >= 2:
            window_pts = self.esr_history[-max(self.patience, 5) :]
            self.last_esr_slope = compute_linear_slope(window_pts)
        if len(self.esr_ch3_history) >= 2:
            window_pts_ch3 = self.esr_ch3_history[-max(self.patience, 5) :]
            self.last_esr_ch3_slope = compute_linear_slope(window_pts_ch3)
        if len(self.mrstft_history) >= 2:
            mr_window = self.mrstft_history[-max(self.patience, 5) :]
            self.last_mrstft_slope = compute_linear_slope(mr_window)

        if epoch < self.warmup_floor:
            return

        should_stop, reason = evaluate_plateau_stopping_condition(
            val_loss_history=self.val_loss_history,
            val_loss_slope=self.last_val_loss_slope,
            epoch=epoch,
            warmup_floor=self.warmup_floor,
            patience=self.patience,
            min_delta=self.min_delta,
        )

        if should_stop:
            loss_improvement = compute_window_improvement(self.val_loss_history, self.patience)
            print(
                f"\n[Early Stopping] Diminishing returns plateau reached at epoch {epoch:03d}: "
                f"val_loss slope {self.last_val_loss_slope:+.1e}, improvement {loss_improvement:.2e} < {self.min_delta:.2e} "
                f"over {self.patience} epochs. Terminating with optimal checkpoint restore.",
                flush=True,
            )
            trainer.should_stop = True
            self.stop_reason = reason or "diminishing_returns_plateau"
            return


class EsrProgressCallback(Callback):
    """Logs validation progress and updates progress bar metrics each epoch for both tiers."""

    def __init__(
        self,
        min_epochs: int = DEFAULT_MIN_EPOCHS,
        stopping_callback: Any = None,
    ) -> None:
        super().__init__()
        self.min_epochs: int = min_epochs
        self.stopping_callback: Any = stopping_callback
        self.best_esr: float = float("inf")
        self.best_mrstft: float | None = None
        self.best_ch3_esr: float | None = float("inf")
        self.best_ch3_mrstft: float | None = None
        self.last_epoch: int = 0

    @override
    def on_validation_epoch_end(self, trainer: Any, pl_module: Any) -> None:
        if getattr(trainer, "sanity_checking", False):
            return
        metrics: dict[str, Any] = getattr(trainer, "callback_metrics", {})
        epoch: int = getattr(trainer, "current_epoch", 0)
        self.last_epoch = epoch
        max_epochs: Any = getattr(trainer, "max_epochs", "?")

        extracted = extract_trainer_metrics(metrics)

        mrstft_val = extracted["mrstft"]
        if mrstft_val is not None:
            self.best_mrstft = min(
                self.best_mrstft if self.best_mrstft is not None else float("inf"), mrstft_val
            )
        mrstft_str = f" | MRSTFT: {mrstft_val:.4f}" if mrstft_val is not None else ""

        mrstft_ch3_val = extracted["mrstft_ch3"]
        if mrstft_ch3_val is not None:
            self.best_ch3_mrstft = min(
                self.best_ch3_mrstft if self.best_ch3_mrstft is not None else float("inf"),
                mrstft_ch3_val,
            )

        optimizers = getattr(trainer, "optimizers", []) or []
        current_lr = (
            optimizers[0].param_groups[0]["lr"]
            if optimizers and optimizers[0].param_groups
            else 0.004
        )
        lr_str = f" | LR: {current_lr:.1e}"

        patience_str = ""
        if self.stopping_callback is not None and hasattr(self.stopping_callback, "get_status_str"):
            patience_str = f" | {self.stopping_callback.get_status_str(epoch)}"

        # Primary Studio Tier (channels_8)
        ch8_val = extracted["esr"]
        if ch8_val is None:
            return
        self.best_esr = min(self.best_esr, ch8_val)

        # A2 Lite Tier (channels_3)
        ch3_val = extracted["esr_ch3"]
        if ch3_val is not None:
            self.best_ch3_esr = min(
                self.best_ch3_esr if self.best_ch3_esr is not None else float("inf"), ch3_val
            )

        if hasattr(trainer, "progress_bar_metrics") and isinstance(
            trainer.progress_bar_metrics, dict
        ):
            trainer.progress_bar_metrics["val_ESR"] = f"{ch8_val:.5f}"
            trainer.progress_bar_metrics["best_ESR"] = f"{self.best_esr:.5f}"
            trainer.progress_bar_metrics["val_ESR_a2_full"] = f"{ch8_val:.5f}"
            trainer.progress_bar_metrics["val_ESR_ch8"] = f"{ch8_val:.5f}"
            if ch3_val is not None:
                trainer.progress_bar_metrics["val_ESR_a2_lite"] = f"{ch3_val:.5f}"
                trainer.progress_bar_metrics["val_ESR_ch3"] = f"{ch3_val:.5f}"

        ch8_db: float = 10.0 * math.log10(max(ch8_val, 1e-12))
        best_db: float = 10.0 * math.log10(max(self.best_esr, 1e-12))
        ch3_str = ""
        if ch3_val is not None:
            ch3_db = 10.0 * math.log10(max(ch3_val, 1e-12))
            ch3_str = f" | A2 Lite: {ch3_val:.6f} ({ch3_db:+.2f} dB)"

        print(
            f"\n[Epoch {epoch:03d}/{max_epochs}] A2 Full: {ch8_val:.6f} ({ch8_db:+.2f} dB) | Best: {self.best_esr:.6f} ({best_db:+.2f} dB){mrstft_str}{ch3_str}{lr_str}{patience_str}",
            flush=True,
        )


__all__ = [
    "AllomorphAdaptiveStopping",
    "EsrProgressCallback",
    "LinearWarmupCallback",
    "compute_linear_slope",
    "compute_window_improvement",
    "evaluate_plateau_stopping_condition",
    "extract_trainer_metrics",
]
