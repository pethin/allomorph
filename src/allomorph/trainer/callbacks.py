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
    DEFAULT_LR_T_MAX,
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

    Extracts float metrics (spectral_ch8, spectral_ch3, esr, mrstft, esr_pre,
    esr_ch3, mrstft_ch3, esr_pre_ch3, hp_high, hp_mid, mb_ch8, mb_ch3, val_loss,
    val_mb_loss, val_spectral_loss) unpacking tensors with .item() or raw scalars,
    and falling back to composite loss if explicit val_loss is absent.
    """
    raw_esr: Any = metrics.get("ESR_packed_1")
    if raw_esr is None:
        raw_esr = metrics.get("ESR")
    esr_val = _to_float(raw_esr)

    raw_mrstft: Any = metrics.get("MRSTFT_packed_1")
    if raw_mrstft is None:
        raw_mrstft = metrics.get("MRSTFT")
    mrstft_val = _to_float(raw_mrstft)

    raw_esr_pre: Any = metrics.get("ESR_pre_packed_1")
    if raw_esr_pre is None:
        raw_esr_pre = metrics.get("ESR_pre")
    if raw_esr_pre is None:
        raw_esr_pre = metrics.get("Pre-emphasized MSE_packed_1")
    esr_pre_val = _to_float(raw_esr_pre)

    raw_hp_high: Any = metrics.get("ESR_HP_high_packed_1")
    if raw_hp_high is None:
        raw_hp_high = metrics.get("ESR_HP_high")
    hp_high_val = _to_float(raw_hp_high)

    raw_hp_mid: Any = metrics.get("ESR_HP_mid_packed_1")
    if raw_hp_mid is None:
        raw_hp_mid = metrics.get("ESR_HP_mid")
    hp_mid_val = _to_float(raw_hp_mid)

    raw_spectral_ch8: Any = metrics.get("Spectral_packed_1")
    if raw_spectral_ch8 is None:
        raw_spectral_ch8 = metrics.get("val_loss_packed_1")
    spectral_ch8_val = _to_float(raw_spectral_ch8)
    if spectral_ch8_val is None and (esr_pre_val is not None or mrstft_val is not None):
        e_pre = esr_pre_val if esr_pre_val is not None else 0.0
        m = mrstft_val if mrstft_val is not None else 0.0
        e_raw = esr_val if esr_val is not None else 0.0
        spectral_ch8_val = 0.50 * e_pre + 0.40 * m + 0.10 * e_raw

    raw_esr_ch3: Any = metrics.get("ESR_packed_0")
    esr_ch3_val = _to_float(raw_esr_ch3)

    raw_mrstft_ch3: Any = metrics.get("MRSTFT_packed_0")
    mrstft_ch3_val = _to_float(raw_mrstft_ch3)

    raw_esr_pre_ch3: Any = metrics.get("ESR_pre_packed_0")
    if raw_esr_pre_ch3 is None:
        raw_esr_pre_ch3 = metrics.get("Pre-emphasized MSE_packed_0")
    esr_pre_ch3_val = _to_float(raw_esr_pre_ch3)

    raw_spectral_ch3: Any = metrics.get("Spectral_packed_0")
    if raw_spectral_ch3 is None:
        raw_spectral_ch3 = metrics.get("val_loss_packed_0")
    spectral_ch3_val = _to_float(raw_spectral_ch3)
    if spectral_ch3_val is None and (esr_pre_ch3_val is not None or mrstft_ch3_val is not None):
        e_pre_3 = esr_pre_ch3_val if esr_pre_ch3_val is not None else 0.0
        m_3 = mrstft_ch3_val if mrstft_ch3_val is not None else 0.0
        e_raw_3 = esr_ch3_val if esr_ch3_val is not None else 0.0
        spectral_ch3_val = 0.50 * e_pre_3 + 0.40 * m_3 + 0.10 * e_raw_3

    raw_val_loss: Any = metrics.get("val_loss")
    if raw_val_loss is None:
        raw_val_loss = metrics.get("val_mb_loss")
    if raw_val_loss is None:
        raw_val_loss = metrics.get("val_spectral_loss")
    val_loss = _to_float(raw_val_loss)

    if val_loss is None:
        if esr_val is not None:
            val_loss = esr_val + (0.50 * esr_ch3_val if esr_ch3_val is not None else 0.0)
        elif spectral_ch8_val is not None:
            val_loss = spectral_ch8_val + (
                0.50 * spectral_ch3_val if spectral_ch3_val is not None else 0.0
            )

    if esr_val is None and val_loss is not None:
        esr_val = val_loss

    raw_mb_ch8 = metrics.get("mb_loss_packed_1")
    if raw_mb_ch8 is None:
        raw_mb_ch8 = metrics.get("val_loss_packed_1")
    mb_ch8_val = _to_float(raw_mb_ch8)

    raw_mb_ch3 = metrics.get("mb_loss_packed_0")
    if raw_mb_ch3 is None:
        raw_mb_ch3 = metrics.get("val_loss_packed_0")
    mb_ch3_val = _to_float(raw_mb_ch3)

    raw_mse = metrics.get("MSE_packed_1")
    if raw_mse is None:
        raw_mse = metrics.get("MSE")
    mse_val = _to_float(raw_mse)

    return {
        "esr": esr_val,
        "mrstft": mrstft_val,
        "esr_pre": esr_pre_val,
        "hp_high": hp_high_val,
        "hp_mid": hp_mid_val,
        "mb_ch8": mb_ch8_val,
        "mb_ch3": mb_ch3_val,
        "mse_full": mse_val,
        "spectral_ch8": spectral_ch8_val,
        "esr_ch3": esr_ch3_val,
        "mrstft_ch3": mrstft_ch3_val,
        "esr_pre_ch3": esr_pre_ch3_val,
        "spectral_ch3": spectral_ch3_val,
        "val_loss": val_loss,
        "val_mb_loss": val_loss,
        "val_spectral_loss": val_loss,
    }


def compute_window_improvement(val_loss_history: Sequence[float], patience: int) -> float:
    """Computes improvement in loss between start of patience window and latest epoch."""
    if patience <= 0 or len(val_loss_history) < patience:
        return 0.0
    return float(val_loss_history[-patience] - val_loss_history[-1])


def compute_pareto_quality_score(
    mb_full_loss: float,
    mse_full: float,
    min_mse_full: float,
    drift_tolerance: float = 1.0,
    drift_penalty_gain: float = 2.5,
) -> float:
    """Computes the Anti-Drift Pareto Electroacoustic Quality Score Q(t).

    In audio digital twin neural modeling, time-domain waveform MSE anchors
    sub-bass fundamental excursion (20-100 Hz) and physical comb cancellation notches (1-3 kHz),
    while high-pass ESR terms drive differentiator tap formation (8-15 kHz).

    Q(t) penalizes checkpoints where time-domain waveform MSE has drifted upward
    beyond drift_tolerance (default 15%) from its post-warmup minimum.
    """
    if (
        min_mse_full <= 0.0
        or min_mse_full == float("inf")
        or mse_full <= min_mse_full * (1.0 + drift_tolerance)
    ):
        return float(mb_full_loss)

    excess = (mse_full - min_mse_full) / min_mse_full - drift_tolerance
    penalty_multiplier = 1.0 + drift_penalty_gain * excess
    return float(mb_full_loss * penalty_multiplier)


def evaluate_plateau_stopping_condition(
    val_loss_history: Sequence[float],
    val_loss_slope: float,
    epoch: int,
    warmup_floor: int,
    patience: int,
    min_delta: float,
    hp_high_slope: float = 0.0,
    mse_drift_ratio: float = 0.0,
    drift_exit_threshold: float = 0.25,
) -> tuple[bool, str | None]:
    """Pure function evaluating diminishing-returns early exit based on history, slope, and window delta."""
    if epoch < warmup_floor:
        return (False, None)
    if patience <= 0 or len(val_loss_history) < patience or epoch < (warmup_floor + patience):
        return (False, None)

    # 1. High-frequency differentiator protection: do not stop while treble differentiator is actively growing
    if hp_high_slope < -5.0e-5:
        return (False, None)

    # 2. Waveform degradation guard: if treble has converged and waveform MSE is drifting, stop early
    if mse_drift_ratio > drift_exit_threshold:
        return (True, "waveform_mse_drift_detected")

    # 3. Diminishing-returns plateau
    improvement = compute_window_improvement(val_loss_history, patience)
    is_flat = (val_loss_slope >= -1.0e-6) and (improvement < min_delta)
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
        lr_t_max: int = DEFAULT_LR_T_MAX,
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

    Guarantees:
    1. Warmup floor: No early stopping before warmup_floor (default: 35 epochs).
    2. Anti-drift Pareto quality score Q(t): Evaluates the Studio Full tier (channels_8)
       directly, penalizing checkpoints where time-domain waveform MSE (which governs
       sub-bass fundamental excursion and comb cancellation notches) has drifted upward
       beyond drift_tolerance (15%) from its post-warmup minimum.
    3. High-frequency differentiator guard: Defers early exit while high-treble ESR (ESR_HP_high)
       is actively descending (slope < -5.0e-5), ensuring differentiator taps have fully formed.
    4. Waveform degradation guard: Exits early if high-frequency differentiator has converged
       and waveform MSE has drifted by >25%, stopping low-end erosion and restoring the optimal
       checkpoint immediately.
    5. Diminishing-returns plateau exit: Exits when Pareto quality score Q(t) fails to improve
       by min_delta for patience consecutive epochs after warmup_floor.
    6. In-memory optimal Pareto checkpoint restore: Automatically caches model state_dict on new
       best Pareto quality score and restores it at training completion (on_train_end).
    """

    def __init__(
        self,
        *cb_args: Any,
        monitor: str = "val_loss",
        warmup_floor: int = DEFAULT_MIN_EPOCHS,
        patience: int = DEFAULT_PATIENCE,
        min_delta: float = DEFAULT_MIN_DELTA,
        drift_tolerance: float = 1.0,
        drift_penalty_gain: float = 2.5,
        drift_exit_threshold: float = 1.50,
        **cb_kwargs: Any,
    ) -> None:
        super().__init__(*cb_args, monitor=monitor, **cb_kwargs)
        self.warmup_floor = warmup_floor
        self.min_epochs = warmup_floor
        self.patience = patience
        self.min_delta = min_delta
        self.drift_tolerance = drift_tolerance
        self.drift_penalty_gain = drift_penalty_gain
        self.drift_exit_threshold = drift_exit_threshold

        # Tracking & State
        self.wait_count: int = 0
        self.best_epoch: int = 0
        self.best_val_loss: float = float("inf")
        self.best_pareto_score: float = float("inf")
        self.best_state_dict: dict[str, Any] | None = None
        self.val_loss_history: list[float] = []
        self.pareto_score_history: list[float] = []
        self.last_val_loss_slope: float = 0.0
        self.last_pareto_slope: float = 0.0

        # Sub-bass waveform tracking
        self.mse_history: list[float] = []
        self.min_mse_full: float = float("inf")
        self.last_mse_drift_ratio: float = 0.0

        # High-frequency differentiator telemetry
        self.hp_high_history: list[float] = []
        self.last_hp_high_slope: float = 0.0
        self.best_hp_high: float = float("inf")

        # Spectral tracking
        self.spectral_ch8_history: list[float] = []
        self.spectral_ch3_history: list[float] = []
        self.best_spectral_ch8: float = float("inf")
        self.best_spectral_ch3: float = float("inf")

        # A2 Full tier (channels_8) history & bests
        self.esr_history: list[float] = []
        self.mrstft_history: list[float] = []
        self.esr_pre_history: list[float] = []
        self.best_esr: float = float("inf")
        self.best_mrstft: float | None = None
        self.best_esr_pre: float | None = None
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
            return f"Warmup: {epoch + 1}/{self.warmup_floor}"
        if self.patience <= 0:
            return "Plateau guard: off"
        best_str = (
            f"ep {self.best_epoch:03d} (Q: {self.best_pareto_score:.5f})"
            if self.best_pareto_score < float("inf")
            else "none"
        )
        drift_str = (
            f", drift: {self.last_mse_drift_ratio * 100:+.1f}%"
            if self.last_mse_drift_ratio > 0
            else ""
        )
        return (
            f"Patience: {self.wait_count}/{self.patience} "
            f"(best: {best_str}{drift_str}, slope: {self.last_pareto_slope:+.1e})"
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

        esr_pre_val = extracted["esr_pre"]
        if esr_pre_val is not None:
            self.best_esr_pre = min(
                self.best_esr_pre if self.best_esr_pre is not None else float("inf"),
                esr_pre_val,
            )
            self.esr_pre_history.append(esr_pre_val)

        spec_ch8 = extracted["spectral_ch8"]
        if spec_ch8 is not None:
            self.best_spectral_ch8 = min(self.best_spectral_ch8, spec_ch8)
            self.spectral_ch8_history.append(spec_ch8)

        spec_ch3 = extracted["spectral_ch3"]
        if spec_ch3 is not None:
            self.best_spectral_ch3 = min(self.best_spectral_ch3, spec_ch3)
            self.spectral_ch3_history.append(spec_ch3)

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

        # High-frequency differentiator tracking
        hp_high_val = extracted.get("hp_high")
        if hp_high_val is not None:
            self.best_hp_high = min(self.best_hp_high, hp_high_val)
            self.hp_high_history.append(hp_high_val)
            if len(self.hp_high_history) >= 2:
                hp_window = self.hp_high_history[-max(self.patience, 4) :]
                self.last_hp_high_slope = compute_linear_slope(hp_window)

        # Waveform MSE tracking (baseline established post-warmup)
        mse_val = extracted.get("mse_full")
        if mse_val is not None and mse_val > 0.0:
            self.mse_history.append(mse_val)
            if epoch >= self.warmup_floor:
                if self.min_mse_full == float("inf"):
                    self.min_mse_full = mse_val
                else:
                    self.min_mse_full = min(self.min_mse_full, mse_val)
                if self.min_mse_full > 0.0:
                    self.last_mse_drift_ratio = max(
                        0.0, (mse_val - self.min_mse_full) / self.min_mse_full
                    )

        # Multi-Band Loss for Full Tier (primary studio model)
        mb_ch8_val = extracted.get("mb_ch8")
        val_loss = extracted["val_loss"]
        if val_loss is None:
            val_loss = esr_val + (esr_ch3_val if esr_ch3_val is not None else 0.0)

        target_studio_loss = mb_ch8_val if mb_ch8_val is not None else val_loss

        # Compute Anti-Drift Pareto Electroacoustic Quality Score Q(t)
        if mse_val is not None and self.min_mse_full < float("inf"):
            pareto_score = compute_pareto_quality_score(
                mb_full_loss=target_studio_loss,
                mse_full=mse_val,
                min_mse_full=self.min_mse_full,
                drift_tolerance=self.drift_tolerance,
                drift_penalty_gain=self.drift_penalty_gain,
            )
        else:
            pareto_score = target_studio_loss

        self.pareto_score_history.append(pareto_score)
        self.val_loss_history.append(val_loss)

        # Check for new global best Pareto Quality Score post-warmup (by at least min_delta)
        if epoch >= self.warmup_floor:
            is_new_best = (self.best_pareto_score - pareto_score) >= self.min_delta
            if is_new_best or self.best_pareto_score == float("inf"):
                self.best_pareto_score = pareto_score
                self.best_val_loss = val_loss
                self.best_epoch = epoch
                self.wait_count = 0
                pl_module = getattr(trainer, "lightning_module", None)
                if pl_module is not None and hasattr(pl_module, "_net"):
                    self.best_state_dict = {
                        k: v.cpu().clone() for k, v in pl_module._net.state_dict().items()
                    }
            else:
                self.wait_count += 1

        # Rolling least-squares regression slopes
        if len(self.pareto_score_history) >= 2:
            pts = self.pareto_score_history[-max(self.patience, 5) :]
            self.last_pareto_slope = compute_linear_slope(pts)
        if len(self.val_loss_history) >= 2:
            pts_val = self.val_loss_history[-max(self.patience, 5) :]
            self.last_val_loss_slope = compute_linear_slope(pts_val)
        if len(self.esr_history) >= 2:
            window_pts = self.esr_history[-max(self.patience, 5) :]
            self.last_esr_slope = compute_linear_slope(window_pts)
        if len(self.esr_ch3_history) >= 2:
            window_pts_ch3 = self.esr_ch3_history[-max(self.patience, 5) :]
            self.last_esr_ch3_slope = compute_linear_slope(window_pts_ch3)
        if len(self.mrstft_history) >= 2:
            mr_window = self.mrstft_history[-max(self.patience, 5) :]
            self.last_mrstft_slope = compute_linear_slope(mr_window)

        if epoch < self.warmup_floor or self.patience <= 0:
            return

        should_stop, reason = evaluate_plateau_stopping_condition(
            val_loss_history=self.pareto_score_history,
            val_loss_slope=self.last_pareto_slope,
            epoch=epoch,
            warmup_floor=self.warmup_floor,
            patience=self.patience,
            min_delta=self.min_delta,
            hp_high_slope=self.last_hp_high_slope,
            mse_drift_ratio=self.last_mse_drift_ratio,
            drift_exit_threshold=self.drift_exit_threshold,
        )

        # High-frequency differentiator protection:
        # If patience expired but high treble ESR is still actively descending, extend patience!
        if self.wait_count >= self.patience:
            if self.last_hp_high_slope < -5.0e-5:
                self.wait_count = max(0, self.patience - 2)
                print(
                    f"\n[Early Stopping] High-frequency differentiator actively descending (slope {self.last_hp_high_slope:+.1e}); extending patience.",
                    flush=True,
                )
                return

            should_stop = True
            reason = "diminishing_returns_plateau"

        if should_stop:
            loss_improvement = compute_window_improvement(self.pareto_score_history, self.patience)
            reason_str = reason or "diminishing_returns_plateau"
            print(
                f"\n[Early Stopping] {reason_str.replace('_', ' ').title()} triggered at epoch {epoch:03d}: "
                f"Pareto Q slope {self.last_pareto_slope:+.1e}, window delta {loss_improvement:.2e} "
                f"(drift: {self.last_mse_drift_ratio * 100:+.1f}%). Terminating with optimal checkpoint restore "
                f"(best epoch: {self.best_epoch:03d}, Pareto Q: {self.best_pareto_score:.6f}, val_loss: {self.best_val_loss:.6f}).\n",
                flush=True,
            )
            trainer.should_stop = True
            self.stop_reason = reason_str
            return

    @override
    def on_train_end(self, trainer: Any, pl_module: Any) -> None:
        if self.best_state_dict is not None and hasattr(pl_module, "_net"):
            pl_module._net.load_state_dict(self.best_state_dict)
            print(
                f"\n[Early Stopping] Successfully restored weights from optimal Pareto checkpoint "
                f"at epoch {self.best_epoch:03d} (Pareto Q: {self.best_pareto_score:.6f}, val_loss: {self.best_val_loss:.6f}).\n",
                flush=True,
            )


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
        self.best_spectral: float = float("inf")
        self.best_spectral_ch8: float = float("inf")
        self.best_spectral_ch3: float = float("inf")
        self.best_esr: float = float("inf")
        self.best_mrstft: float | None = None
        self.best_esr_pre: float | None = None
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

        # Primary Studio Tier (channels_8)
        ch8_spec = extracted["spectral_ch8"]
        if ch8_spec is None:
            ch8_spec = extracted["val_loss"]
        if ch8_spec is not None:
            self.best_spectral = min(self.best_spectral, ch8_spec)
            self.best_spectral_ch8 = min(self.best_spectral_ch8, ch8_spec)

        ch3_spec = extracted["spectral_ch3"]
        if ch3_spec is not None:
            self.best_spectral_ch3 = min(self.best_spectral_ch3, ch3_spec)

        ch8_val = extracted["esr"]
        if ch8_val is not None:
            self.best_esr = min(self.best_esr, ch8_val)

        ch3_val = extracted["esr_ch3"]
        if ch3_val is not None:
            self.best_ch3_esr = min(
                self.best_ch3_esr if self.best_ch3_esr is not None else float("inf"), ch3_val
            )

        esr_pre_val = extracted["esr_pre"]
        if esr_pre_val is not None:
            self.best_esr_pre = min(
                self.best_esr_pre if self.best_esr_pre is not None else float("inf"),
                esr_pre_val,
            )

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

        if hasattr(trainer, "progress_bar_metrics") and isinstance(
            trainer.progress_bar_metrics, dict
        ):
            if ch8_spec is not None:
                trainer.progress_bar_metrics["val_spectral"] = f"{ch8_spec:.5f}"
                trainer.progress_bar_metrics["val_spectral_a2_full"] = f"{ch8_spec:.5f}"
                trainer.progress_bar_metrics["val_spectral_ch8"] = f"{ch8_spec:.5f}"
            if ch8_val is not None:
                trainer.progress_bar_metrics["val_ESR"] = f"{ch8_val:.5f}"
                trainer.progress_bar_metrics["best_ESR"] = f"{self.best_esr:.5f}"
                trainer.progress_bar_metrics["val_ESR_a2_full"] = f"{ch8_val:.5f}"
                trainer.progress_bar_metrics["val_ESR_ch8"] = f"{ch8_val:.5f}"
            if ch3_val is not None:
                trainer.progress_bar_metrics["val_ESR_a2_lite"] = f"{ch3_val:.5f}"
                trainer.progress_bar_metrics["val_ESR_ch3"] = f"{ch3_val:.5f}"
            if ch3_spec is not None:
                trainer.progress_bar_metrics["val_spectral_a2_lite"] = f"{ch3_spec:.5f}"
                trainer.progress_bar_metrics["val_spectral_ch3"] = f"{ch3_spec:.5f}"

        # Multi-Band Loss Metrics
        ch8_mb = extracted.get("mb_ch8")
        ch3_mb = extracted.get("mb_ch3")
        val_loss = extracted.get("val_loss")
        hp_high_val = extracted.get("hp_high")

        if ch8_mb is not None:
            spec_str = f"A2 Full MB: {ch8_mb:.5f}"
        elif ch8_spec is not None:
            spec_str = f"A2 Full Spectral: {ch8_spec:.5f}"
        else:
            spec_str = f"A2 Full ESR: {ch8_val:.6f}" if ch8_val is not None else ""

        hp_str = f" (hp7k: {hp_high_val:.4f})" if hp_high_val is not None else ""

        ch3_str = ""
        if ch3_mb is not None:
            ch3_str = f" | A2 Lite MB: {ch3_mb:.5f}"
        elif ch3_spec is not None:
            ch3_str = f" | A2 Lite Spectral: {ch3_spec:.5f}"
        elif ch3_val is not None:
            ch3_db = 10.0 * math.log10(max(ch3_val, 1e-12))
            ch3_str = f" | A2 Lite: {ch3_val:.6f} ({ch3_db:+.2f} dB)"

        val_str = f" | Val: {val_loss:.5f}" if val_loss is not None else ""

        print(
            f"\n[Epoch {epoch:03d}/{max_epochs}] {spec_str}{hp_str}{mrstft_str}{ch3_str}{val_str}{lr_str}{patience_str}",
            flush=True,
        )


__all__ = [
    "AllomorphAdaptiveStopping",
    "EsrProgressCallback",
    "LinearWarmupCallback",
    "compute_linear_slope",
    "compute_pareto_quality_score",
    "compute_window_improvement",
    "evaluate_plateau_stopping_condition",
    "extract_trainer_metrics",
]
