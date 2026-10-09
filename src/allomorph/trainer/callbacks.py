"""
Allomorph Trainer - Callbacks and Adaptive Early Stopping
PyTorch Lightning callbacks providing linear learning rate warmup,
real-time dual-tier terminal logging, and multi-domain triple-gate early stopping.
"""

import math
from collections.abc import Sequence
from typing import Any, override

from pytorch_lightning.callbacks import Callback, EarlyStopping

from allomorph.trainer.constants import DEFAULT_MIN_EPOCHS


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
    """De-risked adaptive early stopping callback with Triple-Gate Multi-Domain verification.

    1. Warmup floor: No early stopping before warmup_floor (default: 5 epochs linear warmup).
    2. Triple-gate goal exit: Exits when:
       - Gate 1 (Time): Global ESR <= goal_esr (default: 0.00020)
       - Gate 2 (Difference): Delta ESR <= goal_delta_esr (default: 0.020)
       - Gate 3 (Spectral): MRSTFT / max(baseline_mrstft, 1e-6) <= goal_delta_mrstft (default: 0.50)
                           AND MRSTFT <= max_mrstft_ceiling (default: 0.320)
       held for consecutive_patience (default: 3) consecutive validation checks.
    3. De-risked plateau exit: Exits when BOTH time-domain ESR and spectral MRSTFT loss have stalled
       with flat rolling regression slope (dL/dt >= -1e-7) over patience (default: 12) epochs.
    """

    def __init__(
        self,
        *cb_args: Any,
        warmup_floor: int = DEFAULT_MIN_EPOCHS,
        goal_esr: float | None = None,
        goal_delta_esr: float | None = None,
        goal_delta_mrstft: float | None = None,
        max_mrstft_ceiling: float = 0.320,
        goal_esr_lite: float | None = None,
        goal_delta_esr_lite: float | None = None,
        goal_delta_mrstft_lite: float | None = None,
        max_mrstft_ceiling_lite: float = 0.450,
        consecutive_patience: int = 3,
        patience: int = 12,
        min_delta: float = 5e-6,
        baseline_delta_ratio: float = 0.015,
        baseline_mrstft: float = 0.400,
        **cb_kwargs: Any,
    ) -> None:
        super().__init__(*cb_args, **cb_kwargs)
        self.warmup_floor = warmup_floor
        self.min_epochs = warmup_floor
        self.stopping_threshold = goal_esr
        self.goal_esr = goal_esr
        self.goal_delta_esr = goal_delta_esr
        self.target_delta_esr = goal_delta_esr
        self.goal_delta_mrstft = goal_delta_mrstft
        self.max_mrstft_ceiling = max_mrstft_ceiling
        self.goal_esr_lite = goal_esr_lite
        self.goal_delta_esr_lite = goal_delta_esr_lite
        self.goal_delta_mrstft_lite = goal_delta_mrstft_lite
        self.max_mrstft_ceiling_lite = max_mrstft_ceiling_lite
        self.consecutive_patience = consecutive_patience
        self.patience = patience
        self.min_delta = min_delta
        self.baseline_delta_ratio = max(baseline_delta_ratio, 1e-6)
        self.baseline_mrstft = max(baseline_mrstft, 1e-6)
        self.is_identity = baseline_mrstft < 1e-6

        # A2 Full tier (channels_8) history & bests
        self.esr_history: list[float] = []
        self.mrstft_history: list[float] = []
        self.best_esr: float = float("inf")
        self.best_delta_esr: float = float("inf")
        self.best_mrstft: float | None = None
        self.best_diff_mrstft: float | None = None
        self.last_esr_slope: float = 0.0
        self.last_mrstft_slope: float = 0.0

        # A2 Lite tier (channels_3) history & bests
        self.esr_ch3_history: list[float] = []
        self.mrstft_ch3_history: list[float] = []
        self.best_ch3_esr: float = float("inf")
        self.best_ch3_delta_esr: float = float("inf")
        self.best_ch3_mrstft: float | None = None
        self.best_ch3_diff_mrstft: float | None = None
        self.last_esr_ch3_slope: float = 0.0
        self.last_mrstft_ch3_slope: float = 0.0

        self.consecutive_gates_met: int = 0
        self.stop_reason: str = "max_epochs"

    def get_status_str(self, epoch: int) -> str:
        if epoch < self.warmup_floor:
            return f"Warmup: {epoch}/{self.warmup_floor}"
        if self.consecutive_gates_met > 0:
            return f"Triple-Gate: {self.consecutive_gates_met}/{self.consecutive_patience}"
        if self.patience <= 0:
            return "Plateau guard: off"
        pts_count = len(self.esr_history)
        return (
            f"Patience: {min(pts_count, self.patience)}/{self.patience} "
            f"(slopes: full {self.last_esr_slope:+.1e}, lite {self.last_esr_ch3_slope:+.1e})"
        )

    @override
    def _run_early_stopping_check(self, trainer: Any) -> None:
        metrics: dict[str, Any] = getattr(trainer, "callback_metrics", {})
        epoch: int = getattr(trainer, "current_epoch", 0)

        # Studio tier metrics (channels_8)
        raw_esr: Any = metrics.get("ESR_packed_1") or metrics.get("ESR") or metrics.get("val_loss")
        if raw_esr is None:
            return
        esr_val = float(raw_esr.item() if hasattr(raw_esr, "item") else raw_esr)
        delta_esr_val = esr_val / self.baseline_delta_ratio

        raw_mrstft = metrics.get("MRSTFT_packed_1") or metrics.get("MRSTFT")
        mrstft_val = (
            float(raw_mrstft.item() if hasattr(raw_mrstft, "item") else raw_mrstft)
            if raw_mrstft is not None
            else None
        )

        self.best_esr = min(self.best_esr, esr_val)
        self.best_delta_esr = min(self.best_delta_esr, delta_esr_val)
        if mrstft_val is not None:
            self.best_mrstft = min(
                self.best_mrstft if self.best_mrstft is not None else float("inf"), mrstft_val
            )
            diff_mrstft = mrstft_val / self.baseline_mrstft
            self.best_diff_mrstft = min(
                self.best_diff_mrstft if self.best_diff_mrstft is not None else float("inf"),
                diff_mrstft,
            )

        self.esr_history.append(esr_val)
        if mrstft_val is not None:
            self.mrstft_history.append(mrstft_val)

        # A2 Lite tier metrics (channels_3)
        raw_esr_ch3: Any = metrics.get("ESR_packed_0")
        esr_ch3_val = (
            float(raw_esr_ch3.item() if hasattr(raw_esr_ch3, "item") else raw_esr_ch3)
            if raw_esr_ch3 is not None
            else None
        )
        delta_esr_ch3_val = (
            (esr_ch3_val / self.baseline_delta_ratio) if esr_ch3_val is not None else None
        )

        raw_mrstft_ch3 = metrics.get("MRSTFT_packed_0")
        mrstft_ch3_val = (
            float(raw_mrstft_ch3.item() if hasattr(raw_mrstft_ch3, "item") else raw_mrstft_ch3)
            if raw_mrstft_ch3 is not None
            else None
        )

        if esr_ch3_val is not None:
            self.best_ch3_esr = min(self.best_ch3_esr, esr_ch3_val)
            self.esr_ch3_history.append(esr_ch3_val)
        if delta_esr_ch3_val is not None:
            self.best_ch3_delta_esr = min(self.best_ch3_delta_esr, delta_esr_ch3_val)
        if mrstft_ch3_val is not None:
            self.best_ch3_mrstft = min(
                self.best_ch3_mrstft if self.best_ch3_mrstft is not None else float("inf"),
                mrstft_ch3_val,
            )
            diff_mr_ch3 = mrstft_ch3_val / self.baseline_mrstft
            self.best_ch3_diff_mrstft = min(
                self.best_ch3_diff_mrstft if self.best_ch3_diff_mrstft is not None else float("inf"),
                diff_mr_ch3,
            )
            self.mrstft_ch3_history.append(mrstft_ch3_val)

        # Rolling least-squares regression slopes
        if len(self.esr_history) >= 2:
            window_pts = self.esr_history[-max(self.patience, 5):]
            self.last_esr_slope = compute_linear_slope(window_pts)
        if len(self.esr_ch3_history) >= 2:
            window_pts_ch3 = self.esr_ch3_history[-max(self.patience, 5):]
            self.last_esr_ch3_slope = compute_linear_slope(window_pts_ch3)
        if len(self.mrstft_history) >= 2:
            mr_window = self.mrstft_history[-max(self.patience, 5):]
            self.last_mrstft_slope = compute_linear_slope(mr_window)

        if epoch < self.warmup_floor:
            return

        # 1. Dual-Tier Triple-Gate Goal Check (All 6 Gates)
        # A2 Full Tier (channels_8)
        gate1_ok = (self.goal_esr is None) or (esr_val <= self.goal_esr)
        gate2_ok = (self.goal_delta_esr is None) or (delta_esr_val <= self.goal_delta_esr)
        gate3_ok = True
        if not self.is_identity and self.goal_delta_mrstft is not None:
            if mrstft_val is None:
                gate3_ok = False
            else:
                diff_mr = mrstft_val / self.baseline_mrstft
                gate3_ok = (diff_mr <= self.goal_delta_mrstft) and (mrstft_val <= self.max_mrstft_ceiling)

        # A2 Lite Tier (channels_3)
        gate4_ok = (self.goal_esr_lite is None) or (esr_ch3_val is None) or (esr_ch3_val <= self.goal_esr_lite)
        gate5_ok = (self.goal_delta_esr_lite is None) or (delta_esr_ch3_val is None) or (delta_esr_ch3_val <= self.goal_delta_esr_lite)
        gate6_ok = True
        if (
            not self.is_identity
            and self.goal_delta_mrstft_lite is not None
            and mrstft_ch3_val is not None
        ):
            diff_mr_ch3 = mrstft_ch3_val / self.baseline_mrstft
            gate6_ok = (diff_mr_ch3 <= self.goal_delta_mrstft_lite) and (
                mrstft_ch3_val <= self.max_mrstft_ceiling_lite
            )

        all_gates_pass = gate1_ok and gate2_ok and gate3_ok and gate4_ok and gate5_ok and gate6_ok
        if all_gates_pass and (self.goal_esr is not None or self.goal_delta_esr is not None):
            self.consecutive_gates_met += 1
            if self.consecutive_gates_met >= self.consecutive_patience and epoch >= self.warmup_floor:
                esr_db = 10.0 * math.log10(max(esr_val, 1e-12))
                delta_db = 10.0 * math.log10(max(delta_esr_val, 1e-12))
                ch3_info = f", A2 Lite ESR {esr_ch3_val:.6f}" if esr_ch3_val is not None else ""
                mr_info = f", MRSTFT {mrstft_val:.4f}" if mrstft_val is not None else ""
                print(
                    f"\n[Dual-Tier Triple-Gate Achieved] Epoch {epoch:03d}: A2 Full ESR {esr_val:.6f} ({esr_db:+.2f} dB) "
                    f"AND Delta Nuance {delta_esr_val:.6f} ({delta_db:+.2f} dB){mr_info}{ch3_info} "
                    f"held for {self.consecutive_gates_met}/{self.consecutive_patience} consecutive epochs. "
                    "Terminating successfully with official A2 dual-tier studio fidelity!",
                    flush=True,
                )
                trainer.should_stop = True
                self.stop_reason = "dual_triple_gate_converged"
                return
        else:
            self.consecutive_gates_met = 0

        # 2. Joint Adaptive Plateau Check (Both Tiers Flat)
        if (
            self.patience > 0
            and len(self.esr_history) >= self.patience
            and epoch >= (self.warmup_floor + self.patience)
        ):
            recent_esr = self.esr_history[-self.patience:]
            esr_improvement = recent_esr[0] - esr_val
            studio_flat = (self.last_esr_slope >= -1e-7) and (esr_improvement < self.min_delta)

            lite_flat = True
            if len(self.esr_ch3_history) >= self.patience:
                recent_ch3 = self.esr_ch3_history[-self.patience:]
                ch3_improvement = recent_ch3[0] - self.esr_ch3_history[-1]
                lite_flat = (self.last_esr_ch3_slope >= -1e-7) and (ch3_improvement < self.min_delta * 5.0)

            empty_mrstft: list[float] = []
            recent_mrstft = (
                self.mrstft_history[-self.patience:]
                if len(self.mrstft_history) >= self.patience
                else empty_mrstft
            )
            mrstft_flat = (len(recent_mrstft) == 0) or (self.last_mrstft_slope >= -1e-7)

            if studio_flat and lite_flat and mrstft_flat:
                print(
                    f"\n[Early Stopping] Diminishing returns plateau reached at epoch {epoch:03d}: "
                    f"A2 Full slope {self.last_esr_slope:+.1e}, A2 Lite slope {self.last_esr_ch3_slope:+.1e} over {self.patience} epochs. "
                    "Terminating to preserve GPU efficiency.",
                    flush=True,
                )
                trainer.should_stop = True
                self.stop_reason = "joint_plateau_exit"
                return


class EsrProgressCallback(Callback):
    """Logs validation ESR progress and updates progress bar metrics each epoch for both tiers."""

    def __init__(
        self,
        target_esr: float | None = None,
        target_delta_esr: float | None = None,
        min_epochs: int = DEFAULT_MIN_EPOCHS,
        baseline_delta_ratio: float = 0.015,
        baseline_mrstft: float = 0.400,
        stopping_callback: Any = None,
    ) -> None:
        super().__init__()
        self.target_esr: float | None = target_esr
        self.target_delta_esr: float | None = target_delta_esr
        self.min_epochs: int = min_epochs
        self.baseline_delta_ratio: float = max(baseline_delta_ratio, 1e-6)
        self.baseline_mrstft: float = max(baseline_mrstft, 1e-6)
        self.stopping_callback: Any = stopping_callback
        self.best_esr: float = float("inf")
        self.best_delta_esr: float = float("inf")
        self.best_mrstft: float | None = None
        self.best_diff_mrstft: float | None = None
        self.best_ch3_esr: float | None = float("inf")
        self.best_ch3_delta_esr: float | None = float("inf")
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

        target_str: str = ""
        if self.target_esr is not None:
            target_db: float = 10.0 * math.log10(max(self.target_esr, 1e-12))
            min_ep_str = (
                f" (warmup: active after ep {self.min_epochs})"
                if epoch < self.min_epochs
                else " (warmup passed)"
            )
            target_str = f" | Target: {self.target_esr:.6f} ({target_db:+.2f} dB){min_ep_str}"

        raw_mrstft = metrics.get("MRSTFT_packed_1") or metrics.get("MRSTFT")
        mrstft_val = (
            float(raw_mrstft.item() if hasattr(raw_mrstft, "item") else raw_mrstft)
            if raw_mrstft is not None
            else None
        )
        if mrstft_val is not None:
            self.best_mrstft = min(
                self.best_mrstft if self.best_mrstft is not None else float("inf"), mrstft_val
            )
            diff_mrstft = mrstft_val / self.baseline_mrstft
            self.best_diff_mrstft = min(
                self.best_diff_mrstft if self.best_diff_mrstft is not None else float("inf"),
                diff_mrstft,
            )
        mrstft_str = f" | MRSTFT: {mrstft_val:.5f}" if mrstft_val is not None else ""

        raw_mrstft_ch3 = metrics.get("MRSTFT_packed_0")
        mrstft_ch3_val = (
            float(raw_mrstft_ch3.item() if hasattr(raw_mrstft_ch3, "item") else raw_mrstft_ch3)
            if raw_mrstft_ch3 is not None
            else None
        )
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

        gates_str = ""
        if self.stopping_callback is not None:
            sc = self.stopping_callback
            c_met = getattr(sc, "consecutive_gates_met", 0)
            c_pat = getattr(sc, "consecutive_patience", 3)
            if c_met > 0:
                gates_str = f" | [Gates: {c_met}/{c_pat}]"

        patience_str = ""
        if self.stopping_callback is not None and hasattr(self.stopping_callback, "get_status_str"):
            patience_str = f" | {self.stopping_callback.get_status_str(epoch)}"

        # Primary Studio Tier (channels_8)
        raw_ch8: Any = metrics.get("ESR_packed_1") or metrics.get("ESR") or metrics.get("val_loss")
        if raw_ch8 is None:
            return
        ch8_val: float = float(raw_ch8.item() if hasattr(raw_ch8, "item") else raw_ch8)
        self.best_esr = min(self.best_esr, ch8_val)
        delta_esr_val = ch8_val / self.baseline_delta_ratio
        self.best_delta_esr = min(self.best_delta_esr, delta_esr_val)

        # A2 Lite Tier (channels_3)
        raw_ch3: Any = metrics.get("ESR_packed_0")
        ch3_val: float | None = (
            float(raw_ch3.item() if hasattr(raw_ch3, "item") else raw_ch3)
            if raw_ch3 is not None
            else None
        )
        if ch3_val is not None:
            self.best_ch3_esr = min(
                self.best_ch3_esr if self.best_ch3_esr is not None else float("inf"), ch3_val
            )
            delta_ch3 = ch3_val / self.baseline_delta_ratio
            self.best_ch3_delta_esr = min(
                self.best_ch3_delta_esr if self.best_ch3_delta_esr is not None else float("inf"),
                delta_ch3,
            )

        if hasattr(trainer, "progress_bar_metrics") and isinstance(
            trainer.progress_bar_metrics, dict
        ):
            trainer.progress_bar_metrics["val_ESR"] = f"{ch8_val:.5f}"
            trainer.progress_bar_metrics["best_ESR"] = f"{self.best_esr:.5f}"
            trainer.progress_bar_metrics["val_ESR_a2_full"] = f"{ch8_val:.5f}"
            trainer.progress_bar_metrics["val_ESR_ch8"] = f"{ch8_val:.5f}"
            trainer.progress_bar_metrics["delta_ESR"] = f"{delta_esr_val:.5f}"
            if ch3_val is not None:
                trainer.progress_bar_metrics["val_ESR_a2_lite"] = f"{ch3_val:.5f}"
                trainer.progress_bar_metrics["val_ESR_ch3"] = f"{ch3_val:.5f}"

        ch8_db: float = 10.0 * math.log10(max(ch8_val, 1e-12))
        delta_db = 10.0 * math.log10(max(delta_esr_val, 1e-12))
        delta_str = f" | Delta: {delta_esr_val:.5f} ({delta_db:+.2f} dB)"
        ch3_str = ""
        if ch3_val is not None:
            ch3_db = 10.0 * math.log10(max(ch3_val, 1e-12))
            ch3_str = f" | A2 Lite (Ch3): {ch3_val:.6f} ({ch3_db:+.2f} dB)"

        print(
            f"\n[Epoch {epoch:03d}/{max_epochs}] A2 Full ESR: {ch8_val:.6f} ({ch8_db:+.2f} dB){delta_str}{mrstft_str}{ch3_str}{lr_str}{gates_str}{patience_str}{target_str}",
            flush=True,
        )


__all__ = [
    "AllomorphAdaptiveStopping",
    "EsrProgressCallback",
    "LinearWarmupCallback",
    "compute_linear_slope",
]
