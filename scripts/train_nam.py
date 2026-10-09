"""
Allomorph - Neural Amp Modeler (NAM) Architecture 2 (A2) Local Trainer
Trains a high-fidelity analog twin neural model using Apple Silicon Metal (MPS) GPU acceleration.
Includes full source instrument, scale length, and pickup routing metadata in the exported .nam container.
"""

import argparse
import math
import os
import shutil
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, override

REPO_ROOT = Path(__file__).resolve().parent.parent
CIRCUITS_DIR = REPO_ROOT / "circuits"
MODELS_DIR = REPO_ROOT / "models"
AUDIO_DIR = REPO_ROOT / "audio"
SCRIPTS_DIR = REPO_ROOT / "scripts"

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

# ROCm / MIOpen optimizations for AMD GPUs (e.g. RDNA 3/4, gfx1201)
# Bypasses multi-minute solver benchmarking on unchunked validation tensors and silences workspace warnings
os.environ.setdefault("MIOPEN_FIND_MODE", "FAST")
os.environ.setdefault("MIOPEN_LOG_LEVEL", "2")
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

# Parallelize MIOpen kernel compilation across all CPU cores
if "MIOPEN_COMPILE_PARALLEL_LEVEL" not in os.environ:
    cpu_count = os.cpu_count() or 4
    os.environ["MIOPEN_COMPILE_PARALLEL_LEVEL"] = str(min(cpu_count, 16))

# Guarantee MIOpen kernel cache directory exists in project .cache so compiled kernels persist across epochs
_miopen_cache_dir = REPO_ROOT / ".cache" / "miopen"
try:
    _miopen_cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MIOPEN_CACHE_DIR", str(_miopen_cache_dir))
    os.environ.setdefault("MIOPEN_USER_DB_PATH", str(_miopen_cache_dir))
except OSError:
    pass

from allomorph.config import (
    VOICES,
    InstrumentConfig,
    PickupConfig,
    compute_effective_position,
    get_source_pickup,
    load_instrument,
    resolve_voice_coils,
    resolve_voice_pickups,
)
from allomorph.naming import (
    resolve_instruments,
    resolve_voices,
)
from allomorph.pipeline.schema import (
    NamExportMetadata,
    NamSourceInstrumentMeta,
    NamSourcePickupMeta,
    NamTargetVoiceMeta,
    NamTrainingConfig,
    NamTrainingMetadata,
)
from allomorph.version import (
    ALLOMORPH_VERSION,
    DSP_GENERATION,
    get_git_commit,
    is_wet_stem_valid,
    resolve_tri_part_version,
    write_manifest,
)


def find_sweep_input(
    candidate_path: str | Path | None = None,
    version_tag: str | None = None,
) -> Path | None:
    if candidate_path and Path(candidate_path).exists():
        return Path(candidate_path)
    from allomorph.circuit.audio import find_default_input_audio

    return find_default_input_audio(version_tag=version_tag)


DEFAULT_GOAL_ESR = 0.00020  # Gate 1: Studio reference Global ESR (~ -37.0 dB ESR)
DEFAULT_GOAL_DELTA_ESR = 0.020  # Gate 2: Nuance target <= -17.0 dB residual on pickup transformation delta
DEFAULT_GOAL_DELTA_MRSTFT = 0.50  # Gate 3: Differential MRSTFT ratio (MRSTFT / max(baseline_mrstft, 1e-6))
DEFAULT_MAX_MRSTFT_CEILING = 0.320  # Gate 3: Absolute ceiling on validation MRSTFT
DEFAULT_CONSECUTIVE_PATIENCE = 3  # Consecutive validation checks meeting all gates before early exit
DEFAULT_MIN_EPOCHS = 5  # Warmup epoch floor covering initial linear learning rate ramp
DEFAULT_MAX_EPOCHS = 400  # Architecture 2 studio reference epoch safety ceiling
DEFAULT_BATCH_SIZE = 16  # Universal GPU batch size (84 updates/epoch for optimal gradient depth)
DEFAULT_PATIENCE = 12  # Plateau patience window in epochs
DEFAULT_MIN_DELTA = 5e-6  # Minimum loss improvement to reset plateau patience
DEFAULT_PRE_EMPH_WEIGHT = 0.25  # Pre-emphasis loss weight for equalizing high-frequency resonance
DEFAULT_PRE_EMPH_COEF = 0.85  # Pre-emphasis filter coefficient (alpha = 0.85)
DEFAULT_MRSTFT_WEIGHT = 0.0010  # Multi-Resolution STFT loss weight for RLC resonant peak fidelity
DEFAULT_LR_SCHEDULER = "cosine"  # Monotonic Cosine Decay with floor clamp
DEFAULT_ETA_MIN = 1e-5  # Minimum learning rate floor for CosineAnnealingLR
DEFAULT_LR_T_MAX = 35  # Cosine decay period (T_max in epochs)
from allomorph.naming import get_default_input_path

DEFAULT_INPUT_PATH = get_default_input_path()


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


def compute_baseline_delta_ratio(
    ref_wav: str | Path | None,
    tgt_wav: str | Path | None,
    val_samples: int = 432_000,
) -> float:
    """Computes energy ratio of baseline pickup difference on validation segment:
    ||y_val - x_val||^2 / ||y_val||^2.
    """
    if not ref_wav or not tgt_wav:
        return 0.015
    try:
        import numpy as np
        from pedalboard.io import AudioFile

        p_tgt = Path(tgt_wav)
        p_ref = Path(ref_wav)
        if not p_tgt.exists() or not p_ref.exists():
            return 0.015

        with AudioFile(str(p_tgt)) as f_tgt:
            n_frames = f_tgt.frames
            read_len = min(val_samples, n_frames)
            f_tgt.seek(max(0, n_frames - read_len))
            y_val = f_tgt.read(read_len)[0]

        with AudioFile(str(p_ref)) as f_ref:
            n_frames_ref = f_ref.frames
            read_len_ref = min(val_samples, n_frames_ref)
            f_ref.seek(max(0, n_frames_ref - read_len_ref))
            x_val = f_ref.read(read_len_ref)[0]

        min_len = min(len(y_val), len(x_val))
        if min_len == 0:
            return 0.015
        y = y_val[-min_len:].astype(np.float64)
        x = x_val[-min_len:].astype(np.float64)

        y_energy = float(np.sum(y**2))
        if y_energy <= 1e-12:
            return 0.015
        delta_energy = float(np.sum((y - x) ** 2))
        ratio = delta_energy / y_energy
        return max(ratio, 1e-6)
    except (FileNotFoundError, ValueError, RuntimeError, KeyError, OSError, TypeError):
        return 0.015


def compute_baseline_mrstft(
    ref_wav: str | Path | None,
    tgt_wav: str | Path | None,
    val_samples: int = 432_000,
) -> float:
    """Computes MultiResolutionSTFTLoss baseline spectral distance on validation segment:
    MRSTFT(x_val, y_val).
    """
    if not ref_wav or not tgt_wav:
        return 0.400
    try:
        import torch
        from nam._dependencies.auraloss.freq import MultiResolutionSTFTLoss
        from pedalboard.io import AudioFile

        p_tgt = Path(tgt_wav)
        p_ref = Path(ref_wav)
        if not p_tgt.exists() or not p_ref.exists():
            return 0.400

        with AudioFile(str(p_tgt)) as f_tgt:
            n_frames = f_tgt.frames
            read_len = min(val_samples, n_frames)
            f_tgt.seek(max(0, n_frames - read_len))
            y_val = f_tgt.read(read_len)[0]

        with AudioFile(str(p_ref)) as f_ref:
            n_frames_ref = f_ref.frames
            read_len_ref = min(val_samples, n_frames_ref)
            f_ref.seek(max(0, n_frames_ref - read_len_ref))
            x_val = f_ref.read(read_len_ref)[0]

        min_len = min(len(y_val), len(x_val))
        if min_len == 0:
            return 0.400

        y_t = torch.from_numpy(y_val[-min_len:]).float().unsqueeze(0).unsqueeze(0)
        x_t = torch.from_numpy(x_val[-min_len:]).float().unsqueeze(0).unsqueeze(0)

        loss_fn = MultiResolutionSTFTLoss()
        with torch.no_grad():
            res = float(loss_fn(x_t, y_t))
        return max(float(res), 1e-6)
    except (FileNotFoundError, ValueError, RuntimeError, KeyError, OSError, TypeError):
        return 0.400


def configure_a2_architecture(
    nam_core: Any,
    a2_lite_only: bool = True,
    min_epochs: int = DEFAULT_MIN_EPOCHS,
    goal_esr: float | None = DEFAULT_GOAL_ESR,
    goal_delta_esr: float | None = DEFAULT_GOAL_DELTA_ESR,
    goal_delta_mrstft: float | None = DEFAULT_GOAL_DELTA_MRSTFT,
    max_mrstft_ceiling: float = DEFAULT_MAX_MRSTFT_CEILING,
    consecutive_patience: int = DEFAULT_CONSECUTIVE_PATIENCE,
    patience: int = DEFAULT_PATIENCE,
    min_delta: float = DEFAULT_MIN_DELTA,
    pre_emph_weight: float = DEFAULT_PRE_EMPH_WEIGHT,
    pre_emph_coef: float = DEFAULT_PRE_EMPH_COEF,
    mrstft_weight: float = DEFAULT_MRSTFT_WEIGHT,
    lr_scheduler: str = DEFAULT_LR_SCHEDULER,
    eta_min: float = DEFAULT_ETA_MIN,
    lr_t_max: int = DEFAULT_LR_T_MAX,
    reference_wav: str | Path | None = None,
    output_wav: str | Path | None = None,
) -> None:
    """Configure NAM Architecture 2 packed model submodels, loss weighting, and adaptive early stopping.

    Default: a2_lite_only=True isolates channels_8 (A2-Lite studio tier), running ~35% faster on Apple Silicon MPS.
    Injects pre-emphasis loss (alpha=0.85) and MRSTFT (0.0010) to equalize high-frequency pickup resonance gradients.
    Hooks LinearWarmupCallback (5 epochs) and AllomorphAdaptiveStopping (triple-gate + de-risked plateau patience).
    """

    try:
        import torch

        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high")

        if hasattr(torch, "backends") and hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass

    baseline_delta_ratio = compute_baseline_delta_ratio(reference_wav, output_wav)
    baseline_mrstft = compute_baseline_mrstft(reference_wav, output_wav)

    orig_detect_input_version = getattr(
        nam_core, "_orig_detect_input_version", nam_core._detect_input_version
    )
    nam_core._orig_detect_input_version = orig_detect_input_version

    def patched_detect_input_version(path: Any) -> tuple[Any, bool]:
        err_types = (
            getattr(nam_core, "_InputValidationError", ValueError),
            ValueError,
            KeyError,
            RuntimeError,
            OSError,
        )
        try:
            res = orig_detect_input_version(path)
            nam_core._is_dry_wet = False
            return res
        except err_types:
            nam_core._is_dry_wet = True
            version_cls = getattr(nam_core, "_Version", None)
            if version_cls is None:
                from nam.train._version import Version as version_cls

            return version_cls(3, 0, 0), False

    nam_core._detect_input_version = patched_detect_input_version

    orig_check_data = getattr(nam_core, "_orig_check_data", nam_core._check_data)
    nam_core._orig_check_data = orig_check_data

    def patched_check_data(*args: Any, **kwargs: Any) -> Any:
        if getattr(nam_core, "_is_dry_wet", False):
            import nam.train.metadata as meta

            return meta.DataChecks(passed=True, version=3)
        return orig_check_data(*args, **kwargs)

    nam_core._check_data = patched_check_data

    orig_get_data_config = getattr(
        nam_core, "_orig_get_data_config", nam_core._get_data_config
    )
    nam_core._orig_get_data_config = orig_get_data_config

    def patched_get_data_config(*args: Any, **kwargs: Any) -> dict[str, Any]:
        cfg: dict[str, Any] = orig_get_data_config(*args, **kwargs)
        if getattr(nam_core, "_is_dry_wet", False):
            if "train" in cfg:
                cfg["train"]["start_samples"] = 0
            if "validation" in cfg:
                cfg["validation"]["require_input_pre_silence"] = False
        return cfg

    nam_core._get_data_config = patched_get_data_config

    orig_get_packed_model_config = getattr(
        nam_core, "_orig_get_packed_model_config", nam_core._get_packed_model_config
    )
    nam_core._orig_get_packed_model_config = orig_get_packed_model_config

    def get_configured_packed_model_config() -> dict[str, Any]:
        cfg: dict[str, Any] = orig_get_packed_model_config()
        if a2_lite_only:
            cfg["net"]["config"]["submodels"] = [
                s for s in cfg["net"]["config"]["submodels"] if s["name"] == "channels_8"
            ]
        else:
            cfg["net"]["config"]["submodels"] = [
                s
                for s in cfg["net"]["config"]["submodels"]
                if s["name"] in ("channels_3", "channels_8")
            ]
        if "loss" not in cfg:
            new_loss: dict[str, Any] = {}
            cfg["loss"] = new_loss
        if pre_emph_weight > 0.0:
            cfg["loss"]["pre_emph_weight"] = pre_emph_weight
            cfg["loss"]["pre_emph_coef"] = pre_emph_coef
        if mrstft_weight > 0.0:
            cfg["loss"]["mrstft_weight"] = mrstft_weight
        if lr_scheduler == "cosine":
            cfg["lr_scheduler"] = {
                "class": "CosineAnnealingLR",
                "kwargs": {
                    "T_max": lr_t_max,
                    "eta_min": eta_min,
                },
            }
        return cfg

    nam_core._get_packed_model_config = get_configured_packed_model_config

    from pytorch_lightning.callbacks import Callback

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

    stopping_cb_base = getattr(nam_core, "_ValidationStopping", None)
    if stopping_cb_base is None:
        from pytorch_lightning.callbacks import EarlyStopping as stopping_cb_base

    class AllomorphAdaptiveStopping(stopping_cb_base):  # type: ignore[misc, valid-type]
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
            warmup_floor: int = min_epochs,
            goal_esr: float | None = goal_esr,
            goal_delta_esr: float | None = goal_delta_esr,
            goal_delta_mrstft: float | None = goal_delta_mrstft,
            max_mrstft_ceiling: float = max_mrstft_ceiling,
            consecutive_patience: int = consecutive_patience,
            patience: int = patience,
            min_delta: float = min_delta,
            baseline_delta_ratio: float = baseline_delta_ratio,
            baseline_mrstft: float = baseline_mrstft,
            a2_lite_only: bool = a2_lite_only,
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
            self.consecutive_patience = consecutive_patience
            self.patience = patience
            self.min_delta = min_delta
            self.baseline_delta_ratio = max(baseline_delta_ratio, 1e-6)
            self.baseline_mrstft = max(baseline_mrstft, 1e-6)
            self.is_identity = baseline_mrstft < 1e-6
            self.a2_lite_only = a2_lite_only
            self.esr_history: list[float] = []
            self.mrstft_history: list[float] = []
            self.best_esr: float = float("inf")
            self.best_delta_esr: float = float("inf")
            self.best_mrstft: float | None = None
            self.best_diff_mrstft: float | None = None
            self.consecutive_gates_met: int = 0
            self.stop_reason: str = "max_epochs"
            self.last_esr_slope: float = 0.0
            self.last_mrstft_slope: float = 0.0

        def get_status_str(self, epoch: int) -> str:
            if epoch < self.warmup_floor:
                return f"Warmup: {epoch}/{self.warmup_floor}"
            if self.consecutive_gates_met > 0:
                return f"Triple-Gate: {self.consecutive_gates_met}/{self.consecutive_patience}"
            if self.patience <= 0:
                return "Plateau guard: off"
            pts_count = len(self.esr_history)
            return f"Patience: {min(pts_count, self.patience)}/{self.patience} (slope: {self.last_esr_slope:+.1e})"

        def _run_early_stopping_check(self, trainer: Any) -> None:
            metrics: dict[str, Any] = getattr(trainer, "callback_metrics", {})
            epoch: int = getattr(trainer, "current_epoch", 0)

            raw_esr: Any = metrics.get("ESR") if self.a2_lite_only else metrics.get("ESR_packed_1")
            if raw_esr is None:
                raw_esr = metrics.get("val_loss")
            if raw_esr is None:
                return
            esr_val = float(raw_esr.item() if hasattr(raw_esr, "item") else raw_esr)
            delta_esr_val = esr_val / self.baseline_delta_ratio

            raw_mrstft = metrics.get("MRSTFT") or metrics.get("MRSTFT_packed_1") or metrics.get("MRSTFT_packed_0")
            mrstft_val = float(raw_mrstft.item() if hasattr(raw_mrstft, "item") else raw_mrstft) if raw_mrstft is not None else None

            self.best_esr = min(self.best_esr, esr_val)
            self.best_delta_esr = min(self.best_delta_esr, delta_esr_val)
            if mrstft_val is not None:
                self.best_mrstft = min(self.best_mrstft if self.best_mrstft is not None else float("inf"), mrstft_val)
                diff_mrstft = mrstft_val / self.baseline_mrstft
                self.best_diff_mrstft = min(
                    self.best_diff_mrstft if self.best_diff_mrstft is not None else float("inf"),
                    diff_mrstft,
                )

            self.esr_history.append(esr_val)
            if mrstft_val is not None:
                self.mrstft_history.append(mrstft_val)

            if len(self.esr_history) >= 2:
                window_pts = self.esr_history[-max(self.patience, 5):]
                self.last_esr_slope = compute_linear_slope(window_pts)
            if len(self.mrstft_history) >= 2:
                mr_window = self.mrstft_history[-max(self.patience, 5):]
                self.last_mrstft_slope = compute_linear_slope(mr_window)

            if epoch < self.warmup_floor:
                return

            # 1. Triple-Gate Goal Check
            gate1_ok = (self.goal_esr is None) or (esr_val <= self.goal_esr)
            gate2_ok = (self.goal_delta_esr is None) or (delta_esr_val <= self.goal_delta_esr)
            gate3_ok = True
            if not self.is_identity and self.goal_delta_mrstft is not None:
                if mrstft_val is None:
                    gate3_ok = False
                else:
                    diff_mr = mrstft_val / self.baseline_mrstft
                    gate3_ok = (diff_mr <= self.goal_delta_mrstft) and (mrstft_val <= self.max_mrstft_ceiling)

            all_gates_pass = gate1_ok and gate2_ok and gate3_ok
            if all_gates_pass and (self.goal_esr is not None or self.goal_delta_esr is not None):
                self.consecutive_gates_met += 1
                if self.consecutive_gates_met >= self.consecutive_patience and epoch >= self.warmup_floor:
                    esr_db = 10.0 * math.log10(max(esr_val, 1e-12))
                    delta_db = 10.0 * math.log10(max(delta_esr_val, 1e-12))
                    mr_info = f", MRSTFT {mrstft_val:.4f}" if mrstft_val is not None else ""
                    print(
                        f"\n[Triple-Gate Achieved] Epoch {epoch:03d}: Global ESR {esr_val:.6f} ({esr_db:+.2f} dB) "
                        f"AND Delta Nuance {delta_esr_val:.6f} ({delta_db:+.2f} dB){mr_info} "
                        f"held for {self.consecutive_gates_met}/{self.consecutive_patience} consecutive epochs. "
                        "Terminating successfully with studio fidelity!",
                        flush=True,
                    )
                    trainer.should_stop = True
                    self.stop_reason = "triple_gate_converged"
                    return
            else:
                self.consecutive_gates_met = 0

            # 2. De-risked Adaptive Plateau Check
            if (
                self.patience > 0
                and len(self.esr_history) >= self.patience
                and epoch >= (self.warmup_floor + self.patience)
            ):
                recent_esr = self.esr_history[-self.patience:]
                esr_improvement = recent_esr[0] - esr_val
                empty_mrstft: list[float] = []
                recent_mrstft = (
                    self.mrstft_history[-self.patience:]
                    if len(self.mrstft_history) >= self.patience
                    else empty_mrstft
                )
                mrstft_flat = (len(recent_mrstft) == 0) or (self.last_mrstft_slope >= -1e-7)

                if self.last_esr_slope >= -1e-7 and mrstft_flat and (esr_improvement < self.min_delta):
                    print(
                        f"\n[Early Stopping] Diminishing returns plateau reached at epoch {epoch:03d}: "
                        f"ESR slope {self.last_esr_slope:+.1e}, MRSTFT slope {self.last_mrstft_slope:+.1e} over {self.patience} epochs. "
                        "Terminating to preserve GPU efficiency.",
                        flush=True,
                    )
                    trainer.should_stop = True
                    self.stop_reason = "plateau_exit"
                    return

    class EsrProgressCallback(Callback):
        """Logs validation ESR progress and updates progress bar metrics each epoch."""

        def __init__(
            self,
            target_esr: float | None = None,
            target_delta_esr: float | None = None,
            a2_lite_only: bool = True,
            min_epochs: int = DEFAULT_MIN_EPOCHS,
            baseline_delta_ratio: float = 0.015,
            baseline_mrstft: float = 0.400,
            stopping_callback: Any = None,
        ) -> None:
            super().__init__()
            self.target_esr: float | None = target_esr
            self.target_delta_esr: float | None = target_delta_esr
            self.a2_lite_only: bool = a2_lite_only
            self.min_epochs: int = min_epochs
            self.baseline_delta_ratio: float = max(baseline_delta_ratio, 1e-6)
            self.baseline_mrstft: float = max(baseline_mrstft, 1e-6)
            self.stopping_callback: Any = stopping_callback
            self.best_esr: float = float("inf")
            self.best_delta_esr: float = float("inf")
            self.best_mrstft: float | None = None
            self.best_diff_mrstft: float | None = None
            self.best_ch3_esr: float | None = float("inf")
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

            raw_mrstft = metrics.get("MRSTFT") or metrics.get("MRSTFT_packed_1") or metrics.get("MRSTFT_packed_0")
            mrstft_val = float(raw_mrstft.item() if hasattr(raw_mrstft, "item") else raw_mrstft) if raw_mrstft is not None else None
            if mrstft_val is not None:
                self.best_mrstft = min(self.best_mrstft if self.best_mrstft is not None else float("inf"), mrstft_val)
                diff_mrstft = mrstft_val / self.baseline_mrstft
                self.best_diff_mrstft = min(
                    self.best_diff_mrstft if self.best_diff_mrstft is not None else float("inf"),
                    diff_mrstft,
                )
            mrstft_str = f" | MRSTFT: {mrstft_val:.5f}" if mrstft_val is not None else ""

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

            if self.a2_lite_only:
                raw_esr: Any = metrics.get("ESR")
                if raw_esr is None:
                    raw_esr = metrics.get("val_loss")
                if raw_esr is None:
                    return
                esr_val: float = float(raw_esr.item() if hasattr(raw_esr, "item") else raw_esr)
                self.best_esr = min(self.best_esr, esr_val)
                delta_esr_val = esr_val / self.baseline_delta_ratio
                self.best_delta_esr = min(self.best_delta_esr, delta_esr_val)

                if hasattr(trainer, "progress_bar_metrics") and isinstance(
                    trainer.progress_bar_metrics, dict
                ):
                    trainer.progress_bar_metrics["val_ESR"] = f"{esr_val:.5f}"
                    trainer.progress_bar_metrics["best_ESR"] = f"{self.best_esr:.5f}"
                    trainer.progress_bar_metrics["delta_ESR"] = f"{delta_esr_val:.5f}"

                esr_db: float = 10.0 * math.log10(max(esr_val, 1e-12))
                delta_db: float = 10.0 * math.log10(max(delta_esr_val, 1e-12))
                delta_str = f" | Delta: {delta_esr_val:.5f} ({delta_db:+.2f} dB)"

                print(
                    f"\n[Epoch {epoch:03d}/{max_epochs}] Studio ESR: {esr_val:.6f} ({esr_db:+.2f} dB){delta_str}{mrstft_str}{lr_str}{gates_str}{patience_str}{target_str}",
                    flush=True,
                )
            else:
                # Slimmable Architecture 2: channels_8 is the primary studio tier
                raw_ch8: Any = metrics.get("ESR_packed_1")
                raw_ch3: Any = metrics.get("ESR_packed_0")
                if raw_ch8 is None:
                    raw_ch8 = metrics.get("ESR")
                if raw_ch8 is None:
                    raw_ch8 = metrics.get("val_loss")
                if raw_ch8 is None:
                    return
                ch8_val: float = float(raw_ch8.item() if hasattr(raw_ch8, "item") else raw_ch8)
                ch3_val: float | None = (
                    float(raw_ch3.item() if hasattr(raw_ch3, "item") else raw_ch3)
                    if raw_ch3 is not None
                    else None
                )
                self.best_esr = min(self.best_esr, ch8_val)
                delta_esr_val = ch8_val / self.baseline_delta_ratio
                self.best_delta_esr = min(self.best_delta_esr, delta_esr_val)
                if ch3_val is not None:
                    self.best_ch3_esr = min(self.best_ch3_esr if self.best_ch3_esr is not None else float("inf"), ch3_val)

                if hasattr(trainer, "progress_bar_metrics") and isinstance(
                    trainer.progress_bar_metrics, dict
                ):
                    trainer.progress_bar_metrics["val_ESR"] = f"{ch8_val:.5f}"
                    trainer.progress_bar_metrics["best_ESR"] = f"{self.best_esr:.5f}"
                    trainer.progress_bar_metrics["val_ESR_ch8"] = f"{ch8_val:.5f}"
                    trainer.progress_bar_metrics["delta_ESR"] = f"{delta_esr_val:.5f}"
                    if ch3_val is not None:
                        trainer.progress_bar_metrics["val_ESR_ch3"] = f"{ch3_val:.5f}"

                ch8_db: float = 10.0 * math.log10(max(ch8_val, 1e-12))
                delta_db = 10.0 * math.log10(max(delta_esr_val, 1e-12))
                delta_str = f" | Delta: {delta_esr_val:.5f} ({delta_db:+.2f} dB)"
                ch3_str = ""
                if ch3_val is not None:
                    ch3_db = 10.0 * math.log10(max(ch3_val, 1e-12))
                    ch3_str = f" | Ch3 Nano: {ch3_val:.6f} ({ch3_db:+.2f} dB)"

                print(
                    f"\n[Epoch {epoch:03d}/{max_epochs}] Studio ESR: {ch8_val:.6f} ({ch8_db:+.2f} dB){delta_str}{mrstft_str}{ch3_str}{lr_str}{gates_str}{patience_str}{target_str}",
                    flush=True,
                )

    orig_get_callbacks = getattr(nam_core, "_orig_get_callbacks", nam_core.get_callbacks)
    nam_core._orig_get_callbacks = orig_get_callbacks

    def get_callbacks_with_logging(
        threshold_esr: float | None = None,
        *args: Any,
        min_epochs_override: int | None = None,
        **kwargs: Any,
    ) -> list[Any]:
        callbacks: list[Any] = orig_get_callbacks(None, *args, **kwargs)
        effective_min_epochs = (
            min_epochs_override if min_epochs_override is not None else min_epochs
        )

        warmup_cb = LinearWarmupCallback(
            warmup_epochs=5,
            target_lr=0.004,
            lr_t_max=lr_t_max,
            eta_min=eta_min,
            is_cosine=(lr_scheduler == "cosine"),
        )
        callbacks.append(warmup_cb)

        monitor_key = "ESR" if a2_lite_only else "ESR_packed_1"
        effective_goal = threshold_esr if threshold_esr is not None else None
        stopping_cb = None
        if effective_goal is not None:
            stopping_cb = AllomorphAdaptiveStopping(
                monitor=monitor_key,
                stopping_threshold=effective_goal,
                warmup_floor=effective_min_epochs,
                goal_esr=effective_goal,
                goal_delta_esr=goal_delta_esr,
                goal_delta_mrstft=goal_delta_mrstft,
                max_mrstft_ceiling=max_mrstft_ceiling,
                consecutive_patience=consecutive_patience,
                patience=patience,
                min_delta=min_delta,
                baseline_delta_ratio=baseline_delta_ratio,
                baseline_mrstft=baseline_mrstft,
                a2_lite_only=a2_lite_only,
            )
            callbacks.append(stopping_cb)
        nam_core._last_stopping_callback = stopping_cb

        progress_cb = EsrProgressCallback(
            target_esr=effective_goal,
            target_delta_esr=goal_delta_esr if effective_goal is not None else None,
            a2_lite_only=a2_lite_only,
            min_epochs=effective_min_epochs,
            baseline_delta_ratio=baseline_delta_ratio,
            baseline_mrstft=baseline_mrstft,
            stopping_callback=stopping_cb,
        )
        nam_core._last_esr_callback = progress_cb
        callbacks.append(progress_cb)
        return callbacks

    nam_core.get_callbacks = get_callbacks_with_logging


def train_voice(
    instrument: str | InstrumentConfig = "30in",
    voice: str = "precision_active",
    input_wav: str | Path | None = None,
    output_wav: str | Path | None = None,
    reference_wav: str | Path | None = None,
    models_dir: str | Path = MODELS_DIR,
    epochs: int = DEFAULT_MAX_EPOCHS,
    min_epochs: int = DEFAULT_MIN_EPOCHS,
    goal_esr: float | None = DEFAULT_GOAL_ESR,
    goal_delta_esr: float | None = DEFAULT_GOAL_DELTA_ESR,
    goal_delta_mrstft: float | None = DEFAULT_GOAL_DELTA_MRSTFT,
    max_mrstft_ceiling: float = DEFAULT_MAX_MRSTFT_CEILING,
    consecutive_patience: int = DEFAULT_CONSECUTIVE_PATIENCE,
    patience: int = DEFAULT_PATIENCE,
    min_delta: float = DEFAULT_MIN_DELTA,
    pre_emph_weight: float = DEFAULT_PRE_EMPH_WEIGHT,
    pre_emph_coef: float = DEFAULT_PRE_EMPH_COEF,
    mrstft_weight: float = DEFAULT_MRSTFT_WEIGHT,
    lr_scheduler: str = DEFAULT_LR_SCHEDULER,
    eta_min: float = DEFAULT_ETA_MIN,
    lr_t_max: int = DEFAULT_LR_T_MAX,
    batch_size: int = DEFAULT_BATCH_SIZE,
    silent: bool = True,
    save_plot: bool = False,
    fast_dev_run: bool = False,
    basename: str | None = None,
    a2_lite_only: bool = True,
    version_tag: str | None = "auto",
    no_manifest: bool = False,
    include_identity: bool = False,
) -> bool:
    try:
        import nam.train.core as nam_core
        import nam.train.metadata as train_meta
        import torch
        from nam.models.metadata import UserMetadata

        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high")
        if hasattr(torch, "backends") and hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.benchmark = False
    except ImportError:
        print("Error: 'neural-amp-modeler' is not installed in the current environment.")
        print("Please run `uv sync` or install project dependencies:")
        print(f"  uv run python main.py --stage train --voice {voice}")
        return False

    if voice not in VOICES:
        raise KeyError(f"Target voice '{voice}' not found in catalog.")
    vcfg = VOICES[voice]
    voice_name = vcfg.name

    try:
        inst_cfg = (
            instrument if isinstance(instrument, InstrumentConfig) else load_instrument(instrument)
        )
    except (FileNotFoundError, KeyError, ValueError, OSError):
        inst_cfg = InstrumentConfig(
            id=str(instrument),
            name=str(instrument),
            scale_length_in=34.0,
            scale_length_m=0.8636,
            string_wave_speeds=[58.02, 75.88, 99.19, 129.6],
            pickups={},
        )

    inst_ver = getattr(inst_cfg, "version", 1)
    voice_ver = getattr(vcfg, "version", 1)
    tri_part = resolve_tri_part_version(DSP_GENERATION, inst_ver, voice_ver)
    actual_version_tag = (
        tri_part
        if (version_tag == "auto" or version_tag is True)
        else (version_tag if version_tag not in (None, "none", False) else None)
    )

    model_basename = basename or (f"{voice}_{actual_version_tag}" if actual_version_tag else voice)
    inst_id = inst_cfg.id
    inst_name = inst_cfg.name
    scale_length_in = inst_cfg.scale_length_in or 34.0
    try:
        src_pickup = get_source_pickup(inst_cfg, voice)
        src_pickup_name = src_pickup.name
        src_pos_mm = (src_pickup.position_from_bridge_m or 0.0) * 1000.0
    except (KeyError, ValueError):
        src_pickup = PickupConfig(name="Pickup", position_from_bridge_m=0.0)
        src_pickup_name = "Pickup"
        src_pos_mm = 0.0

    inst_models_dir = Path(models_dir) / inst_id
    inst_models_dir.mkdir(parents=True, exist_ok=True)
    target_nam = inst_models_dir / f"{model_basename}.nam"

    input_path = find_sweep_input(input_wav)

    if not output_wav:
        from allomorph.circuit.forward import resolve_target_voicing

        tgt_inst, tgt_v = resolve_target_voicing(voice, instrument=inst_cfg)
        target_slug = (
            tgt_v.tone_name.lower().replace(" ", "_").replace("∕", "_").replace("/", "_")
            if tgt_v.tone_name
            else (tgt_v.id or voice)
        )
        target_wet = AUDIO_DIR / "wet" / tgt_inst.id / f"{target_slug}.wav"
        if not (target_wet.exists() and is_wet_stem_valid(target_wet, base_dry_path=input_path)):
            try:
                from allomorph.circuit import simulate_instrument_voicing

                print(
                    f"[NAM Trainer] Target wet stem missing or stale, auto-simulating: {target_wet.name}"
                )
                output_path = simulate_instrument_voicing(
                    instrument=tgt_inst, voicing=tgt_v, output_wav=target_wet
                )
            except (FileNotFoundError, ValueError, RuntimeError, KeyError, OSError) as e:
                print(f"[NAM Trainer] Warning: Failed to auto-simulate {target_wet.name}: {e}")
                output_path = target_wet
        else:
            output_path = target_wet
    else:
        output_path = Path(output_wav)

    if not input_path or not input_path.exists():
        print(f"Error: Could not find training input file '{input_path}'.")
        return False

    if not output_path.exists():
        print(f"Error: Target output audio '{output_path}' does not exist.")
        print("Please run the simulation stage first:")
        print(f"  uv run python -m allomorph.pipeline.cli --stage sim --voice {voice}")
        return False

    # Resolve reference stem for differential delta ESR (y - x_src)
    if reference_wav:
        reference_path = Path(reference_wav)
    else:
        src_pk = src_pickup.id or getattr(inst_cfg, "default_pickup", None) or "default"
        candidate_source = AUDIO_DIR / "wet" / inst_id / f"{src_pk}.wav"
        if candidate_source.exists() and is_wet_stem_valid(
            candidate_source, base_dry_path=input_path
        ):
            reference_path = candidate_source
        else:
            try:
                from allomorph.circuit.forward import simulate_instrument_voicing

                print(
                    f"[NAM Trainer] Dedicated source stem missing or stale, auto-generating: {candidate_source.name}"
                )
                reference_path = simulate_instrument_voicing(
                    instrument=inst_id,
                    voicing=src_pk,
                    output_wav=candidate_source,
                    input_wav=input_path,
                )
            except (FileNotFoundError, ValueError, RuntimeError, KeyError, OSError) as e:
                print(
                    f"[NAM Trainer] Warning: Failed to auto-generate source pickup stem: {e}. Falling back to input sweep."
                )
                reference_path = input_path

    if not include_identity and reference_path.resolve() == output_path.resolve():
        print(
            f"[NAM Trainer] Skipping identity pair for '{voice}' on {inst_id}: "
            f"source and target audio stems are identical ({reference_path.name}). "
            "Use --include-identity to force training."
        )
        return True

    if goal_esr is not None and goal_esr <= 0:
        threshold_esr = None
    else:
        threshold_esr = goal_esr

    configure_a2_architecture(
        nam_core,
        a2_lite_only=a2_lite_only,
        min_epochs=min_epochs,
        goal_esr=threshold_esr,
        goal_delta_esr=goal_delta_esr,
        goal_delta_mrstft=goal_delta_mrstft,
        max_mrstft_ceiling=max_mrstft_ceiling,
        consecutive_patience=consecutive_patience,
        patience=patience,
        min_delta=min_delta,
        pre_emph_weight=pre_emph_weight,
        pre_emph_coef=pre_emph_coef,
        mrstft_weight=mrstft_weight,
        lr_scheduler=lr_scheduler,
        eta_min=eta_min,
        lr_t_max=lr_t_max,
        reference_wav=reference_path,
        output_wav=output_path,
    )

    train_work_dir = inst_models_dir / f".train_{model_basename}"
    train_work_dir.mkdir(parents=True, exist_ok=True)

    print("\n========================================")
    print("  ALLOMORPH NAM LOCAL A2 TRAINER")
    print(f'  Source Bass: {inst_name} ({inst_id}, {scale_length_in}")')
    print(f"  Source PU:   {src_pickup_name} (pos={src_pos_mm:.1f}mm)")
    print(f"  Target Voice:{voice} ({voice_name})")
    arch_display = (
        "Architecture 2 Lite (channels_8 only, fast)"
        if a2_lite_only
        else "Architecture 2 Slimmable (channels_3 + channels_8, full)"
    )
    print(f"  Model Tier:  {arch_display}")
    print(f"  Batch Size:  {batch_size}")
    print(f"  Input Audio: {input_path.name}")
    print(f"  Output Audio:{output_path.name}")
    if reference_path and reference_path != input_path:
        print(f"  Delta Ref:   {reference_path.name}")
    print(f"  Max Epochs:  {epochs}")
    print(f"  Warmup Ep:   {min_epochs}")
    esr_display = (
        f"{threshold_esr:.6f} (A2-Lite Studio Reference Early Stopping, min {min_epochs} epochs)"
        if (threshold_esr is not None and a2_lite_only)
        else (
            f"{threshold_esr:.6f} (A2 Slimmable Studio Reference Early Stopping, Ch8 <= {threshold_esr:.6f}, min {min_epochs} epochs)"
            if threshold_esr is not None
            else "Disabled (Fixed Epochs)"
        )
    )
    print(f"  Goal ESR:    {esr_display}")
    if goal_delta_esr is not None and goal_delta_esr > 0:
        print(f"  Goal Delta:  {goal_delta_esr:.4f} (-17 dB on pickup delta)")
    if goal_delta_mrstft is not None:
        print(f"  Goal Spectral: MRSTFT/M_base <= {goal_delta_mrstft:.2f} (ceiling {max_mrstft_ceiling:.3f})")
    if patience > 0:
        print(f"  Patience:    {patience} epochs (plateau slope min_delta={min_delta:.1e})")
    if pre_emph_weight > 0:
        print(f"  Pre-Emph:    weight={pre_emph_weight:.2f}, coef={pre_emph_coef:.2f}")
    if mrstft_weight > 0:
        print(f"  MRSTFT:      weight={mrstft_weight:.4f}")
    print(f"  LR Schedule: {lr_scheduler} (T_max={lr_t_max}, eta_min={eta_min:.1e})")
    print(f"  Destination: {target_nam}")
    print("========================================\n")

    model_title = f"{voice_name} [{inst_name}]"
    user_metadata = UserMetadata(
        name=model_title,
        modeled_by="Allomorph (Peter Nguyen <peter@phn.dev>)",
        gear_make=inst_name,
        gear_model=f"{src_pickup_name} -> {voice_name}",
        gear_type="preamp",
        tone_type="clean",
    )

    print("Validating dataset and calibration markers...")
    train_output = nam_core.train(
        input_path=str(input_path),
        output_path=str(output_path),
        train_path=str(train_work_dir),
        epochs=epochs,
        batch_size=batch_size,
        modelname=model_basename,
        silent=silent,
        save_plot=save_plot,
        local=False,
        threshold_esr=threshold_esr,
        user_metadata=user_metadata,
        fast_dev_run=fast_dev_run,
        latency=0,
        ignore_checks=True,
    )

    if train_output is None or train_output.model is None:
        print("Error: Training did not produce a model.")
        return False

    cb = getattr(nam_core, "_last_esr_callback", None)
    stopping_cb = getattr(nam_core, "_last_stopping_callback", None)
    best_diff_esr = (
        cb.best_delta_esr if (cb is not None and cb.best_delta_esr < float("inf")) else None
    )
    best_mrstft = (
        cb.best_mrstft if (cb is not None and cb.best_mrstft < float("inf")) else None
    )
    best_diff_mrstft = (
        cb.best_diff_mrstft if (cb is not None and cb.best_diff_mrstft < float("inf")) else None
    )
    baseline_mrstft = cb.baseline_mrstft if cb is not None else None
    consecutive_gates_met = (
        stopping_cb.consecutive_gates_met if stopping_cb is not None else None
    )
    epochs_trained = cb.last_epoch + 1 if (cb is not None and cb.last_epoch >= 0) else None
    stop_reason = stopping_cb.stop_reason if stopping_cb is not None else None

    raw_meta = train_output.metadata.model_dump()
    if best_diff_esr is not None:
        raw_meta["differential_esr"] = best_diff_esr
    if best_mrstft is not None:
        raw_meta["mrstft_loss"] = best_mrstft
    if baseline_mrstft is not None:
        raw_meta["baseline_mrstft"] = baseline_mrstft
    if best_diff_mrstft is not None:
        raw_meta["differential_mrstft"] = best_diff_mrstft
    if consecutive_gates_met is not None:
        raw_meta["consecutive_gates_met"] = consecutive_gates_met
    if epochs_trained is not None:
        raw_meta["epochs_trained"] = epochs_trained
    if stop_reason is not None:
        raw_meta["stop_reason"] = stop_reason

    print("\nExporting Architecture 2 (.nam) model container with full instrument metadata...")
    nam_meta = NamExportMetadata(
        training=NamTrainingMetadata.model_validate(raw_meta),
        license="PolyForm Noncommercial License 1.0.0 (https://polyformproject.org/licenses/noncommercial/1.0.0)",
        copyright="Copyright 2026 Peter Nguyen <peter@phn.dev>. All commercial rights reserved.",
        author="Peter Nguyen <peter@phn.dev>",
        version=tri_part,
        dsp_version=DSP_GENERATION,
        instrument_version=inst_ver,
        voice_version=voice_ver,
        allomorph_version=ALLOMORPH_VERSION,
        git_commit=get_git_commit(),
        generated_at=datetime.now(UTC).isoformat(),
        source_instrument=NamSourceInstrumentMeta(
            id=inst_id,
            name=inst_name,
            scale_length_in=scale_length_in,
            scale_length_m=inst_cfg.scale_length_m,
            string_wave_speeds=inst_cfg.string_wave_speeds,
            pickup=NamSourcePickupMeta(
                id=src_pickup.id or "",
                name=src_pickup_name,
                position_from_bridge_m=src_pickup.position_from_bridge_m or 0.0,
                position_from_bridge_mm=src_pos_mm,
                aperture_width_in=src_pickup.aperture_width_in,
                coil_spacing_in=src_pickup.coil_spacing_in,
                type=src_pickup.type,
            ),
        ),
        target_voice=NamTargetVoiceMeta(
            id=voice,
            name=voice_name,
            topology=vcfg.topology,
            resonant_frequency_hz=float(vcfg.fr),
            q_factor=float(vcfg.Q),
            target_position_34_m=compute_effective_position(resolve_voice_coils(vcfg)),
            effective_position_m=compute_effective_position(resolve_voice_coils(vcfg)),
            pickups=resolve_voice_pickups(vcfg),
            coils=resolve_voice_coils(vcfg),
            circuit=vcfg.circuit,
        ),
    )
    meta_dump = nam_meta.model_dump()
    other_metadata = {
        train_meta.TRAINING_KEY: meta_dump["training"],
        "license": meta_dump["license"],
        "copyright": meta_dump["copyright"],
        "author": meta_dump["author"],
        "version": meta_dump["version"],
        "dsp_version": meta_dump["dsp_version"],
        "instrument_version": meta_dump["instrument_version"],
        "voice_version": meta_dump["voice_version"],
        "allomorph_version": meta_dump["allomorph_version"],
        "git_commit": meta_dump["git_commit"],
        "generated_at": meta_dump["generated_at"],
        "source_instrument": meta_dump["source_instrument"],
        "target_voice": meta_dump["target_voice"],
    }

    export_net: Any = train_output.model.net
    export_net.export(
        str(inst_models_dir),
        basename=model_basename,
        user_metadata=user_metadata,
        other_metadata=other_metadata,
    )

    # Clean up temporary lightning checkpoint folder
    if train_work_dir.exists():
        shutil.rmtree(train_work_dir, ignore_errors=True)

    if target_nam.exists():
        if not no_manifest:
            write_manifest(
                output_dir=inst_models_dir,
                stage="train",
                files=[target_nam],
                version_tag=tri_part,
            )
        size_kb = target_nam.stat().st_size / 1024
        print("\n[Success] Architecture 2 Model exported successfully!")
        print(f"  Model Path:    {target_nam} ({size_kb:.1f} KB)")
        print(f"  Model Title:   {model_title}")
        print(f"  Source Bass:   {inst_name}")
        print(f"  Source Pickup: {src_pickup_name} ({src_pos_mm:.1f}mm)")
        if train_output.metadata.validation_esr is not None:
            vesr = train_output.metadata.validation_esr
            best_studio_esr = (
                cb.best_esr if (cb is not None and cb.best_esr < float("inf")) else vesr
            )
            esr_status = ""
            if threshold_esr is not None:
                if best_studio_esr <= threshold_esr:
                    esr_status = (
                        f" (Goal Met <= {threshold_esr:.6f}, min {min_epochs} epochs observed)"
                    )
                else:
                    esr_status = f" (Safety ceiling reached at {epochs} epochs)"
            if not a2_lite_only:
                ch8_db = 10.0 * math.log10(max(best_studio_esr, 1e-12))
                print(
                    f"  Validation ESR: {best_studio_esr:.6f} (Ch8 Studio, {ch8_db:+.2f} dB) | {vesr:.6f} (Aggregate){esr_status}"
                )
            else:
                esr_db = 10.0 * math.log10(max(best_studio_esr, 1e-12))
                print(f"  Validation ESR: {best_studio_esr:.6f} ({esr_db:+.2f} dB){esr_status}")
            if best_diff_esr is not None:
                diff_db = 10.0 * math.log10(max(best_diff_esr, 1e-12))
                print(f"  Differential Delta ESR: {best_diff_esr:.6f} ({diff_db:+.2f} dB)")
            if best_mrstft is not None:
                print(f"  Validation MRSTFT: {best_mrstft:.6f}")
            if best_diff_mrstft is not None:
                print(f"  Differential MRSTFT: {best_diff_mrstft:.4f} (ratio of baseline)")
            if stop_reason:
                print(f"  Termination:   {stop_reason}")
        print("  Ready for Darkglass Anagram Block 1 (Preamp) loading.")
        return True
    else:
        print(f"Warning: Expected model file at {target_nam} not found.")
        return False


def main():
    parser = argparse.ArgumentParser(description="Allomorph NAM Architecture 2 Local Trainer")
    parser.add_argument(
        "--instrument",
        "-i",
        default="all",
        help="Source instrument configuration (ID, comma-separated list, 'all', path to .toml, or alias like 30in, 32in; default: 'all')",
    )
    parser.add_argument(
        "--voice",
        default="all",
        help="Target pickup voice (ID, comma-separated list, or 'all'; default: 'all')",
    )
    parser.add_argument(
        "--pickup",
        "-p",
        default=None,
        help="Physical pickup setting for source instrument (default: all pickups on the instrument)",
    )
    parser.add_argument(
        "--input", help="Path to dry training sweep WAV (default: auto-detect optimal_bass_dry.wav)"
    )
    parser.add_argument(
        "--output", help="Path to simulated SPICE output WAV (default: circuits/out_<voice>.wav)"
    )
    parser.add_argument("--models-dir", default=str(MODELS_DIR), help="Output models directory")
    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_MAX_EPOCHS,
        help=f"Maximum number of training epochs (default: {DEFAULT_MAX_EPOCHS} for Architecture 2 studio reference)",
    )
    parser.add_argument(
        "--min-epochs",
        "--warmup-epochs",
        dest="min_epochs",
        type=int,
        default=DEFAULT_MIN_EPOCHS,
        help=f"Minimum warmup training epochs before early stopping can trigger (default: {DEFAULT_MIN_EPOCHS})",
    )
    parser.add_argument(
        "--goal-esr",
        type=float,
        default=DEFAULT_GOAL_ESR,
        help=f"Goal validation ESR for early stopping (default: {DEFAULT_GOAL_ESR} for Architecture 2 studio reference; set to 0 to disable)",
    )
    parser.add_argument(
        "--goal-delta-esr",
        type=float,
        default=DEFAULT_GOAL_DELTA_ESR,
        help=f"Goal validation Differential Delta ESR on pickup delta (default: {DEFAULT_GOAL_DELTA_ESR}; set to 0 to disable)",
    )
    parser.add_argument(
        "--goal-delta-mrstft",
        type=float,
        default=DEFAULT_GOAL_DELTA_MRSTFT,
        help=f"Gate 3: Goal MRSTFT ratio relative to baseline (default: {DEFAULT_GOAL_DELTA_MRSTFT}; set to 0 to disable)",
    )
    parser.add_argument(
        "--max-mrstft-ceiling",
        type=float,
        default=DEFAULT_MAX_MRSTFT_CEILING,
        help=f"Gate 3: Absolute MRSTFT ceiling (default: {DEFAULT_MAX_MRSTFT_CEILING})",
    )
    parser.add_argument(
        "--consecutive-patience",
        type=int,
        default=DEFAULT_CONSECUTIVE_PATIENCE,
        help=f"Consecutive validation epochs satisfying all 3 gates before early exit (default: {DEFAULT_CONSECUTIVE_PATIENCE})",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=DEFAULT_PATIENCE,
        help=f"Plateau patience epochs for early stopping (default: {DEFAULT_PATIENCE}; set to 0 to disable)",
    )
    parser.add_argument(
        "--min-delta",
        type=float,
        default=DEFAULT_MIN_DELTA,
        help=f"Minimum loss improvement to reset plateau patience (default: {DEFAULT_MIN_DELTA})",
    )
    parser.add_argument(
        "--pre-emph-weight",
        type=float,
        default=DEFAULT_PRE_EMPH_WEIGHT,
        help=f"Pre-emphasis loss weight for equalizing high-frequency resonance (default: {DEFAULT_PRE_EMPH_WEIGHT})",
    )
    parser.add_argument(
        "--pre-emph-coef",
        type=float,
        default=DEFAULT_PRE_EMPH_COEF,
        help=f"Pre-emphasis filter coefficient (default: {DEFAULT_PRE_EMPH_COEF})",
    )
    parser.add_argument(
        "--mrstft-weight",
        type=float,
        default=DEFAULT_MRSTFT_WEIGHT,
        help=f"Multi-Resolution STFT loss weight (default: {DEFAULT_MRSTFT_WEIGHT})",
    )
    parser.add_argument(
        "--lr-scheduler",
        choices=["cosine", "exponential"],
        default=DEFAULT_LR_SCHEDULER,
        help=f"Learning rate scheduler type (default: '{DEFAULT_LR_SCHEDULER}')",
    )
    parser.add_argument(
        "--eta-min",
        type=float,
        default=DEFAULT_ETA_MIN,
        help=f"Minimum learning rate floor for cosine annealing (default: {DEFAULT_ETA_MIN})",
    )
    parser.add_argument(
        "--lr-t-max",
        type=int,
        default=DEFAULT_LR_T_MAX,
        help=f"Cosine annealing cycle length T_max in epochs (default: {DEFAULT_LR_T_MAX})",
    )
    parser.add_argument(
        "--reference",
        default=None,
        help="Path to reference source stem for differential delta ESR (default: auto-detected from source pickup)",
    )
    parser.add_argument(
        "--no-goal-esr",
        action="store_true",
        help="Disable goal ESR early stopping and train for the exact number of epochs specified",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Batch size (default: {DEFAULT_BATCH_SIZE})",
    )
    parser.add_argument(
        "--a2-lite-only",
        action="store_true",
        default=None,
        help="Train A2-Lite channels_8 only (default: True, ~35%% faster)",
    )
    parser.add_argument(
        "--full-slimmable",
        action="store_true",
        help="Train full slimmable Architecture 2 (channels_3 + channels_8) instead of default A2-Lite",
    )
    parser.add_argument(
        "--show-plot", action="store_true", help="Display matplotlib validation plot window"
    )
    parser.add_argument(
        "--save-plot", action="store_true", help="Save validation plot as PNG in models/"
    )
    parser.add_argument("--basename", help="Explicit basename for the exported .nam model file")
    parser.add_argument(
        "--fast-dev-run",
        action="store_true",
        help="Run 1-batch dry run for smoke testing NAM training",
    )
    parser.add_argument(
        "--version-tag",
        default="auto",
        help="Semantic version tag (default: 'auto' -> v[dsp].[inst].[voice], or explicit string, or 'none' to disable)",
    )
    parser.add_argument(
        "--no-manifest",
        action="store_true",
        help="Disable generating sidecar manifest.json",
    )
    parser.add_argument(
        "--include-identity",
        action="store_true",
        help="Force training even if source and target stems are identical",
    )
    parser.add_argument("--gui", action="store_true", help="Launch NAM training GUI")
    args = parser.parse_args()

    effective_a2_lite = (
        not args.full_slimmable
        if args.full_slimmable
        else (args.a2_lite_only if args.a2_lite_only is not None else True)
    )

    cli_cfg = NamTrainingConfig.model_validate(
        {
            "instrument": args.instrument,
            "voice": args.voice,
            "input_wav": args.input,
            "output_wav": args.output,
            "reference_wav": args.reference,
            "models_dir": args.models_dir,
            "epochs": args.epochs,
            "min_epochs": args.min_epochs,
            "goal_esr": args.goal_esr,
            "goal_delta_esr": args.goal_delta_esr,
            "goal_delta_mrstft": args.goal_delta_mrstft,
            "max_mrstft_ceiling": args.max_mrstft_ceiling,
            "consecutive_patience": args.consecutive_patience,
            "patience": args.patience,
            "min_delta": args.min_delta,
            "pre_emph_weight": args.pre_emph_weight,
            "pre_emph_coef": args.pre_emph_coef,
            "mrstft_weight": args.mrstft_weight,
            "lr_scheduler": args.lr_scheduler,
            "eta_min": args.eta_min,
            "lr_t_max": args.lr_t_max,
            "no_goal_esr": args.no_goal_esr,
            "batch_size": args.batch_size,
            "show_plot": args.show_plot,
            "save_plot": args.save_plot,
            "basename": args.basename,
            "fast_dev_run": args.fast_dev_run,
            "gui": args.gui,
            "a2_lite_only": effective_a2_lite,
            "full_slimmable": args.full_slimmable,
            "version_tag": args.version_tag,
            "no_manifest": args.no_manifest,
            "include_identity": args.include_identity,
        }
    )

    if cli_cfg.gui:
        try:
            from nam.cli import nam_gui

            nam_gui()
            return
        except ImportError:
            print("Error: 'neural-amp-modeler' GUI could not be loaded.")
            return

    effective_goal_esr = (
        None
        if cli_cfg.no_goal_esr or (cli_cfg.goal_esr is not None and cli_cfg.goal_esr <= 0)
        else cli_cfg.goal_esr
    )

    input_wav_path = cli_cfg.input_wav
    instruments_to_run = resolve_instruments(cli_cfg.instrument)
    voices_to_run = resolve_voices(cli_cfg.voice)
    all_ok = True
    total_runs = len(instruments_to_run) * len(voices_to_run)
    current_run = 0
    for inst in instruments_to_run:
        for idx, voice in enumerate(voices_to_run, 1):
            current_run += 1
            if total_runs > 1:
                print("\n==================================================")
                print(f"  [{current_run}/{total_runs}] Training: {inst} -> {voice}")
                print("==================================================")
            out_wav = (
                cli_cfg.output_wav
                if (len(voices_to_run) == 1 and len(instruments_to_run) == 1)
                else None
            )
            ok = train_voice(
                instrument=inst,
                voice=voice,
                input_wav=input_wav_path,
                output_wav=out_wav,
                reference_wav=cli_cfg.reference_wav,
                models_dir=cli_cfg.models_dir,
                epochs=cli_cfg.epochs,
                min_epochs=cli_cfg.min_epochs,
                goal_esr=effective_goal_esr,
                goal_delta_esr=cli_cfg.goal_delta_esr,
                goal_delta_mrstft=cli_cfg.goal_delta_mrstft,
                max_mrstft_ceiling=cli_cfg.max_mrstft_ceiling,
                consecutive_patience=cli_cfg.consecutive_patience,
                patience=cli_cfg.patience,
                min_delta=cli_cfg.min_delta,
                pre_emph_weight=cli_cfg.pre_emph_weight,
                pre_emph_coef=cli_cfg.pre_emph_coef,
                mrstft_weight=cli_cfg.mrstft_weight,
                lr_scheduler=cli_cfg.lr_scheduler,
                eta_min=cli_cfg.eta_min,
                lr_t_max=cli_cfg.lr_t_max,
                batch_size=cli_cfg.batch_size,
                silent=not cli_cfg.show_plot,
                save_plot=cli_cfg.save_plot,
                fast_dev_run=cli_cfg.fast_dev_run,
                basename=cli_cfg.basename
                if (len(voices_to_run) == 1 and len(instruments_to_run) == 1)
                else None,
                a2_lite_only=cli_cfg.a2_lite_only,
                version_tag=cli_cfg.version_tag,
                no_manifest=cli_cfg.no_manifest,
                include_identity=cli_cfg.include_identity,
            )
            if not ok:
                all_ok = False

    if not all_ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
