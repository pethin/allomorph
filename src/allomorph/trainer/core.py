"""
Allomorph Trainer - Core Architecture 2 Training Engine
Provides dataset preparation, baseline delta & spectral distance computation,
NAM Architecture 2 model configuration, and model export with studio reference metadata.
"""

import contextlib
import math
import os
import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, override

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
    get_default_input_path,
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
from allomorph.trainer.callbacks import (
    AllomorphAdaptiveStopping,
    EsrProgressCallback,
    LinearWarmupCallback,
)
from allomorph.trainer.constants import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_ETA_MIN,
    DEFAULT_LR_SCHEDULER,
    DEFAULT_LR_T_MAX,
    DEFAULT_MAX_EPOCHS,
    DEFAULT_MIN_DELTA,
    DEFAULT_MIN_EPOCHS,
    DEFAULT_MRSTFT_WEIGHT,
    DEFAULT_NUM_WORKERS,
    DEFAULT_PATIENCE,
    DEFAULT_PRE_EMPH_COEF,
    DEFAULT_PRE_EMPH_WEIGHT,
    DEFAULT_PRECISION,
)
from allomorph.version import (
    ALLOMORPH_VERSION,
    DSP_GENERATION,
    get_git_commit,
    is_wet_stem_valid,
    resolve_tri_part_version,
    write_manifest,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
AUDIO_DIR = REPO_ROOT / "audio"
MODELS_DIR = REPO_ROOT / "models"
DEFAULT_INPUT_PATH = get_default_input_path()


def setup_headless_environment() -> None:
    """Configures MIOpen runtime parameters, matplotlib backend, and headless Tkinter shim."""
    # ROCm / MIOpen optimizations for AMD GPUs (e.g. RDNA 3/4, gfx1201)
    os.environ.setdefault("TORCH_BLAS_PREFER_HIPBLASLT", "0")
    os.environ.setdefault("MIOPEN_FIND_MODE", "FAST")
    os.environ.setdefault("MIOPEN_LOG_LEVEL", "2")

    if "MIOPEN_COMPILE_PARALLEL_LEVEL" not in os.environ:
        cpu_count = os.cpu_count() or 4
        os.environ["MIOPEN_COMPILE_PARALLEL_LEVEL"] = str(min(cpu_count, 16))

    miopen_cache_dir = REPO_ROOT / ".cache" / "miopen"
    try:
        miopen_cache_dir.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("MIOPEN_CACHE_DIR", str(miopen_cache_dir))
        os.environ.setdefault("MIOPEN_USER_DB_PATH", str(miopen_cache_dir))
    except OSError:
        pass

    # Ensure headless matplotlib raster backend is active and suppress open figure accumulation
    try:
        import matplotlib
        import matplotlib.pyplot as plt

        matplotlib.use("Agg")
        plt.rc("figure", max_open_warning=0)
    except ImportError:
        pass

    # Suppress upstream PyTorch Lightning / Python 3.14 deprecation warnings
    import warnings

    warnings.filterwarnings("ignore", message=r".*LeafSpec.*")
    # Suppress Lightning 16-mixed model summary warning (informational bit depth notice)
    warnings.filterwarnings("ignore", message=r".*not supported by the model summary.*")

    # Suppress LitLogger promotion tip from Lightning rank_zero logs
    import logging

    class _LitLoggerFilter(logging.Filter):
        @override
        def filter(self, record: logging.LogRecord) -> bool:
            return "litlogger" not in record.getMessage()

    logging.getLogger("lightning_fabric.utilities.rank_zero").addFilter(_LitLoggerFilter())
    logging.getLogger("lightning_utilities.core.rank_zero").addFilter(_LitLoggerFilter())
    logging.getLogger("pytorch_lightning.utilities.rank_zero").addFilter(_LitLoggerFilter())

    # Headless Tkinter fallback: neural-amp-modeler's core trainer imports tkinter at top-level
    if "tkinter" not in sys.modules:
        try:
            import tkinter  # noqa: F401
        except ModuleNotFoundError:
            import types

            def _dummy_mainloop(*args: object, **kwargs: object) -> None:
                pass

            class _DummyTkMisc:
                mainloop = _dummy_mainloop

            class _DummyTkWidget:
                def __init__(self, *args: object, **kwargs: object) -> None:
                    pass

                def __getattr__(self, name: str) -> Any:
                    return _dummy_mainloop

            class _DummyTkModule(types.ModuleType):
                Tk = _DummyTkWidget
                Toplevel = _DummyTkWidget
                Label = _DummyTkWidget
                Button = _DummyTkWidget
                Misc = _DummyTkMisc
                mainloop = _dummy_mainloop

            sys.modules["tkinter"] = _DummyTkModule("tkinter")


setup_headless_environment()


def get_hardware_device_name() -> str:
    """Returns human-readable name of active accelerator or CPU."""
    try:
        import torch

        if torch.cuda.is_available():
            return str(torch.cuda.get_device_name(0))
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "Apple Silicon (MPS)"
        return "CPU"
    except (ImportError, RuntimeError):
        return "CPU"


def resolve_hardware_batch_size(
    batch_size: int | str = DEFAULT_BATCH_SIZE,
    num_train_samples: int = 1350,
) -> int:
    """Dynamically resolves optimal batch size balancing GPU VRAM and gradient update density."""
    if isinstance(batch_size, int) and batch_size > 0:
        return batch_size
    if isinstance(batch_size, str) and batch_size.isdigit():
        return int(batch_size)

    try:
        import torch

        if torch.cuda.is_available():
            _free, total_bytes = torch.cuda.mem_get_info()
            total_gb = total_bytes / (1024**3)
            # High-end GPU (e.g. RX 9070 XT 16GB, RTX 4080/4090): 32
            # Yields ~42 batches/epoch, optimal 17.4s epoch speed
            if total_gb >= 12.0:
                return 32
            if total_gb >= 6.0:
                return 16
            return 8
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            total_ram_gb = (os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")) / (1024**3)
            return 32 if total_ram_gb >= 32.0 else 16
        return 8
    except (ImportError, RuntimeError, OSError):
        return 16


def resolve_hardware_precision(precision: str = DEFAULT_PRECISION) -> str:
    """Dynamically resolves PyTorch Lightning precision based on device capabilities."""
    if precision != "auto":
        return precision

    try:
        import torch

        if torch.cuda.is_available():
            # Check for ROCm/HIP: Composable Kernel in ROCm 7.14 has an upstream bug
            # where bfloat16 grouped convolution backward prints descriptor spam to stdout.
            # 16-mixed is 5.6% faster on RDNA 4 (646ms vs 683ms) and completely silent.
            is_hip = bool(getattr(torch.version, "hip", None))
            if is_hip:
                return "16-mixed"
            if hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
                return "bf16-mixed"
            return "16-mixed"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "32-true"
        return "32-true"
    except (ImportError, RuntimeError):
        return "32-true"


def resolve_hardware_num_workers(num_workers: int | str = DEFAULT_NUM_WORKERS) -> int:
    """Dynamically resolves DataLoader num_workers for audio datasets."""
    if isinstance(num_workers, int) and num_workers >= 0:
        return num_workers
    if isinstance(num_workers, str) and num_workers.isdigit():
        return int(num_workers)
    # Default auto: in-memory contiguous tensor dataset yields zero-copy main-thread slicing
    return 0


def find_sweep_input(
    candidate_path: str | Path | None = None,
    version_tag: str | None = None,
) -> Path | None:
    """Finds the path to the dry sweep input WAV file."""
    if candidate_path and Path(candidate_path).exists():
        return Path(candidate_path)
    from allomorph.circuit.audio import find_default_input_audio

    return find_default_input_audio(version_tag=version_tag)


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
    min_epochs: int = DEFAULT_MIN_EPOCHS,
    patience: int = DEFAULT_PATIENCE,
    min_delta: float = DEFAULT_MIN_DELTA,
    pre_emph_weight: float = DEFAULT_PRE_EMPH_WEIGHT,
    pre_emph_coef: float = DEFAULT_PRE_EMPH_COEF,
    mrstft_weight: float = DEFAULT_MRSTFT_WEIGHT,
    lr_scheduler: str = DEFAULT_LR_SCHEDULER,
    eta_min: float = DEFAULT_ETA_MIN,
    lr_t_max: int = DEFAULT_LR_T_MAX,
    precision: str = DEFAULT_PRECISION,
    num_workers: int | str = DEFAULT_NUM_WORKERS,
    reference_wav: str | Path | None = None,
    output_wav: str | Path | None = None,
) -> None:
    """Configure NAM Architecture 2 packed model submodels, loss weighting, and adaptive early stopping.

    Configures dual-tier slimmable PackedWaveNet (channels_3 A2 Lite + channels_8 A2 Full) for Tone3000 A2 recognition.
    Injects pre-emphasis loss (alpha=0.85) and MRSTFT (0.0010) to equalize high-frequency pickup resonance gradients.
    Hooks LinearWarmupCallback (5 epochs) and AllomorphAdaptiveStopping (composite val_loss plateau).
    """
    resolved_precision = resolve_hardware_precision(precision)
    resolved_workers = resolve_hardware_num_workers(num_workers)

    try:
        import torch

        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high")

        if hasattr(torch, "backends") and hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.benchmark = torch.cuda.is_available()
    except ImportError:
        pass

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

    orig_get_configs = getattr(
        nam_core, "_orig_get_configs", getattr(nam_core, "_get_configs", None)
    )
    if orig_get_configs is not None:
        nam_core._orig_get_configs = orig_get_configs

        def patched_get_configs(
            *args: Any, **kwargs: Any
        ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
            data_config, model_config, learning_config = orig_get_configs(*args, **kwargs)
            if resolved_precision in ("bf16-mixed", "16-mixed", "32-true", "32"):
                learning_config["trainer"]["precision"] = resolved_precision
            learning_config["train_dataloader"]["num_workers"] = resolved_workers
            if resolved_workers > 0:
                learning_config["train_dataloader"]["persistent_workers"] = True
            return data_config, model_config, learning_config

        nam_core._get_configs = patched_get_configs

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

        stopping_cb = AllomorphAdaptiveStopping(
            monitor="val_loss",
            warmup_floor=effective_min_epochs,
            patience=patience,
            min_delta=min_delta,
        )
        callbacks.append(stopping_cb)
        nam_core._last_stopping_callback = stopping_cb

        progress_cb = EsrProgressCallback(
            min_epochs=effective_min_epochs,
            stopping_callback=stopping_cb,
        )
        callbacks.append(progress_cb)
        nam_core._last_esr_callback = progress_cb

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
    patience: int = DEFAULT_PATIENCE,
    min_delta: float = DEFAULT_MIN_DELTA,
    pre_emph_weight: float = DEFAULT_PRE_EMPH_WEIGHT,
    pre_emph_coef: float = DEFAULT_PRE_EMPH_COEF,
    mrstft_weight: float = DEFAULT_MRSTFT_WEIGHT,
    lr_scheduler: str = DEFAULT_LR_SCHEDULER,
    eta_min: float = DEFAULT_ETA_MIN,
    lr_t_max: int = DEFAULT_LR_T_MAX,
    batch_size: int | str = DEFAULT_BATCH_SIZE,
    precision: str = DEFAULT_PRECISION,
    num_workers: int | str = DEFAULT_NUM_WORKERS,
    silent: bool = True,
    save_plot: bool = False,
    fast_dev_run: bool = False,
    basename: str | None = None,
    version_tag: str | None = "auto",
    no_manifest: bool = False,
    include_identity: bool = False,
) -> bool:
    """Trains a Neural Amp Modeler Architecture 2 local model container."""
    try:
        import nam.train.core as nam_core
        import nam.train.metadata as train_meta
        import torch
        from nam.models.metadata import UserMetadata

        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high")
        if hasattr(torch, "backends") and hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.benchmark = torch.cuda.is_available()
    except ImportError as e:
        print(f"Error: Failed to import 'neural-amp-modeler' ({e}).")
        print("Please run `uv sync` to ensure project dependencies are installed:")
        print(f"  uv run python -m allomorph.trainer --voice {voice}")
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

    models_path = Path(models_dir)
    if models_path.name == "nam" or models_path.name == inst_id:
        inst_models_dir = models_path
    else:
        inst_models_dir = models_path / inst_id
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

    # Resolve reference stem
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

    resolved_batch_size = resolve_hardware_batch_size(batch_size)
    resolved_precision = resolve_hardware_precision(precision)
    resolved_num_workers = resolve_hardware_num_workers(num_workers)
    device_name = get_hardware_device_name()

    configure_a2_architecture(
        nam_core,
        min_epochs=min_epochs,
        patience=patience,
        min_delta=min_delta,
        pre_emph_weight=pre_emph_weight,
        pre_emph_coef=pre_emph_coef,
        mrstft_weight=mrstft_weight,
        lr_scheduler=lr_scheduler,
        eta_min=eta_min,
        lr_t_max=lr_t_max,
        precision=resolved_precision,
        num_workers=resolved_num_workers,
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
    arch_display = "Architecture 2 Slimmable (channels_3 + channels_8)"
    print(f"  Model Tier:  {arch_display}")
    print(f"  Hardware:    {device_name}")
    steps_per_epoch = 1350 // resolved_batch_size
    print(f"  Batch Size:  {resolved_batch_size} (auto, {steps_per_epoch} steps/epoch)")
    print(f"  Precision:   {resolved_precision}")
    print(f"  Workers:     {resolved_num_workers} (zero-copy in-memory)")
    print(f"  Input Audio: {input_path.name}")
    print(f"  Output Audio:{output_path.name}")
    print(f"  Schedule:    {lr_scheduler} (epochs={epochs}, T_max={lr_t_max}, eta_min={eta_min:.1e})")
    print(f"  Convergence: Adaptive plateau patience={patience} on composite val_loss (min_delta={min_delta:.1e})")
    if pre_emph_weight > 0:
        print(f"  Pre-Emph:    weight={pre_emph_weight:.2f}, coef={pre_emph_coef:.2f}")
    if mrstft_weight > 0:
        print(f"  MRSTFT:      weight={mrstft_weight:.4f}")
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
    try:
        train_output = nam_core.train(
            input_path=str(input_path),
            output_path=str(output_path),
            train_path=str(train_work_dir),
            epochs=epochs,
            batch_size=resolved_batch_size,
            modelname=model_basename,
            silent=silent,
            save_plot=save_plot,
            local=False,
            threshold_esr=None,
            user_metadata=user_metadata,
            fast_dev_run=fast_dev_run,
            latency=0,
            ignore_checks=True,
        )
    finally:
        with contextlib.suppress(ImportError, RuntimeError):
            import matplotlib.pyplot as plt

            plt.close("all")

    if train_output is None or train_output.model is None:
        print("Error: Training did not produce a model.")
        return False

    cb = getattr(nam_core, "_last_esr_callback", None)
    stopping_cb = getattr(nam_core, "_last_stopping_callback", None)
    best_mrstft = (
        cb.best_mrstft
        if (cb is not None and cb.best_mrstft is not None and cb.best_mrstft < float("inf"))
        else None
    )
    epochs_trained = cb.last_epoch + 1 if (cb is not None and cb.last_epoch >= 0) else None
    stop_reason = stopping_cb.stop_reason if stopping_cb is not None else None

    raw_meta = train_output.metadata.model_dump()
    vesr = raw_meta.get("validation_esr")
    best_studio_esr = (
        cb.best_esr if (cb is not None and cb.best_esr < float("inf")) else vesr
    )
    best_lite_esr = (
        cb.best_ch3_esr
        if (cb is not None and cb.best_ch3_esr is not None and cb.best_ch3_esr < float("inf"))
        else None
    )
    best_lite_mrstft = (
        cb.best_ch3_mrstft
        if (cb is not None and cb.best_ch3_mrstft is not None and cb.best_ch3_mrstft < float("inf"))
        else None
    )

    if best_studio_esr is not None:
        raw_meta["validation_esr"] = best_studio_esr
        raw_meta["validation_esr_a2_full"] = best_studio_esr
        raw_meta["validation_esr_ch8"] = best_studio_esr
    if best_lite_esr is not None:
        raw_meta["validation_esr_a2_lite"] = best_lite_esr
        raw_meta["validation_esr_ch3"] = best_lite_esr
    if vesr is not None:
        raw_meta["validation_esr_aggregate"] = vesr
    if best_lite_mrstft is not None:
        raw_meta["mrstft_loss_ch3"] = best_lite_mrstft
    if best_mrstft is not None:
        raw_meta["mrstft_loss"] = best_mrstft
    if epochs_trained is not None:
        raw_meta["epochs_trained"] = epochs_trained
    if stop_reason is not None:
        raw_meta["stop_reason"] = stop_reason
    raw_meta["batch_size"] = resolved_batch_size
    raw_meta["precision"] = resolved_precision
    raw_meta["num_workers"] = resolved_num_workers
    raw_meta["device_name"] = device_name

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
            vesr_val = float(train_output.metadata.validation_esr)
            best_studio_val: float = (
                cb.best_esr if (cb is not None and cb.best_esr < float("inf")) else vesr_val
            )
            ch8_db = 10.0 * math.log10(max(best_studio_val, 1e-12))
            lite_disp = ""
            if best_lite_esr is not None:
                lite_db = 10.0 * math.log10(max(best_lite_esr, 1e-12))
                lite_disp = f" | {best_lite_esr:.6f} (A2 Lite Ch3, {lite_db:+.2f} dB)"
            agg_disp = f" | {vesr_val:.6f} (Aggregate)"
            print(
                f"  Validation ESR: {best_studio_val:.6f} (A2 Full Ch8, {ch8_db:+.2f} dB){lite_disp}{agg_disp}"
            )
            if best_mrstft is not None:
                print(f"  Validation MRSTFT: {best_mrstft:.6f}")
            if stop_reason:
                print(f"  Termination:   {stop_reason}")
        print("  Ready for Darkglass Anagram Block 1 (Preamp) loading.")
        return True
    else:
        print(f"Warning: Expected model file at {target_nam} not found.")
        return False


def train_voices_from_config(cli_cfg: NamTrainingConfig) -> bool:
    """Executes NAM model training across configured instruments and voices in-process."""
    if cli_cfg.pack:
        from allomorph.pipeline.pack import train_tone_pack

        train_tone_pack(
            pack=cli_cfg.pack,
            voice=cli_cfg.voice or "all",
            overwrite=cli_cfg.overwrite,
            epochs=cli_cfg.epochs,
            min_epochs=cli_cfg.min_epochs,
            patience=cli_cfg.patience,
            min_delta=cli_cfg.min_delta,
            batch_size=cli_cfg.batch_size,
            precision=cli_cfg.precision,
            num_workers=cli_cfg.num_workers,
            lr_scheduler=cli_cfg.lr_scheduler,
            eta_min=cli_cfg.eta_min,
            lr_t_max=cli_cfg.lr_t_max,
            fast_dev_run=cli_cfg.fast_dev_run,
        )
        return True

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
                patience=cli_cfg.patience,
                min_delta=cli_cfg.min_delta,
                pre_emph_weight=cli_cfg.pre_emph_weight,
                pre_emph_coef=cli_cfg.pre_emph_coef,
                mrstft_weight=cli_cfg.mrstft_weight,
                lr_scheduler=cli_cfg.lr_scheduler,
                eta_min=cli_cfg.eta_min,
                lr_t_max=cli_cfg.lr_t_max,
                batch_size=cli_cfg.batch_size,
                precision=cli_cfg.precision,
                num_workers=cli_cfg.num_workers,
                silent=not cli_cfg.show_plot,
                save_plot=cli_cfg.save_plot,
                fast_dev_run=cli_cfg.fast_dev_run,
                basename=cli_cfg.basename
                if (len(voices_to_run) == 1 and len(instruments_to_run) == 1)
                else None,
                version_tag=cli_cfg.version_tag,
                no_manifest=cli_cfg.no_manifest,
                include_identity=cli_cfg.include_identity,
            )
            if not ok:
                all_ok = False

            # Clear GPU memory between consecutive models in batch training
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass

    return all_ok


__all__ = [
    "AUDIO_DIR",
    "DEFAULT_INPUT_PATH",
    "MODELS_DIR",
    "compute_baseline_delta_ratio",
    "compute_baseline_mrstft",
    "configure_a2_architecture",
    "find_sweep_input",
    "setup_headless_environment",
    "train_voice",
    "train_voices_from_config",
]
