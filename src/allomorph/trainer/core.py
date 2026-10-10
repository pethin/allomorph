"""
Allomorph Trainer - Core Architecture 2 Training Engine
Provides dataset preparation, baseline delta & spectral distance computation,
NAM Architecture 2 model configuration, and model export with studio reference metadata.
"""

import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, override

from allomorph.config import InstrumentConfig
from allomorph.naming import (
    get_default_input_path,
    resolve_instruments,
    resolve_voices,
)
from allomorph.pipeline.schema import NamTrainingConfig
from allomorph.trainer.constants import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_ENGINE,
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
    DEFAULT_SEED,
    VALID_ENGINES,
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


def is_mlx_available() -> bool:
    """Checks whether Apple Silicon MLX with Metal GPU acceleration is available."""
    try:
        from allomorph.trainer.engine_mlx import is_mlx_available as _check

        return bool(_check())
    except ImportError, RuntimeError, OSError:
        return False


def resolve_trainer_engine(engine: str = DEFAULT_ENGINE) -> str:
    """Resolves trainer engine ('mlx' or 'torch').
    - 'auto': chooses 'mlx' if is_mlx_available() else 'torch'
    - 'mlx': requires is_mlx_available(), raises RuntimeError if unavailable
    - 'torch': uses PyTorch
    """
    engine_clean = engine.lower().strip()
    if engine_clean not in VALID_ENGINES:
        raise ValueError(f"Invalid engine '{engine}'. Valid options: {VALID_ENGINES}")

    if engine_clean == "auto":
        return "mlx" if is_mlx_available() else "torch"
    if engine_clean == "mlx":
        if not is_mlx_available():
            raise RuntimeError(
                "MLX engine requested, but Apple Silicon MLX with Metal GPU is not available."
            )
        return "mlx"
    return "torch"


def get_hardware_device_name(engine: str = DEFAULT_ENGINE) -> str:
    """Returns human-readable name of active accelerator or CPU."""
    try:
        resolved = resolve_trainer_engine(engine)
        if resolved == "mlx":
            return "Apple Silicon (MLX Metal)"
    except RuntimeError, ValueError:
        pass

    try:
        import torch

        if torch.cuda.is_available():
            return str(torch.cuda.get_device_name(0))
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "Apple Silicon (MPS)"
        return "CPU"
    except ImportError, RuntimeError:
        return "CPU"


def detect_apple_silicon_tier() -> str:
    """Detects Apple Silicon processor tier: 'base', 'pro', 'max', 'ultra', or 'unknown'.

    Tiers correspond to Apple Silicon cache and GPU core topologies:
    - Base (M1-M4): 7-10 GPU cores, 8-16MB SLC, 68-150 GB/s bandwidth.
    - Pro (M1-M4 Pro): 14-20 GPU cores, 24-32MB SLC, 150-273 GB/s bandwidth.
    - Max (M1-M4 Max): 30-40 GPU cores, 48-64MB SLC, 300-410+ GB/s bandwidth.
    - Ultra (M1-M2 Ultra): 60-80 GPU cores, 96-128MB SLC, 800+ GB/s bandwidth.
    """
    if platform.system() != "Darwin":
        return "unknown"

    brand = ""
    try:
        res = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True,
            text=True,
            check=False,
            timeout=1.0,
        )
        brand = res.stdout.strip().lower()
    except subprocess.SubprocessError, OSError:
        pass

    if brand:
        if "ultra" in brand:
            return "ultra"
        if "max" in brand:
            return "max"
        if "pro" in brand:
            return "pro"
        if "apple" in brand:
            return "base"

    # Fallback to physical memory capacity and logical CPU core count
    try:
        ram_gb = (os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")) / (1024**3)
        cpus = os.cpu_count() or 8
        if ram_gb >= 60.0 and cpus >= 20:
            return "ultra"
        if ram_gb >= 30.0 and cpus >= 12:
            return "max"
        if ram_gb >= 16.0 and cpus >= 10:
            return "pro"
        return "base"
    except OSError, ValueError:
        return "base"


def resolve_hardware_batch_size(
    batch_size: int | str = DEFAULT_BATCH_SIZE,
    num_train_samples: int = 1350,
    engine: str = DEFAULT_ENGINE,
) -> int:
    """Dynamically resolves optimal batch size balancing GPU VRAM and gradient update density."""
    if isinstance(batch_size, int) and batch_size > 0:
        return batch_size
    if isinstance(batch_size, str) and batch_size.isdigit():
        return int(batch_size)

    try:
        if engine == "mlx" or (engine == "auto" and is_mlx_available()):
            tier = detect_apple_silicon_tier()
            # Max (30-40 GPU cores, 48-64MB SLC) and Ultra (60-80 GPU cores, 96-128MB SLC)
            # saturate workgroups at batch size 16 without cache spilling.
            # Base (7-10 GPU cores) and Pro (14-20 GPU cores) remain at batch size 8
            # to stay within their 8-24MB L2/SLC cache boundaries.
            if tier in ("max", "ultra"):
                return 16
            return 8

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
    except ImportError, RuntimeError, OSError:
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
    except ImportError, RuntimeError:
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
    except FileNotFoundError, ValueError, RuntimeError, KeyError, OSError, TypeError:
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
    except FileNotFoundError, ValueError, RuntimeError, KeyError, OSError, TypeError:
        return 0.400


def configure_a2_architecture(*args: Any, **kwargs: Any) -> None:
    """Configures NAM Architecture 2 packed model submodels, loss weighting, and adaptive early stopping via PyTorch."""
    from allomorph.trainer.engine_torch import configure_a2_architecture as _cfg

    _cfg(*args, **kwargs)


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
    engine: str = DEFAULT_ENGINE,
    seed: int | None = DEFAULT_SEED,
) -> bool:
    """Trains a Neural Amp Modeler Architecture 2 model container using the configured engine."""
    resolved_engine = resolve_trainer_engine(engine)
    if resolved_engine == "mlx":
        from allomorph.trainer.engine_mlx import train_voice_mlx

        return train_voice_mlx(
            instrument=instrument,
            voice=voice,
            input_wav=input_wav,
            output_wav=output_wav,
            reference_wav=reference_wav,
            models_dir=models_dir,
            epochs=epochs,
            min_epochs=min_epochs,
            patience=patience,
            min_delta=min_delta,
            pre_emph_weight=pre_emph_weight,
            pre_emph_coef=pre_emph_coef,
            mrstft_weight=mrstft_weight,
            lr_scheduler=lr_scheduler,
            eta_min=eta_min,
            lr_t_max=lr_t_max,
            batch_size=batch_size,
            precision=precision,
            num_workers=num_workers,
            silent=silent,
            save_plot=save_plot,
            fast_dev_run=fast_dev_run,
            basename=basename,
            version_tag=version_tag,
            no_manifest=no_manifest,
            include_identity=include_identity,
            seed=seed,
        )

    from allomorph.trainer.engine_torch import train_voice_torch

    return train_voice_torch(
        instrument=instrument,
        voice=voice,
        input_wav=input_wav,
        output_wav=output_wav,
        reference_wav=reference_wav,
        models_dir=models_dir,
        epochs=epochs,
        min_epochs=min_epochs,
        patience=patience,
        min_delta=min_delta,
        pre_emph_weight=pre_emph_weight,
        pre_emph_coef=pre_emph_coef,
        mrstft_weight=mrstft_weight,
        lr_scheduler=lr_scheduler,
        eta_min=eta_min,
        lr_t_max=lr_t_max,
        batch_size=batch_size,
        precision=precision,
        num_workers=num_workers,
        silent=silent,
        save_plot=save_plot,
        fast_dev_run=fast_dev_run,
        basename=basename,
        version_tag=version_tag,
        no_manifest=no_manifest,
        include_identity=include_identity,
        seed=seed,
    )


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
            engine=cli_cfg.engine,
            seed=cli_cfg.seed,
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
                engine=cli_cfg.engine,
                seed=cli_cfg.seed,
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
    "get_hardware_device_name",
    "is_mlx_available",
    "resolve_hardware_batch_size",
    "resolve_hardware_num_workers",
    "resolve_hardware_precision",
    "resolve_trainer_engine",
    "setup_headless_environment",
    "train_voice",
    "train_voices_from_config",
]
