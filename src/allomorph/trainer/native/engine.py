"""Native PyTorch training engine for Allomorph Architecture 2 (A2) NAM models.

Zero-external-framework training loop featuring:
- In-VRAM zero-copy tensor slicing (loading stem audio once directly on GPU/accelerator).
- Self-Normalizing Multi-Band High-Pass ESR Loss with waveform MSE anchoring.
- Anti-Drift Pareto early stopping with in-memory checkpoint restoration.
- Native export to compliant Darkglass Anagram Block 1 slimmable .nam JSON.
"""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from pedalboard.io import AudioFile

from allomorph.config import (
    VOICES,
    InstrumentConfig,
    load_instrument,
)
from allomorph.trainer.callbacks import (
    compute_pareto_quality_score,
    evaluate_plateau_stopping_condition,
)
from allomorph.trainer.constants import (
    DEFAULT_ENGINE,
    DEFAULT_ETA_MIN,
    DEFAULT_LR_T_MAX,
    DEFAULT_MRSTFT_WEIGHT,
    DEFAULT_NUM_WORKERS,
    DEFAULT_PRE_EMPH_COEF,
    DEFAULT_PRE_EMPH_WEIGHT,
    DEFAULT_SEED,
    NATIVE_DEFAULT_BATCH_SIZE,
    NATIVE_DEFAULT_HP_HIGH_CUTOFF,
    NATIVE_DEFAULT_HP_HIGH_WEIGHT,
    NATIVE_DEFAULT_HP_MID_CUTOFF,
    NATIVE_DEFAULT_HP_MID_WEIGHT,
    NATIVE_DEFAULT_LITE_WEIGHT,
    NATIVE_DEFAULT_LR,
    NATIVE_DEFAULT_LR_SCHEDULER,
    NATIVE_DEFAULT_MAX_EPOCHS,
    NATIVE_DEFAULT_MIN_DELTA,
    NATIVE_DEFAULT_MIN_EPOCHS,
    NATIVE_DEFAULT_MSE_WEIGHT,
    NATIVE_DEFAULT_PATIENCE,
    NATIVE_DEFAULT_PRECISION,
    NATIVE_DEFAULT_SLICE_LEN,
    NATIVE_DEFAULT_WEIGHT_DECAY,
)
from allomorph.trainer.native.loss import MultiBandSpectralLoss
from allomorph.trainer.native.models import AllomorphWaveNetA2
from allomorph.version import (
    DSP_GENERATION,
    is_wet_stem_valid,
    resolve_tri_part_version,
)

REPO_ROOT = Path(__file__).resolve().parents[4]
AUDIO_DIR = REPO_ROOT / "audio"
MODELS_DIR = REPO_ROOT / "models"


def load_audio_tensor(wav_path: str | Path) -> tuple[torch.Tensor, int]:
    """Loads an audio file directly into a 1D float32 PyTorch tensor using pedalboard."""
    p = Path(wav_path)
    if not p.exists():
        raise FileNotFoundError(f"Audio file not found: {p}")
    with AudioFile(str(p)) as f:
        sr = int(f.samplerate)
        audio = f.read(f.frames)
    mono = np.mean(audio, axis=0) if audio.shape[0] > 1 else audio[0]
    return torch.from_numpy(mono.astype(np.float32)), sr


def train_voice_native(
    instrument: str | InstrumentConfig = "30in",
    voice: str = "precision_active",
    source_voicing: str | None = None,
    input_wav: str | Path | None = None,
    output_wav: str | Path | None = None,
    reference_wav: str | Path | None = None,
    models_dir: str | Path = MODELS_DIR,
    epochs: int = NATIVE_DEFAULT_MAX_EPOCHS,
    min_epochs: int = NATIVE_DEFAULT_MIN_EPOCHS,
    patience: int = NATIVE_DEFAULT_PATIENCE,
    min_delta: float = NATIVE_DEFAULT_MIN_DELTA,
    mse_weight: float = NATIVE_DEFAULT_MSE_WEIGHT,
    pre_emph_weight: float = DEFAULT_PRE_EMPH_WEIGHT,
    pre_emph_coef: float = DEFAULT_PRE_EMPH_COEF,
    mrstft_weight: float = DEFAULT_MRSTFT_WEIGHT,
    hp_mid_weight: float = NATIVE_DEFAULT_HP_MID_WEIGHT,
    hp_mid_cutoff: float = NATIVE_DEFAULT_HP_MID_CUTOFF,
    hp_high_weight: float = NATIVE_DEFAULT_HP_HIGH_WEIGHT,
    hp_high_cutoff: float = NATIVE_DEFAULT_HP_HIGH_CUTOFF,
    lr_scheduler: str = NATIVE_DEFAULT_LR_SCHEDULER,
    eta_min: float = DEFAULT_ETA_MIN,
    lr_t_max: int = DEFAULT_LR_T_MAX,
    batch_size: int | str = NATIVE_DEFAULT_BATCH_SIZE,
    precision: str = NATIVE_DEFAULT_PRECISION,
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
    device: str | None = None,
    # Dedicated Native Engine Hyperparameters
    lr: float = NATIVE_DEFAULT_LR,
    weight_decay: float = NATIVE_DEFAULT_WEIGHT_DECAY,
    slice_len: int = NATIVE_DEFAULT_SLICE_LEN,
    lite_weight: float = NATIVE_DEFAULT_LITE_WEIGHT,
) -> bool:
    """Trains a Neural Amp Modeler (NAM) Architecture 2 model using Allomorph's Native PyTorch Engine."""
    from allomorph.trainer.core import (
        find_sweep_input,
        get_hardware_device_name,
        resolve_hardware_batch_size,
        resolve_hardware_precision,
        setup_headless_environment,
    )

    setup_headless_environment()

    # 1. Resolve Target Voice
    if voice not in VOICES:
        raise KeyError(f"Target voice '{voice}' not found in catalog.")
    vcfg = VOICES[voice]
    voice_name = vcfg.name

    # 2. Resolve Instrument Configuration
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

    inst_id = inst_cfg.id
    inst_name = inst_cfg.name
    inst_ver = getattr(inst_cfg, "version", 1)
    voice_ver = getattr(vcfg, "version", 1)

    # 3. Resolve Source Voicing
    if source_voicing is not None:
        if source_voicing not in inst_cfg.voicings:
            raise KeyError(
                f"Source voicing '{source_voicing}' not found on instrument '{inst_cfg.id}'. "
                f"Available voicings: {list(inst_cfg.voicings.keys())}"
            )
        src_voicing_id = source_voicing
    elif voice in inst_cfg.voicings:
        src_voicing_id = voice
    elif inst_cfg.voicings:
        src_voicing_id = next(iter(inst_cfg.voicings.keys()))
    else:
        src_voicing_id = "default"

    # 4. Resolve Version Tag and Basename
    tri_part = resolve_tri_part_version(DSP_GENERATION, inst_ver, voice_ver)
    actual_version_tag = (
        tri_part
        if (version_tag == "auto" or version_tag is True)
        else (version_tag if version_tag not in (None, "none", False) else None)
    )

    model_basename = basename or (f"{voice}_{actual_version_tag}" if actual_version_tag else voice)

    # 5. Resolve Output Model Directory
    models_path = Path(models_dir)
    if models_path.name == "nam" or models_path.name == inst_id:
        inst_models_dir = models_path
    else:
        inst_models_dir = models_path / inst_id
    inst_models_dir.mkdir(parents=True, exist_ok=True)
    target_nam = inst_models_dir / f"{model_basename}.nam"

    # 6. Resolve Audio Input / Output Stems
    input_path = find_sweep_input(input_wav)

    if not output_wav:
        from allomorph.config.instruments import resolve_target_voicing

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

                if not silent:
                    print(
                        f"[Native Trainer] Target wet stem missing or stale, auto-simulating: {target_wet.name}"
                    )
                output_path = simulate_instrument_voicing(
                    instrument=tgt_inst, voicing=tgt_v, output_wav=target_wet
                )
            except (FileNotFoundError, ValueError, RuntimeError, KeyError, OSError) as e:
                if not silent:
                    print(f"[Native Trainer] Warning: Failed to auto-simulate {target_wet.name}: {e}")
                output_path = target_wet
        else:
            output_path = target_wet
    else:
        output_path = Path(output_wav)

    if reference_wav:
        reference_path = Path(reference_wav)
    elif input_wav and Path(input_wav).exists():
        reference_path = Path(input_wav)
    else:
        candidate_source = AUDIO_DIR / "wet" / inst_id / f"{src_voicing_id}.wav"
        if candidate_source.exists() and is_wet_stem_valid(
            candidate_source, base_dry_path=input_path
        ):
            reference_path = candidate_source
        else:
            try:
                from allomorph.circuit.forward import simulate_instrument_voicing

                if not silent:
                    print(
                        f"[Native Trainer] Dedicated source stem missing or stale, auto-generating: {candidate_source.name}"
                    )
                src_v = inst_cfg.voicings.get(src_voicing_id)
                if src_v is None:
                    src_v = next(iter(inst_cfg.voicings.values()))
                reference_path = simulate_instrument_voicing(
                    instrument=inst_cfg,
                    voicing=src_v,
                    output_wav=candidate_source,
                )
            except (FileNotFoundError, ValueError, RuntimeError, KeyError, OSError) as e:
                if not silent:
                    print(
                        f"[Native Trainer] Warning: Failed to auto-generate source voicing stem: {e}. Falling back to input sweep."
                    )
                reference_path = input_path or (AUDIO_DIR / "input.wav")

    if not include_identity and reference_path.resolve() == output_path.resolve():
        if not silent:
            print(
                f"[Native Trainer] Skipping identity pair for '{voice}' on {inst_id}: "
                f"source and target audio stems are identical ({reference_path.name}). "
                "Use --include-identity to force training."
            )
        return True

    if not reference_path.exists() or not output_path.exists():
        print(f"Error: Missing audio:\n  Source: {reference_path}\n  Target: {output_path}")
        return False

    # 7. Seed Execution
    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    # 8. Resolve Hardware Device & Parameters
    if device is not None:
        hw_device = torch.device(device)
    elif torch.cuda.is_available():
        hw_device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        hw_device = torch.device("mps")
    else:
        hw_device = torch.device("cpu")

    resolved_bs = resolve_hardware_batch_size(batch_size)
    resolved_precision = resolve_hardware_precision(precision)
    device_name = get_hardware_device_name(engine="native")

    if not silent:
        print("\n=======================================================")
        print(" Allomorph Native PyTorch A2 Trainer")
        print("=======================================================")
        print(f"  Target Voicing: {voice_name} ({voice})")
        print(f"  Source Audio:   {reference_path.name}")
        print(f"  Target Audio:   {output_path.name}")
        print(f"  Output Model:   {target_nam.name}")
        print(f"  Hardware:       {device_name} ({hw_device})")
        print(f"  Batch Size:     {resolved_bs} | Precision: {resolved_precision}")
        print(f"  Schedule:       {lr_scheduler} (max_epochs={epochs}, warmup={min_epochs})")
        print(
            f"  Multi-Band:     MSE={mse_weight:.2f}, Mid(3k)={hp_mid_weight:.2f}, High(7k)={hp_high_weight:.2f}"
        )
        print("-------------------------------------------------------")

    # 9. In-VRAM Zero-Copy Audio Loading
    x_full, sr_x = load_audio_tensor(reference_path)
    y_full, sr_y = load_audio_tensor(output_path)

    if sr_x != sr_y:
        raise ValueError(f"Sample rate mismatch: source={sr_x}Hz, target={sr_y}Hz")
    sample_rate = float(sr_x)

    n_samples = min(len(x_full), len(y_full))
    x_full = x_full[:n_samples]
    y_full = y_full[:n_samples]

    val_samples = min(432_000, max(48000, int(0.10 * n_samples)))
    train_samples = n_samples - val_samples

    # Transfer directly to accelerator VRAM
    x_train = x_full[:train_samples].to(hw_device)
    y_train = y_full[:train_samples].to(hw_device)
    x_val = x_full[train_samples:].to(hw_device)
    y_val = y_full[train_samples:].to(hw_device)

    # Prepend receptive field padding (6346 samples) for exact sample-by-sample alignment
    rf_pad = 6346
    x_train_padded = torch.cat([torch.zeros(rf_pad, device=hw_device), x_train])
    x_val_padded = torch.cat([torch.zeros(rf_pad, device=hw_device), x_val])

    # 10. Model & Loss Initialization
    model = AllomorphWaveNetA2().to(hw_device)
    model.apply_weight_masks()

    loss_fn = MultiBandSpectralLoss(
        mse_weight=mse_weight,
        hp_mid_weight=hp_mid_weight,
        hp_mid_cutoff=hp_mid_cutoff,
        hp_high_weight=hp_high_weight,
        hp_high_cutoff=hp_high_cutoff,
        fs=sample_rate,
    ).to(hw_device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # 11. Slicing Parameters
    ny = slice_len
    nx = ny + rf_pad  # ny + 6346
    steps_per_epoch = max(1, (train_samples - ny) // (ny * resolved_bs))
    if fast_dev_run:
        epochs = 1
        steps_per_epoch = 1

    # Mixed precision context
    use_autocast = resolved_precision in ("bf16-mixed", "16-mixed") and hw_device.type == "cuda"
    autocast_dtype = (
        torch.bfloat16 if resolved_precision == "bf16-mixed" else torch.float16
    )

    # Pre-allocate slice index offset grids in accelerator VRAM
    arange_nx = torch.arange(nx, device=hw_device).unsqueeze(0)
    arange_ny = torch.arange(ny, device=hw_device).unsqueeze(0)

    # State tracking
    best_pareto_score = float("inf")
    best_epoch = 0
    best_val_loss = float("inf")
    best_val_esr = float("inf")
    best_val_esr_lite = float("inf")
    best_state_dict: dict[str, torch.Tensor] | None = None

    val_loss_history: list[float] = []
    hp_high_history: list[float] = []
    min_mse_full = float("inf")
    stop_reason: str | None = None

    t0 = time.time()

    # 12. Training Loop
    for epoch in range(1, epochs + 1):
        epoch_t0 = time.time()
        model.train()

        # Learning Rate Schedule (5-epoch linear warmup + decay)
        warmup_epochs = 5
        target_lr = lr
        if epoch <= warmup_epochs:
            cur_lr = target_lr * (epoch / warmup_epochs)
        elif lr_scheduler == "cosine":
            t_cur = min(epoch - warmup_epochs, lr_t_max)
            cur_lr = eta_min + 0.5 * (target_lr - eta_min) * (
                1.0 + math.cos(math.pi * t_cur / lr_t_max)
            )
        else:
            cur_lr = target_lr * (0.994 ** (epoch - warmup_epochs))

        for param_group in optimizer.param_groups:
            param_group["lr"] = cur_lr

        train_loss_acc = torch.tensor(0.0, device=hw_device)

        for _ in range(steps_per_epoch):
            indices = torch.randint(0, train_samples - ny, (resolved_bs, 1), device=hw_device)
            x_batch = x_train_padded[indices + arange_nx].unsqueeze(1)
            y_batch = y_train[indices + arange_ny]

            optimizer.zero_grad(set_to_none=True)

            if use_autocast:
                with torch.autocast(device_type="cuda", dtype=autocast_dtype):
                    y_hat = model(x_batch)  # (B, 2, ny)
                    l_full, _ = loss_fn(y_hat[:, 1, :], y_batch, return_metrics=False)
                    l_lite, _ = loss_fn(y_hat[:, 0, :], y_batch, return_metrics=False)
                    total_step_loss = l_full + lite_weight * l_lite
            else:
                y_hat = model(x_batch)
                l_full, _ = loss_fn(y_hat[:, 1, :], y_batch, return_metrics=False)
                l_lite, _ = loss_fn(y_hat[:, 0, :], y_batch, return_metrics=False)
                total_step_loss = l_full + lite_weight * l_lite

            total_step_loss.backward()
            model.apply_weight_masks()
            optimizer.step()
            model.apply_weight_masks()

            train_loss_acc = train_loss_acc + total_step_loss.detach()

        train_loss_epoch = float((train_loss_acc / steps_per_epoch).item())

        # 13. Full-Track Validation Pass
        model.eval()
        with torch.no_grad():
            y_val_hat = model(x_val_padded.unsqueeze(0))  # (1, 2, val_samples)
            y_full_pred = y_val_hat[0, 1, :]
            y_lite_pred = y_val_hat[0, 0, :]

            val_l_full, metrics_full = loss_fn(y_full_pred, y_val, return_metrics=True)
            _, _ = loss_fn(y_lite_pred, y_val, return_metrics=False)

            # Full-band ESR calculations
            err_full = y_full_pred - y_val
            err_lite = y_lite_pred - y_val
            target_energy = float(torch.sum(y_val**2).item()) + 1e-12
            cur_esr_full = float(torch.sum(err_full**2).item()) / target_energy
            cur_esr_lite = float(torch.sum(err_lite**2).item()) / target_energy

        # 14. Anti-Drift Pareto Early Stopping Evaluation
        mse_cur = metrics_full["mse"]
        if epoch >= min_epochs:
            if min_mse_full == float("inf"):
                min_mse_full = mse_cur
            else:
                min_mse_full = min(min_mse_full, mse_cur)

        if min_mse_full < float("inf") and min_mse_full > 0.0:
            mse_drift = max(0.0, (mse_cur - min_mse_full) / min_mse_full)
        else:
            mse_drift = 0.0

        mb_val = float(val_l_full.detach().cpu().item())
        val_loss_history.append(mb_val)
        hp_high_history.append(metrics_full.get("hp_high_esr", 0.0))

        if min_mse_full < float("inf"):
            pareto_score = compute_pareto_quality_score(
                mb_full_loss=mb_val,
                mse_full=mse_cur,
                min_mse_full=min_mse_full,
                drift_tolerance=0.15,
                drift_penalty_gain=3.0,
            )
        else:
            pareto_score = mb_val

        # Track global best Pareto checkpoint post-warmup
        if epoch >= min_epochs:
            is_new_best = (best_pareto_score - pareto_score) >= min_delta
            if is_new_best or best_pareto_score == float("inf"):
                best_pareto_score = pareto_score
                best_epoch = epoch
                best_val_loss = mb_val
                best_val_esr = cur_esr_full
                best_val_esr_lite = cur_esr_lite
                best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        # Compute trajectory slopes
        k_eval = 4
        if len(val_loss_history) >= k_eval:
            x_ax = np.arange(k_eval)
            val_slope = float(np.polyfit(x_ax, val_loss_history[-k_eval:], 1)[0])
            hp_slope = float(np.polyfit(x_ax, hp_high_history[-k_eval:], 1)[0])
        else:
            val_slope = 0.0
            hp_slope = 0.0

        if epoch >= min_epochs:
            should_stop, reason = evaluate_plateau_stopping_condition(
                val_loss_history=val_loss_history,
                val_loss_slope=val_slope,
                epoch=epoch,
                warmup_floor=min_epochs,
                patience=patience,
                min_delta=min_delta,
                hp_high_slope=hp_slope,
                mse_drift_ratio=mse_drift,
                drift_exit_threshold=0.25,
            )
        else:
            should_stop, reason = False, None

        epoch_sec = time.time() - epoch_t0
        esr_full_db = 10.0 * math.log10(max(cur_esr_full, 1e-12))
        esr_lite_db = 10.0 * math.log10(max(cur_esr_lite, 1e-12))

        if not silent:
            is_best_str = " *" if epoch == best_epoch else ""
            print(
                f"Epoch {epoch:03d}/{epochs:03d} [{epoch_sec:.2f}s] "
                f"Train: {train_loss_epoch:.4f} | "
                f"Val Loss: {mb_val:.5f} (Q: {pareto_score:.5f}){is_best_str} | "
                f"ESR Ch8: {cur_esr_full:.6f} ({esr_full_db:+.1f}dB) | "
                f"ESR Ch3: {cur_esr_lite:.6f} ({esr_lite_db:+.1f}dB)"
            )

        if should_stop:
            stop_reason = reason
            if not silent:
                print(
                    f"\n[Native Trainer] Early stopping triggered at epoch {epoch:03d} "
                    f"({stop_reason})."
                )
            break

    total_time = time.time() - t0

    # 15. Restore Best Pareto Checkpoint
    if best_state_dict is not None and best_epoch > 0:
        if not silent:
            print(
                f"\n[Native Trainer] Restoring optimal Pareto checkpoint from epoch {best_epoch:03d} "
                f"(Val Loss: {best_val_loss:.5f}, Pareto Q: {best_pareto_score:.5f})"
            )
        model.load_state_dict(best_state_dict)

    # 16. Export Compliant A2 .nam Model
    export_metadata: dict[str, Any] = {
        "model_name": model_basename,
        "instrument_id": inst_id,
        "instrument_name": inst_name,
        "target_voicing": voice,
        "validation_spectral_loss": best_val_loss,
        "validation_esr": best_val_esr,
        "validation_esr_a2_full": best_val_esr,
        "validation_esr_a2_lite": best_val_esr_lite,
        "epochs_trained": best_epoch if best_epoch > 0 else epoch,
        "training_time_seconds": round(total_time, 2),
    }
    if stop_reason:
        export_metadata["stop_reason"] = stop_reason

    model.export_nam(
        out_path=target_nam,
        sample_rate=sample_rate,
        metadata=export_metadata,
    )

    if not silent:
        full_db = 10.0 * math.log10(max(best_val_esr, 1e-12))
        lite_db = 10.0 * math.log10(max(best_val_esr_lite, 1e-12))
        print("\n=======================================================")
        print(f" Native Training Complete in {total_time:.2f}s ({total_time / 60.0:.1f}m)")
        print("=======================================================")
        print(f"  Exported File:  {target_nam.name}")
        print(f"  Best Epoch:     {best_epoch:03d} / {epoch:03d}")
        print(f"  Validation ESR: {best_val_esr:.6f} (A2 Full Ch8, {full_db:+.2f} dB)")
        print(f"  Lite Mode ESR:  {best_val_esr_lite:.6f} (A2 Lite Ch3, {lite_db:+.2f} dB)")
        if stop_reason:
            print(f"  Stop Condition: {stop_reason}")
        print("  Status: Ready for Darkglass Anagram Block 1 loading.")
        print("=======================================================\n")

    return True
