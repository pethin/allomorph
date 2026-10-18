#!/usr/bin/env python3
"""Allomorph - Re-Amped Rig NAM Architecture 2 Native Trainer CLI.

Trains production-grade Neural Amp Modeler (NAM) Architecture 2 (.nam) models
from dry string excitation (audio/input.wav) to hardware re-amped wet audio stems
using Allomorph's high-throughput In-VRAM PyTorch Native Engine.

Specifically configured for non-linear amplifier heads and speaker cabinets:
  - Disables linear pickup regularizer (--no-sine-regularizer) to allow authentic tube/FET saturation
  - Rebalances high-pass loss weighting for steep speaker cabinet roll-offs
  - Performs automatic latency verification / alignment
  - Exports dual-tier A2 (Lite + Full) models for Darkglass Anagram Block 1 and standard NAM hosts.
"""

from __future__ import annotations

import math
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated

import numpy as np
import torch
import torch.nn.functional as F
from cyclopts import App, Parameter

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from allomorph.dsp.constants import FS
from allomorph.dsp.io import read_wav
from allomorph.trainer.core import normalize_training_stem
from allomorph.trainer.native.loss import design_sharp_fir_highpass
from allomorph.trainer.native.models import AllomorphWaveNetA2

app = App(
    name="train-reamp",
    help="Allomorph - Re-Amped Rig NAM Native Trainer.",
    version="1.0.0",
)


def verify_or_align_stems(
    dry: np.ndarray,
    wet: np.ndarray,
    sample_rate: int = FS,
    max_search_ms: float = 120.0,
) -> tuple[int, np.ndarray]:
    """Cross-correlates dry and wet stems and returns (delay_samples, aligned_wet)."""
    max_delay = int(sample_rate * (max_search_ms / 1000.0))
    search_len = min(len(dry), int(sample_rate * 3.0))

    x = dry[:search_len].astype(np.float32)
    y = wet[: search_len + max_delay].astype(np.float32)

    if len(y) <= len(x):
        return 0, wet[: len(dry)]

    corr = np.correlate(y, x, mode="valid")
    delay = int(np.argmax(np.abs(corr)))

    aligned = wet[delay : delay + len(dry)]
    if len(aligned) < len(dry):
        padded = np.zeros(len(dry), dtype=wet.dtype)
        padded[: len(aligned)] = aligned
        aligned = padded

    return delay, aligned


def train_single_reamp_pair(
    dry_path: Path,
    wet_path: Path,
    output_nam: Path,
    epochs: int = 120,
    min_epochs: int = 80,
    patience: int = 15,
    batch_size: int = 16,
    lr: float = 4.0e-3,
    weight_decay: float = 1.0e-4,
    fir_hp_weight: float = 0.75,
    fir_hp_fc: float = 1000.0,
    lite_weight: float = 0.50,
    auto_align: bool = True,
    device_name: str = "auto",
    model_name: str | None = None,
) -> bool:
    """Trains a WaveNet A2 model on a single dry -> wet audio pair."""
    print("------------------------------------------------------------------")
    print(f"  Dry Input:      {dry_path.name}")
    print(f"  Re-Amped Wet:   {wet_path.name}")
    print(f"  Destination:    {output_nam}")
    print("------------------------------------------------------------------")

    # 1. Resolve Device
    if device_name == "auto":
        if torch.cuda.is_available():
            hw_device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            hw_device = torch.device("mps")
        else:
            hw_device = torch.device("cpu")
    else:
        hw_device = torch.device(device_name)

    dev_label = f"{hw_device.type.upper()}"
    if hw_device.type == "cuda":
        dev_label += f" ({torch.cuda.get_device_name(0)})"
    elif hw_device.type == "mps":
        dev_label += " (Apple Silicon GPU)"
    print(f"  Hardware:       {dev_label}")

    # 2. Ingest Audio Stems
    dry_raw, sr_dry = read_wav(dry_path, dtype=np.float32)
    wet_raw, sr_wet = read_wav(wet_path, dtype=np.float32)

    if dry_raw.ndim > 1:
        dry_raw = np.mean(dry_raw, axis=0)
    if wet_raw.ndim > 1:
        wet_raw = np.mean(wet_raw, axis=0)

    if sr_dry != sr_wet:
        raise ValueError(f"Sample rate mismatch: dry={sr_dry} Hz, wet={sr_wet} Hz")
    if sr_dry != FS:
        raise ValueError(f"Sample rate must be {FS} Hz (got {sr_dry} Hz)")

    # 3. Optional Latency Alignment
    if auto_align:
        delay_samples, wet_aligned = verify_or_align_stems(dry_raw, wet_raw, sample_rate=sr_dry)
        delay_ms = (delay_samples / sr_dry) * 1000.0
        if delay_samples > 0:
            print(
                f"  Alignment:      Detected {delay_samples} samples ({delay_ms:.2f} ms) delay — shifted"
            )
        else:
            print("  Alignment:      Zero latency confirmed (0 samples)")
    else:
        wet_aligned = wet_raw[: len(dry_raw)]

    n_samples = min(len(dry_raw), len(wet_aligned))
    print(
        f"  Audio Duration: {n_samples:,} samples ({n_samples / sr_dry:.1f}s / {n_samples / sr_dry / 60.0:.2f} min)"
    )

    # 4. Standardized RMS Normalization (-20.50 dBFS RMS)
    x_norm = normalize_training_stem(dry_raw[:n_samples], sample_rate=sr_dry)
    y_norm = normalize_training_stem(wet_aligned[:n_samples], sample_rate=sr_wet)

    # Split Train (90%) / Validation (10%, up to 432,000 samples = 9s)
    val_samples = min(432_000, max(48000, int(0.10 * n_samples)))
    train_samples = n_samples - val_samples

    x_train_np = x_norm[:train_samples]
    y_train_np = y_norm[:train_samples]
    x_val_np = x_norm[train_samples:n_samples]
    y_val_np = y_norm[train_samples:n_samples]

    rf_pad = 6346  # WaveNet A2 receptive field padding (samples)
    ny = 8192  # Sliced batch output length
    nx = ny + rf_pad  # 14,538 samples

    # Pad training input with silence on the left so step 0 aligns with sample 0
    x_train_padded = np.pad(x_train_np, (rf_pad, 0), mode="constant")
    x_val_padded = np.pad(x_val_np, (rf_pad, 0), mode="constant")

    # Move tensors into accelerator VRAM
    x_train_t = torch.from_numpy(x_train_padded).to(device=hw_device, dtype=torch.float32)
    y_train_t = torch.from_numpy(y_train_np).to(device=hw_device, dtype=torch.float32)
    x_val_t = torch.from_numpy(x_val_padded).to(device=hw_device, dtype=torch.float32)
    y_val_t = torch.from_numpy(y_val_np).to(device=hw_device, dtype=torch.float32)

    # 5. Initialize Model & Closed-Form Wiener Warm-Start
    model = AllomorphWaveNetA2().to(hw_device)
    model.mask_weights()

    # Warm-start output head on first 32k samples
    cal_len = min(32768, train_samples)
    model.warm_start_head(
        x_train_t[: cal_len + rf_pad].unsqueeze(0).unsqueeze(0),
        y_train_t[:cal_len].unsqueeze(0),
    )

    # 6. FIR High-Pass Loss Filter for Amp+Cab
    hp_kernel = design_sharp_fir_highpass(fc=fir_hp_fc, fs=float(sr_dry), num_taps=255).to(
        hw_device
    )

    # 7. Optimizer & Annealing Schedule
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    warmup_epochs = 5
    homotopy_end_epoch = int(0.75 * min_epochs)
    homotopy_tau_init = 0.02

    # State tracking
    best_val_loss = float("inf")
    best_epoch = 0
    best_state_dict = None
    patience_counter = 0

    num_batches_per_epoch = max(1, train_samples // (batch_size * ny))
    print(f"  Batch Config:   Batch Size={batch_size}, {num_batches_per_epoch} steps/epoch")
    print(
        f"  Loss Config:    Waveform MSE=1.00 + FIR HP={fir_hp_weight:.2f} (fc={fir_hp_fc:.0f} Hz)"
    )
    print(f"  Schedule:       Cosine Annealing ({epochs} max epochs, warmup={min_epochs})")
    print("------------------------------------------------------------------")

    start_train_time = time.time()

    # 8. Training Loop
    for epoch in range(1, epochs + 1):
        epoch_t0 = time.time()
        model.train()

        # SiLU-gated activation homotopy schedule (tau: 0.02 -> 0.0)
        if epoch <= homotopy_end_epoch:
            prog = (epoch - 1) / max(1, homotopy_end_epoch)
            model.activation_tau = 0.5 * homotopy_tau_init * (1.0 + math.cos(math.pi * prog))
        else:
            model.activation_tau = 0.0

        # Learning Rate Schedule (5-epoch linear warmup + cosine decay)
        if epoch <= warmup_epochs:
            cur_lr = lr * (epoch / warmup_epochs)
        else:
            t_cur = epoch - warmup_epochs
            t_max = max(1, epochs - warmup_epochs)
            cur_lr = 1e-6 + 0.5 * (lr - 1e-6) * (1.0 + math.cos(math.pi * t_cur / t_max))

        for param_group in optimizer.param_groups:
            param_group["lr"] = cur_lr

        train_loss_acc = 0.0
        max_start = max(1, train_samples - ny)

        for _ in range(num_batches_per_epoch):
            # Sample random batch offsets
            starts = torch.randint(0, max_start, (batch_size,), device=hw_device)

            x_batch = torch.stack([x_train_t[s : s + nx] for s in starts]).unsqueeze(1)
            y_batch = torch.stack([y_train_t[s : s + ny] for s in starts])

            optimizer.zero_grad(set_to_none=True)
            y_hat = model(x_batch)

            pred_lite = y_hat[:, 0, :]
            pred_full = y_hat[:, 1, :]

            # 1. Waveform MSE
            loss_mse_full = F.mse_loss(pred_full, y_batch)
            loss_mse_lite = F.mse_loss(pred_lite, y_batch)

            # 2. FIR High-Pass Loss
            pred_hp_full = F.conv1d(pred_full.unsqueeze(1), hp_kernel)
            pred_hp_lite = F.conv1d(pred_lite.unsqueeze(1), hp_kernel)
            tgt_hp = F.conv1d(y_batch.unsqueeze(1), hp_kernel)

            loss_hp_full = F.mse_loss(pred_hp_full, tgt_hp)
            loss_hp_lite = F.mse_loss(pred_hp_lite, tgt_hp)

            loss = (loss_mse_full + lite_weight * loss_mse_lite) + fir_hp_weight * (
                loss_hp_full + lite_weight * loss_hp_lite
            )

            loss.backward()
            model.mask_gradients()
            optimizer.step()
            model.apply_weight_masks()

            train_loss_acc += loss.item()

        avg_train_loss = train_loss_acc / num_batches_per_epoch
        epoch_dur = time.time() - epoch_t0

        # Validation Pass
        model.eval()
        with torch.no_grad():
            x_val_in = x_val_t[: val_samples + rf_pad].unsqueeze(0).unsqueeze(0)
            y_val_tgt = y_val_t[:val_samples].unsqueeze(0)

            # Sliced inference over validation buffer
            val_preds: list[torch.Tensor] = []
            for v_start in range(0, val_samples, ny):
                v_end = min(v_start + ny, val_samples)
                v_slice = x_val_in[:, :, v_start : v_start + nx]
                if v_slice.shape[2] < nx:
                    v_slice = F.pad(v_slice, (0, nx - v_slice.shape[2]))
                y_val_slice = model(v_slice)[:, :, : v_end - v_start]
                val_preds.append(y_val_slice)

            y_val_hat = torch.cat(val_preds, dim=2)
            val_pred_full = y_val_hat[:, 1, :]

            val_mse = F.mse_loss(val_pred_full, y_val_tgt).item()
            val_esr = (torch.sum((val_pred_full - y_val_tgt) ** 2) / torch.sum(y_val_tgt**2)).item()

        # Terminal Progress
        is_best = val_mse < best_val_loss
        star = " ★" if is_best else ""
        sys.stdout.write(
            f"\r  Epoch {epoch:3d}/{epochs} [{epoch_dur:.2f}s] | "
            f"Train: {avg_train_loss:.5f} | Val MSE: {val_mse:.5f} | ESR: {val_esr:.4f}{star}\n"
        )
        sys.stdout.flush()

        if is_best:
            best_val_loss = val_mse
            best_epoch = epoch
            best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        elif epoch >= min_epochs:
            patience_counter += 1
            if patience_counter >= patience:
                print(
                    f"\n  [Early Stopping] Plateau reached at epoch {epoch} (Best: Epoch {best_epoch})."
                )
                break

    total_train_dur = time.time() - start_train_time
    print(f"\nTraining completed in {total_train_dur:.1f}s ({total_train_dur / 60.0:.2f} min).")

    # 9. Load Best State & Export Container
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    model.eval()
    output_nam.parent.mkdir(parents=True, exist_ok=True)

    clean_name = model_name or wet_path.stem.replace("_", " ").title()
    meta_dict = {
        "name": clean_name,
        "modeled_by": "Allomorph Native Engine",
        "gear_make": "Darkglass Anagram Reamp",
        "gear_model": wet_path.stem,
        "gear_type": "amp",
        "tone_type": "amp_cab",
        "date": datetime.now(tz=UTC).isoformat(),
        "epochs": best_epoch,
        "val_mse": best_val_loss,
    }

    model.export_nam(
        out_path=output_nam,
        sample_rate=float(sr_dry),
        metadata=meta_dict,
    )
    print(f"  ✅ Saved NAM Model: {output_nam} ({output_nam.stat().st_size / 1024:.1f} KB)")
    print("==================================================================\n")
    return True


@app.default
def train_reamp(
    reamp_target: Annotated[
        Path,
        Parameter(
            name=["--reamp", "-r"],
            help="Path to a re-amped wet WAV file or directory containing re-amped WAVs.",
        ),
    ] = REPO_ROOT / "reamp",
    *,
    input_file: Annotated[
        Path,
        Parameter(
            name=["--input", "-i"],
            help="Path to dry excitation input WAV file.",
        ),
    ] = REPO_ROOT / "audio" / "input.wav",
    output_dir: Annotated[
        Path,
        Parameter(
            name=["--output-dir", "-o"],
            help="Directory to save exported .nam models.",
        ),
    ] = REPO_ROOT / "models" / "reamp",
    epochs: Annotated[
        int,
        Parameter(name=["--epochs"], help="Maximum training epochs budget."),
    ] = 120,
    min_epochs: Annotated[
        int,
        Parameter(name=["--min-epochs"], help="Minimum warmup epochs before early stopping."),
    ] = 80,
    patience: Annotated[
        int,
        Parameter(name=["--patience"], help="Early stopping plateau patience epochs."),
    ] = 15,
    batch_size: Annotated[
        int,
        Parameter(name=["--batch-size"], help="Training batch size."),
    ] = 16,
    lr: Annotated[
        float,
        Parameter(name=["--lr"], help="Initial learning rate."),
    ] = 4.0e-3,
    fir_hp_weight: Annotated[
        float,
        Parameter(name=["--fir-hp-weight"], help="High-pass loss weight (tuned for cab roll-off)."),
    ] = 0.75,
    fir_hp_fc: Annotated[
        float,
        Parameter(name=["--fir-hp-fc"], help="High-pass cutoff frequency in Hz."),
    ] = 1000.0,
    no_align: Annotated[
        bool,
        Parameter(name=["--no-align"], help="Skip latency verification / alignment check."),
    ] = False,
    device: Annotated[
        str,
        Parameter(name=["--device"], help="Target compute device ('auto', 'mps', 'cuda', 'cpu')."),
    ] = "auto",
) -> None:
    """Trains production-grade NAM Architecture 2 models from re-amped audio stems using Cyclopts."""
    if not input_file.exists():
        raise FileNotFoundError(f"Dry excitation file '{input_file}' not found.")

    targets: list[Path] = []
    if reamp_target.is_file() and reamp_target.suffix.lower() == ".wav":
        targets.append(reamp_target)
    elif reamp_target.is_dir():
        targets.extend(sorted(reamp_target.glob("*.wav")))
        # Exclude raw stems if save_raw was used
        targets = [t for t in targets if not t.stem.endswith("_raw")]
    else:
        raise FileNotFoundError(f"Target re-amp file or directory '{reamp_target}' not found.")

    if not targets:
        print(f"No .wav files found to train in '{reamp_target}'.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    print("==================================================================")
    print(" Allomorph Re-Amped Rig NAM Native Trainer")
    print("==================================================================")
    print(f"  Excitation Input: {input_file}")
    print(f"  Targets Found:    {len(targets)} WAV stem(s)")
    for idx, t in enumerate(targets, 1):
        print(f"    [{idx}] {t.name}")
    print(f"  Model Output Dir: {output_dir}")
    print("==================================================================\n")

    for idx, target_wav in enumerate(targets, 1):
        model_out = output_dir / f"{target_wav.stem}.nam"
        print(f"\n>>> [{idx}/{len(targets)}] Training Model: {model_out.name} <<<")
        train_single_reamp_pair(
            dry_path=input_file,
            wet_path=target_wav,
            output_nam=model_out,
            epochs=epochs,
            min_epochs=min_epochs,
            patience=patience,
            batch_size=batch_size,
            lr=lr,
            fir_hp_weight=fir_hp_weight,
            fir_hp_fc=fir_hp_fc,
            auto_align=not no_align,
            device_name=device,
        )


if __name__ == "__main__":
    app()
