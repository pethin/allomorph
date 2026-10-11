#!/usr/bin/env python3
"""Benchmark Native PyTorch A2 Engine vs Stock NAM / Lightning Engine.

Compares:
- Training throughput (sec/epoch, total time, peak VRAM)
- Model convergence (MultiBand validation loss, ESR Full, ESR Lite)
- 31-band ISO 1/3-octave frequency response accuracy (Overall MAE, Sub-bass RMSE, Treble RMSE)
- Dynamic crest factor match and transient attack fidelity
across the 4 target voices from 34in_standard_p:
1. Precision Dub (slab_dub)
2. Rickenbacker Clank (bridge_clank)
3. Jazz Pair Active (pair_active)
4. StingRay Parallel (stingray_parallel)
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from pedalboard.io import AudioFile

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from allomorph.dsp.constants import FS
from allomorph.dsp.sweeps import deconvolve_log_sweep
from allomorph.trainer import setup_headless_environment, train_voice
from allomorph.trainer.native.loss import MultiBandSpectralLoss

setup_headless_environment()

VOICES_TO_BENCHMARK = [
    ("precision_dub", "Precision Dub"),
    ("rickenbacker_clank", "Rickenbacker Clank"),
    ("jazz_pair_active", "Jazz Pair Active"),
    ("stingray_parallel", "StingRay Parallel"),
]

ISO_31_FREQS = np.array([
    20.0, 25.0, 31.5, 40.0, 50.0, 63.0, 80.0, 100.0, 125.0, 160.0,
    200.0, 250.0, 315.0, 400.0, 500.0, 630.0, 800.0, 1000.0, 1250.0, 1600.0,
    2000.0, 2500.0, 3150.0, 4000.0, 5000.0, 6300.0, 8000.0, 10000.0, 12500.0, 16000.0,
    20000.0,
])


def load_audio(path: Path) -> np.ndarray:
    with AudioFile(str(path)) as f:
        audio = f.read(f.frames)
    mono = np.mean(audio, axis=0) if audio.shape[0] > 1 else audio[0]
    return mono.astype(np.float32)


def run_model_inference(nam_path: Path, x_dry: np.ndarray) -> np.ndarray:
    """Runs A2 Full submodel inference from a .nam file on a 1D audio array."""
    with open(nam_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    from nam.models import init_from_nam

    sub1_data = data["config"]["submodels"][1]["model"]
    sub1_net = init_from_nam(sub1_data)
    sub1_net.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sub1_net.to(device)

    # WaveNet requires leading receptive field pad
    rf = 6347
    x_padded = np.pad(x_dry, (rf - 1, 0), mode="constant")
    x_t = torch.from_numpy(x_padded).float().unsqueeze(0).to(device)

    with torch.no_grad():
        y_hat_t = sub1_net(x_t)
        # Squeeze batch
        y_hat = y_hat_t.squeeze().cpu().numpy()

    return y_hat[: len(x_dry)].astype(np.float32)


def evaluate_audio_metrics(
    y_pred: np.ndarray, y_tgt: np.ndarray, x_dry: np.ndarray, fs: int = FS
) -> dict[str, float]:
    """Computes full-band ESR, spectral loss, 31-band ISO error, and crest factor."""
    n = min(len(y_pred), len(y_tgt))
    yp = y_pred[:n].astype(np.float64)
    yt = y_tgt[:n].astype(np.float64)

    # 1. Full-band ESR
    err = yp - yt
    tgt_energy = float(np.sum(yt**2)) + 1e-12
    esr = float(np.sum(err**2)) / tgt_energy
    esr_db = 10.0 * math.log10(max(esr, 1e-12))

    # 2. MultiBand Spectral Loss
    loss_fn = MultiBandSpectralLoss(fs=float(fs))
    loss_val, _ = loss_fn(torch.from_numpy(yp).float(), torch.from_numpy(yt).float())
    mb_loss = float(loss_val.item())

    # 3. Frequency Response via Farina log sweep deconvolution on first 16384 samples
    sweep_len = min(16384, n)
    sweep_tgt = yt[:sweep_len]
    sweep_pred = yp[:sweep_len]
    sweep_dry = x_dry[:sweep_len]

    f_bins, H_tgt_c, _ = deconvolve_log_sweep(sweep_tgt, sweep_dry, sr=fs, gate_taps=4096)
    _, H_pred_c, _ = deconvolve_log_sweep(sweep_pred, sweep_dry, sr=fs, gate_taps=4096)

    mag_tgt_db = 20.0 * np.log10(np.maximum(np.abs(H_tgt_c), 1e-6))
    mag_pred_db = 20.0 * np.log10(np.maximum(np.abs(H_pred_c), 1e-6))

    # Interpolate to 31 ISO 1/3-octave center frequencies
    mag_tgt_iso = np.interp(ISO_31_FREQS, f_bins, mag_tgt_db)
    mag_pred_iso = np.interp(ISO_31_FREQS, f_bins, mag_pred_db)

    diff_db = mag_pred_iso - mag_tgt_iso
    overall_mae = float(np.mean(np.abs(diff_db)))

    sub_mask = (ISO_31_FREQS >= 20.0) & (ISO_31_FREQS <= 120.0)
    sub_rmse = float(np.sqrt(np.mean(diff_db[sub_mask] ** 2))) if np.any(sub_mask) else 0.0

    treble_mask = (ISO_31_FREQS >= 3000.0) & (ISO_31_FREQS <= 20000.0)
    treble_rmse = float(np.sqrt(np.mean(diff_db[treble_mask] ** 2))) if np.any(treble_mask) else 0.0

    # 4. Crest Factor on Plucks
    rms_tgt = float(np.sqrt(np.mean(yt**2))) + 1e-12
    peak_tgt = float(np.max(np.abs(yt))) + 1e-12
    crest_tgt_db = 20.0 * math.log10(peak_tgt / rms_tgt)

    rms_pred = float(np.sqrt(np.mean(yp**2))) + 1e-12
    peak_pred = float(np.max(np.abs(yp))) + 1e-12
    crest_pred_db = 20.0 * math.log10(peak_pred / rms_pred)

    delta_crest_db = crest_pred_db - crest_tgt_db

    # 5. Attack Transient ESR (first 20ms = 960 samples)
    att_len = min(960, n)
    att_err = float(np.sum((yp[:att_len] - yt[:att_len]) ** 2))
    att_tgt = float(np.sum(yt[:att_len] ** 2)) + 1e-12
    att_esr_db = 10.0 * math.log10(max(att_err / att_tgt, 1e-12))

    return {
        "esr": esr,
        "esr_db": esr_db,
        "mb_loss": mb_loss,
        "overall_mae": overall_mae,
        "sub_rmse": sub_rmse,
        "treble_rmse": treble_rmse,
        "delta_crest_db": delta_crest_db,
        "att_esr_db": att_esr_db,
    }


def run_benchmark() -> None:
    bundle_dir = REPO_ROOT / "tone3000" / "packs" / "34in_standard_p" / "bundles" / "split"
    dry_wav = bundle_dir / "dry v6.1.1.wav"
    if not dry_wav.exists():
        raise FileNotFoundError(f"Dry bundle file not found: {dry_wav}")

    print("================================================================================")
    print(" ALLOMORPH A2 TRAINER BENCHMARK: NATIVE vs STOCK")
    print(" Hardware: AMD Radeon RX 9070 XT (bf16-mixed)")
    print(" Source Bass: 34in_standard_p [Vintage Open, Split Coil]")
    print("================================================================================\n")

    results: dict[str, dict[str, Any]] = {}

    epochs = 40
    min_epochs = 20
    patience = 5
    batch_size = 16

    for voice_slug, voice_name in VOICES_TO_BENCHMARK:
        target_wav = bundle_dir / f"{voice_name} v6.1.1.wav"
        if not target_wav.exists():
            print(f"Skipping {voice_name}: stem missing at {target_wav}")
            continue

        print(f"\n>>>>>>>>>>>> BENCHMARKING TARGET: {voice_name} ({voice_slug}) <<<<<<<<<<<<")
        results[voice_slug] = {}

        for engine in ["native", "stock"]:
            print(f"\n--- Running Engine: {engine.upper()} on {voice_name} ---")
            out_dir = REPO_ROOT / "scratch" / "benchmark_runs" / engine
            out_dir.mkdir(parents=True, exist_ok=True)

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()

            t0 = time.time()
            _ = train_voice(
                instrument="34in_standard_p",
                voice=voice_slug,
                input_wav=dry_wav,
                reference_wav=dry_wav,
                output_wav=target_wav,
                models_dir=out_dir,
                epochs=epochs,
                min_epochs=min_epochs,
                patience=patience,
                batch_size=batch_size,
                precision="bf16-mixed",
                silent=False,
                engine=engine,
                seed=42,
            )
            elapsed = time.time() - t0

            peak_vram_mb = 0.0
            if torch.cuda.is_available():
                peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)

            # Locate exported model
            model_files = list(out_dir.glob(f"**/*{voice_slug}*.nam"))
            if not model_files:
                model_files = list(out_dir.glob(f"**/*{voice_name}*.nam"))

            if not model_files:
                print(f"Error: Model not found for {engine} on {voice_name}")
                continue

            exported_nam = model_files[0]
            with open(exported_nam, "r", encoding="utf-8") as f:
                model_data = json.load(f)

            meta = model_data.get("metadata", {})
            trained_epochs = meta.get("epochs_trained", epochs)
            stop_reason = meta.get("stop_reason", "schedule_complete")

            # Load audio for validation evaluation
            x_dry = load_audio(dry_wav)
            y_tgt = load_audio(target_wav)

            # Hold out validation segment (~9s = 432,000 samples)
            val_len = min(432_000, len(x_dry))
            x_val = x_dry[-val_len:]
            y_val = y_tgt[-val_len:]

            print(f"Evaluating {engine.upper()} model inference on validation segment...")
            y_pred_val = run_model_inference(exported_nam, x_val)

            # Evaluate audio metrics
            metrics = evaluate_audio_metrics(y_pred_val, y_val, x_val, fs=FS)

            sec_per_epoch = elapsed / max(1, trained_epochs)
            steps_per_sec = (trained_epochs * 84) / elapsed

            run_result = {
                "elapsed_sec": round(elapsed, 2),
                "trained_epochs": trained_epochs,
                "sec_per_epoch": round(sec_per_epoch, 2),
                "steps_per_sec": round(steps_per_sec, 2),
                "peak_vram_mb": round(peak_vram_mb, 1),
                "stop_reason": stop_reason,
                "metrics": metrics,
                "model_path": str(exported_nam),
            }
            results[voice_slug][engine] = run_result

            print(f"[{engine.upper()}] Done in {elapsed:.1f}s ({trained_epochs} ep, {sec_per_epoch:.2f}s/ep)")
            print(f"  VRAM: {peak_vram_mb:.1f} MB | Stop: {stop_reason}")
            print(f"  Val ESR: {metrics['esr_db']:+.2f} dB (lin: {metrics['esr']:.6f})")
            print(f"  31-Band Overall MAE: {metrics['overall_mae']:.2f} dB | Treble RMSE: {metrics['treble_rmse']:.2f} dB")
            print(f"  Crest Match: {metrics['delta_crest_db']:+.2f} dB | Attack ESR: {metrics['att_esr_db']:+.1f} dB")

    # Save raw benchmark JSON
    json_path = REPO_ROOT / "scratch" / "benchmark_results.json"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\n================================================================================")
    print(" BENCHMARK COMPLETED SUCCESSFULLY!")
    print(f" Raw results saved to: {json_path}")
    print("================================================================================")


if __name__ == "__main__":
    run_benchmark()
