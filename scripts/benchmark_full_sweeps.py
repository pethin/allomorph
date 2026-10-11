#!/usr/bin/env python3
"""Comprehensive Synthetic Log-Sine Sweep Benchmark: Native A2 Engine vs Original Stock Engine.

Evaluates:
- 14 Key Electroacoustic Frequencies (60, 100, 250, 500, 1000, 2000, 3000, 4000, 5200, 6000, 8000, 10000, 12000, 15000 Hz)
- 31-Band ISO 1/3-Octave Telemetry (20 to 20,000 Hz)
- Continuous FFT Metrics across 4,097 bins (Overall MAE, Sub-bass RMSE, Low-mid RMSE, Presence RMSE, Treble RMSE, Air RMSE)
- Small-Signal Impulse Response Taps (delta[n] = [0.05, 0, ...])
- Dynamic Validation Plucks Audio Telemetry (Crest Factor tracking, 20ms Attack ESR, Full-Band ESR)
- Computational Throughput (Sec/Epoch, Peak VRAM, Total Training Time, Checkpoint Epoch)

Across the 4 representative target voices from 34in_standard_p:
1. Precision Dub (slab_dub)
2. Rickenbacker Clank (bridge_clank)
3. Jazz Pair Active (pair_active)
4. StingRay Parallel (parallel_classic)
"""

from __future__ import annotations

import argparse
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

from allomorph.circuit.forward import simulate_voicing_dsp
from allomorph.config.instruments import load_instrument, resolve_target_voicing
from allomorph.dsp.conditioning import apply_dc_block
from allomorph.dsp.constants import FS
from allomorph.dsp.sweeps import deconvolve_log_sweep
from allomorph.trainer import setup_headless_environment, train_voice
from allomorph.trainer.constants import (
    STOCK_DEFAULT_HP_HIGH_ESR_WEIGHT,
    STOCK_DEFAULT_HP_MID_ESR_WEIGHT,
    STOCK_DEFAULT_MAX_EPOCHS,
    STOCK_DEFAULT_MIN_DELTA,
    STOCK_DEFAULT_MIN_EPOCHS,
    STOCK_DEFAULT_MSE_WEIGHT,
    STOCK_DEFAULT_PATIENCE,
)

setup_headless_environment()

VOICES_TO_BENCHMARK = [
    ("precision_dub", "Precision Dub"),
    ("rickenbacker_clank", "Rickenbacker Clank"),
    ("jazz_pair_active", "Jazz Pair Active"),
    ("stingray_parallel", "StingRay Parallel"),
]

KEY_FREQUENCIES = [
    60.0, 100.0, 250.0, 500.0, 1000.0, 2000.0, 3000.0, 4000.0, 5200.0, 6000.0,
    8000.0, 10000.0, 12000.0, 15000.0,
]

ISO_31_BANDS = [
    (1, 20.0), (2, 25.0), (3, 31.5), (4, 40.0), (5, 50.0), (6, 63.0),
    (7, 80.0), (8, 100.0), (9, 125.0), (10, 160.0), (11, 200.0), (12, 250.0),
    (13, 315.0), (14, 400.0), (15, 500.0), (16, 630.0), (17, 800.0), (18, 1000.0),
    (19, 1250.0), (20, 1600.0), (21, 2000.0), (22, 2500.0), (23, 3150.0), (24, 4000.0),
    (25, 5000.0), (26, 6300.0), (27, 8000.0), (28, 10000.0), (29, 12500.0), (30, 16000.0),
    (31, 20000.0),
]


def synth_log_sweep(duration_sec: float = 5.0, f_start: float = 20.0, f_end: float = 20000.0, target_dbfs: float = -20.5) -> np.ndarray:
    n_samples = int(duration_sec * FS)
    t = np.linspace(0.0, duration_sec, n_samples, endpoint=False)
    w_start = 2.0 * np.pi * f_start
    w_end = 2.0 * np.pi * f_end
    k = duration_sec / np.log(w_end / w_start)
    phi = w_start * k * (np.exp(t / k) - 1.0)
    sweep = np.sin(phi)
    rms = np.sqrt(np.mean(sweep**2))
    target_rms = 10.0 ** (target_dbfs / 20.0)
    return (sweep * (target_rms / (rms + 1e-12))).astype(np.float32)


def load_audio(path: Path) -> np.ndarray:
    with AudioFile(str(path)) as f:
        audio = f.read(f.frames)
    mono = np.mean(audio, axis=0) if audio.shape[0] > 1 else audio[0]
    return mono.astype(np.float32)


def run_nam_inference(nam_path: Path, input_audio: np.ndarray, submodel_idx: int = 1) -> np.ndarray:
    with open(nam_path, "r", encoding="utf-8") as f:
        container = json.load(f)
    submodels = container.get("config", {}).get("submodels", [])
    if not submodels:
        model_dict = container
    else:
        actual_idx = min(submodel_idx, len(submodels) - 1)
        model_dict = submodels[actual_idx]["model"]

    from nam.models import init_from_nam

    loaded_model = init_from_nam(model_dict)
    loaded_model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaded_model.to(device)

    rf = 6347
    x_padded = np.pad(input_audio, (rf - 1, 0), mode="constant")
    x_t = torch.from_numpy(x_padded).float().unsqueeze(0).to(device)

    with torch.no_grad():
        y_pred_t = loaded_model(x_t)
        y_pred = y_pred_t.squeeze().cpu().numpy()

    return np.asarray(y_pred[: len(input_audio)], dtype=np.float64)


def extract_small_signal_ir(nam_path: Path, submodel_idx: int = 1, num_taps: int = 8) -> list[float]:
    with open(nam_path, "r", encoding="utf-8") as f:
        container = json.load(f)
    submodels = container.get("config", {}).get("submodels", [])
    if not submodels:
        model_dict = container
    else:
        actual_idx = min(submodel_idx, len(submodels) - 1)
        model_dict = submodels[actual_idx]["model"]

    from nam.models import init_from_nam

    loaded_model = init_from_nam(model_dict)
    loaded_model.eval()

    delta = np.zeros(256, dtype=np.float32)
    delta[0] = 0.05
    with torch.no_grad():
        x_t = torch.from_numpy(delta).float().unsqueeze(0)
        y_t = loaded_model(x_t).squeeze().numpy()
    ir = (y_t / 0.05)[:num_taps]
    return [round(float(v), 3) for v in ir]


def evaluate_sweep_metrics(
    y_sweep_pred: np.ndarray,
    src_sweep: np.ndarray,
    f_bins: np.ndarray,
    mag_dsp: np.ndarray,
    gate_taps: int = 8192,
) -> dict[str, Any]:
    sw_blocked = apply_dc_block(y_sweep_pred, sample_rate=FS)
    _, H_m, _ = deconvolve_log_sweep(sw_blocked, src_sweep, gate_taps=gate_taps)
    mag_model = 20.0 * np.log10(np.maximum(np.abs(H_m), 1e-6))

    # 1. 14 Key Electroacoustic Frequencies
    key_telemetry = {}
    for f in KEY_FREQUENCIES:
        idx = int(np.argmin(np.abs(f_bins - f)))
        d_val = float(mag_dsp[idx])
        m_val = float(mag_model[idx])
        key_telemetry[f"{int(f)}Hz"] = {
            "target_db": round(d_val, 2),
            "model_db": round(m_val, 2),
            "error_db": round(m_val - d_val, 2),
        }

    # 2. 31-Band ISO 1/3-Octave Telemetry
    iso_telemetry = {}
    iso_errors = []
    for band_num, f_target in ISO_31_BANDS:
        idx = int(np.argmin(np.abs(f_bins - f_target)))
        d_val = float(mag_dsp[idx])
        m_val = float(mag_model[idx])
        err = m_val - d_val
        iso_errors.append(err)
        iso_telemetry[f"Band_{band_num:02d}_{int(f_target)}Hz"] = {
            "band": band_num,
            "freq_hz": f_target,
            "target_db": round(d_val, 2),
            "model_db": round(m_val, 2),
            "error_db": round(err, 2),
        }

    # 3. Continuous Band Telemetry (20 Hz - 20,000 Hz across all 4,097 bins)
    valid_mask = (f_bins >= 20.0) & (f_bins <= 20000.0)
    err_cont = mag_model[valid_mask] - mag_dsp[valid_mask]
    overall_mae = float(np.mean(np.abs(err_cont)))

    sub_mask = (f_bins >= 20.0) & (f_bins <= 120.0)
    sub_rmse = float(np.sqrt(np.mean((mag_model[sub_mask] - mag_dsp[sub_mask]) ** 2))) if np.any(sub_mask) else 0.0

    low_mid_mask = (f_bins >= 100.0) & (f_bins <= 1000.0)
    low_mid_rmse = float(np.sqrt(np.mean((mag_model[low_mid_mask] - mag_dsp[low_mid_mask]) ** 2))) if np.any(low_mid_mask) else 0.0

    pres_mask = (f_bins >= 1000.0) & (f_bins <= 4000.0)
    pres_rmse = float(np.sqrt(np.mean((mag_model[pres_mask] - mag_dsp[pres_mask]) ** 2))) if np.any(pres_mask) else 0.0

    treble_mask = (f_bins >= 4000.0) & (f_bins <= 10000.0)
    treble_rmse = float(np.sqrt(np.mean((mag_model[treble_mask] - mag_dsp[treble_mask]) ** 2))) if np.any(treble_mask) else 0.0

    air_mask = (f_bins >= 10000.0) & (f_bins <= 20000.0)
    air_rmse = float(np.sqrt(np.mean((mag_model[air_mask] - mag_dsp[air_mask]) ** 2))) if np.any(air_mask) else 0.0

    return {
        "overall_mae": round(overall_mae, 2),
        "sub_rmse": round(sub_rmse, 2),
        "low_mid_rmse": round(low_mid_rmse, 2),
        "presence_rmse": round(pres_rmse, 2),
        "treble_rmse": round(treble_rmse, 2),
        "air_rmse": round(air_rmse, 2),
        "key_frequencies": key_telemetry,
        "iso_31_bands": iso_telemetry,
    }


def evaluate_plucks_dynamics(y_val_pred: np.ndarray, y_val_tgt: np.ndarray) -> dict[str, float]:
    n = min(len(y_val_pred), len(y_val_tgt))
    yp = y_val_pred[:n]
    yt = y_val_tgt[:n]

    # Full-band ESR
    err = yp - yt
    tgt_energy = float(np.sum(yt**2)) + 1e-12
    esr = float(np.sum(err**2)) / tgt_energy
    esr_db = 10.0 * math.log10(max(esr, 1e-12))

    # RMS and Crest Factor
    rms_tgt = float(np.sqrt(np.mean(yt**2))) + 1e-12
    peak_tgt = float(np.max(np.abs(yt))) + 1e-12
    crest_tgt_db = 20.0 * math.log10(peak_tgt / rms_tgt)

    rms_pred = float(np.sqrt(np.mean(yp**2))) + 1e-12
    peak_pred = float(np.max(np.abs(yp))) + 1e-12
    crest_pred_db = 20.0 * math.log10(peak_pred / rms_pred)

    delta_crest_db = crest_pred_db - crest_tgt_db
    delta_rms_db = 20.0 * math.log10(rms_pred / rms_tgt)

    # 20ms Attack Transient ESR (first 960 samples)
    att_len = min(960, n)
    att_err = float(np.sum((yp[:att_len] - yt[:att_len]) ** 2))
    att_tgt = float(np.sum(yt[:att_len] ** 2)) + 1e-12
    att_esr_db = 10.0 * math.log10(max(att_err / att_tgt, 1e-12))

    return {
        "esr": esr,
        "esr_db": round(esr_db, 2),
        "delta_rms_db": round(delta_rms_db, 2),
        "target_crest_db": round(crest_tgt_db, 2),
        "model_crest_db": round(crest_pred_db, 2),
        "delta_crest_db": round(delta_crest_db, 3),
        "att_esr_db": round(att_esr_db, 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Master Native vs Stock NAM Sweep Benchmark")
    parser.add_argument("--epochs", type=int, default=STOCK_DEFAULT_MAX_EPOCHS, help="Schedule budget epochs (60)")
    parser.add_argument("--min-epochs", type=int, default=STOCK_DEFAULT_MIN_EPOCHS, help="Warmup floor epochs (35)")
    parser.add_argument("--patience", type=int, default=STOCK_DEFAULT_PATIENCE, help="Early stopping patience (6)")
    parser.add_argument("--min-delta", type=float, default=STOCK_DEFAULT_MIN_DELTA, help="Min delta (1e-4)")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size (16)")
    parser.add_argument("--skip-train", action="store_true", help="Skip training if models exist")
    parser.add_argument("--only-native", action="store_true", help="Only train/benchmark native engine")
    parser.add_argument("--only-stock", action="store_true", help="Only train/benchmark stock engine")
    parser.add_argument("--voices", nargs="*", default=None, help="Subset of voice slugs to benchmark")
    args = parser.parse_args()

    bundle_dir = REPO_ROOT / "tone3000" / "packs" / "34in_standard_p" / "bundles" / "split"
    dry_wav = bundle_dir / "dry v6.1.1.wav"
    if not dry_wav.exists():
        raise FileNotFoundError(f"Dry bundle file not found: {dry_wav}")

    inst_src = load_instrument("34in_standard_p")
    v_src = inst_src.voicings["vintage_open"]

    print("================================================================================")
    print(" ALLOMORPH A2 ENGINE BENCHMARK: NATIVE vs ORIGINAL STOCK")
    print(f" Settings: epochs={args.epochs}, min_epochs={args.min_epochs}, patience={args.patience}, batch_size={args.batch_size}")
    print(" Loss: MSE=1.00, Mid(3k)=0.50, High(7k)=0.50 | Precision: bf16-mixed")
    print("================================================================================\n")

    # Synthesize benchmark log sweep
    print("[1/3] Synthesizing 5.0-second log-sine sweep (20 - 20,000 Hz, -20.5 dBFS)...")
    sweep_dry = synth_log_sweep(duration_sec=5.0)
    src_sweep = simulate_voicing_dsp(sweep_dry, inst_src, v_src)

    # Load authentic validation plucks
    print("[2/3] Loading dry excitation audio...")
    x_dry_full = load_audio(dry_wav)
    val_len = min(432_000, len(x_dry_full))
    x_val_plucks = x_dry_full[-val_len:]

    voices = [v for v in VOICES_TO_BENCHMARK if (args.voices is None or v[0] in args.voices)]

    results_json = REPO_ROOT / "scratch" / "sweep_benchmark_results.json"
    if results_json.exists():
        try:
            with open(results_json, "r", encoding="utf-8") as f:
                all_results: dict[str, Any] = json.load(f)
        except (json.JSONDecodeError, OSError):
            all_results = {}
    else:
        all_results = {}

    stock_dir = REPO_ROOT / "scratch" / "benchmark_runs" / "stock_orig"
    native_dir = REPO_ROOT / "scratch" / "benchmark_runs" / "native_calibrated"
    stock_dir.mkdir(parents=True, exist_ok=True)
    native_dir.mkdir(parents=True, exist_ok=True)

    engines_to_run = []
    if not args.only_native:
        engines_to_run.append(("stock", stock_dir))
    if not args.only_stock:
        engines_to_run.append(("native", native_dir))

    for voice_slug, voice_name in voices:
        target_wav = bundle_dir / f"{voice_name} v6.1.1.wav"
        if not target_wav.exists():
            print(f"Skipping {voice_name}: stem missing at {target_wav}")
            continue

        print("\n################################################################################")
        print(f" TARGET VOICING: {voice_name} ({voice_slug})")
        print("################################################################################")

        # Ground truth DSP target simulation
        inst_tgt, v_tgt = resolve_target_voicing(voice_slug, instrument=inst_src)
        tgt_sweep = simulate_voicing_dsp(sweep_dry, inst_tgt, v_tgt)

        gate_taps = 8192
        f_bins, H_dsp, _ = deconvolve_log_sweep(tgt_sweep, src_sweep, gate_taps=gate_taps)
        mag_dsp = 20.0 * np.log10(np.maximum(np.abs(H_dsp), 1e-6))

        # Target validation plucks
        y_tgt_full = load_audio(target_wav)
        y_val_tgt = y_tgt_full[-val_len:]

        voice_res: dict[str, Any] = all_results.get(
            voice_slug,
            {
                "voice_slug": voice_slug,
                "voice_name": voice_name,
                "models": {},
            },
        )

        for engine_name, out_dir in engines_to_run:
            print(f"\n>>> Running Engine: {engine_name.upper()} on {voice_name} <<<")

            model_files = list(out_dir.glob(f"**/*{voice_slug}*.nam")) + list(out_dir.glob(f"**/*{voice_name}*.nam"))

            if args.skip_train and model_files:
                exported_nam = model_files[0]
                print(f"Reusing existing trained model: {exported_nam}")
                train_time = 0.0
                peak_vram_mb = 0.0
            else:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats()

                t0 = time.time()
                ok = train_voice(
                    instrument="34in_standard_p",
                    voice=voice_slug,
                    input_wav=dry_wav,
                    reference_wav=dry_wav,
                    output_wav=target_wav,
                    models_dir=out_dir,
                    epochs=args.epochs,
                    min_epochs=args.min_epochs,
                    patience=args.patience,
                    min_delta=args.min_delta,
                    mse_weight=STOCK_DEFAULT_MSE_WEIGHT,
                    hp_mid_weight=STOCK_DEFAULT_HP_MID_ESR_WEIGHT,
                    hp_high_weight=STOCK_DEFAULT_HP_HIGH_ESR_WEIGHT,
                    batch_size=args.batch_size,
                    precision="bf16-mixed",
                    silent=False,
                    engine=engine_name,
                    seed=42,
                )
                train_time = time.time() - t0

                peak_vram_mb = 0.0
                if torch.cuda.is_available():
                    peak_vram_mb = float(torch.cuda.max_memory_allocated() / (1024 * 1024))

                if not ok:
                    print(f"Error: Training failed for {engine_name} on {voice_name}")
                    continue

                model_files = list(out_dir.glob(f"**/*{voice_slug}*.nam")) + list(out_dir.glob(f"**/*{voice_name}*.nam"))
                if not model_files:
                    print(f"Error: Model not found in {out_dir}")
                    continue
                exported_nam = model_files[0]

            with open(exported_nam, "r", encoding="utf-8") as f:
                nam_meta = json.load(f).get("metadata", {})

            trained_epochs = nam_meta.get("epochs_trained", args.epochs)
            stop_reason = nam_meta.get("stop_reason", "schedule_complete")

            # Evaluate Sweep Deconvolution
            print(f"Evaluating {engine_name.upper()} log sweep deconvolution...")
            # Full tier (channels_8, submodel 1)
            y_sw_full = run_nam_inference(exported_nam, src_sweep, submodel_idx=1)
            full_sweep_metrics = evaluate_sweep_metrics(y_sw_full, src_sweep, f_bins, mag_dsp, gate_taps=gate_taps)

            # Lite tier (channels_3, submodel 0)
            y_sw_lite = run_nam_inference(exported_nam, src_sweep, submodel_idx=0)
            lite_sweep_metrics = evaluate_sweep_metrics(y_sw_lite, src_sweep, f_bins, mag_dsp, gate_taps=gate_taps)

            # Small-signal impulse response
            ir_full = extract_small_signal_ir(exported_nam, submodel_idx=1)
            ir_lite = extract_small_signal_ir(exported_nam, submodel_idx=0)

            # Evaluate Plucks Audio Dynamics
            print(f"Evaluating {engine_name.upper()} validation plucks dynamics...")
            y_val_pred = run_nam_inference(exported_nam, x_val_plucks, submodel_idx=1)
            dynamics_metrics = evaluate_plucks_dynamics(y_val_pred, y_val_tgt)

            sec_per_epoch = round(train_time / max(trained_epochs, 1), 2) if train_time > 0 else 0.0

            voice_res["models"][engine_name] = {
                "nam_path": str(exported_nam),
                "epochs_trained": trained_epochs,
                "stop_reason": stop_reason,
                "train_time_seconds": round(train_time, 2),
                "sec_per_epoch": sec_per_epoch,
                "peak_vram_mb": round(peak_vram_mb, 1),
                "ir_full": ir_full,
                "ir_lite": ir_lite,
                "sweep_full": full_sweep_metrics,
                "sweep_lite": lite_sweep_metrics,
                "plucks_dynamics": dynamics_metrics,
            }

        all_results[voice_slug] = voice_res

    # Save complete JSON benchmark telemetry
    results_json = REPO_ROOT / "scratch" / "sweep_benchmark_results.json"
    with open(results_json, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    # Print summary comparison table
    print("\n=========================================================================================================================")
    print(" MASTER ENGINE BENCHMARK SUMMARY (Stock vs Native Calibrated)")
    print("=========================================================================================================================")
    header = f"{'Voice':<20} | {'Engine':<7} | {'Epochs':<6} | {'Time (s)':<8} | {'s/ep':<5} | {'Full MAE':<8} | {'Sub-RMSE':<8} | {'Treb-RMSE':<9} | {'Air-RMSE':<8} | {'ΔCrest':<7} | {'Att-ESR':<7}"
    print(header)
    print("-" * len(header))
    for v_slug, v_data in all_results.items():
        v_name = v_data.get("voice_name", v_slug)
        models = v_data.get("models", {})
        for eng in ["stock", "native"]:
            m = models.get(eng)
            if not m:
                continue
            sw_full = m.get("sweep_full", {})
            dyn = m.get("plucks_dynamics", {})
            ep = f"{m.get('epochs_trained', '-')}"
            t_s = f"{m.get('train_time_seconds', 0.0):.1f}"
            s_ep = f"{m.get('sec_per_epoch', 0.0):.2f}"
            mae = f"{sw_full.get('overall_mae', 0.0):.2f} dB"
            sub = f"{sw_full.get('sub_rmse', 0.0):.2f} dB"
            treb = f"{sw_full.get('treble_rmse', 0.0):.2f} dB"
            air = f"{sw_full.get('air_rmse', 0.0):.2f} dB"
            crest = f"{dyn.get('delta_crest_db', 0.0):+.2f} dB"
            att = f"{dyn.get('att_esr_db', 0.0):.2f} dB"
            print(f"{v_name:<20} | {eng.upper():<7} | {ep:<6} | {t_s:<8} | {s_ep:<5} | {mae:<8} | {sub:<8} | {treb:<9} | {air:<8} | {crest:<7} | {att:<7}")
        print("-" * len(header))

    print("\n================================================================================")
    print(f" BENCHMARK COMPLETE — Results saved to {results_json}")
    print("================================================================================\n")


if __name__ == "__main__":
    main()
