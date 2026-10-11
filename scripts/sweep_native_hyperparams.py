#!/usr/bin/env python3
"""Targeted Hyperparameter Sweeps for Full Native Trainer Optimization.

Sweeps:
1. Batch Size: B in [16, 32, 48, 64]
2. Learning Rate & Scheduler: lr in [0.002, 0.004, 0.006, 0.008], scheduler in ['exponential', 'cosine']
3. Submodel Multi-Task Weighting: w_lite in [0.10, 0.25, 0.50, 0.75, 1.00]
4. Spectral Loss Tuning: w_mid, w_high, cutoffs
5. Slice Window Length: ny in [4096, 8192, 16384]

Evaluates:
- Training throughput (sec/epoch, total seconds, peak VRAM)
- Convergence epoch and stop reason
- Validation ESR (Full and Lite)
- Synthetic 5.0-second log sweep deconvolution:
  - Overall MAE (20 to 20,000 Hz)
  - Sub-bass RMSE (20 to 120 Hz)
  - Low-mid RMSE (100 to 1,000 Hz)
  - Presence RMSE (1,000 to 4,000 Hz)
  - Treble RMSE (4,000 to 10,000 Hz)
  - Air RMSE (10,000 to 20,000 Hz)
  - Key frequency tracking (60 Hz, 500 Hz, 3 kHz, 10 kHz)
  - Dynamic Peak Crest Factor error
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

from benchmark_full_sweeps import (
    evaluate_plucks_dynamics,
    evaluate_sweep_metrics,
    extract_small_signal_ir,
    load_audio,
    run_nam_inference,
    synth_log_sweep,
)

from allomorph.circuit.forward import simulate_voicing_dsp
from allomorph.config.instruments import load_instrument
from allomorph.trainer import setup_headless_environment
from allomorph.trainer.native.engine import train_voice_native

setup_headless_environment()

SWEEP_RESULTS_DIR = REPO_ROOT / "scratch" / "native_hyperparam_sweeps"
SWEEP_RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def evaluate_model_on_sweep(nam_path: Path, tgt_inst_id: str, tgt_voice_id: str, submodel_idx: int = 1) -> dict[str, Any]:
    inst_cfg = load_instrument(tgt_inst_id)
    from allomorph.config.instruments import resolve_target_voicing

    v_src = inst_cfg.voicings["vintage_open"]
    _, tgt_v = resolve_target_voicing(tgt_voice_id, instrument=inst_cfg)

    sweep_dry = synth_log_sweep(duration_sec=5.0)
    src_sweep = simulate_voicing_dsp(sweep_dry, inst_cfg, v_src)
    tgt_sweep = simulate_voicing_dsp(sweep_dry, inst_cfg, tgt_v)

    from allomorph.dsp.sweeps import deconvolve_log_sweep

    f_bins, H_tgt, _ = deconvolve_log_sweep(tgt_sweep, src_sweep, gate_taps=8192)
    mag_dsp = 20.0 * np.log10(np.maximum(np.abs(H_tgt), 1e-6))

    y_sweep_pred = run_nam_inference(nam_path, sweep_dry, submodel_idx=submodel_idx)
    metrics = evaluate_sweep_metrics(
        y_sweep_pred=y_sweep_pred,
        src_sweep=src_sweep,
        f_bins=f_bins,
        mag_dsp=mag_dsp,
        gate_taps=8192,
    )

    # Dynamic Plucks
    bundle_dir = REPO_ROOT / "tone3000" / "packs" / "34in_standard_p" / "bundles" / "split"
    dry_wav = bundle_dir / "dry v6.1.1.wav"
    tgt_wav = bundle_dir / f"{tgt_v.tone_name} v6.1.1.wav"

    plucks_metrics: dict[str, float] = {}
    if dry_wav.exists() and tgt_wav.exists():
        x_dry = load_audio(dry_wav)
        y_tgt = load_audio(tgt_wav)
        val_len = min(432_000, len(x_dry))
        x_val = x_dry[-val_len:]
        y_val_tgt = y_tgt[-val_len:]
        y_val_pred = run_nam_inference(nam_path, x_val, submodel_idx=submodel_idx)
        plucks_metrics = evaluate_plucks_dynamics(y_val_pred, y_val_tgt)

    metrics["plucks"] = plucks_metrics
    metrics["ir_taps"] = extract_small_signal_ir(nam_path, submodel_idx=submodel_idx)
    return metrics


def run_single_experiment(
    exp_name: str,
    voice_id: str = "jazz_pair_active",
    source_inst: str = "34in_standard_p",
    batch_size: int = 16,
    lr: float = 0.002,
    lr_scheduler: str = "exponential",
    lr_t_max: int = 45,
    epochs: int = 60,
    min_epochs: int = 35,
    patience: int = 6,
    weight_decay: float = 1e-4,
    slice_len: int = 8192,
    hp_mid_weight: float = 0.75,
    hp_mid_cutoff: float = 3000.0,
    hp_high_weight: float = 0.50,
    hp_high_cutoff: float = 7000.0,
    lite_weight: float = 0.75,
) -> dict[str, Any]:
    exp_dir = SWEEP_RESULTS_DIR / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[Sweep] Starting Experiment: {exp_name} ({voice_id})")
    print(f"        bs={batch_size}, lr={lr}, sched={lr_scheduler}, ny={slice_len}")
    print(f"        hp_mid_w={hp_mid_weight} ({hp_mid_cutoff}Hz), hp_high_w={hp_high_weight} ({hp_high_cutoff}Hz)")

    t0 = time.time()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    nam_candidate = exp_dir / source_inst / f"{voice_id}.nam"
    if not nam_candidate.exists():
        nam_candidate = exp_dir / f"{voice_id}.nam"

    if nam_candidate.exists():
        print(f"[Sweep] Found existing trained model at {nam_candidate}, evaluating directly...")
        success = True
        train_time = 0.0
        peak_vram_mb = 0.0
        nam_path = nam_candidate
    else:
        success = train_voice_native(
            instrument=source_inst,
            voice=voice_id,
            models_dir=exp_dir,
            basename=voice_id,
            epochs=epochs,
            min_epochs=min_epochs,
            patience=patience,
            batch_size=batch_size,
            lr=lr,
            lr_scheduler=lr_scheduler,
            lr_t_max=lr_t_max,
            weight_decay=weight_decay,
            slice_len=slice_len,
            hp_mid_weight=hp_mid_weight,
            hp_mid_cutoff=hp_mid_cutoff,
            hp_high_weight=hp_high_weight,
            hp_high_cutoff=hp_high_cutoff,
            lite_weight=lite_weight,
            silent=False,
        )
        train_time = round(time.time() - t0, 2)
        peak_vram_mb = round(torch.cuda.max_memory_allocated() / (1024 * 1024), 1) if torch.cuda.is_available() else 0.0

        nam_path = exp_dir / source_inst / f"{voice_id}.nam"
        if not nam_path.exists():
            nam_path = exp_dir / f"{voice_id}.nam"

    if not (success and nam_path.exists()):
        print(f"[Sweep] FAILED to find model for {exp_name} at {nam_path}")
        return {"exp_name": exp_name, "success": False, "error": "Training or export failed"}

    eval_full = evaluate_model_on_sweep(nam_path, source_inst, voice_id, submodel_idx=1)
    eval_lite = evaluate_model_on_sweep(nam_path, source_inst, voice_id, submodel_idx=0)

    result = {
        "exp_name": exp_name,
        "voice_id": voice_id,
        "success": True,
        "hyperparams": {
            "batch_size": batch_size,
            "lr": lr,
            "lr_scheduler": lr_scheduler,
            "lr_t_max": lr_t_max,
            "weight_decay": weight_decay,
            "slice_len": slice_len,
            "hp_mid_weight": hp_mid_weight,
            "hp_mid_cutoff": hp_mid_cutoff,
            "hp_high_weight": hp_high_weight,
            "hp_high_cutoff": hp_high_cutoff,
            "lite_weight": lite_weight,
            "min_epochs": min_epochs,
            "max_epochs": epochs,
            "patience": patience,
        },
        "telemetry": {
            "train_time_sec": train_time,
            "peak_vram_mb": peak_vram_mb,
        },
        "sweep_full": eval_full,
        "sweep_lite": eval_lite,
    }

    c_err = eval_full.get("plucks", {}).get("delta_crest_db", 0.0)
    print(f"[Sweep] Finished {exp_name}: Time={train_time}s, Full MAE={eval_full['overall_mae']} dB, Treble={eval_full['treble_rmse']} dB, Crest Err={c_err} dB")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Native Trainer Hyperparameter Sweeps")
    parser.add_argument(
        "--sweep-type",
        choices=["batch_size", "lr_sched", "lite_weight", "spectral_loss", "slice_len", "all"],
        default="batch_size",
    )
    parser.add_argument("--voice", default="jazz_pair_active")
    args = parser.parse_args()

    json_path = SWEEP_RESULTS_DIR / f"sweep_{args.sweep_type}_{args.voice}.json"
    results: dict[str, Any] = {}
    if json_path.exists():
        try:
            with open(json_path, encoding="utf-8") as f:
                results = json.load(f)
        except (json.JSONDecodeError, OSError):
            results = {}

    def record_experiment(name: str, **kwargs: Any) -> None:
        if name in results and results[name].get("success"):
            print(f"[Sweep] Skipping {name} (already recorded in {json_path.name})")
            return
        res = run_single_experiment(name, **kwargs)
        results[name] = res
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

    if args.sweep_type in ("batch_size", "all"):
        print("\n=======================================================")
        print("SWEEP 1: Batch Size (16, 32, 48, 64)")
        print("=======================================================")
        for bs in [16, 32, 48, 64]:
            exp_name = f"bs_{bs}_{args.voice}"
            record_experiment(exp_name, voice_id=args.voice, batch_size=bs)

    if args.sweep_type in ("lr_sched", "all"):
        print("\n=======================================================")
        print("SWEEP 2: Learning Rate & Scheduler")
        print("=======================================================")
        for lr_val in [0.002, 0.004, 0.006, 0.008]:
            for sched in ["exponential", "cosine"]:
                exp_name = f"lr_{lr_val}_{sched}_{args.voice}"
                record_experiment(exp_name, voice_id=args.voice, lr=lr_val, lr_scheduler=sched)

    if args.sweep_type in ("lite_weight", "all"):
        print("\n=======================================================")
        print("SWEEP 3: Submodel Multi-Task Lite Weighting")
        print("=======================================================")
        for lw in [0.10, 0.25, 0.50, 0.75, 1.00]:
            exp_name = f"lw_{lw:.2f}_{args.voice}"
            record_experiment(exp_name, voice_id=args.voice, lite_weight=lw)

    if args.sweep_type in ("spectral_loss", "all"):
        print("\n=======================================================")
        print("SWEEP 4: Multi-Band Spectral Loss Tuning")
        print("=======================================================")
        spec_configs = [
            ("mid25_high25", 0.25, 3000.0, 0.25, 7000.0),
            ("mid50_high50_base", 0.50, 3000.0, 0.50, 7000.0),
            ("mid75_high50", 0.75, 3000.0, 0.50, 7000.0),
            ("mid50_high75", 0.50, 3000.0, 0.75, 7000.0),
            ("cutoff_low", 0.50, 2500.0, 0.50, 6000.0),
            ("cutoff_high", 0.50, 3500.0, 0.50, 8000.0),
        ]
        for cname, mw, mc, hw, hc in spec_configs:
            exp_name = f"spec_{cname}_{args.voice}"
            record_experiment(
                exp_name,
                voice_id=args.voice,
                hp_mid_weight=mw,
                hp_mid_cutoff=mc,
                hp_high_weight=hw,
                hp_high_cutoff=hc,
            )

    if args.sweep_type in ("slice_len", "all"):
        print("\n=======================================================")
        print("SWEEP 5: Audio Slice Window Length")
        print("=======================================================")
        for sl in [4096, 8192, 16384]:
            exp_name = f"slice_{sl}_{args.voice}"
            record_experiment(exp_name, voice_id=args.voice, slice_len=sl)

    print(f"\n[Sweep] All results saved to {json_path}")


if __name__ == "__main__":
    main()
