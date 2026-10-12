# Allomorph Neural Trainer Documentation

Welcome to the **Allomorph Neural Amp Modeler (NAM) Trainer** documentation suite. This directory contains detailed architectural specifications, mathematical derivations, hardware optimization benchmarks, and convergence proofs for training high-fidelity Architecture 2 (A2) neural digital twins tailored for the **Darkglass Anagram** pedalboard (Block 1) and DAW plugin hosts.

---

## 1. Architecture Overview: Dual-Engine Design

Allomorph maintains two decoupled training engines within [`src/allomorph/trainer/`](../../src/allomorph/trainer/):

```
src/allomorph/trainer/
├── constants.py           # Engine defaults: STOCK_DEFAULT_MAX_EPOCHS and NATIVE_DEFAULT_*
├── core.py                # Universal NAM metadata, packaging, and validation helpers
├── stock/                 # Standard upstream PyTorch Lightning / NAM trainer engine
│   └── engine.py          # Stock 200-epoch training wrapper with A2 export & ESR logging
└── native/                # Custom high-throughput PyTorch A2 engine
    ├── engine.py          # In-VRAM vectorized training loop and early stopping
    ├── loss.py            # Multi-band spectral loss and FIR design
    └── models.py          # Unified 11-channel masked WaveNet A2 model
```

| Dimension | Stock Engine (`allomorph.trainer.stock`) | Native Engine (`allomorph.trainer.native`) |
| :--- | :--- | :--- |
| **Framework** | PyTorch Lightning + `neural-amp-modeler` | Pure PyTorch (Zero external trainer dependencies) |
| **Data Ingestion** | Multi-process CPU `DataLoader` (IPC queues) | In-VRAM contiguous buffer with 2D GPU index gather |
| **Model Execution** | Upstream NAM training passes | Unified 11-channel block-masked Conv1D (Lite + Full in single pass) |
| **Throughput (RX 9070 XT)** | $6.52\text{ s / epoch}$ (65% GPU saturation) | **$3.15\text{ s / epoch}$ (85–92% GPU saturation)** — **$2.48\times$ faster** |
| **Early Stopping** | None (upstream fixed 200-epoch schedule) | Integrated post-warmup Pareto stopping ($39.2\%$ compute saved) |
| **Target Output** | Dual-tier A2 `.nam` file | Dual-tier A2 `.nam` file (Identical C++ core schema) |

---

## 2. Documentation Directory Layout

### Master Optimization Process (Dual-Engine Playbook)
- **[Dual-Engine Optimization Process](optimization_process.md):**  
  Comprehensive end-to-end technical reference detailing the optimization journeys of both the **Stock Engine** (local Lightning vs Tone3000 Cloud, $500\text{ Hz}$ comb notch carving, $10\text{--}15\text{ kHz}$ treble restoration) and the **Native Engine** (in-VRAM 2D index gather, zero dynamic weight allocations, $2.48\times$ GPU throughput speedup, and $3.02\times$ overall workflow acceleration).

### Native Engine Technical Guides (`docs/trainer/native/`)
- **[Engine Architecture & Neural Model](native/engine_architecture.md):**  
  Detailed WaveNet A2 mathematical specifications, $6,347$-sample ($132.2\text{ ms}$) receptive field derivation, unified 11-channel block-diagonal masking, zero-copy in-VRAM slicing, and standalone `.nam` JSON serialization.
- **[Hyperparameter Optimization & Empirical Sweeps](native/hyperparameter_optimization.md):**  
  Exhaustive 5-dimensional optimization campaign across batch size, learning rate, scheduler, submodel multi-task weighting, spectral loss, and slice length; hardware analysis of AMD RDNA 4 Infinity Cache and MIOpen unrolling thresholds.
- **[Multi-Band Spectral Loss & Dynamics](native/spectral_loss_and_dynamics.md):**  
  Physical string harmonic energy decay ($-43.5\text{ dB}$ harmonic drop-off), Multi-Resolution STFT Loss (MRSTFT) with bit-exact power clamping, full-band MSE anchoring, and the Anti-Drift Pareto early stopping architecture.
- **[GPU Throughput Optimization & Benchmarks](native/gpu_throughput_optimization.md):**  
  Hardware profiling on AMD Radeon RX 9070 XT, resolution of three major GPU bottlenecks (PCIe host syncs, sequence slicing, and dynamic weight masking), Adam velocity ceiling proofs ($\ge 35$ epochs physical warmup floor), and cross-catalog 4-voice sweep benchmark matrix ($3.02\times$ workflow acceleration).
- **[What Didn't Work: Forensic Analysis & Regression Prevention](native/what_did_not_work_and_regression_prevention.md):**  
  Authoritative, permanent reference documenting 9 major anti-patterns, electroacoustic failures, and hardware bottlenecks (flat $22\text{ kHz}$ chirps, MRSTFT differentiator crushing, independent LUFS flattening, probe gain calibration mismatch, Gibbs boundary transients, and Infinity Cache spilling) with explicit mathematical root-cause proofs and regression prevention invariants.

### Stock Engine Documentation (`docs/trainer/stock/`)
- **[Stock Engine Architecture](stock/README.md):**  
  Upstream NAM / PyTorch Lightning baseline, fixed 200-epoch standard schedule, A2 slimmable container packaging, and terminal progress logging.

---

## 3. CLI Quick Reference

The trainer is invoked via `scripts/train_nam.py` or the `allomorph-train` entrypoint.

### Training with the Native Engine (Default Recommended):
```bash
uv run python scripts/train_nam.py \
  --source-inst 34in_standard_p \
  --voice-id precision_dub \
  --engine native \
  --epochs 100 \
  --min-epochs 60 \
  --patience 10 \
  --precision bf16-mixed
```

### Training with the Upstream Stock Engine:
```bash
uv run python scripts/train_nam.py \
  --source-inst 34in_standard_p \
  --voice-id precision_dub \
  --engine stock \
  --epochs 200 \
  --precision bf16-mixed
```

### Key CLI Flags:
| Flag | Description | Engine | Default |
| :--- | :--- | :---: | :---: |
| `--engine` | Selects training engine (`native` or `stock`) | Both | `native` |
| `--epochs` | Maximum schedule budget | Both | `100` (Native) / `200` (Stock) |
| `--min-epochs` | Warmup floor before early stopping evaluates | Native | `60` |
| `--patience` | Early stopping patience epochs | Native | `10` |
| `--min-delta` | Early stopping minimum improvement delta | Native | `1e-6` |
| `--native-lr` | Native Adam learning rate | Native | `0.004` |
| `--native-weight-decay` | Native Adam weight decay | Native | `3.17e-7` |
| `--native-slice-len` | Target audio head window length | Native | `8192` |
| `--precision` | Floating-point precision (`bf16-mixed`, `16-mixed`, `32-true`) | Both | `bf16-mixed` |
