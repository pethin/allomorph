# Allomorph Neural Trainer Documentation

Welcome to the **Allomorph Neural Amp Modeler (NAM) Trainer** documentation suite. This directory contains detailed architectural specifications, mathematical derivations, hardware optimization benchmarks, and convergence proofs for training high-fidelity Architecture 2 (A2) neural digital twins tailored for the **Darkglass Anagram** pedalboard (Block 1) and DAW plugin hosts.

---

## 1. Architecture Overview: Dual-Engine Design

Allomorph maintains two decoupled training engines within [`src/allomorph/trainer/`](../../src/allomorph/trainer/):

```
src/allomorph/trainer/
├── constants.py           # Immutable defaults: STOCK_DEFAULT_* and NATIVE_DEFAULT_*
├── core.py                # Universal NAM metadata, packaging, and validation helpers
├── stock/                 # Legacy PyTorch Lightning / upstream NAM trainer engine
│   └── engine.py          # Stock multi-process DataLoader and Lightning Trainer
└── native/                # Custom high-throughput PyTorch A2 engine
    ├── engine.py          # In-VRAM vectorized training loop and early stopping
    ├── loss.py            # Pure-NumPy FIR design and energy-normalized spectral loss
    └── models.py          # Unified 11-channel masked WaveNet A2 model
```

| Dimension | Stock Engine (`allomorph.trainer.stock`) | Native Engine (`allomorph.trainer.native`) |
| :--- | :--- | :--- |
| **Framework** | PyTorch Lightning + `neural-amp-modeler` | Pure PyTorch (Zero external trainer dependencies) |
| **Data Ingestion** | Multi-process CPU `DataLoader` (IPC queues) | In-VRAM contiguous buffer with 2D GPU index gather |
| **Model Execution** | Sequential / dynamic submodel passes | Unified 11-channel block-masked Conv1D (Lite + Full in single pass) |
| **Throughput (RX 9070 XT)** | $6.52\text{ s / epoch}$ (65% GPU saturation) | **$3.15\text{ s / epoch}$ (85–92% GPU saturation)** — **$2.48\times$ faster** |
| **Early Stopping** | Anti-Drift Pareto with Lightning callbacks | Integrated post-warmup Pareto stopping ($39.2\%$ compute saved) |
| **Target Output** | Dual-tier A2 `.nam` file | Dual-tier A2 `.nam` file (Identical C++ core schema) |

---

## 2. Documentation Directory Layout

### Master Optimization Process (Dual-Engine Playbook)
- **[Dual-Engine Optimization Process](optimization_process.md):**  
  Comprehensive end-to-end technical reference detailing the optimization journeys of both the **Stock Engine** (local Lightning vs Tone3000 Cloud, $500\text{ Hz}$ comb notch carving, $10\text{--}15\text{ kHz}$ treble restoration in 35 epochs) and the **Native Engine** (in-VRAM 2D index gather, zero dynamic weight allocations, $2.48\times$ GPU throughput speedup, and $3.02\times$ overall workflow acceleration).

### Native Engine Technical Guides (`docs/trainer/native/`)
- **[Engine Architecture & Neural Model](native/engine_architecture.md):**  
  Detailed WaveNet A2 mathematical specifications, $6,347$-sample ($132.2\text{ ms}$) receptive field derivation, unified 11-channel block-diagonal masking, zero-copy in-VRAM slicing, and standalone `.nam` JSON serialization.
- **[Hyperparameter Optimization & Empirical Sweeps](native/hyperparameter_optimization.md):**  
  Exhaustive 5-dimensional optimization campaign across batch size, learning rate, scheduler, submodel multi-task weighting, spectral loss, and slice length; hardware analysis of AMD RDNA 4 Infinity Cache and MIOpen unrolling thresholds.
- **[Multi-Band Spectral Loss & Dynamics](native/spectral_loss_and_dynamics.md):**  
  Physical string harmonic energy decay ($-43.5\text{ dB}$ harmonic drop-off), pure-NumPy Type-I linear-phase FIR highpass filter design (`design_fir_highpass`), self-normalizing energy ESR loss, and the Anti-Drift Pareto early stopping architecture.
- **[GPU Throughput Optimization & Benchmarks](native/gpu_throughput_optimization.md):**  
  Hardware profiling on AMD Radeon RX 9070 XT, resolution of three major GPU bottlenecks (PCIe host syncs, sequence slicing, and dynamic weight masking), Adam velocity ceiling proofs ($\ge 35$ epochs physical warmup floor), and cross-catalog 4-voice sweep benchmark matrix ($3.02\times$ workflow acceleration).

### Stock Engine Technical Guides (`docs/trainer/stock/`)
- **[Stock Spectral Loss Optimization](stock/trainer_spectral_loss_optimization.md):**  
  Historical analysis of cloud training limitations vs local training, spectral loss formulation in PyTorch Lightning, and bass-weighted loss components.
- **[Stock Early Stopping Architecture](stock/trainer_early_stopping_architecture.md):**  
  Upstream PyTorch Lightning callback architecture, multi-metric Pareto scoring, and plateau detection.

---

## 3. CLI Quick Reference

The trainer is invoked via `scripts/train_nam.py` or the `allomorph-train` entrypoint.

### Training with the Native Engine (Default Recommended):
```bash
uv run python scripts/train_nam.py \
  --source-inst 34in_standard_p \
  --voice-id precision_dub \
  --engine native \
  --epochs 60 \
  --min-epochs 35 \
  --patience 6 \
  --precision bf16-mixed
```

### Training with the Authentic Stock Engine:
```bash
uv run python scripts/train_nam.py \
  --source-inst 34in_standard_p \
  --voice-id precision_dub \
  --engine stock \
  --epochs 60 \
  --min-epochs 35 \
  --precision bf16-mixed
```

### Key CLI Flags:
| Flag | Description | Engine | Default |
| :--- | :--- | :---: | :---: |
| `--engine` | Selects training engine (`native` or `stock`) | Both | `native` |
| `--epochs` | Maximum schedule budget | Both | `60` |
| `--min-epochs` | Warmup floor before early stopping evaluates | Both | `35` |
| `--patience` | Early stopping patience epochs | Both | `6` |
| `--native-lr` | Native Adam learning rate | Native | `0.002` |
| `--native-weight-decay` | Native Adam weight decay | Native | `1e-4` |
| `--native-slice-len` | Target audio head window length | Native | `8192` |
| `--precision` | Floating-point precision (`bf16-mixed`, `16-mixed`, `32-true`) | Both | `bf16-mixed` |
