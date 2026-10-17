# Allomorph Neural Amp Modeler (NAM) Training Architecture & Pairings

This document is the primary reference for **Neural Amp Modeler (NAM) Architecture 2 (A2)** training in Allomorph, covering Forward Digital Twin audio pairings, non-linear harmonic restoration theory, excitation sweeps, dual-engine architecture (`native` vs `stock`), hyperparameter standards, and `allomorph train` / `allomorph audit` CLI workflows.

---

## 1. Quick Reference: Training Audio Pairings

| Modeling System | Stage / Role | Input Audio (Dry Excitation) | Target (Output Wet) Audio | Exported NAM Model | Learned Transformation |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Direct NAM Training** | **Direct Single-Block Digital Twin** | Source stem `audio/stems/<src_inst>/<src_voicing>.wav` | Target stem `audio/stems/<tgt_inst>/<tgt_voicing>.wav` | `models/<src_inst>/<src_voicing>/<tgt_inst>/<tgt_voicing>.nam` | Direct stem $\to$ stem neural modeling mapping source voicing to target voicing |
| **Tone3000 Upload Bundle** | **Single Dry + Multi Wet Stems** | `bundles/<pickup>/dry.wav` | `bundles/<pickup>/<target_name>.wav` | Trained via Tone3000 cloud trainer | $W = X_{\text{dry}} * h_{\text{aperture}} * h_{\text{circuit}}$ with $-0.09\text{ dBFS}$ true-peak ceiling |
| **Custom Pair Training** | **Explicit Audio Files** | `--input <path_to_dry.wav>` | `--output <path_to_wet.wav>` | `models/<basename>.nam` | Direct mapping via user-specified excitation and response |

---

## 2. Signal Flow Mathematical Proof & Neural Harmonic Restoration

### Direct Single-Block Hardware Operation
When a player plugs their physical bass into the pedalboard (Darkglass Anagram Block 1), the instrument produces an electrical voltage governed by its physical string excitation $X_{\text{string}}$ and pickup/circuit response $H_{\text{src}}$:
$$X_{\text{bass}} = X_{\text{string}} \cdot H_{\text{src}}$$

The monolithic NAM digital twin evaluates the end-to-end forward transformation $H_{\text{direct}} = \dfrac{H_{\text{tgt}}}{H_{\text{src}}}$:
$$Y_{\text{Block 1}} = X_{\text{bass}} \cdot H_{\text{direct}} = (X_{\text{string}} \cdot H_{\text{src}}) \cdot \frac{H_{\text{tgt}}}{H_{\text{src}}} = X_{\text{string}} \cdot H_{\text{tgt}}$$

The output is the authentic physical **Target Voice** with full dynamic feel, magnetic drag, core hysteresis, and circuit loading, synthesized in a single zero-latency neural forward pass.

### Non-Linear Harmonic Restoration vs. Linear IR Noise Amplification

A fundamental electroacoustic limitation of linear Impulse Responses (IRs) or finite impulse response (FIR) filters is their inability to restore attenuated or missing high-frequency harmonics without catastrophic noise floor degradation:

1. **The Linear Inversion Noise Penalty:**
   Any linear time-invariant (LTI) system evaluates $Y(f) = X(f) \cdot H(f)$. If the input signal includes source pickup thermal Johnson noise and active preamp hiss $N_{\text{src}}(f)$:
   $$Y(f) = S_{\text{src}}(f) \cdot H(f) + N_{\text{src}}(f) \cdot H(f)$$
   When converting a warm, rolled-off source pickup (e.g. an active neck pickup or deep soapbar whose response drops by $-20\text{ to }-40\text{ dB}$ above $6\text{ kHz}$) into an aggressive, clanky target (e.g. Rickenbacker 4003 or Dingwall bridge pickup with strong $8\text{--}12\text{ kHz}$ treble), a linear IR must apply $+20\text{ to }+40\text{ dB}$ of linear high-frequency boost. This explodes the background noise floor $N_{\text{src}}(f)$ into an unbearable wash of high-frequency hiss. Furthermore, if a physical harmonic is cancelled entirely by a pickup aperture comb null ($H_{\text{src}}(f_0) = 0$), linear multiplication by zero cannot recover it.

2. **Neural Harmonic Generation in Direct Stem $\to$ Stem NAM:**
   Unlike a linear filter, a **Neural Amp Modeler (WaveNet Architecture 2)** network is a deep non-linear convolutional model:
   $$y[n] = \mathcal{N}_{\theta}(x_{\text{src}}[n])$$
   Trained directly on paired stems ($X_{\text{src}} \to Y_{\text{tgt}}$), the neural network learns the non-linear relationship between physical string excursion and harmonic overtone production. Rather than applying a massive linear gain multiplier to a noisy, attenuated high-frequency band, the network **actively generates phase-coherent upper harmonics directly from the strong fundamental and low-order string overtones present in the source signal**, restoring missing high end while leaving the uncorrelated input noise floor untouched.

---

## 3. Excitation Sweeps & Level Calibration

1. **Master Base Excitation (`audio/input.wav`):**
   - **Origin:** Synthesized by [`allomorph.dsp.generate_optimal_bass_dry`](../../src/allomorph/dsp/dry.py) (see [`docs/dsp_synthesis_engine.md`](../dsp_synthesis_engine.md)).
   - **Contents:** Multi-tier logarithmic sine chirps ($15\text{ Hz}$ to $22\text{ kHz}$) with $1200\text{ Hz}$ string tilt ($1/\sqrt{1 + (f/1200)^2}$), 5-step dynamic velocity ladder on open E1 ($pp$ to $ff$), modal plucks with attack pitch sag, pick strikes, slap thumb pops, palm-muted staccato, percussive ghost notes ($<40\text{ ms}$), natural harmonics, calibrated CCIF dual-tone intermodulation probes ($19\text{ kHz} + 20\text{ kHz}$, $0.150$ amplitude), and the Multi-Technique Bass Performance Suite.
   - **Leveling:** Calibrated to the official NAM training reference standard of **$-20.50\text{ LUFS}$** (ITU-R BS.1770-4 gated integrated loudness) / **$-20.50\text{ dBFS}$ RMS** and **$-6.30\text{ dBFS}$ True Peak**, providing $>20.4\text{ dB}$ of clean digital headroom.
2. **Source & Target Instrument Stems (`audio/stems/<inst_id>/<voicing>.wav`):**
   - **Origin:** Generated by [`allomorph.circuit.forward.simulate_instrument_voicing`](../../src/allomorph/circuit/forward.py) (`allomorph stems generate`).
   - **Leveling:** On-disk wet stems preserve raw physical transfer levels bounded strictly by the **$0.9900$ ($-0.09\text{ dBFS}$) 4x Oversampled True-Peak safety ceiling** via transparent whole-file linear peak scaling. At the start of NAM training (`train_voice_native` and `train_voice_stock`), both source and target stems are normalized via [`allomorph.trainer.core.normalize_training_stem`](../../src/allomorph/trainer/core.py) to the calibrated broadband RMS standard ($-20.50\text{ dBFS}$ RMS, `TARGET_RMS_DBFS`), guaranteeing $0.00\text{ dB}$ unity RMS insertion gain into Block 2 downstream pedals and amps.

---

## 4. Dual-Engine Architecture & Submodule Layout

Allomorph maintains two decoupled training backends within [`src/allomorph/trainer/`](../../src/allomorph/trainer/):

```
src/allomorph/trainer/
├── constants.py           # Canonical defaults: STOCK_DEFAULT_* and NATIVE_DEFAULT_*
├── core.py                # Universal NAM metadata, packaging, and validation helpers
├── stock/                 # Standard upstream PyTorch Lightning / NAM trainer engine
│   └── engine.py          # Stock 200-epoch training wrapper with A2 export & ESR logging
└── native/                # Custom high-throughput PyTorch A2 engine
    ├── engine.py          # In-VRAM vectorized training loop and early stopping
    ├── loss.py            # Multi-band spectral loss and 255-tap Blackman FIR design
    └── models.py          # Unified 11-channel masked WaveNet A2 model
```

| Dimension | Stock Engine (`allomorph.trainer.stock`) | Native Engine (`allomorph.trainer.native`) |
| :--- | :--- | :--- |
| **Framework** | PyTorch Lightning + `neural-amp-modeler` | Pure PyTorch (Zero external trainer dependencies) |
| **Data Ingestion** | Multi-process CPU `DataLoader` (IPC queues) | In-VRAM contiguous buffer with 2D GPU index gather |
| **Model Execution** | Upstream NAM training passes | Unified 11-channel block-masked Conv1D (Lite + Full in single pass) |
| **Throughput (RX 9070 XT)** | $6.52\text{ s / epoch}$ (65% GPU saturation) | **$2.26\text{--}3.15\text{ s / epoch}$ (85–100% GPU saturation)** |
| **Early Stopping** | None (upstream fixed 200-epoch schedule) | Integrated post-warmup Anti-Drift Pareto stopping |
| **Target Output** | Dual-tier A2 `.nam` file | Dual-tier A2 `.nam` file (Identical C++ core schema) |

### Hyperparameter Standards & Defaults (`src/allomorph/trainer/constants.py`)

| Parameter | Stock Engine (`STOCK_DEFAULT_*`) | Native Engine (`NATIVE_DEFAULT_*`) | Description |
| :--- | :---: | :---: | :--- |
| **Architecture** | `A2` | `A2` | Slimmable Architecture 2 (`channels_3` + `channels_8`) |
| **Max Epoch Schedule** (`--epochs`) | `200` | `120` | Training epoch budget |
| **Min Warmup Epochs** (`--min-epochs`) | N/A | `100` | Warmup floor before early stopping evaluates |
| **Patience** (`--patience`) | N/A | `15` | Early stopping plateau patience window |
| **Min Delta** (`--min-delta`) | N/A | `1.0e-6` | Composite validation loss improvement threshold |
| **Loss Formulation** | Standard Upstream NAM (MSE/ESR) | Pure MSE + 255-tap Blackman FIR GEMM | Time-domain MSE with highpass regularizer ($w=2.5, f_c=1200\text{Hz}$) |
| **Waveform MSE (`MSE`)** | N/A (Standard NAM) | `1.00` | Full-band ($0\text{--}24\text{ kHz}$) time-domain MSE anchor weight |
| **FIR Highpass (`FIR HP`)** | N/A | `2.50` | 255-tap linear-phase Blackman FIR high-pass MSE weight (`padding="valid"`) |
| **FIR Highpass Cutoff (`FIR HP FC`)** | N/A | `1200.0 Hz` | $-6\text{ dB}$ cutoff frequency isolating RLC peaks ($>1.2\text{ kHz}$) and penalizing harmonic buzz |
| **Linearity Reg (`sine_regularizer`)** | N/A | `True` ($170.0\text{ s}$) | 3 tilted log chirps ($20\text{ Hz}\to 14\text{ kHz}$) + $28\text{ freqs} \times 5\text{ amps}$ pure sines ($40\text{--}7000\text{ Hz}$) with gain alignment |
| **Wiener Head Warm-Start** | N/A | `True` | Closed-form Tikhonov-regularized ($\lambda_{\text{rel}}=10^{-2}$) Step 0 `head_rechannel` solver |
| **$C^\infty$ Activation Homotopy** | N/A | `0.02` ($\tau_0$) | Zero-centered SiLU-gated LeakyReLU homotopy ($\tau: 0.02 \to 0.0$ over $75\%$ of `min_epochs`) |
| **In-VRAM EMA Shadow** | N/A | `0.995` | Exponential moving average (`torch._foreach_lerp_`) evaluated during post-warmup validation |
| **Unlucky Spike Restart (CL-37)** | N/A | `500.0x` (`max=3`) | Detects catastrophic loss explosions ($\ge 500\times \mathcal{L}_{\min}$, floor $0.02$) or `NaN`s and restarts with $0.85\times$ LR backoff |
| **Base Learning Rate** (`--lr`) | `1.0e-2` (Stock Adam) | `4.0e-3` (Native Adam) | Optimizer initial learning rate |
| **Learning Rate Schedule** | Exponential ($0.994^t$) | Cosine Annealing ($T_{\max}=115, \eta_{\min}=10^{-6}$) | Smooth annealing extinguishing LeakyReLU zero-crossing buzz |
| **Weight Decay** (`--weight-decay`) | `0.0` | `3.17e-7` | Optimizer L2 kernel regularization |
| **Slice Length** (`--slice-len`) | `8192` (DataLoader) | `8192` (In-VRAM Permuted) | Audio segment length in samples |
| **Batch Size** | `auto` | `auto` (`16` on GPU) | Fits 64 MB Infinity Cache boundary; prevents GDDR6 memory thrashing |
| **Precision** (`--precision`) | `auto` | `auto` | `bf16-mixed` on modern GPUs, `32-true` on CPU |

### How `Linearity Reg`, `MSE + FIR HP (1200 Hz)`, and the `Cosine + Homotopy Schedule` Work Together

The native engine's training dynamics rely on a three-part synergy between **data space** (`Linearity Reg`), **loss space** (`MSE` + `FIR HP`), and **time/optimizer space** ($C^\infty$ Homotopy + Cosine Annealing):

1. **Data Space — `Linearity Reg` ($170\text{ s}$ Analytical Linear Probes):**
   Before Epoch 1, [`train_voice_native`](../../src/allomorph/trainer/native/engine.py) synthesizes 3 continuous tilted log chirps ($20\text{ Hz}\to 14\text{ kHz}$ at amplitudes `0.40, 0.20, 0.08`) and $140$ pure-sine probes ($28\text{ frequencies} \times 5\text{ amplitudes}$ from `0.25` down to `0.008` / $-42\text{ dBFS}$) through the exact **linear** (`apply_saturation=False`) Source and Target Digital Twin solvers, scales them by `probe_gain_correction` to match the stem RMS gain ratio, uses the `0.20` chirp to solve the Step-0 Wiener-Hopf output head in closed form, and appends all $170\text{ s}$ directly to the in-VRAM training buffer.
2. **Loss Space — Dual-Band Synergy of `MSE` ($1.0$) + `FIR HP` ($2.5, f_c=1200\text{ Hz}$):**
   - **Above $1200\text{ Hz}$:** Natural bass string energy drops by $-30\text{ to }-45\text{ dB}$ above $1.2\text{ kHz}$. `FIR HP` strips out the massive $<500\text{ Hz}$ fundamentals via a 255-tap linear-phase Blackman high-pass filter and magnifies high-frequency gradients by $2.5\times$, forcing the network to track RLC resonant peaks ($1.8\text{--}4.5\text{ kHz}$), Neodymium presence, and pick clank with exact sample-level phase.
   - **Below $1200\text{ Hz}$ (Anti-Buzz THD Penalty):** For every low-frequency sine wave ($40\text{--}880\text{ Hz}$) in `Linearity Reg`, the ideal linear target has **zero energy above $1200\text{ Hz}$** ($h_{\text{HP}} * y_{\text{sine}} \equiv 0$). Any spurious crossover distortion or high-order odd harmonics generated by the WaveNet's `LeakyReLU(0.01)` kinks pass straight through the $1200\text{ Hz}$ high-pass filter and are heavily penalized ($2.5\times$) against a `0.0` target.
3. **Time / Optimizer Space — Two-Phase $C^\infty$ Homotopy & Cosine Annealing:**
   - **Epochs 1–75 (Smooth Manifold Shaping):** While the learning rate is high ($4\times 10^{-3} \to 1.5\times 10^{-3}$), the $C^\infty$ SiLU-gated activation homotopy ($\tau: 0.02 \to 0.0$) rounds off the `LeakyReLU` corner so gradients flow smoothly across $x = 0$ without kink chatter, locking in macro EQ and comb-notch inversions.
   - **Epochs 75–120 (Kink-Locking & Micro-Polishing):** At Epoch 75, $\tau$ reaches `0.0` (exact hardware `LeakyReLU(0.01)`), while the Cosine schedule decays the learning rate down to $\eta_{\min} = 10^{-6}$ alongside in-VRAM EMA ($\beta = 0.995$). Microscopic step sizes combined with `FIR HP` + `Linearity Reg` gently settle every channel bias term so zero-crossings never chatter—achieving $-37\text{ to }-40\text{ dB}$ validation ESR.

*For full mathematical derivations, see **[Native Multi-Band Spectral Loss & Convergence Dynamics](native/spectral_loss_and_dynamics.md)**.*

---

## 5. Deep-Dive Trainer Documentation (`docs/trainer/`)

- **[Dual-Engine Optimization Process](optimization_process.md):** End-to-end technical reference detailing the optimization journeys of both the Stock Engine and the Native Engine.
- **[Native Engine Architecture & Neural Model](native/engine_architecture.md):** WaveNet A2 mathematical specifications, $6,347$-sample ($132.2\text{ ms}$) receptive field derivation, unified 11-channel block-diagonal masking, zero-copy in-VRAM slicing, and standalone `.nam` JSON serialization.
- **[Hyperparameter Optimization & Empirical Sweeps](native/hyperparameter_optimization.md):** 5-dimensional optimization campaign across batch size, learning rate, scheduler, submodel multi-task weighting, spectral loss, and slice length.
- **[Multi-Band Spectral Loss & Dynamics](native/spectral_loss_and_dynamics.md):** Physical string harmonic energy decay ($-43.5\text{ dB}$ harmonic drop-off), Multi-Resolution STFT Loss (MRSTFT), full-band MSE anchoring, and Anti-Drift Pareto early stopping.
- **[GPU Throughput Optimization & Benchmarks](native/gpu_throughput_optimization.md):** Hardware profiling on AMD Radeon RX 9070 XT, resolution of PCIe host syncs, sequence slicing, and dynamic weight masking.
- **[What Didn't Work: Forensic Analysis & Regression Prevention](native/what_did_not_work_and_regression_prevention.md):** Authoritative reference documenting 9 major anti-patterns and electroacoustic failures with mathematical root-cause proofs and regression prevention invariants.
- **[Stock Engine Architecture](stock/README.md):** Upstream NAM / PyTorch Lightning baseline, fixed 200-epoch standard schedule, and A2 slimmable container packaging.

---

## 6. CLI Workflows (`allomorph train` & `allomorph audit`)

### Training NAM Models (`allomorph train`)
```bash
# Train a standalone instrument voicing with the native engine (default):
uv run allomorph train instrument:34in_standard_p:vintage_open --source 30in_emg_mmtw:precision_active --engine native

# Train all target voices for a tone pack with adaptive early stopping:
uv run allomorph train pack:34in_standard_p --epochs 120 --min-epochs 100 --patience 15

# Train with the upstream stock NAM engine (standard 200-epoch baseline):
uv run allomorph train instrument:34in_standard_p:slab_dub --source 34in_standard_p:vintage_open --engine stock --epochs 200
```

### Audio QA Telemetry & Catalog Auditing (`allomorph audit`)
```bash
# Audit entire stems audio catalog across all instruments:
uv run allomorph audit --verbose
```
- **Loudness Compliance:** Target stems must evaluate within $\pm 0.50\text{ LUFS}$ of the $-20.50\text{ LUFS}$ target ($[-21.00, -20.00]\text{ LUFS}$).
- **True-Peak Safety:** 4x oversampled true peak must stay strictly $\le 0.9900$ ($-0.09\text{ dBFS}$).
- **Dynamic Crest Factor:** Stems maintain $\ge 12.0\text{ dB}$ crest factor ($20\log_{10}(\text{Peak}_{\text{true}} / \text{RMS})$).
- **DC Offset Immunity:** Mean DC offset must stay below $-60.0\text{ dBFS}$ ($< 0.0010$).
