# Neural Amp Modeler (NAM) Training Architecture & Audio Pairings

This document details the audio excitation files, target wet sweeps, training pairings, and CLI workflows for training **NAM Architecture 2 (A2)** models under the **Direct Single-Block Digital Twin System** (Architecture D) and Tone3000 upload bundles.

---

## 1. Quick Reference: Training Audio Pairings

| Modeling System | Stage / Role | Input Audio (Dry Excitation) | Target (Output Wet) Audio | Exported NAM Model | Learned Transformation |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Direct NAM Training** | **Direct Single-Block Digital Twin** | Source wet stem `audio/wet/<src_inst>/<pickup>.wav` | Target wet stem `audio/wet/<tgt_inst>/<voice_slug>.wav` | `models/<src_inst>/<voice_slug>.nam` | Direct wet $\to$ wet neural modeling mapping source pickup to target voice |
| **Tone3000 Upload Bundle** | **Single Dry + Multi Wet Stems** | `bundles/<pickup>/dry.wav` | `bundles/<pickup>/<target_name>.wav` | Trained via Tone3000 cloud trainer | $W = X_{\text{dry}} * h_{\text{aperture}} * h_{\text{circuit}}$ with $-0.09\text{ dBFS}$ true-peak ceiling |
| **Custom Pair Training** | **Explicit Audio Files** | `--input <path_to_dry.wav>` | `--output <path_to_wet.wav>` | `models/<basename>.nam` | Direct mapping via user-specified excitation and response |

---

## 2. Signal Flow Mathematical Proof

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
   Trained directly on paired wet stems ($X_{\text{src}} \to Y_{\text{tgt}}$), the neural network learns the non-linear relationship between physical string excursion and harmonic overtone production. Rather than applying a massive linear gain multiplier to a noisy, attenuated high-frequency band, the network **actively generates phase-coherent upper harmonics directly from the strong fundamental and low-order string overtones present in the source signal**. 

This generates genuine treble air, metallic attack snap, and pickup bite—faithfully restoring the missing high end while leaving the uncorrelated input noise floor completely untouched.

---

## 3. Excitation Sweeps & Level Calibration

### 1. Master Base Excitation (`audio/input.wav`)
- **Origin:** Synthesized by [`allomorph.dsp.generate_optimal_bass_dry`](../src/allomorph/dsp/dry.py).
- **Contents:** The single unvoiced string vibration signal in the entire system. Contains multi-tier logarithmic sine chirps ($15\text{ Hz}$ to $22\text{ kHz}$) with $1200\text{ Hz}$ string tilt ($1/\sqrt{1 + (f/1200)^2}$), 5-step dynamic velocity ladder on open E1 ($pp$ to $ff$), modal plucks with attack pitch sag, plectrum pick strikes, slap thumb pops, palm-muted staccato, percussive ghost notes ($<40\text{ ms}$), natural harmonics, calibrated CCIF dual-tone intermodulation probes ($19\text{ kHz} + 20\text{ kHz}$, $0.150$ amplitude), and the Multi-Technique Bass Performance Suite replacing unrealistic pink noise to prevent loudness measurement skew.
- **Leveling:** Calibrated to the official NAM training reference standard of **$-20.50\text{ LUFS}$** (ITU-R BS.1770-4 gated integrated loudness) / **$-20.50\text{ dBFS}$ RMS** and **$-6.30\text{ dBFS}$ True Peak**, providing $>20.4\text{ dB}$ of clean digital headroom for natural high-resonance pickup peaks and scale snap without digital clipping or premature limiter engagement.

### 2. Source Instrument Pickup Stems (`audio/wet/<inst_id>/<pickup>.wav`)
- **Origin:** Generated by [`allomorph.circuit.forward.simulate_instrument_voicing`](../src/allomorph/circuit/forward.py).
- **Contents:** The simulated physical output of a source instrument's pickup aperture and circuit, serving as the input stem for wet $\to$ wet paired modeling.
- **Leveling:** Matches master excitation levels ($-20.50\text{ LUFS}$) strictly bounded by the **$0.9900$ ($-0.09\text{ dBFS}$) 4x Oversampled True-Peak safety ceiling** via transparent linear peak scaling to avoid double compression.

### 3. Target Wet Stems (`audio/wet/<inst_id>/<voice_slug>.wav`)
- **Origin:** Synthesized via native forward simulation and non-linear magnetic ODE solving (`allomorph --stage sim` or `simulate_instrument_voicing`).
- **Contents:** Fully simulated target voice responses featuring dynamic magnetic string pull damping, back-EMF string braking, dynamic reluctance inductance modulation, core curvature wobble, dynamic aperture bloom ($\kappa_{\text{ap}}$), branch-wise pre-summing magnetic saturation for composite pickups with causal arrival delay $\tau_i$, and Stage 8 oversampled non-linear magnetic saturation.
- **Leveling:** Calibrated ITU-R BS.1770-4 gated integrated loudness matching ($-20.50\text{ LUFS}$) guarantees uniform stage gain across all target voices, strictly bounded by the linear peak-based safety ceiling at **$-0.09\text{ dBFS}$ (0.9900 True Peak)** without double-compressing transient dynamics.

---

## 4. Training CLI Workflows (`scripts/train_nam.py`)

Allomorph includes an automated trainer targeting Apple Silicon Metal (MPS) and AMD ROCm GPU acceleration:

### A. Train Direct NAM Models
Trains a monolithic A2 model mapping directly from the source instrument pickup to the target voice (wet $\to$ wet paired modeling):
```bash
# Train direct model from 30" EMG MMTWX to Precision Active:
uv run python scripts/train_nam.py --instrument 30in_emg_mmtw --voice precision_active

# Train all target voices for an instrument:
uv run python scripts/train_nam.py --instrument 34in_standard_p --voice all

# Train with explicit input/output files:
uv run python scripts/train_nam.py --input audio/wet/30in_emg_mmtw/mmtw_dual.wav \
                                  --output audio/wet/34in_standard_pj/precision_active.wav \
                                  --basename precision_active

# Or simulate and train via the allomorph pipeline:
uv run allomorph --stage train --instrument 30in_emg_mmtw --voice precision_active
```

### B. Tone3000 Upload Bundles (`bundles/<pickup>/`)
For web training on Tone3000, bundles are generated via `allomorph --stage pack`:
- Each bundle contains a single `dry.wav` and all target wet stems mapped to that pickup switch position.
- Model names follow the zero-scroll display budget ($\le 34$ characters) and filename ceiling ($\le 64$ characters).

---

## 5. Training Engines, Standards & Submodule Layout

Allomorph provides two dedicated training backends in `src/allomorph/trainer/`:
1. **Native Engine (`allomorph.trainer.native`):** Bespoke in-VRAM zero-copy PyTorch engine with unified 11-channel masked convolutions, multi-resolution STFT spectral loss guidance, and fast convergence with adaptive early stopping. See [`docs/trainer/native/engine_architecture.md`](trainer/native/engine_architecture.md) and [`docs/trainer/native/spectral_loss_and_dynamics.md`](trainer/native/spectral_loss_and_dynamics.md).
2. **Stock Engine (`allomorph.trainer.stock`):** Upstream Neural Amp Modeler / PyTorch Lightning engine providing a standard 200-epoch baseline with A2 slimmable submodel extraction and terminal progress logging. See [`docs/trainer/stock/`](trainer/stock/).

### Hyperparameter Standards & Defaults

| Parameter | Stock Engine Default (`STOCK_DEFAULT_*`) | Native Engine Default (`NATIVE_DEFAULT_*`) | Description |
| :--- | :---: | :---: | :--- |
| **Architecture** | `A2` | `A2` | Slimmable Architecture 2 (`channels_3` + `channels_8`) |
| **Max Epoch Schedule** | `200` | `100` | Training epoch budget |
| **Early Stopping** | None (fixed 200-epoch schedule) | Anti-Drift Pareto | Early termination on plateau convergence |
| **Min Warmup Epochs** | N/A | `60` | Warmup floor before early stopping engages |
| **Patience** | N/A | `10` | Early stopping plateau patience window |
| **Min Delta** | N/A | `1.0e-6` | Composite validation loss improvement threshold |
| **Loss Formulation** | Standard Upstream NAM (MSE/ESR) | Multi-Resolution STFT + MSE | Loss function for parameter optimization |
| **Waveform MSE Weight**| N/A (Standard NAM) | `1.00` | Full-band time-domain MSE anchor weight |
| **MRSTFT Weight** | N/A | `0.0005` | Multi-scale log-magnitude and spectral convergence weight |
| **Base Learning Rate** | `1.0e-2` (Stock Adam) | `4.0e-3` (Native Adam) | Optimizer initial learning rate |
| **Weight Decay** | `0.0` | `3.17e-7` | Optimizer L2 kernel regularization |
| **Slice Length** | `8192` (DataLoader) | `8192` (In-VRAM Permuted) | Audio segment length in samples |
| **Precision** | `auto` | `auto` | `bf16-mixed` on modern GPUs, `32-true` on CPU |
| **Engine Choice** | `--engine stock` | `--engine native` | Training backend selection (default: `auto`) |

---

## 6. Audio QA Telemetry & Catalog Auditing (`allomorph --stage audit`)

To maintain pristine audio quality across all 25 target voices and source instrument digital twins, Allomorph includes an integrated audio QA telemetry suite:

```bash
# Audit entire wet audio catalog across all instruments:
uv run allomorph --stage audit

# Audit with verbose telemetry and strict pass/fail status:
uv run allomorph --stage audit --verbose
```

### Telemetry Guardrail Standards
- **Loudness Compliance:** Target stems must evaluate within $\pm 0.50\text{ LUFS}$ of the $-20.50\text{ LUFS}$ target ($[-21.00, -20.00]\text{ LUFS}$).
- **True-Peak Safety:** 4x oversampled true peak must stay strictly $\le 0.9900$ ($-0.09\text{ dBFS}$) to guarantee inter-sample clipping immunity during downstream reconstruction.
- **Dynamic Crest Factor:** Stems maintain $\ge 12.0\text{ dB}$ crest factor ($20\log_{10}(\text{Peak}_{\text{true}} / \text{RMS})$) ensuring that transients remain punchy without double-compression.
- **DC Offset Immunity:** Mean DC offset must stay below $-60.0\text{ dBFS}$ ($< 0.0010$) to prevent DAC bias and asymmetry in non-linear pedalboard models.

