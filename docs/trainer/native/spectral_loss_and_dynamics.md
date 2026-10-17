# Native Multi-Band Spectral Loss & Convergence Dynamics

**Document Version:** 1.0.0  
**Target Hardware:** Darkglass Anagram Block 1 (Preamp / NAM Host)  
**Package Implementation:** [`src/allomorph/trainer/native/`](../../../src/allomorph/trainer/native/)  
**Primary Loss Module:** [`loss.py`](../../../src/allomorph/trainer/native/loss.py) | **Callbacks:** [`callbacks.py`](../../../src/allomorph/trainer/callbacks.py)

---

## 1. Physical Motivation: Overcoming Physical String Harmonic Decay

Electric bass strings exhibit an extreme physical energy drop-off across frequency registers:
- Fundamental register ($41\text{--}120\text{ Hz}$): Excursion envelopes reach up to $\pm 2.5\text{ mm}$, carrying $>99.5\%$ of total mechanical vibration energy.
- Treble sheen and attack harmonics ($3\text{--}15\text{ kHz}$): Vibrational excursion drops by $-43.5\text{ dB}$, representing less than $0.005\%$ of signal power.

```
Harmonic Energy Differential:
Fund (60 Hz):   ================================================= [ 0 dB ]
Low-Mid (500 Hz): =======================                         [ -18 dB ]
Clank (3 kHz):  ========                                          [ -32 dB ]
Air (7-12 kHz): ==                                                [ -44 dB ] (22,000x energy drop)
```

Under naive time-domain Mean Squared Error (MSE) loss:
$$\mathcal{L}_{\text{MSE}} = \frac{1}{N} \sum_{n=0}^{N-1} (y_{\text{pred}}[n] - y_{\text{tgt}}[n])^2$$

The gradient $\nabla_\theta \mathcal{L}_{\text{MSE}}$ is dominated almost exclusively by low-frequency excursion. High-frequency differentiator taps in the neural network receive gradients attenuated by $>20,000\times$, delaying treble and attack convergence until epoch 150–200. Conversely, forcing unnormalized pre-emphasis filters introduces severe sub-bass phase drift and low-end mush.

---

## 2. Composite Time-Domain Loss: Waveform MSE + 255-Tap Blackman FIR High-Pass (`FIR HP`)

To supervise both fundamental bass weight and high-frequency pickup resonance without phase distortion, the native engine evaluates a strictly time-domain composite loss in [`engine.py`](../../../src/allomorph/trainer/native/engine.py) and [`loss.py`](../../../src/allomorph/trainer/native/loss.py):

$$\mathcal{L}_{\text{total}} = w_{\text{MSE}} \cdot \Big(\mathcal{L}_{\text{MSE}}(\hat{y}_{\text{full}}, y) + w_{\text{lite}} \mathcal{L}_{\text{MSE}}(\hat{y}_{\text{lite}}, y)\Big) + w_{\text{HP}} \cdot \Big(\mathcal{L}_{\text{MSE}}(h_{\text{HP}} * \hat{y}_{\text{full}}, h_{\text{HP}} * y) + w_{\text{lite}} \mathcal{L}_{\text{MSE}}(h_{\text{HP}} * \hat{y}_{\text{lite}}, h_{\text{HP}} * y)\Big)$$

where:
- **`MSE` ($w_{\text{MSE}} = 1.00$, `NATIVE_DEFAULT_MSE_WEIGHT`):** Full-band sample-by-sample Mean Squared Error across $0\text{ Hz}$ to $24\text{ kHz}$. Anchors fundamental bass frequencies ($30\text{--}500\text{ Hz}$), comb-filter notch inversions ($550\text{--}800\text{ Hz}$), overall phase alignment, and dynamic transient envelopes.
- **`FIR HP` ($w_{\text{HP}} = 2.50$, `NATIVE_DEFAULT_FIR_HP_WEIGHT`):** Convolves both predicted $\hat{y}[n]$ and target $y[n]$ with a Type-I linear-phase **255-tap Blackman-windowed FIR high-pass filter** (`padding="valid"`) on the GPU (`F.conv1d`) and weights the high-passed MSE by $2.50\times$.
- **`FIR HP FC` ($f_c = 1200.0\text{ Hz}$, `NATIVE_DEFAULT_FIR_HP_FC`):** The $-6\text{ dB}$ cutoff frequency of the 255-tap high-pass kernel. $1200\text{ Hz}$ matches the physical string overtone corner frequency (`f_corner` in [`docs/dsp_synthesis_engine.md`](../../dsp_synthesis_engine.md)), blocking the massive $<500\text{ Hz}$ fundamental energy ($>58\text{ dB}$ Blackman stopband rejection) so the optimizer receives a dedicated, high-gain gradient microscope across the $1.2\text{--}14\text{ kHz}$ band where passive RLC pickup resonances ($1.8\text{--}4.5\text{ kHz}$), Neodymium presence peaks, eddy-current damping slopes, and active treble shelves reside.

### 2.1 255-Tap Blackman FIR High-Pass Formulation (`design_sharp_fir_highpass`)

For a filter with $M = 255$ taps and cutoff frequency $f_c = 1200.0\text{ Hz}$ at sampling rate $f_s = 48,000\text{ Hz}$:

1. **Normalized Cutoff:** $\omega_c = 2\pi \dfrac{f_c}{f_s}$.
2. **Symmetric Time Indices:** $m = n - \dfrac{M - 1}{2}$ for $n \in [0, M-1]$.
3. **Ideal High-Pass Impulse Response:**
   $$h_{\text{ideal}}[m] = \text{sinc}(m) - \frac{\omega_c}{\pi} \text{sinc}\left(\frac{\omega_c m}{\pi}\right)$$
4. **Blackman Tapering (`np.blackman`):**
   $$w[n] = 0.42 - 0.50 \cos\left(\frac{2\pi n}{M - 1}\right) + 0.08 \cos\left(\frac{4\pi n}{M - 1}\right), \quad h_{\text{win}}[n] = h_{\text{ideal}}[n] \cdot w[n]$$
5. **Unit Nyquist Gain Normalization:**
   $$S_{\text{Nyq}} = \sum_{n=0}^{M-1} h_{\text{win}}[n] \cdot \cos(\pi m), \quad h_{\text{HP}}[n] = \frac{h_{\text{win}}[n]}{S_{\text{Nyq}}}$$

Because a symmetric odd-tap FIR filter is strictly **linear-phase** and evaluated in the time domain via `F.conv1d(..., padding="valid")`, `FIR HP` enforces **both exact magnitude and exact sample-accurate phase** above $1200\text{ Hz}$ in a single fused GPU GEMM call—avoiding the phase blindness and host-sync overhead of FFT/STFT losses.

---

## 3. Parseval Linearity Regularizer (`Linearity Reg`) & Three-Way Convergence Loop

### 3.1 What `Linearity Reg` (`sine_regularizer = True`) Does

A 23-layer WaveNet A2 network contains non-linear `LeakyReLU(0.01)` activations at every residual layer. When trained solely on complex musical stems and high-level sweeps, the network can minimize waveform MSE on loud transients while leaving subtle zero-crossing kinks (crossover distortion), intermodulation products, or static EQ/phase drift in the small-signal regime (where notes ring out and decay).

Before training begins ([`engine.py` Step 9c](../../../src/allomorph/trainer/native/engine.py#L434-L586)), **Linearity Reg** synthesizes **$170.0\text{ seconds}$** ($8,160,000\text{ samples}$) of clean, analytical reference signals and passes them directly through the exact **linear** electroacoustic + MNA circuit solver (`simulate_voicing_dsp(..., apply_saturation=False, apply_dither=False)`) for both the Source Digital Twin and the Target Digital Twin:

1. **3 Multi-Amplitude Continuous Tilted Log Chirps ($3 \times 10\text{ s} = 30\text{ s}$):**
   - Sweeps $20\text{ Hz} \to 14,000\text{ Hz}$ with a $2000\text{ Hz}$ string-tilt envelope at three amplitude tiers: **`0.40`**, **`0.20`**, and **`0.08`**.
   - Provides dense, continuous frequency-response and phase supervision across the entire audio spectrum at multiple operating levels.
2. **140 Multi-Amplitude Pure-Sine Probes ($28\text{ frequencies} \times 5\text{ amplitudes} \times 1.0\text{ s} = 140\text{ s}$):**
   - **28 Frequencies (`NATIVE_DEFAULT_SINE_FREQUENCIES`):** `(40.0, 80.0, 81.4, 110.0, 160.0, 162.8, 220.0, 300.0, 329.6, 330.0, 440.0, 500.0, 550.0, 586.2, 600.0, 650.0, 659.2, 700.0, 880.0, 1000.0, 1100.0, 1318.5, 1320.0, 1760.0, 2000.0, 3000.0, 5000.0, 7000.0)` Hz. Specifically targets open-string fundamentals ($E_1/A_1/D_2/G_2$ harmonics), dual-pickup comb-filter notch basins ($500\text{--}700\text{ Hz}$), and passive RLC resonant peaks ($1.76\text{--}7.0\text{ kHz}$).
   - **5 Amplitude Tiers (`NATIVE_DEFAULT_SINE_AMPLITUDES`):** **`(0.25, 0.10, 0.05, 0.03, 0.008)`**, spanning from normal playing level down to $-42\text{ dBFS}$ (`0.008`) to force clean linear behavior across small-signal note tails.
3. **Broadband Gain Coherence (`probe_gain_correction`):**
   - Because the musical training stems (`x_train`, `y_train`) include non-linear magnetic/circuit saturation and are normalized to the target RMS standard ($-18.0\text{ dBFS}$ internal target scaling), there is a small broadband gain offset ($0.5\text{--}2.5\text{ dB}$) between the full-track stem RMS ratio and the raw unnormalized linear DSP ratio.
   - Linearity Reg computes `probe_gain_correction = stem_gain_ratio / raw_dsp_gain_ratio` (logged as `gain align: +X.XX dB`) and scales the target probes to match `y_train` with $100\%$ gain coherence so the optimizer never sees conflicting gain targets between musical stems and linear probes.
4. **Step-0 Wiener-Hopf Head Warm-Start (`warm_start_head`):**
   - Before Epoch 1, the `0.20`-amplitude linear calibration chirp (`cal_chirp_x`, `cal_chirp_y`) is passed through the frozen 23-layer dilated basis to solve the 16-tap `head_rechannel` weights in closed form via Tikhonov-regularized Cholesky normal equations, placing the model at unity gain and approximate linear EQ transfer at Step 0.
5. **In-VRAM Concatenation & Full-Track Permutation:**
   - The $170\text{ s}$ of paired linear probes `(x_sines_t, y_sines_t)` are concatenated directly onto `(x_train, y_train)` in GPU VRAM. Under epoch block permutation, every training epoch mixes real non-linear bass performances with pure single-frequency sines and chirps.

### 3.2 How `Linearity Reg`, `MSE + FIR HP`, and the `Cosine + Homotopy Schedule` Fit Together

The three components form a tightly coupled system where each resolves a failure mode of the others:

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ 1. DATA SPACE: Linearity Reg (170s Pure Sines + Tilted Chirps appended to Stems)      │
│    Provides clean single-frequency & swept targets from 40 Hz to 14 kHz across 5 amps  │
└──────────────────────────────────────────┬─────────────────────────────────────────────┘
                                           ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ 2. LOSS SPACE: Full-Band MSE (w=1.0) + 1200 Hz FIR High-Pass MSE (w=2.5)               │
│    • MSE (w=1.0) locks in the 40–1000 Hz fundamentals & comb-notch inversions          │
│    • FIR HP (w=2.5, fc=1200 Hz) magnifies the >1.2 kHz chirps, sines, & pick clank     │
│      while penalizing any spurious high-frequency harmonic buzz from low-freq sines!   │
└──────────────────────────────────────────┬─────────────────────────────────────────────┘
                                           ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ 3. TIME / OPTIMIZER SPACE: C^∞ Homotopy (Ep 1–75) + Cosine LR Annealing (Ep 5–120)     │
│    • Ep 1–75: Smooth SiLU-gated activation (tau: 0.02 -> 0) shapes global EQ & phase   │
│    • Ep 75–120: Exact LeakyReLU(0.01) + decaying Cosine LR (4e-3 -> 1e-6) freezes      │
│      zero-crossing kinks so FIR HP + Linearity Reg can polish out -60 dB fizz          │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

#### A. The Dual-Band Synergy Between `FIR HP (1200 Hz)` and `Linearity Reg`
`FIR HP` interacts with `Linearity Reg` in two complementary ways depending on whether a probe's frequency is **above** or **below** $f_c = 1200\text{ Hz}$:
1. **High-Frequency Probes ($f_0 \ge 1200\text{ Hz}$ — $1.32\text{--}7.0\text{ kHz}$ sines & $1.2\text{--}14\text{ kHz}$ chirp bands):**
   Because the log chirps have a physical string tilt and passive pickup targets roll off above their RLC resonance peak, raw high-frequency energy is much smaller than low-frequency energy. Without `FIR HP`, full-band `MSE` would spend $>99\%$ of its gradient budget fitting the $40\text{--}500\text{ Hz}$ sines. `FIR HP` blocks everything below $1200\text{ Hz}$ and multiplies the high-band error by $2.5\times$, forcing the WaveNet to nail the exact target RLC resonant peak height, $Q$ factor, and phase.
2. **Low-Frequency Probes ($f_0 < 1200\text{ Hz}$ — $40\text{--}880\text{ Hz}$ sines):**
   When a pure $110\text{ Hz}$ or $330\text{ Hz}$ sine wave from `Linearity Reg` passes through the linear target solver, the target has **zero energy above $1200\text{ Hz}$** ($h_{\text{HP}} * y_{\text{sine}} \equiv 0$). However, if the WaveNet's `LeakyReLU(0.01)` layers introduce tiny zero-crossing kinks (crossover distortion / high-order odd harmonics at $11 f_0, 13 f_0, 15 f_0 > 1200\text{ Hz}$), those spurious harmonics pass straight through the $1200\text{ Hz}$ `FIR HP` filter! Because the high-passed target is literally `0.0`, **`FIR HP` acts as a $2.5\times$-weighted Total Harmonic Distortion (THD) penalty** on every low-frequency sine wave in `Linearity Reg`.

#### B. How the `Cosine Schedule` + `$C^\infty$ Homotopy` Completes the Loop
Even with `Linearity Reg` and `FIR HP` penalizing high-frequency distortion and EQ error, training a sharp `LeakyReLU(0.01)` network at a constant or high learning rate causes **kink oscillation**: magnified $2.5\times$ by `FIR HP`, steep high-frequency gradients cause Adam updates to jump back and forth across the $x = 0$ bias threshold of individual WaveNet channels, creating a persistent high-frequency "fizz" plateau around $-30\text{ dB}$ ESR.

The two-phase schedule in [`engine.py`](../../../src/allomorph/trainer/native/engine.py#L711-L735) eliminates this:
1. **Phase 1 — Smooth Manifold & Filterbank Shaping (Epochs 1–75):**
   - The **$C^\infty$ Activation Homotopy** ($\tau: 0.02 \to 0.0$ via a cosine curve over $75\%$ of `min_epochs`) rounds off the `LeakyReLU` corner into a $C^\infty$ `SiLU`-gated curve ($\psi_\tau(x) = 0.01x + 0.99\tau\,\text{silu}(x/\tau)$) while the **Cosine Learning Rate** is in its high-to-mid range ($4\times 10^{-3} \to 1.5\times 10^{-3}$).
   - Gradients flow smoothly across $x = 0$ without kink chatter, allowing the 6,347-sample dilated filterbank to rapidly lock in the macro transfer function (comb-notch inversions, bass body, and RLC peak placement) driven by `MSE` + `FIR HP`.
2. **Phase 2 — Exact Hardware Kink-Locking & Micro-Polishing (Epochs 75–120):**
   - At Epoch 75 (`0.75 * min_epochs`), $\tau$ reaches `0.0`, locking the network into the exact `LeakyReLU(0.01)` activation executed by the Darkglass Anagram C++ engine.
   - Simultaneously, the **Cosine LR Schedule** ($T_{\max}=115$) smoothly decays the learning rate down toward $\eta_{\min} = 1.0\times 10^{-6}$ ($4,000\times$ smaller than peak LR), while the **In-VRAM EMA shadow weights** ($\beta = 0.995$) average out step-to-step variance.
   - With microscopic step sizes and `FIR HP` + `Linearity Reg` continuously penalizing any energy above $1200\text{ Hz}$ on low-frequency sines, the optimizer gently settles the channel bias terms so zero-crossings never chatter—driving validation ESR down to **$-37\text{ to }-40\text{ dB}$** with $\le 0.16\text{ dB}$ spectral band tracking error.

---

## 3.3 Historical Context: Why Normalized High-Pass ESR & Online MRSTFT Were Replaced

### The Small-Signal Singularity of Normalized High-Pass ESR

Previously, high-frequency emphasis was attempted via `NormalizedHighPassESRLoss`, which computes high-pass error normalized by target band energy:
$$\mathcal{L}_{\text{hp}}(y, \hat{y}) = \frac{\sum_n (h * (y - \hat{y}))[n]^2}{\sum_n (h * y)[n]^2 + \epsilon}$$

While effective on full-amplitude plucks, this formulation exhibits an electroacoustic singularity on decaying harmonics, quiet passages, and chords. As target high-frequency energy decays toward the noise floor ($E_{\text{target}} \to 10^{-7}$), the denominator collapses. The resulting gradient:
$$\frac{\partial \mathcal{L}_{\text{hp}}}{\partial \hat{y}} \propto \frac{1}{\sum (h * y)^2 + \epsilon}$$
explodes by several orders of magnitude. Because a WaveNet contains non-linear activations, the optimizer drives internal convolutional weights into heavy saturation to generate spurious upper harmonics out of thin air, producing severe audible distortion on small signals. Unnormalized `FIR HP` ($w_{\text{HP}} = 2.50$) completely avoids denominator division, scaling gradients smoothly with signal amplitude.

### Offline Multi-Resolution STFT (`MultiResolutionSTFTLoss`) for Telemetry Only

`MultiResolutionSTFTLoss` in [`loss.py`](../../../src/allomorph/trainer/native/loss.py) is retained strictly for **offline post-training telemetry** (`compute_baseline_mrstft`) and stock engine diagnostics, and is never included in the active native training loop:

| Component | Default Weight | Target Domain | Acoustic & Hardware Purpose |
| :--- | :---: | :---: | :--- |
| **Waveform MSE (`MSE`)** | `1.00` | Time Domain | Anchors fundamental phase, sub-bass excursion, comb-notch inversion, and DC stability |
| **FIR Highpass (`FIR HP`)** | `2.50` ($f_c = 1200\text{ Hz}$) | Time Domain Conv1D | 255-tap Blackman FIR (`padding="valid"`). Enforces exact magnitude and phase above $1200\text{ Hz}$ and penalizes high-order harmonic distortion on low-frequency `Linearity Reg` sines. |
| **Linearity Reg (`sine_regularizer`)** | `Enabled` ($170\text{ s}$) | In-VRAM Training Buffer | 3 tilted log chirps ($20\text{ Hz}\to 14\text{ kHz}$) + $28\text{ freqs} \times 5\text{ amps}$ pure sines ($40\text{--}7000\text{ Hz}$) with dynamic gain coherence (`probe_gain_correction`). |
| **MRSTFT Loss** | `0.00` (Offline only) | Multi-Scale Log-STFT | **Excluded from training:** Phase-blind log-STFT crushes differentiator taps on pure sines. Used strictly for post-training QA telemetry (`mrstft_loss`, `differential_mrstft`). |

> [!CAUTION]
> **Regression Prevention:** For complete forensic proofs of why online MRSTFT, untilted sweeps, independent LUFS matching, and `padding="same"` failed, consult [What Didn't Work: Forensic Analysis & Regression Prevention](what_did_not_work_and_regression_prevention.md).

---

## 4. Anti-Drift Pareto Early Stopping Architecture

### 4.1 Coupling Validation Loss with the 255-Tap Blackman FIR High-Pass Objective

During validation, the native engine evaluates both broadband time-domain MSE ($\text{MSE}_{\text{val}}$) and the 255-tap Blackman FIR high-pass MSE ($\mathcal{L}_{\text{fir\_val}}$, `padding="valid"`, $f_c = 1200\text{ Hz}$), forming the composite validation objective:
$$\mathcal{L}_{\text{val}}(t) = w_{\text{mse}} \cdot \text{MSE}_{\text{val}}(t) + w_{\text{fir}} \cdot \mathcal{L}_{\text{fir\_val}}(t)$$

### 4.2 Pareto Quality Metric

The native engine continuously monitors multi-objective validation loss via `compute_pareto_quality_score` (`drift_tolerance=0.15`, `drift_penalty_gain=2.5`) and `evaluate_plateau_stopping_condition` (`drift_exit_threshold=0.25`):
$$\mathcal{Q}(t) = \mathcal{L}_{\text{val}}(t) \cdot \left( 1.0 + 2.5 \cdot \max\left(0, \frac{\text{MSE}_{\text{val}}(t) - \text{MSE}_{\min}}{\text{MSE}_{\min}} - 0.15\right)^2 \right)$$

This allows the network up to $15\%$ broadband MSE trade-off post-warmup (`epoch >= min_epochs`) as it refines high-frequency differentiator taps on the 255-tap FIR high-pass objective, while quadratically penalizing excessive low-frequency waveform drift and halting if drift exceeds $25\%$.

### 4.3 In-Memory Checkpoint Restoration

When training completes or early stopping engages, the trainer restores the in-memory EMA-evaluated state dict corresponding to the optimal post-warmup Pareto score $\mathcal{Q}(t)$ with `model.activation_tau = 0.0` (exact `LeakyReLU(0.01)`):

```python
if best_state_dict is not None and best_epoch > 0:
    print(
        f"\n[Native Trainer] Restoring optimal Pareto checkpoint from epoch {best_epoch:03d} "
        f"(Val Loss: {best_val_loss:.5f}, Pareto Q: {best_pareto_score:.5f})"
    )
    model.load_state_dict(best_state_dict)
```

### 4.4 Unlucky Loss Spike Detection & Automatic Training Restart (CL-37)

In deep 23-layer dilated WaveNet architectures trained with `bf16-mixed` precision and sharp highpass FIR regularization, an occasional unlucky stochastic mini-batch alignment can trigger a catastrophic gradient explosion (e.g. `train_loss` jumping from $0.00004 \to 0.0343 \to 14.89 \to 109.72$ and validation ESR spiking from $-35.1\text{ dB}$ to $+16.2\text{ dB}$). Once the Adam second-moment accumulator ($v_t$) and EMA shadow weights (`ema_params`, $\beta = 0.995$) are polluted by a $10^4\times$ gradient spike, the remaining epochs are wasted slowly decaying the corrupted moments.

To guarantee clean, uninterrupted convergence without wasting GPU compute or falsely triggering on benign transient wobbles ($0.001\text{--}0.005$), the native engine evaluates `evaluate_loss_spike_condition` (`callbacks.py`) after every training epoch and validation pass:
- **Trigger Condition:** Fires immediately if `train_loss_epoch` or `mb_val` is non-finite (`NaN`/`Inf`), or if either exceeds `spike_ratio_threshold` ($500.0\times$, `NATIVE_DEFAULT_SPIKE_RATIO_THRESHOLD`) times its running minimum (`min_train_loss` or `min_val_loss_seen`) while above the absolute floor $\mathcal{L}_{\text{floor}} = 0.02$.
- **Automatic Recovery:** Aborts the corrupted trajectory immediately and restarts training from Epoch 1 (up to `NATIVE_DEFAULT_MAX_SPIKE_RESTARTS = 3` attempts) with fresh WaveNet A2 weights, fresh closed-form Tikhonov `warm_start_head` initialization, reset Adam moments and EMA shadow buffers, and a $0.85\times$ learning rate backoff ($\eta_{\text{attempt}} = 0.85^{\text{attempt}} \cdot \eta_0$) to prevent re-entering the same unstable trajectory.

---

## 5. Native vs. Stock Hyperparameter Specifications

The native engine operates under an isolated hyperparameter regime tuned for rapid convergence, locked 100% GPU compute saturation, and distortion-free electroacoustic tracking:

| Hyperparameter | Upstream Stock Engine (`STOCK_DEFAULT_MAX_EPOCHS`) | Native Engine (`NATIVE_DEFAULT_*`) | Rationale |
| :--- | :---: | :---: | :--- |
| **Linearity Regularizer**| None (standard upstream NAM) | **GPU GEMM FIR ($w=2.5, f_c=1200\text{Hz}$)** | Eliminates LeakyReLU zero-crossing rectification buzzing on harmonics |
| **Spectral Loss** | None (standard upstream NAM) | **Pure Time-Domain MSE (`0.00` MRSTFT)** | Prevents differentiator crushing; eliminates ROCm hipFFT launch stalls |
| **Max Epoch Schedule** | `200` | **`120`** | Pareto-optimal convergence achieved in 105–115 epochs with $\le -36.6\text{ dB}$ ESR |
| **Min Warmup Epochs** | N/A | **`100`** | Guarantees full differentiator tap formation before early stopping |
| **Scheduler & Annealing**| Exponential ($0.994^t$, Upstream) | **Cosine Annealing ($T_{\max}=115, \eta_{\min}=10^{-6}$)**| Smoothly anneals learning rate to $10^{-6}$ to extinguish small-signal buzz |
| **Patience Window** | N/A | **`15`** | Plateau patience evaluated on composite val_loss |
| **Optimizer & LR** | Adam ($1.0\text{e-2}$, Upstream) | **Adam ($4.0\text{e-3}$)** | Stable step size preventing parameter oscillation around narrow spectral basins |
| **L2 Weight Decay** | `0.0` | **`3.17e-7`** | Matches upstream weight decay regularization standard |
| **Slice Length ($ny$)** | $8,192\text{ samples}$ | **$8,192\text{ samples}$** | Captures $170.7\text{ ms}$ continuous audio windows |
| **Batch Size** | `auto` (Upstream) | **`16` (Locked)** | Fits inside 64 MB Infinity Cache; prevents cache spilling cliffs |
| **Precision** | `auto` (`bf16-mixed`) | **`auto` (`bf16-mixed`)** | Universal bfloat16 hardware acceleration with FP32 fallback |

*For the complete mathematical proof of the warmup floor, throughput scaling, and Adam velocity ceiling, see [Native Engine GPU Throughput Profiling & Hardware Acceleration](gpu_throughput_optimization.md) and [What Didn't Work: Forensic Analysis & Regression Prevention](what_did_not_work_and_regression_prevention.md).*

