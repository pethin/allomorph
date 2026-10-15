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

## 2. Zero-Phase Blackman FIR High-Pass & Multi-Harmonic Sine Regularizer

To isolate harmonic bands without phase distortion, the native engine designs Type-I linear-phase Blackman-windowed FIR filters directly in pure NumPy (`design_fir_highpass` in `loss.py`, $M = 255$ taps in `engine.py`, `padding="valid"`):

### 2.1 Formulation

For a filter with $M$ taps (where $M$ is odd, $M=255$ in `engine.py` and $M=31$ default in `loss.py`) and cutoff frequency $f_c = 1200.0\text{ Hz}$ at sampling rate $f_s = 48,000\text{ Hz}$:

1. **Normalized Cutoff:** $\omega_c = 2\pi \dfrac{f_c}{f_s}$.
2. **Symmetric Time Indices:** $m = n - \dfrac{M - 1}{2}$ for $n \in [0, M-1]$.
3. **Ideal High-Pass Impulse Response:**
   $$h_{\text{ideal}}[m] = \text{sinc}(m) - \frac{\omega_c}{\pi} \text{sinc}\left(\frac{\omega_c m}{\pi}\right)$$
4. **Blackman Tapering (`np.blackman` in `engine.py` / Hamming in `loss.py`):**
   $$w[n] = 0.42 - 0.50 \cos\left(\frac{2\pi n}{M - 1}\right) + 0.08 \cos\left(\frac{4\pi n}{M - 1}\right)$$
   $$h_{\text{win}}[n] = h_{\text{ideal}}[n] \cdot w[n]$$
5. **Unit Nyquist Gain Normalization:**
   $$S_{\text{Nyq}} = \sum_{n=0}^{M-1} h_{\text{win}}[n] \cdot \cos(\pi m)$$
   $$h[n] = \frac{h_{\text{win}}[n]}{S_{\text{Nyq}}}$$

Combined with the 5-harmonic small-signal sine regularizer (`sine_frequencies = (659.25, 1318.5, 1977.8, 2637.0, 3296.3)` Hz, `sine_amplitudes = (0.25, 0.10, 0.05, 0.03, 0.008)`), this guarantees zero phase distortion, exact $0.0\text{ dB}$ gain at Nyquist, and suppression of `LeakyReLU(0.01)` zero-crossing buzz.

---

## 3. Multi-Resolution STFT (MRSTFT) & Composite Loss Formulation

### 3.1 The Small-Signal Singularity of Normalized High-Pass ESR

Previously, high-frequency emphasis was attempted via `NormalizedHighPassESRLoss`, which computes high-pass error normalized by target band energy:
$$\mathcal{L}_{\text{hp}}(y, \hat{y}) = \frac{\sum_n (h * (y - \hat{y}))[n]^2}{\sum_n (h * y)[n]^2 + \epsilon}$$

While effective on full-amplitude plucks, this formulation exhibits an electroacoustic singularity on decaying harmonics, quiet passages, and chords. As target high-frequency energy decays toward the noise floor ($E_{\text{target}} \to 10^{-7}$), the denominator collapses. The resulting gradient:
$$\frac{\partial \mathcal{L}_{\text{hp}}}{\partial \hat{y}} \propto \frac{1}{\sum (h * y)^2 + \epsilon}$$
explodes by several orders of magnitude. 

Because a WaveNet contains non-linear gated activations ($\tanh(Wx) \odot \sigma(Vx)$), the optimizer discovers an unphysical shortcut: it drives internal convolutional weights into heavy non-linear saturation to generate spurious upper harmonics out of thin air to satisfy the high-pass loss. This produced severe audible distortion on small signals (up to **`16.4%`** THD on native and **`67.5%`** on stock engines at $-40\text{ dBFS}$).

### 3.2 Multi-Resolution STFT Loss Formulation

To overcome this singularity while preserving full high-frequency deconvolution accuracy ($+18.6\text{ dB}$ at $10\text{ kHz}$), the native engine deploys `MultiResolutionSTFTLoss`. MRSTFT operates in the log-spectral domain, where a $10\text{ dB}$ error at $10\text{ kHz}$ incurs the exact same numerical penalty as a $10\text{ dB}$ error at $100\text{ Hz}$, providing balanced gradient pressure to develop linear differentiator taps without non-linear distortion.

For $M$ simultaneous Short-Time Fourier Transform (STFT) resolutions:
$$\mathcal{L}_{\text{MRSTFT}}(y, \hat{y}) = \frac{1}{M} \sum_{m=1}^M \left( \mathcal{L}_{\text{sc}}^{(m)}(y, \hat{y}) + \mathcal{L}_{\text{log\_mag}}^{(m)}(y, \hat{y}) \right)$$

1. **Spectral Convergence Loss ($\mathcal{L}_{\text{sc}}$):**
   $$\mathcal{L}_{\text{sc}}(y, \hat{y}) = \frac{\| |Y| - |\hat{Y}| \|_F}{\| |Y| \|_F + \epsilon}$$
2. **Log-Magnitude L1 Loss ($\mathcal{L}_{\text{log\_mag}}$):**
   $$\mathcal{L}_{\text{log\_mag}}(y, \hat{y}) = \frac{1}{K} \sum_{k=0}^{K-1} \left| \log |Y_k| - \log |\hat{Y}_k| \right|$$

#### Resolution Multi-Scale Grid:
- **Resolution 1 (Bass / Fundamental):** FFT size $N=2048$, hop size $H=240$, Hann window length $W=1200$ ($25.0\text{ ms}$).
- **Resolution 2 (Mid-Band / Body):** FFT size $N=1024$, hop size $H=120$, Hann window length $W=600$ ($12.5\text{ ms}$).
- **Resolution 3 (Transients / Air):** FFT size $N=512$, hop size $H=50$, Hann window length $W=240$ ($5.0\text{ ms}$).

### 3.3 Zero-Silence Gradient Proof ($C^\infty$ Power Clamping)

In naive STFT loss implementations, magnitude is computed via $\sqrt{\text{power} + \epsilon_{\text{small}}}$ with $\epsilon \approx 10^{-14}$. When computing $\log |\hat{Y}|$, the gradient evaluated at silence ($|\hat{Y}| \approx 10^{-7}$) explodes:
$$\frac{d}{d\hat{Y}} \log |\hat{Y}| = \frac{1}{|\hat{Y}|} \approx 10,000,000$$
This instability previously necessitated hard gradient norm clipping (`max_norm = 1.0`), which in turn suppressed tiny high-frequency gradient updates.

The Allomorph native engine adopts the bit-exact `auraloss` formulation:
$$P = \text{STFT}_{\text{real}}^2 + \text{STFT}_{\text{imag}}^2$$
$$P_{\text{clamped}} = \text{clamp}(P, \min=1.0\times 10^{-8})$$
$$|\hat{Y}| = \sqrt{P_{\text{clamped}}}$$

**Mathematical Proof of Zero Silence Gradient:**  
For any sub-audible noise or silence where $P < 1.0 \times 10^{-8}$, the clamp function enters its constant boundary:
$$\frac{d}{dP} \text{clamp}(P, \min=c) = 0 \quad \text{for } P < c$$
By the chain rule:
$$\frac{\partial \mathcal{L}_{\text{log\_mag}}}{\partial \hat{Y}} = \frac{\partial \mathcal{L}}{\partial |\hat{Y}|} \cdot \frac{1}{2\sqrt{P_{\text{clamped}}}} \cdot \frac{d P_{\text{clamped}}}{dP} \cdot \frac{\partial P}{\partial \hat{Y}} = \frac{\partial \mathcal{L}}{\partial |\hat{Y}|} \cdot \frac{1}{2\sqrt{c}} \cdot 0 \cdot \frac{\partial P}{\partial \hat{Y}} \equiv 0.0$$

Silence and sub-audible noise bins evaluate to mathematically exact **`0.0`** gradient. Gradients cannot explode, eliminating the need for gradient clipping and allowing the Adam optimizer to directly integrate high-frequency linear differentiator updates.

### 3.4 Composite Loss Weighting Allocation & FIR Regularizer

$$\mathcal{L}_{\text{total}} = w_{\text{mse}} \cdot \mathcal{L}_{\text{mse}} + w_{\text{fir}} \cdot \left( \mathcal{L}_{\text{fir\_full}} + w_{\text{lite}} \cdot \mathcal{L}_{\text{fir\_lite}} \right) + w_{\text{mrstft}} \cdot \mathcal{L}_{\text{mrstft}}$$

| Component | Default Weight | Target Domain | Acoustic & Hardware Purpose |
| :--- | :---: | :---: | :--- |
| **Waveform MSE** | `1.00` | Time Domain | Anchors fundamental phase, sub-bass excursion, and DC stability |
| **FIR Highpass GEMM** | `2.50` | Time Domain Conv1D | 255-tap Blackman FIR ($f_c = 1200\text{ Hz}$, `padding="valid"`). Forces WaveNet into linear subspaces across chime harmonics H2–H5 ($1318\text{--}3296\text{ Hz}$) without FFT launch bubbles. |
| **MRSTFT Loss** | `0.00` (Bypassed) | Multi-Scale Log-STFT | **Bypassed:** When evaluated on pure sines, log-STFT crushes differentiator taps, muffling bright voices (presence $-6.9\text{ dB}$, treble $-4.3\text{ dB}$). Furthermore, `torch.stft` micro-kernels fragment ROCm GPU pipelines, dropping compute usage to $50\%$. |
| **Pre-Emphasis MSE** | `0.00` (Bypassed) | Time Domain | Bypassed to prevent sub-bass phase distortion. |
| **High-Pass ESR** | `0.00` (Bypassed) | Band-Passed ESR | Bypassed to prevent small-signal denominator singularities ($E \to 10^{-7}$) that force non-linear diode distortion. |

> [!CAUTION]
> **Regression Prevention:** For complete forensic proofs of why MRSTFT, untilted sweeps, independent LUFS matching, and `padding="same"` failed, consult [What Didn't Work: Forensic Analysis & Regression Prevention](what_did_not_work_and_regression_prevention.md).

---

## 4. Anti-Drift Pareto Early Stopping Architecture

### 4.1 The Late-Epoch Waveform Drift Phenomenon

In long training schedules (e.g. 100–300 epochs), neural networks often over-fit subtle high-frequency residual noise at the expense of fundamental waveform phase alignment. This manifests as:
- Deteriorating full-band waveform MSE ($>+10\%\text{ drift}$)
- Flabby low-end response and phase incoherence
- Unnatural transient blurring

### 4.2 Pareto Quality Metric

The native engine continuously monitors multi-objective validation loss:
$$\mathcal{Q}(t) = \mathcal{L}_{\text{val}}(t) \cdot \left( 1.0 + \max\left(0, \frac{\text{MSE}(t) - \text{MSE}_{\text{min}}}{\text{MSE}_{\text{min}}}\right) \right)$$

If $\text{MSE}(t)$ drifts $>15\%$ above its historical minimum, the model incurs a quadratic Pareto penalty, halting training and triggering immediate checkpoint restoration.

### 4.3 In-Memory Checkpoint Restoration

When early stopping engages, the trainer does not save degraded late-epoch weights. Instead, it restores the in-memory state dict corresponding to the optimal $\mathcal{Q}(t)$:

```python
if best_state_dict is not None:
    print(
        f"\n[Native Trainer] Restoring optimal Pareto checkpoint from epoch {best_epoch:03d} (val_loss: {best_val_loss:.6f})"
    )
    model.load_state_dict(best_state_dict)
```

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

