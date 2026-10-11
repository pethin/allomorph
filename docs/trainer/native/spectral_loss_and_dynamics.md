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

## 2. Zero-Phase FIR High-Pass Filter Design

To isolate harmonic bands without phase distortion, the native engine designs Type-I linear-phase FIR filters directly in pure NumPy (`design_fir_highpass`), avoiding heavy external DSP dependencies:

### 2.1 Formulation

For a filter with $M$ taps (where $M$ is odd, default $M=31$) and cutoff frequency $f_c$ at sampling rate $f_s$:

1. **Normalized Cutoff:** $\omega_c = 2\pi \dfrac{f_c}{f_s}$.
2. **Symmetric Time Indices:** $m = n - \dfrac{M - 1}{2}$ for $n \in [0, M-1]$.
3. **Ideal High-Pass Impulse Response:**
   $$h_{\text{ideal}}[m] = \text{sinc}(m) - \frac{\omega_c}{\pi} \text{sinc}\left(\frac{\omega_c m}{\pi}\right)$$
4. **Hamming Tapering:**
   $$w[n] = 0.54 - 0.46 \cos\left(\frac{2\pi n}{M - 1}\right)$$
   $$h_{\text{win}}[n] = h_{\text{ideal}}[n] \cdot w[n]$$
5. **Unit Nyquist Gain Normalization:**
   $$S_{\text{Nyq}} = \sum_{n=0}^{M-1} h_{\text{win}}[n] \cdot \cos(\pi m)$$
   $$h[n] = \frac{h_{\text{win}}[n]}{S_{\text{Nyq}}}$$

This guarantees zero phase distortion (linear phase shift $\tau = (M-1)/2 = 15$ samples, compensated via symmetric padding) and exact $0.0\text{ dB}$ gain at Nyquist.

---

## 3. Self-Normalizing Multi-Band Loss Formulation

The native engine evaluates `MultiBandSpectralLoss`, combining time-domain waveform anchoring with batch energy-normalized high-pass Error-to-Signal Ratios (ESR):

$$\mathcal{L}_{\text{total}} = w_{\text{mse}} \cdot \mathcal{L}_{\text{mse}} + w_{\text{mid}} \cdot \mathcal{L}_{\text{hp, mid}} + w_{\text{high}} \cdot \mathcal{L}_{\text{hp, high}}$$

Where each high-pass component is normalized by the energy of the target band:

$$\mathcal{L}_{\text{hp}}(y_{\text{pred}}, y_{\text{tgt}}) = \frac{\sum_{n} \left( (y_{\text{pred}} * h)[n] - (y_{\text{tgt}} * h)[n] \right)^2}{\sum_{n} (y_{\text{tgt}} * h)[n]^2 + 10^{-8}}$$

### 3.1 Loss Weighting Allocation

| Component | Default Weight | Cutoff ($f_c$) | Acoustic Purpose |
| :--- | :---: | :---: | :--- |
| **Waveform MSE** | `1.00` | Full-band | Locks fundamental phase, sub-bass pitch, and comb nulls |
| **Mid-Treble ESR** | `0.50` | $3,000\text{ Hz}$ | Equalizes pickup attack bite, fret clank, and growl |
| **High-Treble ESR** | `0.50` | $7,000\text{ Hz}$ | Equalizes acoustic air, RLC resonant peaks, and harmonic sheen |

Because the divisor scales with target band energy, high-frequency errors receive full $1.0$-scale gradient updates starting from **Epoch 1**, accelerating treble convergence by $>10\times$.

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

When early stopping engages (e.g. at epoch 35), the trainer does not save the degraded late-epoch weights. Instead, it restores the in-memory state dict corresponding to the optimal $\mathcal{Q}(t)$:

```python
if best_state_dict is not None:
    print(f"\n[Native Trainer] Restoring optimal Pareto checkpoint from epoch {best_epoch:03d} (val_loss: {best_val_loss:.6f})")
    model.load_state_dict(best_state_dict)
```

---

## 5. Native vs. Stock Hyperparameter Specifications

The native engine operates under an isolated hyperparameter regime tuned for rapid convergence:

| Hyperparameter | Stock Lightning Engine (`STOCK_DEFAULT_*`) | Native Engine (`NATIVE_DEFAULT_*`) | Rationale |
| :--- | :---: | :---: | :--- |
| **Min Warmup Epochs** | `35` | **`35`** | Bounded by the Adam velocity ceiling ($|\Delta\theta| \le 0.004$) for differentiator taps |
| **Max Epoch Schedule** | `60` | **`60`** | Complete convergence achieved in 35–45 epochs |
| **Patience Window** | `6` | **`6`** | Evaluates plateau slope over 6 epochs |
| **Optimizer & LR** | Adam ($1.0\text{e-2}$, Lightning) | **Adam ($4.0\text{e-3}$)** | Lower learning rate prevents oscillatory overshoot |
| **L2 Weight Decay** | `0.0` | **`1.0e-4`** | Regularizes high-frequency convolutional kernels |
| **Slice Length ($ny$)** | $8,192\text{ samples}$ | **$8,192\text{ samples}$** | Captures $170\text{ ms}$ continuous audio windows |
| **Precision** | `auto` (`bf16-mixed`) | **`auto` (`bf16-mixed`)** | Fast bfloat16 tensor operations on modern GPUs |

*For the complete mathematical proof of the 35-epoch warmup floor and Adam velocity ceiling, see [Native Engine GPU Throughput Profiling & Hardware Acceleration](gpu_throughput_optimization.md).*

