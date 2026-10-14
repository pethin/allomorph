# Allomorph Clamp & Limit Registry

In accordance with **Guardrail 5.2.3 (Catalog-Driven Empirical Analysis for Clamps & Limits)**, all clamping, limiting, thresholding, and regularization mechanisms across the Allomorph codebase must be systematically analyzed across the entire catalog matrix to prevent arbitrary squashing of authentic electroacoustic dynamics.

This document serves as the master living registry tracking every active clamp, limit, threshold, and regularizer in the Forward Digital Twin pipeline.

---

## 1. Summary Status Matrix

| ID | Parameter / Mechanism | Location | Formulation | Catalog Extrema | Status |
|---|---|---|---|---|---|
| **CL-04** | Spatial Bridge Displacement (`alg4`) | `aperture.py:114` | Unified $C^\infty$ Partition-of-Unity Algebraic Limiter (`alg4`) | $[-9.11\text{ dB}, +9.11\text{ dB}]$ | ✅ **Evaluated & Verified** (Asymmetric $+12.0\text{ dB}$ boost / $-16.0\text{ dB}$ cut relative to $\eta_0 = 0.1082677$) |
| **CL-10** | Audio Drive Peak Normalization | `forward.py:679` | `min(in_peak * 0.687, 0.70)` | In-peak $\in (0.10, 1.00]$ | ✅ **Evaluated & Verified** (Linear scalar drive ceiling into non-linear ODE solver) |
| **CL-13** | Analog Power Rail Ceiling | `saturation.py:638` | $x / (1 + (|x|/V_{\text{sat}})^8)^{1/8}$ with $V_{\text{sat}} = 0.985$ | $\|x\| \le 1.0$ | ✅ **Evaluated & Verified** ($< 0.05\%$ compression for normal levels) |
| **CL-14** | Buffer Slew Rate Ceiling | `saturation.py:147` | $f_{\text{slew}} = 16\text{ kHz}$, `max_delta = 2*pi*f*vsat/sr` | Slew rate limit | ✅ **Evaluated & Verified** (Engages only on extreme pick spikes) |
| **CL-16** | True-Peak PCM Ceiling | `forward.py:553, 825` | `filtered * (CALIBRATION_PEAK_CEILING / max_peak)` | Maximum audio sample | ✅ **Evaluated & Verified** (Linear peak-based scaling paired with $-20.50\text{ LUFS}$ reference) |
| **CL-17** | 24-bit PCM Integer Clamp | `dsp/pcm.py` | `np.clip(x * 8388607, -8388608, 8388607)` | Output integer array | 🔒 **PCM Container Guard** (Prevents integer overflow wrap) |
| **CL-18** | Potentiometer Position Bounds | `parser.py` | `np.clip(pos, 0.0, 1.0)` | Wiper $[0.0, 1.0]$ | 🔒 **Physical Potentiometer Wiper Datum** |
| **CL-19** | Multi-Pickup Coherence Decay Sigmoid | `aperture.py:322`, `dataframe.py` | $0.5 \cdot (1 - \tanh((f - f_{\text{mid}}) / f_{\text{sigma}}))$ | Smooth $C^\infty$ blend | ✅ **Evaluated & Verified** (Guardrail 5.1.2: engages when $\lambda \le d$) |
| **CL-20** | Inharmonicity Frequency Floor | `strings.py` | $\max(f_0, 15.0\text{ Hz})$ | Fundamental $f_0$ | ✅ **Evaluated & Verified** (Prevents $\log_2(0)$ divergence) |
| **CL-21** | Multi-Rate Anti-Aliasing Decimation Mask | `saturation.py:258` | `1.0 - cinf_smoothstep((f - 18000) / 6000)` | Transition $[18\text{ kHz}, 24\text{ kHz}]$ | ✅ **Evaluated & Verified** ($C^\infty$ decimation lowpass, vanishes all boundary derivatives at Nyquist) |
| **CL-22** | Dynamic Lenz Saturation Core Drag | `saturation.py:85, 113` | Softplus excess ($\alpha=16.0$) + $V_{\text{sat}} \cdot \tanh(\text{excess}_{\text{raw}}/V_{\text{sat}})$ | In-core signal envelope $e \ge 0$ | ✅ **Evaluated & Verified** ($C^\infty$ softplus + rational drag $1 / (1 + \dots \cdot \text{excess})$) |
| **CL-23** | Homomorphic Real-Cepstrum FIR Tail Window & Power Floor | `dsp/fir.py:37, 46` | Power floor $10^{-12}$ ($-120\text{ dB}$); `1.0 - cinf_smoothstep(t)` on trailing 15% taps | Normalized tap index $t \in [0, 1]$ | ✅ **Evaluated & Verified** ($C^\infty$ mollifier; $-120\text{ dB}$ floor preserves deep tone-rolled stopbands) |
| **CL-26** | Luthier Pickup Isolation Leveling | `aperture.py:168`, `forward.py:617` | $M \cdot \tanh(\text{excess} / M)$ with shifted softplus ($\alpha=2.0, M=6.0\text{ dB}$), mean-normalized in blends | $[0.00\text{ dB}, +5.26\text{ dB}]$ | ✅ **Evaluated & Verified** (Mean-normalized across active branches in multi-pickup blends) |
| **CL-27** | Dynamic BS.1770-4 LUFS Matching | `forward.py`, `dsp/loudness.py` | ITU-R BS.1770-4 / EBU R128 dual-gated integrated loudness ($\Delta G = L_{\text{tgt}} - L_{\text{wet}}$) | Target $-20.50\text{ LUFS}$ | ✅ **Evaluated & Verified** (Eliminates passive tone capacitor RMS loudness skew) |
| **CL-28** | 4x Oversampled True-Peak Ceiling | `forward.py`, `dsp/loudness.py` | 4x zero-padded FFT sinc interpolation, linear scale if $tp > 0.9900$ | $tp \le 0.9900$ ($-0.087\text{ dBFS}$) | ✅ **Evaluated & Verified** (Zero DAC reconstruction inter-sample clipping) |
| **CL-29** | Dynamic Magnetic Aperture Bloom | `saturation.py`, `parser.py` | $\kappa_{\text{ap}} \cdot \text{excess} \cdot \tanh(x/V_{\text{sat}})$ | $\kappa_{\text{ap}} \in [0.000, 0.045]$ | ✅ **Evaluated & Verified** (Alnico/Ceramic displacement aperture expansion, exact $0.00$ on small signals) |
| **CL-30** | Schedule-Driven Natural Convergence | `trainer/core.py`, `trainer/constants.py` | Cosine Annealing ($T_{\max}=115$, $\eta_{\min}=10^{-6}$) + Anti-Drift Pareto early exit | All target voicings | ✅ **Evaluated & Verified** (Deep annealing to micro-step basin) |
| **CL-31** | Dynamic Hardware Execution Tuning | `trainer/core.py`, `trainer/constants.py` | VRAM-scaled batch size ($8/16/32$), auto precision (`bf16-mixed`/`16-mixed`/`32-true`) | GPU / MPS / CPU | ✅ **Evaluated & Verified** (Maintains $[35, 55]$ gradient steps/epoch) |

---

## 2. Evaluated & Verified Forward Digital Twin Invariants

The following mechanisms have been audited and verified across the Forward Digital Twin catalog:

1. **Spatial Bridge Excursion Limiter (`alg4`, CL-04):**
   Unified partition-of-unity algebraic formulation: $\Delta G_{\text{soft}} = dg / (1 + (dg / g_{\text{eff}})^4)^{0.25}$ with $g_{\text{eff}} = \sigma(dg) \cdot 12 + (1 - \sigma(dg)) \cdot 16$ and $\sigma(dg) = 0.5(1 + \tanh(dg / 4.0))$. Eliminates piecewise boundary kinks, allowing Mudbucker neck ($+9.11\text{ dB}$) and close bridge pickups ($-9.11\text{ dB}$) to retain authentic fundamental standing-wave excursion via the $220\text{ Hz}$ low-shelf $H_{\text{pos}}(f)$ with zero artificial high-frequency counter-shelves.
2. **Audio Drive Peak Normalization (CL-10):**
   ```python
   target_drive_peak = min(in_peak * 0.687, 0.70)
   audio_mono = (audio_mono / max(in_peak, 1e-9)) * target_drive_peak
   ```
   When `in_peak > 0.10` (full-scale audio playback), this linearly scales the audio peak into the non-linear ODE solver so forte plucks experience authentic magnetic saturation while preserving small-signal linearity ($\le 0.10$).
3. **Multi-Rate Anti-Aliasing Decimation Mask (CL-21):**
   $C^\infty$ lowpass filter mask evaluated via $1.0 - S_\infty((f - 18\text{ kHz}) / 6\text{ kHz})$ combined with exact $1/m$ polyphase decimation normalization (`out = np.fft.irfft(Y_down, n_sig) / float(m)`). All derivatives vanish identically at $18\text{ kHz}$ and Nyquist ($24\text{ kHz}$), eliminating $O(1/n^2)$ algebraic boundary truncation leakage and guaranteeing $0.00\text{ dB}$ gain parity across `oversample = 1, 2, 4`.
4. **Dynamic Lenz Saturation Core Drag (CL-22):**
   Evaluated in ODE state loops via $C^\infty$ softplus excess ($\alpha=16.0$) with $V_{\text{sat}} \cdot \tanh(\text{excess}_{\text{raw}}/V_{\text{sat}})$ and strictly positive rational velocity drag $1.0 / (1.0 + \dots \cdot \text{excess})$. Eliminates $C^0$ conditional slope kinks (`if e > vsat:`) while ensuring smooth differentiability across the core saturation boundary.
5. **Homomorphic Real-Cepstrum FIR Synthesis Tail Window & Power Floor (CL-23):**
   Cepstral power regularization floor $\epsilon = 10^{-12}$ ($-120\text{ dB}$) matches the visualizer's $10^{-6}$ amplitude floor so deep tone-rolled passive circuits (`precision_dub`, `jazz_neck_warm`) retain their authentic stopband roll-off without premature $-80\text{ dB}$ shelving. Tail windowing across the final 15% of FIR taps uses $1.0 - S_\infty(t)$, vanishing all boundary derivatives at the window transition ($t=0$) and buffer end ($t=1$).
6. **True-Peak PCM Linear Peak Scaling (CL-16 & CL-28):**
   Peak-based linear whole-file scaling (`filtered * (CALIBRATION_PEAK_CEILING / max_peak)`) is paired with the calibrated $-20.50\text{ LUFS}$ excitation reference standard (providing $> 20.4\text{ dB}$ crest factor headroom), guaranteeing that 100% of target voicings pass through cleanly below $-0.09\text{ dBFS}$ ($0.9900$) with zero double-compression.
7. **Luthier Individual Pickup Isolation Leveling (CL-26):**
   Models authentic luthier setup in multi-pickup blends (raising bridge pickups closer to strings to compensate for fractional standing-wave displacement $\eta = x/L$). Formulated as a smooth, shifted softplus ($\alpha = 2.0$) with asymptotic $\tanh$ ceiling ($M = 6.0\text{ dB}$) and mean-normalized across active branches ($\tilde{K}_{\text{iso}, i} = K_{\text{iso}, i} / \overline{K}_{\text{iso}}$) in composite blends (`len(active_branches) > 1`).
8. **Schedule-Driven Natural Convergence Architecture (CL-30):**
   Deterministic Cosine Annealing learning rate schedule ($T_{\max} = 115$ epochs, warmup floor $100$ epochs, $\eta_{\min} = 10^{-6}$) paired with Anti-Drift Pareto early stopping on validation loss.
9. **Dynamic Hardware Execution Tuning (CL-31):**
   Auto-resolves batch size ($8/16/32$), precision (`bf16-mixed`, `16-mixed`, `32-true`), and zero-copy in-VRAM slicing based on detected accelerator and VRAM headroom.

