# Allomorph Clamp & Limit Registry

In accordance with **Guardrail 5.2.3 (Catalog-Driven Empirical Analysis for Clamps & Limits)**, all clamping, limiting, thresholding, and regularization mechanisms across the Allomorph codebase must be systematically analyzed across the entire catalog matrix (735 playable source-to-target pairs) to prevent arbitrary squashing of authentic electroacoustic dynamics.

This document serves as the master living registry tracking every clamp, limit, threshold, and regularizer in the pipeline.

---

## 1. Summary Status Matrix

| ID | Parameter / Mechanism | Location | Formulation | Catalog Extrema | Status |
|---|---|---|---|---|---|
| **CL-01** | Scale Tension Snap Zero-Center | `prefilter.py:456`, `dataframe.py:221` | `if ΔL <= 0: H=1.0` else `3.5*tanh(1.8*ΔL/14.0)` | $[0.00\text{ dB}, +2.51\text{ dB}]$ | ✅ **Evaluated & Verified** (Bit-exact $0.000\text{ dB}$ on $\Delta L \le 0$) |
| **CL-02** | Circuit Deconvolution Headroom | `solver.py:879` | `smooth_soft_knee_db(h_db, thresh=5.5, ceiling=8.0)` | $[0.00\text{ dB}, +6.77\text{ dB}]$ | ✅ **Evaluated & Verified** (Passive split-P reaches $+6.77\text{ dB}$) |
| **CL-03** | Differential Prefilter & Output Limiter | `solver.py:879`, `prefilter.py:253, 433` | `smooth_soft_knee_db(raw_db, thresh=6.0, ceiling=8.0)` | $[0.00\text{ dB}, +8.00\text{ dB}]$ | ✅ **Evaluated & Verified** (100% linear $< +6.0\text{ dB}$) |
| **CL-04** | Spatial Bridge Displacement (`alg4`) | `aperture.py:115` | Unified $C^\infty$ Partition-of-Unity Algebraic Limiter (`alg4`) | $[-9.11\text{ dB}, +9.11\text{ dB}]$ | ✅ **Evaluated & Verified** ($< 0.004\text{ dB}$ deviation from legacy piecewise curve, zero kinks) |
| **CL-05** | Saddle Witness-Point Coupling | `prefilter.py:301, 501` | Direct ratio $H_{\text{saddle, tgt}} / H_{\text{saddle, src}}$ | $[-1.76\text{ dB}, +1.38\text{ dB}]$ | ✅ **Evaluated & Verified** (Redundant `tanh` removed) |
| **CL-06** | Acoustic Aperture Positive Boost | `prefilter.py:253, 433` | `smooth_soft_knee_db(q_db, thresh=6.0, ceiling=g_max)` | $[0.00\text{ dB}, +4.89\text{ dB}]$ | ✅ **Evaluated & Verified** (100% linear for all magnetic pairs) |
| **CL-07** | String Damping Positive Boost | `strings.py:89` | `smooth_soft_knee_db(r_db, thresh=5.0, ceiling=8.0)` | $[0.00\text{ dB}, +2.54\text{ dB}]$ | ✅ **Evaluated & Verified** (100% linear for stainless clank) |
| **CL-08** | Acoustic Aperture Negative Cut | `prefilter.py:254, 434` | `-smooth_soft_knee_db(-q_db, thresh=10.0, ceiling=14.0)` | $[-8.10\text{ dB}, 0.00\text{ dB}]$ | ✅ **Evaluated & Verified** (100% linear across all 735 catalog pairs) |
| **CL-09** | String Damping Negative Cut | `strings.py:90` | `-smooth_soft_knee_db(-r_db, thresh=24.0, ceiling=36.0)` | $[-28.23\text{ dB}, 0.00\text{ dB}]$ | ✅ **Evaluated & Verified** (100% linear down to $-24\text{ dB}$, $-28.06\text{ dB}$ on heavy flats) |
| **CL-10** | Audio Drive Peak Normalization | `audio.py:54`, `simulation.py:207` | `min(in_peak * 0.687, 0.70)` | In-peak $\in [0.10, 1.00]$ | ⚠️ **Heuristic Scaling Constant** (Squashes forte drive peak) |
| **CL-11** | Differential Metallurgy Denominator | `simulation.py:938, 996` | `1.0 - min(0.85, tgt_vsat / src_vsat) + 0.15` | Ratio $\in [0.45, 1.00]$ | ℹ️ **Numerical Guard** (Prevents division by zero when $V_{\text{sat}}$ matches) |
| **CL-12** | Differential Circuit HF Shelving | `solver.py:881-890` | `excess_boost * (1 - cinf_smoothstep(...))` | Max $20\text{ kHz} = +1.49\text{ dB}$ | ✅ **Evaluated & Verified** ($C^\infty$ roll-off above $8\text{ kHz}$; Guardrail 5.3.6: $< +2.0\text{ dB}$ at $20\text{ kHz}$) |
| **CL-13** | Analog Power Rail Ceiling | `saturation.py:1001` | $x / (1 + (|x|/V_{\text{sat}})^8)^{1/8}$ with $V_{\text{sat}} = 0.985$ | $\|x\| \le 1.0$ | ✅ **Evaluated & Verified** ($< 0.05\%$ compression for normal levels) |
| **CL-14** | Buffer Slew Rate Ceiling | `saturation.py:147` | $f_{\text{slew}} = 16\text{ kHz}$, `max_delta = 2*pi*f*vsat/sr` | Slew rate limit | ✅ **Evaluated & Verified** (Engages only on extreme pick spikes) |
| **CL-15** | Sub-Audible DC Floor | `prefilter.py:263, 439` | $\max(\eta, 10^{-4})$, $\max(\cdot, 10^{-6})$ | $20\text{ Hz} \in [-5.85, +7.10]\text{ dB}$ | ✅ **Evaluated & Verified** (Guardrail 5.3.6: $[-12, +12]\text{ dB}$) |
| **CL-16** | True-Peak PCM Ceiling | `simulation.py:65`, `staging.py:18` | `CALIBRATION_PEAK_CEILING = 0.9900` ($-0.087\text{ dBFS}$) | Maximum audio sample | 🔒 **DAC Hardware Guard** (Prevents inter-sample clipping) |
| **CL-17** | 24-bit PCM Integer Clamp | `dsp.py:128`, `simulation.py:534` | `np.clip(x * 8388608, -8388608, 8388607)` | Output integer array | 🔒 **PCM Container Guard** (Prevents integer overflow wrap) |
| **CL-18** | Potentiometer Position Bounds | `parser.py:260, 414, 425, 438` | `np.clip(pos, 0.0, 1.0)` | Wiper $[0.0, 1.0]$ | 🔒 **Physical Potentiometer Wiper Datum** |
| **CL-19** | Multi-Pickup Coherence Decay Sigmoid | `aperture.py:352-353`, `dataframe.py:200` | $0.5 \cdot (1 - \tanh((f - f_{\text{mid}}) / f_{\text{sigma}}))$ | Smooth $C^\infty$ blend | ✅ **Evaluated & Verified** (Guardrail 5.1.2: engages when $\lambda \le d$) |
| **CL-20** | Inharmonicity Frequency Floor | `strings.py:155` | $\max(f_0, 15.0\text{ Hz})$ | Fundamental $f_0$ | ✅ **Evaluated & Verified** (Prevents $\log_2(0)$ divergence) |
| **CL-21** | Multi-Rate Anti-Aliasing Decimation Mask | `saturation.py:100` | `1.0 - cinf_smoothstep((f - 18000) / 6000)` | Transition $[18\text{ kHz}, 24\text{ kHz}]$ | ✅ **Evaluated & Verified** ($C^\infty$ decimation lowpass, vanishes all boundary derivatives at Nyquist) |
| **CL-22** | Dynamic Lenz Saturation Core Drag | `saturation.py:114, 185` | Softplus excess ($\alpha=16.0$) + $w \cdot \tanh(\text{excess}/w)$ | In-core signal envelope $e \ge 0$ | ✅ **Evaluated & Verified** ($C^\infty$ softplus replacing $C^0$ conditional `if e > vsat:`) |
| **CL-23** | Homomorphic Real-Cepstrum FIR Tail Window | `dsp.py:108` | `1.0 - cinf_smoothstep(t)` on trailing 15% taps | Normalized tap index $t \in [0, 1]$ | ✅ **Evaluated & Verified** ($C^\infty$ mollifier replacing $C^1$ cosine; eliminates $O(1/n^2)$ boundary leakage) |
| **CL-24** | Acoustic Aperture Comb-Null De-Combing Taper | `prefilter.py:221, 401` | `1.0 - cinf_smoothstep((f - f_start) / (f_end - f_start))` | Transition $[f_{\text{start}}, f_{\text{end}}]$ | ✅ **Evaluated & Verified** ($C^\infty$ de-combing taper, zero slope kinks at boundary) |
| **CL-25** | Differential Circuit Deconvolution HF Taper | `solver.py:885-886` | `1.0 - cinf_smoothstep((f - 8000) / 12000)` | Transition $[8\text{ kHz}, 20\text{ kHz}]$ | ✅ **Evaluated & Verified** ($C^\infty$ high-frequency deconvolution shelf to $0.00\text{ dB}$) |

---

## 2. In-Depth Analysis of Identified Candidates

### Candidate 1: Acoustic Aperture Negative Cut (`prefilter.py:254, 434`)
- **Current Formulation:**
  ```python
  g_min_db = -14.0
  f_neg = g_min_db * np.tanh(q_db / g_min_db)
  ```
- **Catalog Finding:**
  - Across all 735 source-to-target pairs, 310 pairs require an acoustic cut deeper than $-6.0\text{ dB}$.
  - The deepest legitimate physical aperture cut in the catalog is **$-8.10\text{ dB}$** (e.g. Dingwall bridge/middle pickup into modern P/MM active voices).
  - Because `f_neg` uses an unthresholded `tanh` scaled by $-14.0\text{ dB}$, compression starts immediately at $0.0\text{ dB}$.
  - At $-8.10\text{ dB}$, $-14.0 \cdot \tanh(-8.10 / -14.0) = -7.31\text{ dB}$, causing **$0.80\text{ dB}$ of unphysical loss** on legitimate aperture transformations.
  - At $-6.00\text{ dB}$, it squashes to $-5.65\text{ dB}$ ($0.35\text{ dB}$ loss).
- **Recommendation:**
  Replace with strictly $C^\infty$ thresholded soft-knee limiting:
  ```python
  # Linear passband transparency down to -10.0 dB, smoothly saturating to -14.0 dB floor
  f_neg = -smooth_soft_knee_db(-q_db, thresh=10.0, ceiling=abs(g_min_db), alpha=2.0)
  ```
  All 735 pairs in the catalog (minimum $-8.10\text{ dB}$) will achieve **100% linear passband transparency** ($< 10^{-4}\text{ dB}$ error), while non-invertible comb nulls deeper than $-10.0\text{ dB}$ will be smoothly regularized without infinite attenuation.

---

### Candidate 2: String Viscoelastic Damping Negative Cut (`strings.py:90`)
- **Current Formulation:**
  ```python
  g_min_db = -36.0
  f_neg = g_min_db * np.tanh(r_db / g_min_db)
  ```
- **Catalog Finding:**
  - When transforming from stainless steel roundwounds (`roundwound_stainless_clank`) to vintage heavy flatwounds (`flatwound_vintage_heavy`), the analytical damping ratio reaches **$-28.23\text{ dB}$** at $20\text{ kHz}$.
  - Because `f_neg` uses unthresholded `tanh(r_db / -36.0)`, compression starts immediately at $0.0\text{ dB}$:
    - At $-10.0\text{ dB}$, $-36 \cdot \tanh(-10 / -36) = -9.77\text{ dB}$ ($0.23\text{ dB}$ loss).
    - At $-20.0\text{ dB}$, $-36 \cdot \tanh(-20 / -36) = -18.15\text{ dB}$ ($1.85\text{ dB}$ loss).
    - At $-28.23\text{ dB}$, $-36 \cdot \tanh(-28.23 / -36) = -23.59\text{ dB}$ (**$4.64\text{ dB}$ of unphysical loss**).
- **Recommendation:**
  Apply thresholded soft-knee limiting:
  ```python
  # Linear passband transparency down to -24.0 dB, smoothly saturating to -36.0 dB floor
  f_neg = -smooth_soft_knee_db(-r_db, thresh=24.0, ceiling=abs(g_min_db), alpha=2.0)
  ```
  Legitimate flatwound damping roll-off down to $-24\text{ dB}$ will remain 100% linear, with smooth asymptotic saturation preventing $-\infty$ numerical singularities.

---

### Candidate 3: Audio Drive Peak Normalization (`audio.py:54`, `simulation.py:207`)
- **Current Formulation:**
  ```python
  target_drive_peak = min(max_in * 0.687, 0.70)
  ```
- **Physical Context:**
  - $0.687$ corresponds to $-3.26\text{ dBFS}$, and $0.70$ corresponds to $-3.10\text{ dBFS}$.
  - When `max_in > 0.10` (full-scale audio playback), this scales the audio peak into the non-linear ODE solver so forte plucks experience $1.5\text{--}2.5\text{ dB}$ of magnetic saturation.
  - While physically motivated to avoid digital clipping before the saturation block, the hard clamp `min(..., 0.70)` introduces a non-smooth derivative ceiling if `max_in * 0.687 > 0.70` (i.e. `max_in > 1.018`).
- **Recommendation:**
  Ensure that when inputs approach full scale ($> 1.0$), scaling utilizes the $C^\infty$ algebraic limiter rather than a hard `min()`.

---

## 3. Evaluated & Verified Invariants

The following mechanisms have been fully audited and verified against the complete catalog matrix:

1. **Scale Tension Snap Zero-Center:**
   Verified bit-exact $1.0000$ ($0.000\text{ dB}$) across all frequencies whenever $L_{\text{src}} \ge L_{\text{tgt}}$. Bounded at $+2.51\text{ dB}$ for the maximum 30"-to-37" transformation.
2. **Circuit Deconvolution Headroom:**
   $6.0\text{ dB} \to 8.0\text{ dB}$ expansion allows passive split-P pickups to authentically reach their $+6.77\text{ dB}$ resonance peak when deconvolved against active or wideband reference circuits without premature clipping.
3. **Differential Prefilter & Output Limiter:**
   `smooth_soft_knee_db(raw_db, thresh=6.0, ceiling=8.0)` guarantees 100% linear passband transparency below $+6.0\text{ dB}$ while bounding peak differential gain to $\le +8.0\text{ dB}$.
4. **Spatial Bridge Excursion Limiter (`alg4`):**
   Unified partition-of-unity algebraic formulation: $\Delta G_{\text{soft}} = dg / (1 + (dg / g_{\text{eff}})^4)^{0.25}$ with $g_{\text{eff}} = \sigma(dg) \cdot 12 + (1 - \sigma(dg)) \cdot 16$. Eliminates piecewise boundary kinks, differing from legacy by $< 0.004\text{ dB}$ across all catalog pairs. Mudbucker neck ($+9.11\text{ dB}$) and 60s J bridge ($-9.11\text{ dB}$) retain full excursion while sub-audible DC gain remains strictly within $[-5.85\text{ dB}, +7.10\text{ dB}] \subset [-12\text{ dB}, +12\text{ dB}]$.
5. **Acoustic Aperture Positive Boost:**
   Thresholded at $+6.0\text{ dB}$ with $+8.0\text{ dB}$ ceiling ($+12.0\text{ dB}$ for direct sensors). All 735 magnetic pairs peak at $\le +4.89\text{ dB}$, ensuring 100% linear passband transparency.
6. **String Damping Positive Boost:**
   Thresholded at $+5.0\text{ dB}$ with $+8.0\text{ dB}$ ceiling. Stainless steel roundwound clank ($+2.54\text{ dB}$) passes with $< 2 \times 10^{-9}\text{ dB}$ error.
7. **Saddle Witness-Point Boundary Layer:**
   Redundant `tanh` squashing removed; ratio evaluated directly as $10^{r_{\text{saddle}}/20}$, verified bounded within $[-1.76\text{ dB}, +1.38\text{ dB}]$.
8. **Differential Circuit High-Frequency Shelving & Ultrasonic Rolloff:**
   $C^\infty$ roll-off evaluated via `cinf_smoothstep` above $8\text{ kHz}$. Verified across all instruments: $20\text{ kHz}$ ultrasonic gain is strictly between $-16.71\text{ dB}$ and $+1.49\text{ dB} < +2.0\text{ dB}$.
9. **Multi-Rate Anti-Aliasing Decimation Mask:**
   $C^\infty$ lowpass filter mask evaluated via $1.0 - S_\infty((f - 18\text{ kHz}) / 6\text{ kHz})$. All derivatives vanish identically at $18\text{ kHz}$ and Nyquist ($24\text{ kHz}$), eliminating $O(1/n^2)$ algebraic boundary truncation leakage during $2\times$ and $4\times$ decimation.
10. **Dynamic Lenz Saturation Core Drag:**
    Evaluated in ODE state loops via $C^\infty$ softplus excess ($\alpha=16.0$) with $w \cdot \tanh(\text{excess}/w)$. Eliminates $C^0$ conditional slope kinks (`if e > vsat:`) while ensuring smooth differentiability across the core saturation boundary.
11. **Homomorphic Real-Cepstrum FIR Synthesis Tail Window:**
    Tail windowing across the final 15% of FIR taps uses $1.0 - S_\infty(t)$ instead of half-cosine. Vanishes all boundary derivatives at the window transition ($t=0$) and buffer end ($t=1$), eliminating Gibbs truncation ripples in minimum-phase impulse responses.
12. **Acoustic Aperture Comb-Null De-Combing Tapers:**
    Smooth de-combing tapers in both upright bass piezo and standard magnetic prefilters use $1.0 - S_\infty(t)$, providing a strictly $C^\infty$ transition to bridge reference without piecewise slope kinks.
13. **Differential Circuit Deconvolution HF Taper:**
    High-frequency shelving from $8\text{ kHz}$ to $20\text{ kHz}$ uses $1.0 - S_\infty(t)$, smoothly damping out-of-band deconvolution to $0.00\text{ dB}$ with zero derivative discontinuity.
