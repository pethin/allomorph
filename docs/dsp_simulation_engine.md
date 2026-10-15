# Allomorph - Native Virtual Analog (VA) Circuit Simulation Engine

This document provides the technical architecture and execution reference for the native Virtual Analog (VA) circuit simulation engine implemented in [`src/allomorph/circuit/`](../src/allomorph/circuit/) and exposed via the `allomorph-sim` CLI entrypoint. The engine executes analytical Modified Nodal Analysis (MNA) circuit solving and state-space non-linear magnetic saturation at $>1500\times$ real-time speed.

> [!NOTE]
> - For the canonical physical equations of passive RLC networks, active preamps, impedance dispersion, and the 15-parameter non-linear magnetic metallurgy model, see [`docs/circuit_theory.md`](circuit_theory.md).
> - For spatial pickup aperture integrals, scale-length continuum physics, and arrival delays, see [`docs/aperture_math.md`](aperture_math.md).
> - For master string excitation, homomorphic real-cepstrum FIR synthesis, ITU-R BS.1770-4 gated loudness calibration, and 24-bit PCM WAV I/O, see [`docs/dsp_synthesis_engine.md`](dsp_synthesis_engine.md).

---

## 1. Engine Architecture & Branch-Wise Forward Signal Flow

In physical multi-pickup bass systems (e.g. Jazz Bass neck + bridge, PJ, P/MM, Dingwall multi-scale), vibrating strings induce independent magnetic flux variations across physically separated apertures before electrical current summation at the control harness. Allomorph executes this in [`simulate_instrument_voicing`](../src/allomorph/circuit/forward.py):

```
[Input Audio: 24-bit 48 kHz (audio/input.wav)]
          │
          ▼
┌────────────────────────────────────────────────────────┐
│ 1. Per-Branch Acoustic Aperture & Loaded Circuit FIR   │
│    • 2D cylindrical rod / 1D blade aperture kernel     │
│    • 24-point log wave-speed continuum broadcasting    │
│    • Close-bridge saddle stiffness (x < 75 mm)         │
│    • Vectorized MNA loaded circuit transfer (H_elec,i) │
│    • Causal integer arrival delay shifting (tau_i)     │
└────────────────────────────────────────────────────────┘
          │ (Per-pickup linear branch signals)
          ▼
┌────────────────────────────────────────────────────────┐
│ 2. Branch-Wise Non-Linear Transducer Dynamics          │
│    • Block 1: Oversampled (2x/4x) magnetic pole/core   │
│      ODE saturation (Lenz drag, Dahl hysteresis,       │
│      string pull, pitch sag, reluctance quack,         │
│      Steinmetz loss, back-EMF, 2f0 orbit, slew limit)  │
│    • Exact 1/m polyphase decimation with C^inf mask    │
│    • Block 2: Active pickup internal op-amp dynamics   │
│      (x_series 2.40V vs classic 0.50V) with order-8    │
│      branchless fsqrt rail limiter                     │
│    • Bit-exact linear bypass for small signals <= 0.10 │
└────────────────────────────────────────────────────────┘
          │ (Saturated branch signals)
          ▼
┌────────────────────────────────────────────────────────┐
│ 3. Current Summation & Spatial Cross-Coherence         │
│    • Time-domain branch sum with mean-normalized       │
│      luthier isolation weights (K_iso)                 │
│    • Minimum-phase spatial coherence filter (N=1024)   │
│      when inter-pickup delay delta_samples > 0         │
└────────────────────────────────────────────────────────┘
          │
          ▼
┌────────────────────────────────────────────────────────┐
│ 4. Downstream Preamp EQ & String Wrap Damping          │
│    • Active onboard preamp EQ (flat DC H(0) >= 1.0)    │
│    • Viscoelastic string wrap damping (H_wrap)         │
│    • Sub-audible 8.0 Hz DC-blocking filter             │
│    • -108 dBFS RLC-shaped Johnson thermal noise dither │
└────────────────────────────────────────────────────────┘
          │
          ▼
┌────────────────────────────────────────────────────────┐
│ 5. ITU-R BS.1770-4 Loudness & 4x True-Peak Staging     │
│    • Gated loudness matching to -20.50 LUFS            │
│    • 4x oversampled true-peak ceiling (<= 0.9900)      │
│      via transparent whole-file linear peak scaling    │
│    • Vectorized 24-bit little-endian PCM WAV export    │
└────────────────────────────────────────────────────────┘
```

---

## 2. Vectorized MNA Circuit Solver Optimizations (`solver.py`, `mna/`)

The MNA solver evaluates complex frequency responses across log-spaced frequency vectors ($s = j\omega$) using vectorized NumPy array operations. For the complete mathematical derivations of Foster core diffusion, Alnico skin dispersion, Jordan permeability relaxation, Cole-Davidson dielectric absorption, distributed hyperbolic coil admittance, coupled $2\times 2$ mutual matrices, and interactive potentiometer wiper splitting, see [`docs/circuit_theory.md`](circuit_theory.md) (Sections 8–14).

### Key SIMD & Numerical Implementations
1. **Pre-Factored Cole-Davidson Admittance:**
   To avoid raising complex frequency vectors to fractional powers, dielectric relaxation is factored into a scalar complex phase constant $\kappa = C \cdot e^{j(\alpha - 1)\pi/2}$ and a cached real power vector:
   $$s_{\text{norm}} = \max\left(\frac{\omega}{\omega_0}, 10^{-6}\right), \quad Y_C(s) = \kappa \cdot s \cdot s_{\text{norm}}^{\alpha - 1}$$
2. **Multi-Channel Unity Branch Weighting:**
   When a voicing declares a multi-channel MNA circuit with distinct branch curves, branch FIRs are evaluated with unity weight ($p_{\text{weight}} = 1.0$) so the MNA nodal network governs physical current division without $-6\text{ dB}$ double-attenuation.
3. **Decoupled Active Preamp Biquad Generation (`compute_active_preamp_biquads`):**
   In addition to frequency-domain evaluation (`compute_active_preamp_eq`, constrained to flat DC transmission $H_{\text{preamp}}(0) \ge 1.0$), `compute_active_preamp_biquads(eq_config, sample_rate)` synthesizes decoupled second-order section (SOS) biquad filter coefficients ($b_0, b_1, b_2, a_1, a_2$) for bass and treble shelves via bilinear transform.
4. **Closed-Form Analytical Jacobians (`compute_core_impedance_jacobians`):**
   Evaluates exact partial derivatives ($\partial Z_L / \partial L$, $\partial Z_L / \partial \chi_\mu$, $\partial Z_L / \partial k_{\text{skin}}$) in microseconds directly on complex NumPy vectors without finite-difference perturbation noise (see [`docs/circuit_theory.md#10-complex-magnetic-permeability-dispersion-muomega-jordan-after-effect`](circuit_theory.md#10-complex-magnetic-permeability-dispersion-muomega-jordan-after-effect)).

---

## 3. Displacement-Domain Conditioning & Multi-Rate Oversampling (`saturation.py`)

For the physical derivations of the 15 non-linear magnetic parameters (`vsat`, `alpha`, `alpha3`, `k_sag`, `k_eddy`, `k_pull`, `lambda_L`, `k_stein`, `k_emf`, `beta_curv`, `eta_hyst`, `kappa_orbit`, `kappa_geom`, `tau_touch`, `kappa_ap`), see [`docs/circuit_theory.md`](circuit_theory.md) (Sections 8 and 15–25).

### 3.1 Displacement-Domain Pre/De-Emphasis Weighting
Directly saturating raw pickup velocity signals generates unnatural intermodulation distortion (IMD) hash in the high treble. To replicate physical string dynamics, saturation operates in the **displacement domain**:
1. **Pre-Filter:** $H_{\text{pre}}(s) = \left(\frac{\omega_c}{s + \omega_c}\right)^{0.55}$ ($\omega_c = 2\pi \cdot 40\text{ Hz}$) converts velocity into physical string displacement $x_{\text{disp}}$.
2. **Non-Linear State-Space Mapping:** Hysteresis, orbital bloom, conformal clearance, dipole expansion, and soft-knee saturation are applied in the displacement domain.
3. **De-Emphasis Filter:** $H_{\text{de}}(s) = 1 / H_{\text{pre}}(s)$ restores velocity calibration with zero high-frequency harmonic smearing.

### 3.2 Multi-Rate Anti-Aliasing Oversampling (2x / 4x) & Exact $1/m$ Decimation
Non-linear polynomial and rational mappings generate upper harmonics. To prevent digital alias foldback:
1. **Zero-Padded FFT Upsampling:** Signals are upsampled to $96\text{ kHz}$ ($2\times$) or $192\text{ kHz}$ ($4\times$).
2. **$C^\infty$ Real-Analytic Decimation Mask:** Before downsampling back to $48\text{ kHz}$, ultrasonic content across $[22\text{ kHz}, 24\text{ kHz}]$ is attenuated via the canonical real-analytic mollifier (`cinf_smoothstep`):
   $$M_{\text{aa}}(f) = 1.0 - S_\infty\left(\frac{f - 22000\text{ Hz}}{2000\text{ Hz}}\right)$$
   Because all derivatives of $S_\infty$ vanish identically at $22\text{ kHz}$ and Nyquist ($24\text{ kHz}$), the audible passband through $20\text{ kHz}$ remains $100\%$ flat while algebraic $O(1/n^2)$ boundary truncation leakage is eliminated ($>100\text{ dB}$ alias rejection with zero time-domain pre-ringing).
3. **Exact $1/m$ Polyphase Decimation Scaling:** Inverse real FFT decimation normalizes by the oversampling factor (`out = np.fft.irfft(Y_down, n_sig) / float(m)`), guaranteeing $0.00\text{ dB}$ gain parity across `oversample = 1, 2, 4`.

---

## 4. High-Performance SIMD Directives & Concurrency

### 4.1 Numba JIT Acceleration (`@njit(fastmath=True)`)
Never write interpreted Python loops over audio sample buffers. Recursive sample-by-sample ODE state solvers in [`saturation.py`](../src/allomorph/circuit/saturation.py) are compiled with Numba (`@njit(fastmath=True)`):
- `_dahl_core`: State-space domain-wall pinning hysteresis recurrence.
- `_lenz_envelope_core`: Two-stage attack/release ($\tau_{\text{att}} = 6\text{ ms}, \tau_{\text{rel}} = 45\text{ ms}$) envelope detector.
- `_lenz_velocity_drag_core`: $750\text{ Hz}$ crossover filter, $C^\infty$ softplus excess, and multi-factor velocity drag.
- `_slew_limit_core`: Recursive $\tanh$ Barkhausen domain-wall slew limiter ($f_{\text{slew}} = 16\text{ kHz}$).

### 4.2 Single-Pass Frequency-Domain Stage Fusion & Caching
Rather than executing separate inverse and forward FFTs for displacement conversion, de-emphasis, and anti-aliasing, the engine fuses linear stages into single-pass spectral multiplications:
$$Y_{\text{up}}(f) = \text{rfft}(v_{\text{sat}}) \cdot \frac{H_{\text{de}}(f)}{\text{scale}} \cdot M_{\text{aa}}(f)$$
1. **Passband-Constrained Multiplication:** Because $M_{\text{aa}}(f) \equiv 0$ for all $f \ge 24\text{ kHz}$, spectral multiplications are restricted strictly to audible passband bins ($k \le k_{\text{stop}} = \lceil 24000 \cdot N_{\text{up}} / f_{s,\text{up}} \rceil$), eliminating millions of redundant floating-point operations on ultrasonic bins.
2. **LRU-Cached Filter Packs & Netlists:** `@functools.lru_cache` caches parsed MNA netlists and polar displacement filter packs (`_get_saturation_filter_pack`).
3. **SIMD Horner Polynomial Evaluation:** Evaluates 3rd-order dipole expansion via Horner's rule ($x \cdot (1.0 + x \cdot (\alpha + x \cdot \alpha_3))$).
4. **Multi-Channel Mono Forward-FFT Broadcast (`fft_convolve_multi`):** Computes the mono input forward FFT once and broadcasts it across all channel FIRs, reducing FFT operations from $2C$ to $C + 1$.

### 4.3 Multi-Process Batch Pool & Frame Bounding
- **Parallel Execution (`ProcessPoolExecutor`):** Exposes `-j` / `--jobs` across CLI batch runners.
- **Bounded Frame Simulation (`--max-samples`):** Restricts simulation to the first $N$ frames for rapid test verification (`2,400–48,000` samples).

---

## 5. `allomorph-sim` CLI Reference

```bash
# Simulate a single target voice from raw audio:
uv run allomorph-sim --voice precision_active --instrument 30in_emg_mmtw

# Simulate all target voices in parallel across 8 CPU cores:
uv run allomorph-sim --voice all --instrument 30in_emg_mmtw -j 8

# Simulate with custom volume/tone potentiometer wiper loading:
uv run allomorph-sim --voice jazz_pair_open --vol 0.8 --tone 0.5

# Rapid prototyping run on a bounded 2-second slice (96,000 samples):
uv run allomorph-sim --voice stingray_parallel --max-samples 96000
```
