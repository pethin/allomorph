# Allomorph - Architectural Guardrails & Mathematical Modeling Handbook

This document provides the definitive mathematical specifications, physical equations, circuit theorems, and numerical derivations underlying the Allomorph modeling pipeline. It serves as the primary technical reference companion to `AGENTS.md`.

---

## 0. Meta-Architecture & Invariant Hierarchy

To preserve fast agent reasoning and prevent LLM context exhaustion, Allomorph maintains a strict two-tier architecture:
1. **`AGENTS.md` (Normative Core):** Contains only high-level rules, negative constraints ("Never do X"), and boundary conditions kept under **$18\text{ KB}$**.
2. **`docs/architectural_guardrails.md` (Unabridged Handbook):** Contains all extensive LaTeX equations, matrix derivations, parameter mappings, and physics proofs.
3. **`tests/test_guardrails.py` (Executable Invariants):** Programmatically enforces all rules on every test run.
4. **Primacy of Physical Modeling over Hardcoded Test Values:** The physical equations and analytical derivations model real physical behavior and serve as the ground truth. When refining or correcting physical modeling (e.g. implementing $C^\infty$ smoothness, removing piecewise steps, regularizing singularities, or fixing multi-pickup delays), hardcoded test values (dB thresholds, frequency bins, gain bounds) are viable and expected to change. Never warp, tune, or artificially clamp genuine physical formulas to fit obsolete or heuristic legacy test assertions; instead, update the tests to reflect the verified physical behavior.
5. **Catalog-Driven Empirical Analysis for Clamps, Limiters & Thresholds:** Never pick arbitrary constants, clipping bounds, or soft-knee thresholds by intuition, guesswork, or isolated single-voice tests. Every clamp, gain bound, soft-knee threshold, and regularization floor introduced into the modeling pipeline must be empirically evaluated across the entire Cartesian product of the catalog (all $N$ playable source pickups $\times$ all $M$ target voices, encompassing all 735+ catalog transformations). The analysis must strictly enforce:
   - **100% Linear Passband Transparency:** Authentic physical and electrical transformations must remain uncompressed (linear identity) within normal musical operating bounds.
   - **Thresholds Above Genuine Maxima:** Soft-knee thresholds and headroom ceilings must be calibrated above the highest legitimate electroacoustic peaks observed across the catalog (such as differential passive circuit resonance peaks up to $+6.9\text{ dB}$ or aperture displacement ratios from neck-to-bridge routing).
   - **Physical Asymmetry Preservation:** If the underlying physics is directional or asymmetric (such as bridge displacement scaling, where strings have tighter boundary limits near the bridge than toward the neck), limiters must model that physical asymmetry (e.g. an asymmetric algebraic limiter like `alg4`) rather than imposing an artificial symmetric clamp (`tanh(x/g)`) that prematurely compresses the valid physical direction.
   - **Strictly Targeted Regularization:** Clamping and saturation must strictly target genuine out-of-band divergence, non-invertible comb nulls, high-frequency Johnson noise/coil hiss amplification, or 24-bit fixed-point PCM integer overflow—never suppressing valid in-band acoustic or circuit response.
   - **Living Inventory Registry:** All pipeline clamps, limits, ceilings, and floors must be registered, classified, and tracked in [`docs/clamp_and_limit_registry.md`](file:///Users/peter/Projects/pethin/passivizer/docs/clamp_and_limit_registry.md).

When adding new modeling features, write the mathematical derivations into this document, declare the rule in `AGENTS.md`, and add an invariant check to `tests/test_guardrails.py`.

---

## 1. Spatial Acoustics, Scale Physics & Tuning/Gauge Invariance

### 1.1 Dynamic Comb & De-Combing Boundaries
Never hardcode static cutoff frequencies for wave/delay phenomena. Derive spatial transition boundaries dynamically from actual impulse peak delays or physical pickup datums:
$$f_{\text{notch}} = \frac{1}{2\Delta\tau} = \frac{f_s}{2 \cdot \Delta\text{peaks}}, \quad f_{\text{start}} = f_{\text{notch}}, \quad f_{\text{end}} = 1.7 \cdot f_{\text{notch}}$$
$$f_{\text{peak, src}} = \frac{\bar{c}}{x_{\text{src}}}, \quad f_{\text{taper\_start}} = \min(f_{\text{peak, src}}, 2500\text{ Hz}), \quad f_{\text{taper\_end}} = \min(1.8 f_{\text{taper\_start}}, 4500\text{ Hz})$$

### 1.2 Wavelength-Dependent Coherence Decay ($\lambda \le d$)
Evaluate humbucker cross-coherence decay dynamically per string/continuum wave speed $v$ based on acoustic wavelength $\lambda = v/f$ relative to coil spacing $d$:
$$f_{\text{mid}} = 1.4 \cdot \frac{v}{d}, \quad f_\sigma = \max\left(0.4 \cdot \frac{v}{d}, 1.0\right), \quad \gamma(f, v) = \frac{1}{2}\left[1 - \tanh\left(\frac{f - f_{\text{mid}}}{f_\sigma}\right)\right]$$
The transition begins right after the constructive peak, smoothly blending ($C^\infty$) into incoherent power summation without piecewise cosine clipping. Always engage coherence decay for any non-zero sample delay:
$$\text{has\_spatial\_delay} = (\text{len}(\text{channels}) > 1 \land \Delta\text{samples} > 0)$$

### 1.3 Spatial Arrival Delays & Vector Causal Normalization
When transforming between multi-pickup configurations with differing pickup spacings, the relative arrival delay across sensing channels must be normalized causally across the entire pickup array:
$$\tau_{\text{diff}, i} = \tau_{\text{tgt}, i} - \tau_{\text{src}, i}, \quad \tau_i = \tau_{\text{diff}, i} - \min_{j} \tau_{\text{diff}, j}$$
This guarantees strict physical causality ($\min(\tau_i) \equiv 0$, zero non-causal negative sample shifts) while preserving the exact differential delay between pickups ($\tau_0 - \tau_1 = \Delta\tau_{\text{phys}}$). Delays are applied strictly via causal discrete sample shifting:
$$\text{delay\_samples}_i = \lfloor \tau_i \cdot f_s + 0.5 \rfloor, \quad \text{fir}_i = [0]^{\text{delay\_samples}_i} + \text{fir}_i[:N - \text{delay\_samples}_i]$$
Never clamp early pickups to zero or invert spatial delays. Never implement fractional arrival delays on causal minimum-phase FIR filters via circular FFT phase rotation ($H(f) \cdot e^{-j 2\pi f \tau_i}$). Because the minimum-phase impulse response peak is concentrated at tap 0, continuous-time sinc interpolation wraps the negative-time non-causal sinc tail around to the end of the circular buffer. Slicing the buffer back to $N$ taps discards this wrapped tail, convolving the frequency spectrum with a Dirichlet kernel and injecting artificial periodic Gibbs truncation ripples ($\Delta f = 1/\tau_i$) across the high frequencies ($8\text{--}20\text{ kHz}$).

### 1.4 Sidewinder Architecture
If coils feed a single central row of pole pieces under the string ($\Delta x = 0$, e.g. Gibson EB Mudbucker), configure a single coil entry with effective center position ($x = x_{\text{center}}$) and expanded aperture width ($w \approx 1.25''$), never a multi-coil spaced array.

### 1.5 Scale-Normalized Fractional Coordinates ($\eta = x / L$)
Never subtract raw millimeters across different scale lengths. Calculate displacement using fractional coordinates normalized to standard 34" equivalent inches:
$$\eta_{\text{tgt}} = \frac{x_{\text{tgt}}}{L_{\text{tgt}}}, \quad \eta_{\text{src}} = \frac{x_{\text{src}}}{L_{\text{src}}}, \quad \Delta x_{\text{in}} = (\eta_{\text{tgt}} - \eta_{\text{src}}) \times 34.0''$$

### 1.6 Dynamic Target Scale Resolution & Tension Snap
Dynamically resolve effective target scale length ($L_{\text{tgt}} = 37.0''$ for multiscale, $34.0''$ otherwise) and apply proportional tension snap whenever $L_{\text{src}} < L_{\text{tgt}}$:
$$\text{snap\_db} = \min\left(3.5\text{ dB}, 1.8 \cdot \frac{L_{\text{tgt}} - L_{\text{src}}}{4.0''}\right)$$

### 1.7 Continuous Wave-Speed Continuum ($f_0 \in [30.87, 100]\text{ Hz}$)
Never constrain acoustic spatial filtering to 4 discrete open-string wave speeds or note-name strings (`["E", "A"]`). Integrate acoustic aperture responses ($H_{\text{composite}}$) across a continuous, log-spaced distribution ($N \ge 24$ points, uniform $1/N$ weight) spanning Low B ($30.87\text{ Hz}$) to open G ($100.00\text{ Hz}$). Route continuum points geometrically via register halves (`[1, 2]` treble vs `[3, 4]` bass), never note names. On multi-scale instruments (e.g. 34"-37" Dingwall NG, 32"-35" Dingwall SP1), vibrating string length $L(f_0)$ smoothly interpolates logarithmically from $L_{\text{max}}$ at Low B ($30.87\text{ Hz}$) down to $L_{\text{min}}$ at High G ($100.0\text{ Hz}$):
$$t(f_0) = \frac{\log_2(f_0) - \log_2(f_{\text{min}})}{\log_2(f_{\text{max}}) - \log_2(f_{\text{min}})}, \quad L(f_0) = L_{\text{max}} - t(f_0) \cdot (L_{\text{max}} - L_{\text{min}}), \quad v_0(f_0) = 2 \cdot L(f_0) \cdot f_0$$
Derive string stiffness $B_s(f_0)$ via an exact $C^\infty$ Gaussian Radial Basis Function (RBF) solver across empirical laboratory anchors ($f_{0,i} \in \{27.50, 30.87, \dots, 196.00\}\text{ Hz}$):
$$\mathbf{\Phi}_{i,j} = \exp\left(-\left(\epsilon \cdot \left|\log_2 f_{0,i} - \log_2 f_{0,j}\right|\right)^2\right), \quad \mathbf{\Phi} \mathbf{w} = \log_2 \mathbf{B}_s, \quad B_s(f_0) = 2^{\sum_j w_j \exp\left(-\left(\epsilon \cdot \left|\log_2 f_0 - \log_2 f_{0,j}\right|\right)^2\right)}$$
with $\epsilon = 0.5$, reproducing calibration anchors to $< 10^{-10}$ relative error while guaranteeing strictly decreasing monotonicity across bass fundamental registers without piecewise linear slope kinks. The RBF evaluation across all continuum points is fully vectorized ($\mathbf{\Phi}_{\text{eval}} \mathbf{w}$). Calculate mean propagation delay using register centroid $\bar{f}_0 = 66.9045\text{ Hz}$ ($\bar{c} = 2 \bar{L} \bar{f}_0$).

To eliminate nested scalar iterations across frequency bins and registers, continuum wave speeds are broadcast into a 2D tensor matrix ($\mathbf{V}_{\text{disp}} \in \mathbb{R}^{K \times M}$ where $K=24$ register rows and $M$ frequency columns):
$$\mathbf{V}_{\text{disp}} = \mathbf{v}_0 \odot \sqrt{1 + \mathbf{B}_s \odot \frac{(\mathbf{F} \oslash \mathbf{f}_0)^2}{1 + (\mathbf{F} / 3500\text{ Hz})^2}}$$
evaluating all aperture and spatial standing-wave responses across all continuum registers in a single SIMD vector pass.

### 1.8 Longitudinal Wave Transmission & Core Percussion ($H_{\text{long}}(f)$)
Plucking an electric bass string excites longitudinal compression waves propagating through the steel core wire ($c_L \approx 5100\text{ m/s}$), producing a distinct resonant clank at $f_L = c_L / (2L) \approx 2.7\text{--}3.3\text{ kHz}$. When target voicing string has higher longitudinal coupling than source ($\Delta k_{\text{long}} = \max(k_{\text{long,tgt}} - k_{\text{long,src}}, 0) > 0$):
$$H_{\text{long}}(f) = 1.0 + \Delta k_{\text{long}} \cdot \frac{f / f_L}{Q_L \sqrt{(1 - (f/f_L)^2)^2 + (f / (Q_L f_L))^2}} \cdot e^{-(f/6000.0)^2}, \quad Q_L = 8.0$$
Returns exact $1.0000$ ($0.00\text{ dB}$) when source matches target.

### 1.9 Cylindrical Rod vs Blade 2D Sensing Aperture
Differentiate magnetic pole geometry across sensing coils using a unified geometric sensing aperture kernel:
$$H_{\text{aperture}}(f, v; \beta, w) = \frac{1}{\sqrt{1 + \beta \cdot \left(\frac{2\pi (w/2) f}{v}\right)^2}}$$
- **Cylindrical Rod Poles (Vintage Jazz/P, Music Man):** $\beta = 0.25$, evaluating circular 2D Airy disc sensitivity ($r_p = w_m / 2$):
  $$\text{ap\_cyl}(f, v) = \frac{1}{\sqrt{1 + 0.25 \cdot \left(\frac{2\pi r_p f}{v}\right)^2}}$$
- **Blade / Slit Sensors (Active EMG, Dual-Rails):** $\beta = 1/3$, evaluating 1D rectangular integration across aperture width $w_m$ ($r_p = w_m / 2 \implies 2\pi r_p = \pi w_m$):
  $$\text{ap\_blade}(f, v) = \frac{1}{\sqrt{1 + \frac{1}{3} \cdot \left(\frac{\pi w_m f}{v}\right)^2}}$$
Both formulations evaluate to exact $1.0000$ at DC ($f=0$) and exact $0.00\text{ dB}$ identity matching, unified under a single branchless SIMD kernel.

### 1.10 Bridge Saddle Witness-Point Boundary Layer Stiffness ($H_{\text{saddle}}$)
Bass strings have finite flexural bending stiffness ($E I$), creating an exponential boundary layer ($l_b \approx \sqrt{B_s} \cdot L \approx 2.0\text{--}3.5\text{ mm}$) at the saddle witness point. For pickups situated close to the bridge ($x < 0.075\text{ m}$):
$$f_{\text{saddle}}(x) = 7200.0 \cdot \left(\frac{x}{0.075\text{ m}}\right) + 1200.0\text{ Hz}, \quad H_{\text{saddle}}(f, x) = \frac{1}{\sqrt{1 + (f / f_{\text{saddle}}(x))^2}}$$
Evaluated differentially ($H_{\text{saddle,tgt}} / H_{\text{saddle,src}}$). Evaluates to exact $1.0000$ ($0.00\text{ dB}$) when $x \ge 0.075\text{ m}$ or on identity matching, gently rolling off brittle ultra-high-frequency artifacts ($> 7\text{ kHz}$) on close-bridge pickups without dulling mid growl.

---

## 2. Mathematical Smoothness, Regularization & Boundary Continuity ($C^1 / C^\infty$)

### 2.1 Regularized Denominators, Asymptotic Limiters & Smooth Norms
Never clamp transfer ratio denominators with premature floors (e.g. `np.maximum(mag, 0.05)`) or apply hard rectangular clipping (`np.clip(..., 0.15, 3.0)`). Use regularized denominators ($\max(\text{mag}, 10^{-6})$) so identical profiles evaluate to exact $1.0000$ ($0.00\text{ dB}$).
- **Branchless 3-Stage `fsqrt` Algebraic Rail Limiter ($p=8$):** Output limiting on analog wet audio buffers is governed by:
  $$f(x) = \frac{x}{\left(1 + \left(\frac{x}{V_{\text{sat}}}\right)^8\right)^{1/8}} = \frac{x}{\sqrt{\sqrt{\sqrt{1 + u^4}}}}, \quad u = \left(\frac{x}{V_{\text{sat}}}\right)^2, \quad V_{\text{sat}} = 0.985$$
  Because $(x / V_{\text{sat}})^8 \ge 0$ for all real $x$, the absolute value $|x|$ is eliminated with zero sign checks or branching. The 4th power $u^4 = (u^2)^2$ evaluates with two fast multiplications, and the $1/8$-th power $(\cdot)^{1/8}$ is resolved via three successive hardware square roots (`fsqrt`). On modern SIMD and Apple Silicon (`arm64`) architectures, three pipelined hardware `fsqrt` instructions execute up to $4.8\times$ faster than general-purpose transcendental power routines (`np.power`) while maintaining exact mathematical equivalence ($\Delta < 10^{-14}$) down to the floating-point noise floor. The limiter guarantees strict boundedness $|f(x)| \le V_{\text{sat}}$, $C^1$ and $C^2$ continuity everywhere, and small-signal linearity ($|f(x) - x| < 10^{-8}$ for $|x| \le 0.10$).
- **Fused Analytical String Damping Ratio:** Differential viscoelastic string damping between target and source is evaluated in closed form without separate reciprocal divisions:
  $$\frac{H_{\text{damp, tgt}}(f)}{H_{\text{damp, src}}(f)} = \sqrt{\frac{1 + (f / f_{d,\text{src}})^{2 n_{\text{src}}}}{1 + (f / f_{d,\text{tgt}})^{2 n_{\text{tgt}}}}}$$
  eliminating separate reciprocal calculations, halving floating-point division count, and guaranteeing strictly positive denominators for zero-risk logarithmic evaluation.
- **Charbonnier Pseudo-Norm:** Discontinuous $|x|$ derivatives are regularized via:
  $$\|x\|_\epsilon = \sqrt{x^2 + \epsilon^2} - \epsilon, \quad \frac{d\|x\|_\epsilon}{dx} = \frac{x}{\sqrt{x^2 + \epsilon^2}}$$
  vanishing continuously at $x=0$.
- **Quadratic Steinmetz Core:** The infinite gradient singularity of $|x|^{1.6}$ at origin is eliminated via quadratic regularization:
  $$S(u) = \left(\frac{u^2}{1 + u^2}\right)^{0.8}$$
- **Strictly $C^\infty$ Thresholded Soft-Knee Saturation (`smooth_soft_knee_db`):** When soft-limiting positive gain boost, unthresholded $g \cdot \tanh(r_{\text{db}} / g)$ begins compressing immediately at $0\text{ dB}$, artificially blunting legitimate low-to-mid boost (e.g. compressing $+4\text{ dB}$ to $+3.7\text{ dB}$). Furthermore, piecewise thresholded conditionals (`np.where(r > thresh, ...)`) introduce higher-derivative kinks at the boundary. Instead, Allomorph evaluates a strictly $C^\infty$ infinitely differentiable thresholded soft knee with smooth softplus excess:
  $$\text{excess} = \frac{1}{\alpha} \ln\left(1 + e^{\alpha(r_{\text{db}} - \text{thresh})}\right) = \frac{1}{\alpha} \text{logaddexp}\left(0, \alpha(r_{\text{db}} - \text{thresh})\right), \quad \alpha = 2.0$$
  $$\text{sat\_excess} = w \cdot \tanh\left(\frac{\text{excess}}{w}\right), \quad w = \text{ceiling} - \text{thresh}$$
  $$r_{\text{soft\_db}} = r_{\text{db}} - \text{excess} + \text{sat\_excess}$$
  where $\text{thresh}$ is chosen above the highest legitimate catalog peak (e.g. $\text{thresh} = 6.0\text{ dB}$, $\text{ceiling} = 8.0\text{ dB}$, $w = 2.0\text{ dB}$).
  Properties:
  - **Strictly $C^\infty$ Everywhere:** Continuous derivatives of all orders ($\frac{d^n y}{dx^n} \in C(\mathbb{R})$) with zero piecewise conditionals, zero `np.where`, and zero boundary kinks.
  - **100% Linear Passband Transparency:** For $r_{\text{db}} \le \text{thresh} - 1.0\text{ dB}$, $|r_{\text{soft\_db}} - r_{\text{db}}| < 10^{-4}\text{ dB}$ (virtually bit-exact linear passband).
  - **Strict Monotonicity:** $\frac{dy}{dx} = 1 - \sigma(\alpha(x - \text{thresh})) \cdot \tanh^2\left(\frac{\text{excess}}{w}\right) \in (0, 1]$ everywhere.
  - **Asymptotic Boundedness:** As $r_{\text{db}} \to +\infty$, $r_{\text{soft\_db}} \to \text{ceiling}$ without overshoot.

### 2.2 Smooth $C^\infty$ Transition Across $0\text{ dB}$ Threshold
Never use piecewise conditionals (`np.where(h > 0, h * s, h)`) which create first-derivative slope kinks at $0\text{ dB}$. Use smooth softplus blending:
$$\text{excess\_boost} = \frac{1}{\beta} \ln\left(1 + e^{\beta \cdot h_{\text{db\_soft}}}\right) = \frac{1}{\beta} \text{logaddexp}\left(0, \beta \cdot h_{\text{db\_soft}}\right), \quad \beta = 1.2$$
$$h_{\text{db\_final}} = h_{\text{db\_soft}} - (1.0 - s) \cdot \text{excess\_boost}$$

### 2.3 Quadrature Regularization Floor at Comb Nulls
Never evaluate multi-coil humbucker cancellation as a raw rectified phasor sum ($|\cos(\pi f d / v)|$), which creates non-differentiable V-shaped cusps at nulls. Account for 3D flux fringing with a quadrature regularized floor ($\epsilon_{\text{quad}} \approx 0.18$):
$$p_{\text{coh\_reg}} = p_{\text{coh}} + \epsilon_{\text{quad}}^2 \cdot p_{\text{incoh}}, \quad m_{\text{blend}} = \frac{\sqrt{\gamma \cdot p_{\text{coh\_reg}} + (1 - \gamma) p_{\text{incoh}}}}{\text{dc\_norm}}$$

### 2.4 Absolute Transfer Ratios (No Mid-Band Reference Normalization)
Never normalize differential circuit curves by dividing by an arbitrary mid-frequency bin like 1 kHz (`h_diff / h_diff[1 kHz]`). Passive circuits naturally attenuate high frequencies; mid-band normalization artificially projects attenuation into false low-frequency boost, clamping tone-rolled profiles. Evaluate curves in absolute gain units:
$$h_{\text{db}} = 20 \log_{10}\left(\max(h_{\text{diff}}, 10^{-6})\right)$$

---

## 3. True Differential Circuit Deconvolution & Staging Integrity

### 3.1 True Differential Deconvolution ($H_{\text{diff}} = H_{\text{target}} / H_{\text{source}}$)
- Never use ad-hoc identity bypass conditionals (`if is_identity: return 1.0`) in place of true deconvolution, and never treat active instruments as unvoiced generic EMGs. Define explicit source circuit models in `[pickups.<id>.circuit]` for all active instruments with onboard preamps (StingRay 2-band, Dingwall FD3n) directly in `config/instruments/*.toml`.
- Directly evaluate model equality: $\text{if } \text{allclose}(H_{\text{tgt}}, H_{\text{src}}): H_{\text{diff}} \equiv 1.000$ ($0.00\text{ dB}$ identity across all frequencies).
- Universally evaluate differential transfer functions whenever a source circuit is present.

### 3.2 Strict Prevention of Double Voicing
- Automatically inspect input filenames: if `Path(input_wav).name.startswith("aperture_")`, automatically set `prefiltered = True` to guarantee `compute_voice_prefilter_firs` is never re-convolved.
- When target voice declares a multi-channel circuit, evaluate branch FIRs with unity weighting ($p_{\text{weight}} = 1.0$), letting the SPICE nodal network evaluate physical current division without $-6\text{ dB}$ double-attenuation.

### 3.3 Sub-Audible DC Decoupling & Gibbs Truncation Ripple Prevention
- Active bass preamps feature DC-blocking capacitors ($10\text{--}47\ \mu\text{F}$) solely for power rail DC offset isolation. Never include sub-audible AC-coupling differentiator poles ($s / (s + \omega_{\text{sub}})$ where $\omega_{\text{sub}} \le 2\pi \cdot 20\text{ rad/s}$) in active preamp transfer functions (`compute_active_preamp_eq`).
- Forcing $H_{\text{source}}(s) \to 0$ as $s \to 0$ causes differential deconvolution against passive targets ($H_{\text{target}}(0) > 0$) to synthesize an unphysical $>30\text{ dB}$ sub-audible inversion step ($0\text{--}5.86\text{ Hz}$, $T > 170\text{ ms}$).
- In finite-tap minimum-phase FIR synthesis ($N = 2048$, duration $42.6\text{ ms}$ at $48\text{ kHz}$), truncating such long-period waves truncates the impulse response mid-cycle, convolving the spectrum with a Dirichlet kernel and generating periodic Gibbs truncation ripples:
  $$\Delta f = \frac{f_s}{N} = \frac{48000\text{ Hz}}{2048} = 23.4375\text{ Hz}$$
  across $20\text{--}300\text{ Hz}$.
- Active preamp models must strictly represent musical shelving contours ($H_{\text{bass}} \cdot H_{\text{treble}}$) with flat, finite DC transmission ($H_{\text{preamp}}(0) \ge 1.0$), ensuring smooth, ripple-free differential curves down to $20\text{ Hz}$.

### 3.4 Architecture C Normative Invariants (Two-Stage Pipeline & Canonical Intermediate)
1. **Two-Stage Signal Flow Decoupling:**
   - **Stage 1 (Frontend IR):** Deconvolutes source instrument pickup, aperture sinc, and loaded RLC circuit into the Canonical Intermediate baseline. Synthesized as a 2048-tap minimum-phase causal FIR loaded into Block 1 (IR Loader). Consumes 0% neural CPU and introduces 0 ms algorithmic latency.
   - **Stage 2 (Backend NAM):** Captures target pickup loaded RLC circuit and non-linear magnetic feel from the Canonical Intermediate baseline. Loaded into Block 2 (NAM Preamp, A2-Lite architecture).
       - **A2 Studio Reference Standard:** Governed by $\text{ESR} \le 0.0080$ ($\approx -21\text{ dB}$ residual error on `optimal_bass_dry.wav`) with a $400$ max epoch safety ceiling and batch size $32$, monitoring the studio `channels_8` submodel (`ESR_packed_1`). In time-domain mean-squared error ($\text{ESR} = \sum (y - \hat{y})^2 / \sum y^2$), low-frequency fundamentals carry $80\text{--}90\%$ of total signal power. With the optimal bass dry calibration signal, training enables the full slimmable Architecture 2 container (`channels_3` + `channels_8`) by default, yielding lower ESR and superior dynamic fidelity over single-tier models. Isolating the 8-channel submodel alone is selectable via `--a2-lite-only`. Deep convergence ensures the optimizer resolves high-frequency pickup RLC resonant peaks ($3\text{--}5\text{ kHz}$), pick attack transients, and dual-pickup comb filtering without premature cutoff or perceptual high-frequency haziness.
2. **Canonical Intermediate Baseline Datum:**
   - Scale Length: Standard $34.0''$ ($863.6\text{ mm}$).
   - Spatial Sensing Envelope: Single narrow magnetic aperture slit ($w = 0.75''$, $d = 0$, zero comb nulls) centered at the **$93.5\text{ mm}$ ($3.68''$) acoustic median** from the bridge saddle.
   - Electrical Circuit: Wideband passive reference pickup ($L = 3.2\text{ H}, R_{\text{dc}} = 6.5\text{ k}\Omega, R_{\text{eddy}} = 150\text{ k}\Omega, C_{\text{coil}} = 90\text{ pF}, f_r \approx 4.8\text{ kHz}, Q \approx 0.75$) with $500\text{k}\Omega$ volume pot, $500\text{k}\Omega$ tone pot ($47\text{ nF}$ cap), and $330\text{ pF}$ instrument cable loading. Eliminates unphysical $+40\text{ dB}$ ultrasonic deconvolution spikes and high-frequency double roll-offs when modeling passive instruments.
   - Dynamic Feel & Magnet Metallurgy: Custom ideal reference magnet (`magnet_type = "ideal"`, $\alpha = 0.0, \alpha_3 = 0.0, k_{\text{sag}} = 0.0, k_{\text{eddy}} = 0.0, k_{\text{core}} = 0.0, V_{\text{sat}} = 10.0\text{V}$) providing zero magnetic non-linearity, zero Lenz core sag, zero dynamic eddy de-Qing, zero Foster network core perturbation, and maximum linear headroom.
3. **Canonical Intermediate Headroom & Dynamic Range Theorems:**
   - The Canonical Intermediate sweep is strictly calibrated to **$-1.50\text{ dBFS}$ True Peak** and **$-16.50\text{ dBFS}$ Nominal RMS**.
   - Preserves $>106.5\text{ dB}$ signal-to-noise ratio while guaranteeing a $1.5\text{ dB}$ anti-clipping margin preventing digital inter-sample overs through downstream high-Q resonant filters.
4. **Strictly Positive Polarity Invariant:**
   - Every synthesized frontend IR must enforce $\text{sign}\left(\sum_{n=0}^{16} h_{\text{front}}[n]\right) > 0$. If negative, invert $h = -h$ before 24-bit PCM export to guarantee zero phase cancellation when mixed in parallel with dry DI or analog preamps.
5. **Physical Source Knobs at 100% Invariant:**
   - Source deconvolution mathematically requires physical source volume and tone pots wide open ($100\%$). Tone cap roll-offs (22nF, 47nF Motown, 100nF Dub) and loading are selected in Block 2.

### 3.5 First-Class Transducer Taxonomy & Zero-Conditional Deconvolution Invariant
1. **Transducer Physical Taxonomy:**
   Transducers are categorized strictly by physical transduction principles (`sensor_type`):
   - `magnetic`: Velocity sensing via inductive magnetic coils at position $x$ with spatial aperture $\sin(k x) \cdot \operatorname{sinc}(k w / 2)$.
   - `bridge_force`: Direct piezo force sensing at the bridge witness point with velocity-to-force leaky integration ($+6\text{ dB/oct}$ from $70\text{--}250\text{ Hz}$) and spruce acoustic damping.
   - `direct`: Pure studio DI or dry string vibration baseline with an acoustic transfer function identically flat across all audible frequencies ($H_{\text{tgt, acoustic}}(f) \equiv 1.0$) and flat active buffer circuitry ($H_{\text{circuit}}(f) \equiv 1.0$).
2. **Prohibition of Procedural Deconvolution Bypasses:**
   Never short-circuit physical deconvolution using hardcoded voice ID branches (e.g. `if voice_id == "15_neutral_character": return [impulse]`). All acoustic transformations must flow through the universal regularized quotient:
   $$H_{\text{quotient}}(f) = \frac{H_{\text{tgt, acoustic}}(f) \cdot H_{\text{src, macro}}(f)}{H_{\text{src, macro}}(f)^2 + \epsilon^2}$$
   When $H_{\text{tgt, acoustic}}(f) = 1.0$, this equation naturally and stably evaluates the inverse macro-aperture ($1 / H_{\text{src}}$) of the source instrument without special-case logic.
3. **Direct Output Invariant:**
   In output mode, direct sensor targets must evaluate to bit-exact $0.00\text{ dB}$ across all frequency bins.

### 3.6 Fail-Fast Declarative Integrity & Zero Silent Fallbacks Invariant
1. **Zero Silent Fallback Policy:**
   Every physical entity in Allomorph (instruments, pickups, coils, circuits, scales, strings, and magnet types) is defined declaratively through validated configuration schemas. The simulation, visualizer, and deconvolution pipelines must adhere to a strict fail-fast invariant: never substitute silent defaults, heuristics, or unverified hardware configurations when an explicit parameter or table is missing or invalid.
2. **Normative Invariants:**
   - **Passive Pickup Circuits:** Any passive instrument (`electronics = "passive"`) evaluated in circuit simulation, visualizer difference mode, or frontend IR export must declare an explicit `[circuit]` model for the requested source pickup. A missing circuit model must immediately raise a `ValueError` identifying the missing block. Never default to an arbitrary 34" P-Bass split coil or generic biquad.
   - **Pickup Option Validation:** Specifying a pickup via CLI or API (`--pickup <name>` or `pickup="<name>"`) must resolve to a valid key in the instrument's `[pickups]` table. Unrecognized pickup names must raise `KeyError` listing available pickups. Never silently fall back to `default_pickup`.
   - **Instrument Default Pickups:** An instrument's `default_pickup` must exist in its `[pickups]` table. If an instrument specifies neither `default_pickup` nor a matching `pickup_mapping`, resolving the source pickup must raise `ValueError`. Never pick the first dictionary key arbitrarily.
   - **Scale Resolution:** Scale lengths and multi-scale boundaries must resolve to valid presets in `config/scales.toml` or explicit numeric dimensions. Unknown scale names or malformed scale dictionaries must raise `ValueError`. Never silently default unknown scale names to 34" ($0.8636\text{ m}$).
   - **String Presets & Magnet Metallurgy:** String presets and magnet types must exist in `config/strings.toml` and `MAGNET_PROPERTIES`. Unknown presets or magnet strings must raise `KeyError` listing available options. Never silently default unknown strings to standard nickel roundwounds or unknown magnets to Alnico V.
   - **Composite Pickup Geometry:** All components in composite pickups referencing other pickups via `pickup = "<name>"` must exist in the parent instrument. Unrecognized references must raise `KeyError`.

### 3.7 Visualizer-Pipeline DSP Staging & IR Fidelity Invariants

#### 3.7.1 Signal Flow Inspector (Block 1 + Block 2 Digital Twin)
The interactive Signal Flow Inspector visualizer (`docs/frequency_responses/{instrument}.html`) is not an illustrative approximation; it is an analytical digital twin of the Darkglass Anagram physical signal chain relative to the standardized Canonical Intermediate datum ($0.00\text{ dB}$):
$$\text{Stage 1: Source Bass Input} \xrightarrow{H_{\text{front}}} \text{Stage 2: Block 1 Deconvolution} \to \text{Stage 3: Canonical Intermediate (0 dB)} \xrightarrow{H_{\text{back}}} \text{Stage 4: Block 2 Target Voicing} \to \text{Stage 5: Target Voice Output}$$
1. **Normative Equivalence to `export_frontend_ir`:** Stage 2 (`Block 1 Deconvolution`) evaluates the exact frequency response of the 2048-tap minimum-phase causal FIR filter synthesized by `export_frontend_ir(inst_id, pickup_key)`. The magnitude response plotted in Stage 2 must match the FFT of the actual exported FIR WAV file within $< 0.5\text{ dB}$ across $20\text{ Hz}$ to $20\text{ kHz}$.
2. **Wiener Regularization & HF Clamping:** Stage 2 must strictly enforce Wiener regularization and soft-knee boost clamping ($\le +8.0\text{ dB}$ peak, $< +2.0\text{ dB}$ at $20\text{ kHz}$). Unbounded theoretical inverse boosts ($> +8\text{ dB}$) are strictly prohibited.
3. **Canonical Intermediate Neutralization:** Stage 1 (`Source Bass Input`, $\text{db\_src} = -\text{db\_front}$) and Stage 2 (`Block 1 Deconvolution`, $\text{db\_front}$) neutralize the source pickup into the standardized Canonical Intermediate datum:
4. **Target Voicing & Output Staging:** Stage 4 (`Block 2 Target Voicing`) represents the universal target transfer function ($H_{\text{back}} = H_{\text{tgt}} / H_{\text{can}}$) from the Canonical Intermediate datum. Evaluated relative to the standardized Canonical Intermediate datum ($0.00\text{ dB}$), Stage 5 (`Target Voice Output`) represents the resulting acoustic target voice output:
   $$\text{Stage 3} + \text{Stage 4} = \text{Stage 5} \quad (0.00\text{ dB} + \text{db\_back} = \text{db\_out})$$
   ensuring end-to-end additive signal flow consistency across all frequency bins.

#### 3.7.2 Frontend Deconvolutions (Block 1 IR Fidelity & DC Transmission)
The standalone Frontend Deconvolution curves (`docs/frequency_responses/{instrument}_frontend.html` and `docs/frequency_responses/frontend_deconvolutions.html`):
1. **FIR Synthesis Fidelity:** Magnitude curves generated by `build_instrument_frontend_dataframe` and `build_frontend_deconvolutions_dataframe` must match the FFT of the actual 2048-tap minimum-phase FIR from `export_frontend_ir` within $< 0.5\text{ dB}$ across $20\text{ Hz}$ to $20\text{ kHz}$ across all instruments (passive split-P, passive Jazz bridge, active EMG MMTW).
2. **Wiener Gain Limits:** Peak boost is clamped to $\le +8.0\text{ dB}$ with smooth cosine roll-off above $8.0\text{ kHz}$, keeping $20\text{ kHz}$ ultrasonic gain $< +2.0\text{ dB}$.
3. **Finite Sub-Audible DC Transmission:** DC transmission at $20\text{ Hz}$ must be finite and bounded within $[-12.0\text{ dB}, +12.0\text{ dB}]$, strictly preventing unphysical $>30\text{ dB}$ sub-audible DC differentiation steps and Gibbs truncation ripples.
4. **Master-Single Consistency:** `build_frontend_deconvolutions_dataframe()` must match `build_instrument_frontend_dataframe()` bit-exact across all playable instruments.

#### 3.7.3 Universal Target Voicings (Block 2 Loaded RLC & Aperture Transfer)
The Universal Target Voicings visualizer (`docs/frequency_responses/universal_targets.html` via `build_universal_targets_dataframe`):
1. **Catalog Completeness:** Must encompass all 23 target voices relative to the Canonical Intermediate baseline ($H_{\text{backend}} = H_{\text{target}} / H_{\text{canonical}}$), containing exactly 600 frequency points per voice.
2. **Physical Electroacoustic Boundedness:** Decibels must be finite and non-null (zero NaNs, zero Infs), with peak resonant boost $< +25.0\text{ dB}$ and deep tone-cap roll-off attenuation $> -100.0\text{ dB}$.
3. **Source Invariance:** Universal target voicings are defined purely relative to the Canonical Intermediate baseline and are strictly source-instrument invariant.
4. **Pipeline Consistency:** For non-matching target voicings, Stage 4 in `build_composite_instrument_dataframe` matches `build_universal_targets_dataframe` within 0.01 dB rounding precision.

#### 3.7.4 Performance & Vectorization Mandate
All visualizer curves must be evaluated using precomputed, globally cached universal target matrices (`get_cached_target_dfs`) and vectorized NumPy array operations without invoking multi-rate FFT solvers inside per-pickup loops. Full generation of all charts must complete in $< 5.0\text{ seconds}$ (verified in `tests/test_guardrails.py`).

---

## 4. Differential Non-Linear Metallurgy, Magnetic Dynamics & Analog Realism

### 4.1 Differential Magnetic Softening
Never bypass saturation with blanket conditionals (`is_passive or has_source_circuit`). Evaluate differential metallurgy between source and target:
$$\Delta\alpha = \max(\alpha_{\text{tgt}} - \alpha_{\text{src}}, 0), \quad \Delta\alpha_3 = \max(\alpha_{3,\text{tgt}} - \alpha_{3,\text{src}}, 0), \quad \Delta\eta_{\text{hyst}} = \max(\eta_{\text{tgt}} - \eta_{\text{src}}, 0), \quad \Delta k_{\text{sag}} = \max(k_{\text{sag,tgt}} - k_{\text{sag,src}}, 0)$$
$$\Delta k_{\text{eddy}} = \max(k_{\text{eddy,tgt}} - k_{\text{eddy,src}}, 0), \quad \Delta\kappa_{\text{orbit}} = \max(\kappa_{\text{orbit,tgt}} - \kappa_{\text{orbit,src}}, 0), \quad \Delta\beta_{\text{curv}} = \max(\beta_{\text{curv,tgt}} - \beta_{\text{curv,src}}, 0)$$
$$\Delta k_{\text{pull}} = \max(k_{\text{pull,tgt}} - k_{\text{pull,src}}, 0), \quad \Delta\tau_{\text{touch}} = \max(\tau_{\text{touch,tgt}} - \tau_{\text{touch,src}}, 0), \quad \Delta\kappa_{\text{geom}} = \max(\kappa_{\text{geom,tgt}} - \kappa_{\text{geom,src}}, 0), \quad \Delta k_{\text{stein}} = \max(k_{\text{stein,tgt}} - k_{\text{stein,src}}, 0)$$
$$\Delta k_{\text{emf}} = \max(k_{\text{emf,tgt}} - k_{\text{emf,src}}, 0), \quad \Delta\lambda_L = \max(\lambda_{L,\text{tgt}} - \lambda_{L,\text{src}}, 0)$$

$$V_{\text{sat,eff}} = \begin{cases} V_{\text{sat,tgt}} & \text{if active source} \\ \frac{V_{\text{sat,tgt}}}{1.0 - \min\left(0.85, \frac{V_{\text{sat,tgt}}}{V_{\text{sat,src}}}\right) + 0.15} & \text{if passive source with } V_{\text{sat,tgt}} < V_{\text{sat,src}} \\ 10.0 & \text{otherwise} \end{cases}$$
Engage softening if and only if $(\text{not is\_identity}) \land (\text{not is\_passive} \lor \text{is\_target\_more\_saturated})$. Bypass saturation on small signals ($\le 0.10$ peak) to preserve bit-exact test linearity.

### 4.2 Nonlinear Magnetic String Pull & Attack Pitch Sag ($k_{\text{pull}}$)
Evaluate dynamic pole pull damping and attack pitch sag in `_lenz_velocity_drag_core`, dynamically weighted by register excursion ratio:
$$w_{\text{reg}} = 0.70 + 0.60 \cdot \frac{|x_{\text{low}}[n]|}{\max(|x_{\text{low}}[n]| + |x_{\text{high}}[n]|, 10^{-6})}$$
$$\text{pull\_damping} = k_{\text{pull}} \cdot w_{\text{reg}} \cdot \text{excess} \cdot \tanh\left(\frac{\max(x[n], 0)}{V_{\text{sat}}}\right), \quad \text{pitch\_sag} = -k_{\text{pull}} \cdot w_{\text{reg}} \cdot \text{excess} \cdot (x_{\text{high}}[n] - x_{\text{high}}[n-1])$$
$$\text{drag}_{\text{high}} = 1.0 - (k_{\text{sag}} + k_{\text{eddy}} + \text{pull\_damping} + \text{stein\_damping}) \cdot \text{excess}$$
$$x_{\text{out}}[n] = \text{drag}_{\text{low}} x_{\text{low}}[n] + \text{drag}_{\text{high}} (x_{\text{high}}[n] + \text{wobble} + \text{pitch\_sag})$$
Thick lower strings ($f_0 \le 60\text{ Hz}$) experience up to $1.3\times$ pull damping and transient pitch sag, capturing authentic Alnico pole drag without choking high-register sustain ($0.7\times$).

### 4.3 Excursion-Dependent Dynamic Touch Spectral Tilt ($\tau_{\text{touch}}$)
In saturation stage, inject highpass attack harmonics modulated by displacement envelope:
$$H_{\text{hp}}(s) = \frac{s}{s + 2\pi \cdot 400.0}, \quad \text{touch\_mod} = \tau_{\text{touch}} \cdot \tanh\left(\frac{|x_{\text{disp}}|}{V_{\text{sat}}}\right) \cdot x_{\text{disp,hp}}, \quad x_{\text{disp}} = x_{\text{disp}} + \text{touch\_mod}$$

### 4.4 Dynamic Eddy De-Qing, Orbital Bloom & Inductance Curvature Wobble
- **Eddy current core de-Qing:** $\text{eddy\_factor} = k_{\text{eddy}} \cdot \text{excess} \cdot \tanh(|x_{\text{high}}| / V_{\text{sat}})$.
- **2D Elliptical string orbit bloom:** $x_{\text{quad}} = x \cdot \mathcal{H}\{x\}$, $x_{\text{out}} = x + \Delta\kappa_{\text{orbit}} \cdot \tanh(|x| / V_{\text{sat}}) \cdot x_{\text{quad}}$ (zero DC bias $2f_0$ bloom).
- **Core inductance curvature wobble:** $\text{wobble} = \beta_{\text{curv}} \cdot \tanh(x^2 / V_{\text{sat}}^2) \cdot (x_{\text{high}}[n] - x_{\text{high}}[n-1])$.

### 4.5 Transient Magnetic Slew-Rate Limiting
Bound domain-wall displacement delta via soft-knee saturation (`_slew_limit_core`):
$$\Delta x_{\text{max}} = \frac{2\pi f_{\text{slew}} V_{\text{sat}}}{f_s}, \quad f_{\text{slew}} = 16000.0\text{ Hz}, \quad \Delta x_{\text{slew}}[n] = \Delta x_{\text{max}} \cdot \tanh\left(\frac{x[n] - x_{\text{slewed}}[n-1]}{\Delta x_{\text{max}}}\right)$$

### 4.6 Thermal Dither & Body Coupling
- Inject calibrated $-108\text{ dBFS}$ RLC-shaped Johnson noise dither to prevent hardware fixed-point neural gating pops (bypassed on small signals $\le 0.10$).
- Model diffuse body microphonics on unpotted vintage passive pickups: $f_b = 6200.0\text{ Hz}, Q_b = 1.8, f_{\text{damp}} = 9500.0\text{ Hz}$.

### 4.7 Conformal Geometric Clearance & Dynamic Steinmetz AC Core Loss
- **Conformal clearance divergence:** $x_{\text{disp}} / (1 - \kappa_{\text{geom}} \tanh(x_{\text{disp}} / V_{\text{sat}}))$ modeling pole proximity growl and pushback without negative rail clipping.
- **Dynamic Steinmetz core loss:** $\text{stein\_damping} = k_{\text{stein}} \cdot \text{excess} \cdot (|dx_{\text{high}}/dt| / V_{\text{sat}})^{0.6}$ softening harsh flux spikes on attack plucks.

### 4.8 Dynamic Reluctance Inductance Modulation & Back-EMF String Braking
- **Dynamic Reluctance Inductance Modulation ($\lambda_L$ "Vowel Quack"):**
  Plucking a ferromagnetic string near a pickup pole momentarily compresses the magnetic air gap, decreasing magnetic reluctance ($\mathcal{R}_{\text{gap}} \propto g$) and dynamically increasing instantaneous coil inductance $L(t) = L_0 (1 + \Delta\lambda_L \cdot \text{excess} \cdot \tanh(|x[n]| / V_{\text{sat}}))$. This dynamically shifts the pickup resonance $f_r(t)$ downward on hard pick/finger plucks before recovering during sustain. In the recursive state formulation:
  $$\text{ind\_mod} = -\Delta\lambda_L \cdot \text{excess} \cdot \tanh\left(\frac{|x[n]|}{V_{\text{sat}}}\right) \cdot (x_{\text{high}}[n] - x_{\text{high}}[n-1])$$
- **Electromechanical Back-EMF String Braking ($k_{\text{emf}}$):**
  Induced coil current circulating through the pickup circuit creates an opposing Lenz Lorentz force acting back on the physical string ($\mathbf{F}_{\text{EMF}} \propto i_{\text{coil}}(t)$), providing instantaneous electromechanical braking on extreme high-frequency attack clank transients:
  $$\text{emf\_damping} = \Delta k_{\text{emf}} \cdot \text{excess} \cdot \tanh\left(\frac{|x_{\text{high}}[n]|}{V_{\text{sat}}}\right)$$
  $$\text{drag}_{\text{high}} = 1.0 - (k_{\text{sag}} + k_{\text{eddy}} + \text{pull\_damping} + \text{stein\_damping} + \text{emf\_damping}) \cdot \text{excess}$$

---

## 5. Complex Electrical Impedance & Inter-Coil Transmission Modeling

### 5.1 Fractional-Order Dielectric Absorption
Model tone capacitor and cable admittance via Cole-Davidson fractional frequency scaling ($\alpha_{\text{tone}} \approx 0.988, \alpha_{\text{cable}} \approx 0.994, \omega_0 = 2\pi \cdot 1000\text{ rad/s}$):
$$s_{\text{norm}} = \max\left(\frac{\omega}{\omega_0}, 10^{-6}\right), \quad Y_C(s) = s \cdot C \cdot s_{\text{norm}}^{\alpha - 1} \cdot e^{j(\alpha - 1)\pi / 2}$$

### 5.2 Coupled $2\times 2$ Nodal Transfer Matrix
For parallel dual-coil configurations, solve the coupled mutual system ($M = k_m \sqrt{L_n L_b}, Z_m = s M, Y_m = s C_m, \Delta_Z = Z_n Z_b - Z_m^2$):
$$H_{n \to 2}(s) = \frac{Z_b - Z_m}{\Delta_Z(Y_{\text{eff}2} + Y_m) + Z_n + Z_b - 2 Z_m}, \quad H_{b \to 2}(s) = \frac{Z_n - Z_m}{\Delta_Z(Y_{\text{eff}2} + Y_m) + Z_n + Z_b - 2 Z_m}$$

### 5.3 Complex Magnetic Permeability Dispersion ($\mu^*(\omega)$)
In `compute_core_impedance`, model causal Jordan core relaxation via logarithmic dispersion:
$$\mu_{\text{rel}}(s) = 1.0 - \chi_{\mu} \ln\left(1.0 + \frac{s}{\omega_{\mu}}\right), \quad \omega_{\mu} = 2\pi \cdot 1200.0\text{ rad/s}, \quad Z_L(s) = \mu_{\text{rel}}(s) \cdot \left[s L_{\infty} + \frac{s L_{\text{core}} R_{\text{core}}}{s L_{\text{core}} + R_{\text{core}}}\right]$$

### 5.4 Distributed Inter-Winding Transmission Line Admittance
Replace lumped parallel coil admittance with the hyperbolic transmission factor ($k_{\text{dist}} \in [0.00, 0.05], \omega_{\text{dist}} = 2\pi \cdot 10000\text{ rad/s}$):
$$\gamma_{\text{dist}} = k_{\text{dist}} \sqrt{\frac{s}{\omega_{\text{dist}}}}, \quad Y_{\text{coil}}(s) = (s C_{\text{coil}} + G_{\text{coil}}) \cdot \frac{\tanh(\gamma_{\text{dist}})}{\gamma_{\text{dist}}}$$

### 5.5 Interactive Potentiometer Wiper Division & Cable Loading
Model Volume and Tone pot wiper positions ($P_{\text{vol}}, P_{\text{tone}} \in [0, 1]$). Rolling volume down splits $R_{\text{vol}}$ into series $R_{\text{top}}$ and shunt $R_{\text{bot}}$, interacting with cable capacitance $C_{\text{cable}}$; rolling tone down reduces series resistance into $C_{\text{tone}}$. Exactly preserves netlist defaults when $P_{\text{vol}} = 1.0, P_{\text{tone}} = 1.0$.

### 5.6 Solid Alnico Pole Eddy Skin-Effect Fractional Dispersion ($Z_{\text{skin}}(s)$)
In pickups with solid conducting metallic pole pieces (e.g. Alnico rod magnets in Fender Jazz and Precision basses), eddy currents circulating inside the cylindrical magnet volume force high-frequency magnetic flux to the cylinder periphery. This classical skin effect increases the series AC resistance of the inductive branch proportionally to $\sqrt{\omega}$:
$$Z_{\text{skin}}(s) = R_{\text{dc}} \cdot k_{\text{skin}} \cdot \left(\sqrt{1 + \frac{s}{\omega_{\text{skin}}}} - 1\right), \quad \omega_{\text{skin}} = 2\pi f_{\text{skin}}$$
- **Exact DC Transparency:** At DC ($s = 0$), $Z_{\text{skin}}(0) \equiv 0$, guaranteeing exact bit-exact preservation of measured DC coil resistance $R_{\text{dc}}$ and $0.00\text{ dB}$ DC transfer matching.
- **High-Frequency Loss:** At high frequencies ($f \gg f_{\text{skin}}$), $Z_{\text{skin}}(s) \to R_{\text{dc}} \cdot k_{\text{skin}} \cdot \sqrt{s / \omega_{\text{skin}}}$, introducing the authentic $45^\circ$ phase angle and $\sqrt{\omega}$ loss that gently softens top-end hash without inducing un-damped resonant peaking or Gibbs truncation ripples.

---

## 6. High-Performance Audio DSP & SIMD Directives

### 6.1 Buffer & Recurrence Execution Acceleration
- **No Interpreted Loops Over Audio Buffers:** Never write scalar Python loops (`for i in range(n)`) over audio buffers. Always accelerate recursive ODE state solvers (Lenz drag, Dahl hysteresis, slew limiting) using Numba JIT compilation (`@njit(fastmath=True)`) with graceful pure-Python fallback ($220\times$ speedup).
- **NumPy View Byte Packing:** Vectorize 24-bit little-endian WAV packing in C via NumPy view slicing (`scaled.astype("<i4").view(np.uint8).reshape(-1, 4)[:, :3].tobytes()`, $135\times$ speedup).
- **SIMD Circuit Transfer Evaluation:** Formulate all nodal impedances, admittances, and voltage divider ratios directly on complex NumPy frequency vectors ($s = 1j \cdot \omega$) rather than scalar loops ($12\times$ speedup).

### 6.2 Pure Real-FFT Homomorphic Cepstrum Pipeline
Minimum-phase causal FIR synthesis from log-magnitude curves ($N = 4096$ taps, $N_{\text{fft}} = 8192$) strictly utilizes a pure real-FFT pipeline (`irfft` $\to$ causal folding $\to$ `rfft` $\to$ `irfft`):
1. **Real Cepstrum Computation:** Given real, conjugate-symmetric log-magnitude spectrum $L[k] = \frac{1}{2}\ln(|H[k]|^2 + \epsilon^2)$ for $k \in [0, N]$:
   $$\hat{c}[n] = \text{irfft}(L)[n]$$
   Because $L$ is purely real and even, $\hat{c}[n]$ is purely real by definition, eliminating all complex allocations and negative-frequency mirroring buffer concatenations.
2. **Causal Liftering:** The minimum-phase analytic spectrum is obtained by zeroing non-causal negative cepstral time indices and doubling positive time indices:
   $$c_{\text{min}}[n] = \begin{cases} \hat{c}[0], & n = 0 \\ 2\hat{c}[n], & 1 \le n < N \\ \hat{c}[N], & n = N \\ 0, & N < n < 2N \end{cases}$$
3. **Analytic Spectrum & Direct Inverse Real FFT:**
   $$S[k] = \text{rfft}(c_{\text{min}})[k] = \ln|H[k]| + j \theta_{\text{min}}[k]$$
   $$h[n] = \text{irfft}(\exp(S))[:N]$$
This guarantees exact mathematical equivalence ($\Delta < 10^{-14}$) to classical complex-FFT Hilbert synthesis while eliminating $100\%$ of imaginary leakage, cutting intermediate array allocations by half, and speeding up minimum-phase FIR generation across all stages.

### 6.3 Frequency-Domain Stage Fusion & Caching
- **Fuse Linear Stages in Frequency Domain:** Avoid redundant FFT/IRFFT round-trips. Apply displacement pre-filters ($X_{\text{up}} \cdot H_{\text{pre}}$) before inverse FFT, and combine de-emphasis ($H_{\text{de}} / \text{scale}$) and anti-aliasing lowpass ($aa\_mask$) into a single product before decimation.
- **Passband-Constrained Oversampling:** Zero-padded frequency-domain multiplications in saturation oversampling are restricted strictly to audible passband bins ($k \le k_{\text{stop}} = \lceil 24000 \cdot N / f_s \rceil$), eliminating millions of redundant floating-point multiplications on known-zero ultrasonic frequencies.
- **Broadcast Input FFTs:** In multi-pickup instruments, precompute the mono input forward FFT once across all channels using `max_ir_len` and broadcast it across channel FIRs (`fft_convolve_multi`).
- **In-Memory Netlist LRU Caching:** Wrap SPICE netlist parsing with `@functools.lru_cache(maxsize=128)` and return shallow copies (`cached.model_copy()`), eliminating redundant disk reads and regex tokenization.

### 6.4 Parallel Concurrency & Test Bounding
- **Multi-Process Concurrency:** Expose parallel process execution using `concurrent.futures.ProcessPoolExecutor` with `-j/--jobs` CLI flags (defaulting to `min(4, os.cpu_count())`), dropping 16-voice batch simulation from $7+\text{ minutes}$ to $80\text{ seconds}$.
- **Decouple Target vs Differential Curve Generation:** Precompute global target output voice master dataframe once globally. Compute source-to-target difference dataframes once per instrument, reducing dataframe builds by $73\%$.
- **Frame-Bounded Audio Processing (`max_samples`):** Support bounded frame prefixes (`max_samples = 4800` to `48000`) in unit tests to drop test execution from $22\text{s}$ down to $0.15\text{s}$ while preserving complete signal pipeline verification.

### 6.5 Visualizer Vectorization, Caching & Vega-Lite Payload Bounding (Commit `7c6e634`)
- **Prohibition of Multi-Rate FFTs in Signal Flow Loops:** Never invoke `build_voice_dataframe(mode="difference")`, FIR filter synthesis, or multi-rate FFTs inside per-pickup/per-voice loops within `build_composite_instrument_dataframe`. Across 11 playable instruments with multiple pickup switch positions and 23 target voices, iterative synthesis executes $> 700$ redundant 2048-tap FIR convolutions and 8192-point FFTs, blowing up chart generation from $< 3\text{ s}$ to $> 20\text{ s}$.
- **Decoupled Universal Backend Caching:** Universal target voicings are defined relative to the Canonical Intermediate baseline ($H_{\text{backend}} = H_{\text{target}} / H_{\text{canonical}}$). Because the Canonical Intermediate is fixed (34" scale, 93.5mm datum, wideband passive reference circuit), target curves are strictly source-invariant and must be precomputed and cached globally once via `get_cached_target_dfs(step=step)`:
  $$\text{db\_back\_dict}[vid] = \text{np.round}(\text{db\_tgt} - \text{db\_can}, 2)$$
- **Analytical Mirror Reflection for Identity Matches:** For authentic source-to-target digital twin matches (`is_match`), set the mirror reflection directly in vector form:
  $$\text{db\_back} = -\text{db\_front}, \quad \text{db\_out} = 0.00\text{ dB}$$
  This simultaneously guarantees:
  1. Exact $0.00\text{ dB}$ flat identity output across all frequency points.
  2. Authentic Wiener-regularized frontend deconvolution ($\le +6\text{ dB}$ clamping) without unregularized $+24\text{ dB}$ ultrasonic spikes.
  3. Microsecond execution speed with zero redundant multi-rate FFTs.
- **For Non-Matching Voicings:** Evaluate directly as $\text{db\_back} = \text{db\_back\_dict}[vid]$ and $\text{db\_out} = \text{np.round}(\text{db\_front} + \text{db\_back}, 2)$ via vectorized NumPy array additions.
- **Payload Bounding & Downsampling:** Downsample composite chart curves using `step = 3` (200 log-spaced points from $20\text{ Hz}$ to $20\text{ kHz}$) and round decibel magnitudes to 2 decimal places (`np.round(..., 2)`). Filter stages using Vega-Lite `PredicateComposition` (`(pickup_select & pred_stages_12) | (pickup_select & voice_select & pred_stages_34)`). This compresses standalone HTML payloads from $> 66\text{ MB}$ down to $\sim 5\text{ MB}$ ($92.5\%$ reduction) and prevents browser DOM freezes.
- **Strict Latency Budget:** Full execution of `generate_all_charts` across all 11 playable instruments must complete in $< 5.0\text{ seconds}$ (verified in `tests/test_guardrails.py`).

### 6.6 Baked Transformations Visualizer Architecture & Single-File Invariants
- **Monolithic 1-Block Differential Curves:** The Baked Transformations tab (`docs/frequency_responses/baked_responses.html` and `#baked` in the portal) visualizes the full differential transfer functions ($H_{\text{diff}} = H_{\text{tgt}} / H_{\text{src}}$) corresponding directly to the dynamic-tier baked outputs.
- **Self-Contained Embedded JSON:** To avoid browser `file://` CORS security restrictions when viewing documentation locally without an HTTP server, the 276-curve matrix (12 playable instruments $\times$ 23 target voices) is embedded directly into the HTML as `<script id="baked-data" type="application/json">` and parsed in-memory via `JSON.parse(document.getElementById("baked-data").textContent)`.
- **Matrix Serialization & Payload Bounding:** Downsampling with `step = 3` (200 log-spaced points from $20\text{ Hz}$ to $20\text{ kHz}$) and 2-decimal rounding (`np.round(..., 2)`) bounds the complete 276-curve JSON payload to $\sim 390\text{ KB}$ and produces a self-contained HTML page under $450\text{ KB}$.
- **Sub-Millisecond Multi-Select Vega Streaming:** Client-side selection filtering uses native JavaScript table assembly and `vegaView.change('table', vega.changeset().remove(...).insert(...)).runAsync()`, updating interactive multi-selected instrument $\times$ voicing combinations instantaneously without redrawing or recreating DOM elements.
- **Physical Bounds & Identity Invariants:** Identity pairings evaluate to exact $0.00\text{ dB}$, while all other transformations strictly adhere to physical electroacoustic bounds ($< +30\text{ dB}$ boost, $> -100\text{ dB}$ attenuation per Section 5.3).

