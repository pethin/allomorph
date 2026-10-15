# Allomorph - Architectural Guardrails & Mathematical Primitives

This document defines the cross-cutting mathematical primitives, software invariants, and Single-Source-of-Truth (SSOT) verification index for the Allomorph modeling pipeline.

---

## 0. Meta-Architecture & Progressive Disclosure Hierarchy

To prevent context exhaustion and eliminate contradictory documentation drift, Allomorph enforces a strict Single-Source-of-Truth (SSOT) progressive-disclosure hierarchy:

1. **Root [`AGENTS.md`](../AGENTS.md) & Scoped `AGENTS.md` Routers:** Minimal navigation guides (`< 4 KB` root, `< 2 KB` scoped) that map directories and modules to their canonical documentation and test suites without duplicating formulas or numeric constants.
2. **Cross-Cutting Mathematical & Architectural Invariants (This File):** Defines universal $C^\infty$ smoothness primitives (`cinf_smoothstep`, `compute_pickup_isolation_leveling`, `alg4`, `fsqrt` rail limiter), fail-fast declarative integrity, transducer taxonomy, and visualizer contracts, while indexing all domain invariants to their canonical domain documentation.
3. **Canonical Domain Documentation (`docs/*.md`):** Every domain equation, circuit topology, or naming taxonomy is defined in exactly one canonical document:
   - Spatial acoustics, scale continuum & arrival delays $\to$ [`docs/aperture_math.md`](aperture_math.md)
   - MNA circuit theory, impedance dispersion & magnetic metallurgy $\to$ [`docs/circuit_theory.md`](circuit_theory.md)
   - SIMD circuit simulation pipeline, oversampling & `allomorph-sim` $\to$ [`docs/dsp_simulation_engine.md`](dsp_simulation_engine.md)
   - Master dry excitation, real-cepstrum FIR, BS.1770-4 LUFS & PCM I/O $\to$ [`docs/dsp_synthesis_engine.md`](dsp_synthesis_engine.md)
   - NAM A2 pairing theory, native/stock trainers & QA audit $\to$ [`docs/trainer/README.md`](trainer/README.md)
   - Tone naming grammar, character budgets & storefront packs $\to$ [`docs/tone_packs/naming.md`](tone_packs/naming.md)
   - Catalog-wide active clamps, limits & thresholds $\to$ [`docs/clamp_and_limit_registry.md`](clamp_and_limit_registry.md)
4. **Executable Verification ([`tests/test_guardrails.py`](../tests/test_guardrails.py)):** Programmatically enforces all architectural invariants on every test run.
5. **Primacy of Physical Modeling over Hardcoded Test Values:** First-principles physical equations serve as ground truth. When refining physical modeling (e.g. enforcing $C^\infty$ smoothness or regularizing singularities), hardcoded legacy test values (dB thresholds, frequency bins) are expected to be updated to match verified physical behavior. Never warp or artificially clamp physical formulas to appease obsolete test numbers.
6. **Catalog-Driven Empirical Analysis for Clamps & Limits:** Never pick arbitrary constants or clipping bounds by intuition. Every clamp, soft-knee threshold, or regularization floor must be evaluated across the entire Cartesian product of playable source pickups $\times$ target voices, preserving 100% linear passband transparency and physical asymmetry, and must be registered in [`docs/clamp_and_limit_registry.md`](clamp_and_limit_registry.md).

---

## 1. Master Architectural Invariant Index

| Guardrail ID | Invariant Summary | Canonical SSOT Reference | Automated Verification |
| :--- | :--- | :--- | :--- |
| **5.1.1** | Dynamic spatial comb boundaries ($v_0 / \Delta x$) & quadrature null floor ($\epsilon_{\text{quad}} = 0.18$) | [`docs/aperture_math.md` (§2–3)](aperture_math.md#3-dual-coil-humbucker-spatial-comb-filtering-d) | `tests/physics/test_aperture.py` |
| **5.1.2** | Causal integer arrival delay shifting ($\tau_i$), $C^\infty$ coherence decay ($\lambda \le d$), 2D rod vs 1D blade, saddle stiffness ($H_{\text{saddle}}$), & luthier leveling ($K_{\text{iso}}$) | [`docs/aperture_math.md` (§3, §8–12)](aperture_math.md#12-multi-pickup-spatial-propagation-delays-causal-shifting--luthier-isolation-leveling) | `tests/physics/test_aperture.py` |
| **5.1.3** | Fractional standing-wave excursion ($\eta = x/L$ vs $\eta_0 = 0.1082677$), asymmetric `alg4` limiter ($+12\text{ dB}/-16\text{ dB}$), & $220\text{ Hz}$ low-shelf $H_{\text{pos}}(f)$ | [`docs/aperture_math.md` (§4.C)](aperture_math.md#c-scale-normalized-spatial-bridge-proximity-low-shelf-h_textpos) | `test_guardrail_5_1_asymmetric_displacement_limiter` |
| **5.1.4** | 24-point log wave-speed continuum ($30.87\text{--}100\text{ Hz}$) & Gaussian RBF stiffness inharmonicity ($B_s$, $<10^{-6}$ anchor error, strict monotonicity) | [`docs/aperture_math.md` (§2)](aperture_math.md#2-multi-string-wave-speed-problem--the-allomorph-solution) | `test_guardrail_5_1_inharmonicity_rbf_calibration_and_monotonicity` |
| **5.2.1** | Order-8 `fsqrt` rail limiter ($V_{\text{sat}} \le 0.9900$), $C^\infty$ shifted softplus luthier isolation leveling (`compute_pickup_isolation_leveling`), & $-120\text{ dB}$ ($10^{-6}$) denominator floor | §2 below & [`docs/clamp_and_limit_registry.md`](clamp_and_limit_registry.md) | `test_guardrail_5_2_algebraic_rail_limiter_order_8`, `test_guardrail_5_2_smooth_soft_knee_c_inf` |
| **5.2.2–3** | Absolute gain units ($20\log_{10}(\max(|H|, 10^{-6}))$, no mid-band normalization) & catalog-wide clamp validation | §2 below & [`docs/clamp_and_limit_registry.md`](clamp_and_limit_registry.md) | `tests/visualizer/`, `tests/test_guardrails.py` |
| **5.3.1–2** | Standalone Forward Digital Twin synthesis (`input.wav` $\to$ stem), NAM wet-to-wet pairing, & unity multi-channel MNA branch FIR weighting | [`docs/trainer/README.md` (§1–3)](trainer/README.md), [`docs/dsp_simulation_engine.md` (§1–2)](dsp_simulation_engine.md) | `test_guardrail_5_3_difference_high_frequency_ceiling` |
| **5.3.3** | Active preamp finite flat DC transmission ($|H_{\text{preamp}}(0)| \ge 1.0$, zero sub-audible differentiator poles) | [`docs/circuit_theory.md` (§23)](circuit_theory.md#23-sub-audible-dc-blocking-filter-80-hz--gibbs-truncation-ripple-prevention) | `test_guardrail_5_3_active_preamp_dc_transmission` |
| **5.3.4** | First-class physical transducer taxonomy (`sensor_type = "magnetic" \| "bridge_force"`) with zero voice-ID bypasses | §3.1 below & [`docs/aperture_math.md` (§6)](aperture_math.md#6-upright-bass-bridge-saddle-force-transducer-physics-sensor_type--bridge_force) | `test_guardrail_5_3_transducer_taxonomy` |
| **5.3.5** | Fail-fast declarative integrity & zero silent fallbacks (`ValueError`/`KeyError` on unknown instruments, voices, scales, pickups, or magnets) | §3.2 below & [`docs/configuration_reference.md`](configuration_reference.md) | `test_guardrail_5_3_zero_silent_fallbacks` |
| **5.3.6** | Zero-latency homomorphic cepstrum FIRs, $-20.50\text{ LUFS}$ BS.1770-4 gated matching, $0.9900$ 4x true-peak linear ceiling, & Tone3000 affinity bundles | [`docs/dsp_synthesis_engine.md` (§5–6)](dsp_synthesis_engine.md#5-itu-r-bs1770-4-loudness-matching--peak-safety-loudnesspy) | `tests/dsp/`, `tests/pipeline/` |
| **5.4.1–2** | 15-parameter non-linear magnetic feel, 2-block active op-amp headroom (`x_series` vs `classic`), $1/m$ polyphase decimation, & small-signal linear bypass ($\le 0.10$) | [`docs/circuit_theory.md` (§8, §15–25)](circuit_theory.md#24-two-block-active-pickup-metallurgy--internal-op-amp-dynamics) | `test_guardrail_5_4_small_signal_linearity` |
| **5.5.1–2** | Cole-Davidson dielectric relaxation, Jordan core dispersion, solid Alnico skin effect ($\sqrt{s}$), hyperbolic winding admittance, & coupled $2\times 2$ nodal matrix | [`docs/circuit_theory.md` (§9–14)](circuit_theory.md#9-solid-pole-eddy-skin-effect-fractional-dispersion-z_textskins) | `tests/circuit/` |
| **5.6** | Tone naming grammar (`[Family] [Configuration] [Voicing Modifier]`), zero-scroll pedalboard budget ($\le 34$ chars), & Tone3000 filename limit ($\le 64$ chars) | [`docs/tone_packs/naming.md`](tone_packs/naming.md) | `test_guardrail_5_6_tone_naming_pedalboard_budget` |
| **6.1–6** | Numba `@njit` ODE acceleration, pure real-FFT cepstrum, stage fusion, & visualizer global caching (`get_cached_target_dfs`, `step=3`, $< 5.0\text{s}$) | §3.3 below & [`docs/dsp_simulation_engine.md` (§4)](dsp_simulation_engine.md#4-high-performance-simd-directives--concurrency) | `tests/visualizer/` |
| **7.1–2** | Progressive disclosure `AGENTS.md` byte budgets (`< 4 KB` root, `< 2 KB` scoped) & repository markdown link integrity | §0 above & [`AGENTS.md`](../AGENTS.md) | `test_guardrail_agents_md_progressive_disclosure_budgets`, `test_guardrail_documentation_markdown_links_valid` |

---

## 2. Cross-Cutting $C^\infty$ Mathematical Smoothness Primitives

### 2.1 Regularized Denominators & Absolute Gain Units
Never clamp transfer magnitudes or cepstral power with premature floors (`0.05` or `1e-8`). Use regularized floors ($\max(|H(f)|, 10^{-6})$, power floor $10^{-12}$, $-120\text{ dB}$) so identical profiles evaluate to exact $1.0000$ ($0.00\text{ dB}$) and deep passive tone-cap stopbands are preserved without premature shelving. Always evaluate frequency curves in absolute gain units:
$$h_{\text{db}}(f) = 20 \log_{10}\left(\max(|H(f)|, 10^{-6})\right)$$
Never normalize forward or differential curves by subtracting or dividing by an arbitrary mid-frequency band ($100\text{--}800\text{ Hz}$ median or $1\text{ kHz}$ bin).

### 2.2 Universal Real-Analytic Mollifier (`cinf_smoothstep`)
Standard raised-cosine tapers ($w(t) = 0.5(1 + \cos(\pi t))$) have discontinuous second derivatives ($w''(0) = -\pi^2/2 \ne 0$), causing algebraic $O(1/n^2)$ Gibbs spectral leakage. Allomorph standardizes on the canonical bump-derived mollifier `cinf_smoothstep(t)`:
$$f(t) = \begin{cases} e^{-1/t} & t > 0 \\ 0 & t \le 0 \end{cases}, \quad S_\infty(t) = \begin{cases} 0 & t \le 0 \\ 1 & t \ge 1 \\ \frac{f(t)}{f(t) + f(1 - t)} & 0 < t < 1 \end{cases}$$
$S_\infty(t)$ is strictly $C^\infty$ on $\mathbb{R}$, with all derivatives of all orders vanishing identically at both transition boundaries ($\left.\frac{d^n S_\infty}{dt^n}\right|_{t=0^+} = \left.\frac{d^n S_\infty}{dt^n}\right|_{t=1^-} = 0$). It is used across:
1. **Homomorphic Real-Cepstrum FIR Tail Windowing (`dsp/fir.py`):** Trailing 15% of taps tapered via $1.0 - S_\infty(t)$.
2. **Multi-Rate Decimation Anti-Aliasing (`circuit/saturation.py`):** $H_{\text{aa}}(f) = 1.0 - S_\infty((f - 22000)/2000)$ over $[22\text{ kHz}, 24\text{ kHz}]$.
3. **Saddle Witness-Point Boundary Coupling (`physics/aperture.py`):** Transitions over $[35\text{ mm}, 75\text{ mm}]$ via $S_\infty((x - 0.035)/0.040)$.

### 2.3 Branchless 3-Stage `fsqrt` Order-8 Algebraic Rail Limiter (`apply_algebraic_rail_limiter`)
Analog rail saturation is governed by the branchless order-8 algebraic limiter:
$$f(x) = \frac{x}{\left(1 + \left(\frac{x}{V_{\text{sat}}}\right)^8\right)^{1/8}} = \frac{x}{\sqrt{\sqrt{\sqrt{1 + u^4}}}}, \quad u = \left(\frac{x}{V_{\text{sat}}}\right)^2, \quad V_{\text{sat}} = 0.985$$
Evaluating $(\cdot)^{1/8}$ via three pipelined hardware square roots (`fsqrt`) executes up to $4.8\times$ faster than `np.power` while maintaining exact mathematical equivalence ($\Delta < 10^{-14}$), strict peak bounding ($|f(x)| \le V_{\text{sat}}$), $C^\infty$ smoothness, and passband linearity ($< 0.005\text{ dB}$ deviation for $|x| \le 0.3 V_{\text{sat}}$).

### 2.4 Strictly $C^\infty$ Shifted-Softplus Saturation (`compute_pickup_isolation_leveling`, `_lenz_velocity_drag_core`)
Never use piecewise conditional soft knees (`np.where(x > thresh, ...)` or `if e > vsat:`). Soft-limiting positive excursion or pickup isolation compensation uses a shifted $C^\infty$ softplus (`logaddexp`) with an asymptotic $\tanh$ ceiling ($M = 6.0\text{ dB}, \alpha = 2.0$):
$$\text{excess} = \max\left(\frac{1}{\alpha}\left(\ln\left(1 + e^{\alpha \cdot \Delta_{\text{raw}}}\right) - \ln 2\right), 0.0\right), \quad g_{\text{iso,dB}} = M \cdot \tanh\left(\frac{\text{excess}}{M}\right)$$
Similarly, dynamic Lenz core drag (`_lenz_velocity_drag_core`, $\alpha_{\text{soft}} = 16.0$) evaluates $\text{excess}_{\text{raw}} = \frac{1}{\alpha_{\text{soft}}}\ln(1 + e^{\alpha_{\text{soft}}(e - V_{\text{sat}})})$ and $\text{excess} = V_{\text{sat}}\tanh(\text{excess}_{\text{raw}}/V_{\text{sat}})$, and non-differentiable $|x|$ norms at origin are regularized via the **Charbonnier pseudo-norm** ($\sqrt{x^2 + \epsilon^2} - \epsilon$).

### 2.5 Asymmetric Order-4 Partition-of-Unity Limiter (`alg4`, `soft_clamp_displacement_ratio`)
To bound directional physical phenomena (such as neck-ward vs bridge-ward standing-wave displacement ratio $\Delta G = 20\log_{10}(\eta / \eta_0)$) without derivative kinks at $\Delta G = 0$:
$$\sigma(dg) = \frac{1}{2}\left(1 + \tanh\left(\frac{dg}{4.0}\right)\right), \quad g_{\text{eff}}(dg) = \sigma(dg) \cdot g_{\text{pos}} + (1 - \sigma(dg)) \cdot g_{\text{neg}}$$
$$\Delta G_{\text{soft}}(dg) = \frac{dg}{\left(1 + \left(\frac{dg}{g_{\text{eff}}(dg)}\right)^4\right)^{1/4}}, \quad g_{\text{pos}} = 12.0\text{ dB}, \quad g_{\text{neg}} = 16.0\text{ dB}$$

---

## 3. Cross-Cutting Software & Visualizer Contracts

### 3.1 First-Class Transducer Taxonomy (`sensor_type`)
All instruments and voicings must declare a valid physical `sensor_type` (never use procedural `if voice_id == ...` bypasses):
- `"magnetic"`: Velocity sensing via inductive magnetic coils at position $x$ with spatial aperture $\sin(kx)\operatorname{sinc}(kw/2)$ (see [`docs/aperture_math.md`](aperture_math.md)).
- `"bridge_force"`: Piezo force sensing at the bridge witness point ($x = 0$) with maple bridge rocking resonance ($800\text{ Hz}$) and wood mass roll-off ($3.2\text{ kHz}$) (see [`docs/aperture_math.md` (§6)](aperture_math.md#6-upright-bass-bridge-saddle-force-transducer-physics-sensor_type--bridge_force)).

### 3.2 Fail-Fast Declarative Integrity & Zero Silent Fallbacks
Every physical entity (instruments, pickups, coils, circuits, scales, strings, and magnet types) is defined declaratively through validated schemas (`src/allomorph/config/`). Missing pickup/harness blocks, unrecognized voice or instrument IDs, unknown scale presets, unrecognized wiring terminals or component override keys, or unknown string/magnet keys must immediately raise explicit diagnostic `ValueError` or `KeyError` exceptions. Never silently default unknown scales to 34" or substitute fallback pickups.

### 3.3 Visualizer Vectorization, Caching & Single-File Portal Contracts (`src/allomorph/visualizer/`)
1. **Three-Line Differential Evaluation:** Evaluates Source Forward ($20\log_{10}|H_{\text{src}}|$), Target Forward ($20\log_{10}|H_{\text{tgt}}|$), and Differential ($20\log_{10}|H_{\text{tgt}}| - 20\log_{10}|H_{\text{src}}|$), yielding bit-exact $0.00\text{ dB}$ on identity pairings and finite bounded curves ($[-120\text{ dB}, +25\text{ dB}]$) across all catalog pairs.
2. **Global Voicing Curve Caching Mandate:** Never invoke FIR synthesis or multi-rate FFTs inside per-pickup/per-voice loops. Cache 1D forward voicing curves via `_VOICING_CURVE_CACHE` in `dataframe.py` and evaluate differential curves via vectorized subtraction (`tgtMags[i] - srcMags[i]`).
3. **Payload & Latency Bounds:** Export full 600-point (`step = 1`) log-spaced resolution from $20\text{ Hz}$ to $20\text{ kHz}$ across all 60 catalog voicings in a 1D `<script id="voicings-data" type="application/json">` payload for local `file://` compatibility (`< 450 KB`), and maintain full `generate_all_charts` execution under $< 5.0\text{ seconds}$.
