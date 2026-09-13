# Allomorph - Agent Development Guidelines

Welcome to **Allomorph**. This repository houses an analog digital twin and universal modeling pipeline designed to transform active, passive, and acoustic bass pickup and transducer configurations (P-Bass, Jazz Bass, StingRay, Rickenbacker, P/J, P/MM, upright bass piezos) into authentic high-impedance passive and active target voicings tailored for the **Darkglass Anagram** pedalboard, Neural Amp Modeler (NAM), and DAW plugin hosts.

---

## 1. Physical Hardware Specifications & Datums

### Source Instruments
1. **Current Bass (30" Short Scale):**
   - Pickup: Single active 18V EMG MMTW (dual-mode: MM dual coil or J single-coil, coil spacing $d=0.90''$, coil aperture $w=0.75''$).
   - Physical Datum: Centerline located **$77.5\text{ mm}$ from the bridge** ($303.5\text{ mm}$ from the 12th fret; neck coil $88.9\text{ mm}$, bridge coil $66.1\text{ mm}$).
   - Character: Warm low-mid bloom ($180\text{--}250\text{ Hz}$), reduced string tension compared to 34", selectable J-bridge growl or MM authority.

2. **Planned Bass (32" Medium Scale):**
   - Routing: Reverse EMG PX split ($122.8\text{ mm}$ from bridge) + EMG MMTWX ($62.2\text{ mm}$ centerline, $50.8\text{ mm}$ bridge coil) with EMG ABCX active blend.
   - Reference: `/Users/peter/Projects/pethin/medium-scale-bass/Pickup Placement - P_MM.md`.

### Target Instruments & Tonal Scalings
- **Standard 34" Scale:** Standard Fender ($125.0\text{ mm}$ P, $155.6\text{ mm}$ 60s J neck, $63.5\text{ mm}$ 60s J bridge) and Music Man ($66.0\text{ mm}$ StingRay).
- **34"-37" Multi-Scale (Dingwall Fanned Fret):** High-tension wave speeds ($77.4\text{--}169.3\text{ m/s}$), angled bridge sweet spot ($48.0\text{ mm}$ datum), FD3 parallel dual-coil SPICE netlist ($f_r = 3.4\text{ kHz}, Q = 1.5$), and stainless-steel roundwound harmonic extension.

---

## 2. Tech Stack & Engineering Constraints

When contributing to or maintaining this repository, strictly adhere to these architectural standards:

- **Package & Environment Manager:** Always use **`uv`** (`uv run`, `uv add`).
- **Tabular Data & Visualizations:** Strictly use **`polars`** for dataframes (`pl.DataFrame`, `pl.concat`) and **`altair`** (Vega-Lite declarative charts compiling to standalone HTML in `docs/`). Do **NOT** use `pandas` or `matplotlib`.
- **Numerical Math & DSP:** Strictly use **`numpy`** for 1D frequency responses, acoustic/electrical vector math, and FFTs (`np.fft`).
- **Audio DSP & I/O:**
  - 24-bit audio file I/O uses **Spotify's `pedalboard`** (JUCE-backed SIMD C++ engine).
  - Minimum-phase FIR synthesis uses our internal homomorphic real-cepstrum Hilbert transform engine vectorized with NumPy.
  - Do **NOT** add `scipy` or `soundfile` as dependencies (they carry legacy C/Fortran bloat).
- **Linting, Code Quality & Static Typing:**
  - Strictly use **`ruff`** (`uv run ruff check`, `uv run ruff check --fix`) for linting, code formatting, and import sorting targeting Python 3.14 (`py314`).
  - Strictly use **`pyrefly`** (`uv run pyrefly check`) with the strict preset (`preset = "strict"` in `pyproject.toml`) for type checking across the entire codebase. Do NOT disable rules, downscale presets, or suppress type checks with `# type: ignore` unless proven fundamentally impossible. Use `pydantic` for structured schemas and runtime validation where appropriate.
- **Git & Contribution Governance:** All commits must be signed off using `git commit -s` (or `Signed-off-by: Legal Name <email>`) to satisfy `CLA.md` and pass automated DCO CI checks.
- **Circuit Simulation Engine:**
  - **Native WAV SPICE Engine:** Built-in Apple Silicon (`arm64`) WAV SPICE circuit simulation engine (`src/allomorph/circuit/`, `allomorph-sim`) providing zero-external-dependency SPICE netlist parsing, analytical nodal RLC solving, regularized differential SPICE transfer functions ($H_{\text{diff}} = H_{\text{target}} / H_{\text{source}}$), dynamic core saturation bypass, and vector soft-knee saturation ($V_{\text{sat}} \cdot \tanh(v/V_{\text{sat}})$) directly on audio waveforms at >1500x speed.

---

## 3. Repository Layout

```
allomorph/
├── .github/
│   ├── workflows/dco.yml     # Hardened injection-safe DCO / CLA sign-off check
│   └── PULL_REQUEST_TEMPLATE.md # Contributor pull request checklist
├── AGENTS.md                 # Antigravity project rules and context (this file)
├── CLA.md                    # Harmony-based Contributor License Agreement (Dual-licensing)
├── CONTRIBUTING.md           # Contributor guide, setup, and sign-off instructions
├── LICENSE                   # PolyForm Noncommercial License 1.0.0
├── pyproject.toml            # Project metadata (polars, altair, pedalboard)
├── README.md                 # Comprehensive architecture, CLI usage, roadmap
├── src/                      # Core reusable library package
│   └── allomorph/            # Modern package (config/, physics/, circuit/, visualizer/, pipeline/)
├── audio/                    # Generated 24-bit audio digital twins (audio/<instrument>/)
├── config/                   # Modular TOML configuration files
│   ├── instruments/          # Source bass definitions with embedded [pickups.<id>.circuit]
│   ├── preamps.toml          # Reusable active preamp catalog (Sadowsky, StingRay, Aguilar, Dingwall)
│   ├── scales.toml           # Standard scale wave speeds
│   ├── strings.toml          # Physical string core/wrap presets
│   └── voices/               # Target voice TOMLs with embedded declarative [circuit] tables
├── docs/                     # Technical documentation & interactive charts
│   ├── architectural_guardrails.md # Master mathematical reference handbook & derivations
│   ├── tone_naming_guidelines.md # Normative tone naming rules, character limits & bassist lexicon
│   ├── voice_catalog.md      # Passive pickup models, RLC parameters, character
│   ├── circuit_theory.md     # Differential equations, 500k volume pot, treble bleed
│   ├── aperture_math.md      # Aperture sinc, wave speeds, multi-scale filters
│   ├── anagram_workflow.md   # Darkglass Anagram Block 1 routing & banks
│   ├── frequency_responses.html # Master interactive portal
│   └── frequency_responses/     # Per-instrument standalone Altair visualizations
├── models/                   # Trained Neural Amp Modeler (.nam) models (models/<instrument>/)
├── scripts/                  # Workflow utilities & CLI entrypoints
│   ├── analyze_voices.py     # Polars + Altair frequency curve visualizer
│   ├── train_nam.py          # Local NAM A2 PyTorch/MPS GPU trainer
│   └── generate_tone3000_artwork.py # Tone3000 storefront artwork generator
├── tests/                    # Hierarchical test suite (circuit/, config/, dsp/, physics/, pipeline/, tone3000/, visualizer/, test_guardrails.py)
└── tone3000/                 # Tone3000 storefront packs, artwork, and documentation
    ├── assets/               # Production-ready vector SVG and 1024x1024 JPG artwork
    └── docs/                 # Standardized storefront product descriptions and catalog
```

---

## 4. Signal Flow on Darkglass Anagram

The models produced by Allomorph are loaded into **Block 1** (as a high-impedance passive/transducer front-end before preamp and drive):
```
[Bass: Active / Passive / Piezo] -> [Block 1: Allomorph NAM] -> [Block 2: Darkglass Preamp/Drive] -> [Block 3: Cab IR Loader] -> [FOH/Audio Interface]
```
- **Block 1 (NAM Preamp):** Load trained `.nam` neural model (feather/nano architecture) capturing full RLC resonance, eddy currents, and non-linear magnetic feel.
- **Block 2 (Darkglass Preamp/Drive):** Microtubes B7K, Vintage Ultra, or Alpha·Omega for bass saturation.
- **Block 3 (Cabinet IR Loader):** Downstream speaker cabinet impulse responses (4x10, 8x10, 2x12).

---

## 5. Architectural Guardrails (Core Invariants)

All code contributions must strictly satisfy the following normative invariants (verified automatically via `tests/test_guardrails.py`). Complete mathematical proofs and derivations are documented in [`docs/architectural_guardrails.md`](file:///Users/peter/Projects/pethin/passivizer/docs/architectural_guardrails.md).

### 5.1 Spatial Acoustics, Scale Physics & Continuum Routing
1. **Dynamic Spatial Boundaries:** Derive comb null and de-combing taper boundaries dynamically from delays ($\bar{c}/x_{\text{src}}$); never hardcode static cutoff bins.
2. **Coherence Decay & Vector Causal Normalization:** Multi-pickup spatial arrival delays ($\tau_i$) must be normalized causally via Vector Causal Normalization ($\tau_i = \Delta\tau_i - \min_j \Delta\tau_j$) and applied via causal integer sample shifts ($[0]*k + \text{fir}[:-k]$); never clamp early pickups to zero or rotate phase via circular FFT ($e^{-j 2\pi f \tau_i}$) which injects high-frequency Gibbs truncation ripples. Cross-coherence decay across dual coils must transition smoothly ($C^\infty$ $\tanh$ sigmoid) into incoherent summation when $\lambda \le d$ (engage whenever $\Delta\text{samples} > 0$). Model sidewinders as a single centered coil ($w \approx 1.25''$). Distinguish 2D cylindrical rod poles ($J_1(k r_p)/(k r_p)$) from 1D blade slits, and model saddle witness-point boundary stiffness ($H_{\text{saddle}}$) for close bridge pickups ($x < 75\text{ mm}$).
3. **Fractional Coordinates, Low-Shelf Displacement Ratio & Scale Snap:** Evaluate bridge proximity fundamental excursion via scale-normalized fractional standing-wave displacement ratios ($\Delta G = 20 \log_{10}(\eta_{\text{tgt}} / \eta_{\text{src}})$) implemented through an asymptotically flat single low-shelf filter ($H_{\text{pos}}(f)$, corner 220 Hz); never apply artificial symmetrical high-frequency counter-shelves that boost bridge clank or cut neck treble. Apply proportional high-frequency tension snap when $L_{\text{src}} < L_{\text{tgt}}$, and model steel core longitudinal compression clank ($H_{\text{long}}$ at $2.7\text{--}3.3\text{ kHz}$).
4. **Wave-Speed Continuum & Inharmonicity RBF:** Integrate aperture responses across a continuous 24-point log continuum ($30.87\text{ to } 100\text{ Hz}$) routed geometrically by register halves (`[1, 2]` treble vs `[3, 4]` bass), never note names. Interpolate empirical string stiffness inharmonicity anchors ($B_s$) via an exact $C^\infty$ Gaussian RBF solver reproducing calibration table values to $< 10^{-10}$ relative error with strictly decreasing monotonicity across bass fundamental registers. Keep local neural network training in `scripts/train_nam.py` strictly identical to T3K baseline under Architecture 2 ($\text{ESR} \le 0.0080, \text{epochs} \le 400$).

### 5.2 Mathematical Smoothness & Regularization ($C^\infty$)
1. **Regularized Denominators, Smooth Limiters & Blending:** Never clamp denominators with premature hard floors; use $\max(\text{mag}, 10^{-6})$ so identical setups yield exact $0.00\text{ dB}$. Bound analog rail saturation via an asymptotic $C^\infty$ algebraic limiter ($x / (1 + (|x|/V_{\text{sat}})^8)^{1/8}$ with $V_{\text{sat}} = 0.985$). Use smooth softplus blending (`logaddexp`) across $0\text{ dB}$ (no piecewise conditionals or slope kinks). Bound boost/cut via smooth partition-of-unity convex blends or asymptotic soft-knee saturation ($g \cdot \tanh(r_{\text{db}}/g)$). Model asymmetric norms using Charbonnier pseudo-norms ($\sqrt{x^2 + \epsilon^2} - \epsilon$) and regularized quadratic Steinmetz loss ($(x^2 / (1 + x^2))^{0.8}$) to eliminate infinite gradient singularities at $x=0$.
2. **Quadrature Null Floors & Absolute Units:** Preserve 3D flux fringing with a quadrature floor ($\epsilon_{\text{quad}} \approx 0.18$) at comb nulls. Evaluate curves in absolute gain units ($20\log_{10}(\max(h_{\text{diff}}, 10^{-6}))$); never normalize by an arbitrary mid-frequency bin.

### 5.3 True Differential Circuit Deconvolution & Staging Integrity
1. **True Differential Deconvolution:** Evaluate $H_{\text{diff}} = H_{\text{tgt}} / H_{\text{src}}$ using dedicated declarative circuit models in `config/instruments/` for active/buffered instruments. Exact equality must evaluate to $0.00\text{ dB}$ across all bins.
2. **Double Voicing Prevention:** Automatically bypass prefiltering if audio file starts with `aperture_`. Multi-channel target circuits evaluate branch FIRs with unity weight ($p_{\text{weight}} = 1.0$), letting the SPICE nodal network evaluate current division.
3. **Sub-Audible DC Transmission:** Active preamps must feature flat, finite DC transmission ($H_{\text{preamp}}(0) \ge 1.0$). Strictly omit sub-audible AC-coupling differentiators ($s / (s + \omega_{\text{sub}})$) from preamp EQ models to prevent Gibbs truncation ripples ($\Delta f = f_s / N = 23.4\text{ Hz}$) across $20\text{--}300\text{ Hz}$.
4. **Transducer Taxonomy & Zero-Conditional Deconvolution:** Model all sensors strictly through a first-class physical taxonomy (`sensor_type = "magnetic" | "bridge_force" | "direct"`). Never inject ad-hoc voice ID conditionals (`if voice_id == ...`) or conditional impulse bypasses. Direct sensors define flat acoustic transfer ($H_{\text{tgt, acoustic}} \equiv 1.0$) and flat active circuit response ($H_{\text{circuit}} \equiv 1.0$), naturally solving the inverse macro-aperture ($1 / H_{\text{src}}$) through the universal Wiener regularized quotient.
5. **Fail-Fast Declarative Integrity & Zero Silent Fallbacks:** Never silently substitute fallback models, arbitrary pickups, assumed scales, or unvoiced circuits when a configuration block or parameter is missing or invalid. Missing passive pickup circuits, invalid voice IDs, unknown scale strings, unmapped pickups, unknown string presets, and unrecognized magnet types must immediately raise explicit, diagnostic `ValueError` or `KeyError` exceptions. Silent fallbacks mask configuration errors, violate declarative reproducibility, and corrupt downstream deconvolution filters.
6. **Visualizer-Pipeline DSP Staging & IR Fidelity:** The visualizer's Signal Flow Inspector (`docs/frequency_responses/{instrument}.html`), Frontend Deconvolutions (`{instrument}_frontend.html`, `frontend_deconvolutions.html`), and Universal Target Voicings (`universal_targets.html`) must strictly replicate the authentic two-stage DSP pipeline:
   - **Frontend Deconvolutions & Block 1 IR:** Stage 2 / frontend curves must match the frequency response of the actual impulse response synthesized by `export_frontend_ir` within $< 0.5\text{ dB}$ across $20\text{ Hz}$ to $20\text{ kHz}$, strictly enforcing Wiener noise regularization and soft-knee boost clamping ($\le +8.0\text{ dB}$ peak, $< +2.0\text{ dB}$ at $20\text{ kHz}$). Unbounded theoretical inverse boosts are strictly prohibited. Sub-audible DC transmission at $20\text{ Hz}$ must be finite and bounded within $[-12\text{ dB}, +12\text{ dB}]$.
   - **Universal Target Voicings (Block 2):** Must encompass all 23 target voices relative to Canonical Intermediate baseline, maintaining physical electroacoustic bounds ($< +25\text{ dB}$ boost, $> -100\text{ dB}$ attenuation), and strictly matching Stage 4 target curves in the Signal Flow Inspector.
   - **Signal Flow Staging & Canonical Intermediate Datum:** Stages are evaluated relative to the standardized Canonical Intermediate datum ($0.00\text{ dB}$, wideband passive reference baseline at $34''$ @ $93.5\text{ mm}$, $f_r = 4.8\text{ kHz}, Q = 0.75$). Stage 1 (Source Bass Input) + Stage 2 (Block 1 Deconvolution) = Stage 3 (Canonical Intermediate, flat $0.00\text{ dB}$) bit-exact, and Stage 3 (Canonical Intermediate, flat $0.00\text{ dB}$) + Stage 4 (Block 2 Target Voicing) = Stage 5 (Target Voice Output) bit-exact across all frequency bins.

### 5.4 Differential Non-Linear Metallurgy & Dynamics
1. **Differential Metallurgy Softening:** Soften differentially based on relative target vs source metallurgy ($\Delta\alpha, \Delta k_{\text{sag}}, \Delta k_{\text{eddy}}, \Delta\kappa_{\text{geom}}, \Delta k_{\text{stein}}, \Delta k_{\text{emf}}, \Delta\lambda_L$). Bypass on small signals ($\le 0.10$) to preserve bit-exact test linearity.
2. **Dynamic Magnetic Feel:** Model register-weighted magnetic string pull damping and attack pitch sag ($w_{\text{reg}}$ in `_lenz_velocity_drag_core`), electromechanical back-EMF string braking ($k_{\text{emf}}$), dynamic reluctance inductance modulation ($\lambda_L$ "vowel quack"), displacement-modulated touch highpass tilt ($\tau_{\text{touch}}$), eddy de-Qing, $2f_0$ orbit bloom, core curvature wobble, soft-knee slew limiting, conformal clearance divergence, and $-108\text{ dBFS}$ Johnson noise dither.

### 5.5 Complex Electrical Impedance & Harnesses
1. **Fractional Dielectric, Permeability & Skin Dispersion:** Model tone cap and cable admittance via continuous Cole-Davidson fractional frequency scaling down to DC ($f=0\text{ Hz}$, $\alpha_{\text{tone}} \approx 0.988, \alpha_{\text{cable}} \approx 0.994$), causal Jordan core relaxation ($\mu^*(\omega)$), and solid Alnico pole eddy skin-effect fractional series dispersion ($Z_{\text{skin}}(s) \propto \sqrt{s}$).
2. **Coupled System & Harness Controls:** Solve coupled $2\times 2$ nodal transfer matrix for parallel coils, hyperbolic transmission line admittance ($\tanh(\gamma)/\gamma$), and interactive volume/tone wiper splitting with cable capacitance loading ($P_{\text{vol}}, P_{\text{tone}}$).

### 5.6 Tone Naming, Bassist Lexicon & Display Character Budgets
All target voice naming, Tone3000 model exports (`.nam`), baked stems, and storefront listings must satisfy the normative guidelines in [`docs/tone_naming_guidelines.md`](file:///Users/peter/Projects/pethin/passivizer/docs/tone_naming_guidelines.md):
1. **First-Principles Grammar:** Every tone name must follow the strict 2-to-3 token grammar: `[Family] [Configuration] [Voicing Modifier]`. Combined with physical switch positions and version tags, the full model name is: `[Tone Name] [Pickup Position] v[dsp].[inst].[voice]`.
2. **Pedalboard Zero-Scroll Target ($\le 34$ chars):** The Darkglass Anagram screen displays 34 characters without scrolling. All 24 target voices must achieve 100% zero-scroll ($\le 34$ characters) on stage including pickup position and version tag.
3. **Character Length Hard Limit ($\le 64$ chars):** The Tone3000 uploader enforces a strict 64-character ceiling on model filenames (excluding `.nam`). Filenames exceeding 64 characters raise diagnostic `ValueError` exceptions in `get_t3k_basename()`.
4. **Permitted Taxonomy Vocabulary:** Draw tokens strictly from the approved taxonomy dictionary:
   - Families: `Jazz`, `Precision`, `StingRay`, `PJ`, `P∕MM`, `Rickenbacker`, `Mudbucker`, `Dingwall`, `Upright`, `Studio`, `Canonical`.
   - Configurations: `Pair`, `Bridge`, `Neck`, `Parallel`, `Series` (omitted when inherent, as in `Precision`, `Upright`, `Studio`).
   - Modifiers: `Open`, `Mids`, `Warm`, `Dub`, `Vintage`, `Modern`, `Active`, `Passive`, `Growl`, `Clank`, `Deep`, `Acoustic`, `Direct`.
5. **Musician-Facing Sonic Descriptors (No Component Values):** Strictly omit electrical component values (`22nF`, `47nF`, `100nF`) and ad-hoc artist nicknames. Replace them with functional musical descriptors (`Mids`, `Warm`, `Dub`, `Growl`).
6. **No Redundant Source Instrument Names:** Omit the source bass name from the tone name (e.g. `Precision Vintage [Split] v2.1.1` instead of `34in Standard P Precision Vintage [Split] v2.1.1`). Tone packs are dedicated to a specific source edition; repeating the instrument name wastes 15-25 characters and triggers scrolling.
7. **Standard Bracketed Physical Switch Selectors:** Multi-pickup source basses must use standardized physical selector tags: `[Parallel]`, `[Series]`, `[Neck]`, `[Bridge]`, `[Split]`. Single-pickup basses and character presets omit bracketed tags automatically.
8. **Filesystem Safety:** Replace `/` and `\` with the Unicode Division Slash (`∕`, `\u2215`) to maintain flat directories.

---

## 6. High-Performance Audio DSP & SIMD Directives

To maintain simulation speeds exceeding $>1500\times$ real time:
1. **Buffer & Recurrence Acceleration:** Never execute interpreted Python loops over audio buffers. Compile recursive ODE state solvers with Numba (`@njit(fastmath=True)`). Pack 24-bit little-endian WAV bytes via C view slicing (`.astype("<i4").view(np.uint8)`). Formulate nodal circuit transfers on complex NumPy vectors ($s = 1j \cdot \omega$).
2. **Stage Fusion & Caching:** Fuse linear filter stages in the frequency domain. Broadcast input forward FFTs across channels. Cache parsed netlists with `@functools.lru_cache`. Precompute global target voice dataframes once.
3. **Automated Verification & Test Evolution:** All changes must satisfy automated property assertions in `tests/test_guardrails.py` as well as zero errors under `uv run ruff check` and `uv run pyrefly check`. Hardcoded test values (dB thresholds, frequency ranges, bounds) are viable and expected to change when refining physical modeling or mathematical representations, provided the underlying formulas are justifiably sound. Never warp, tune, or artificially clamp genuine physical equations to appease obsolete or heuristic legacy test assertions; instead, update the tests to reflect the verified physical behavior.
4. **Visualizer Vectorization & Payload Bounding (Commit `7c6e634`):** Never invoke `build_voice_dataframe(mode="difference")`, FIR synthesis, or multi-rate FFTs inside per-pickup/per-voice loops of `build_composite_instrument_dataframe`. Always leverage globally cached universal target voicings (`get_cached_target_dfs(step=step)`) and analytical identity reflection ($\text{db\_back} = -\text{db\_front}, \text{db\_out} = 0.00\text{ dB}$) with vectorized NumPy array math. Ensure `step=3` downsampling (200 points) and 2-decimal float rounding to keep composite chart HTML payloads under $6\text{ MB}$ and `test_generate_all_charts` runtime strictly under $5.0\text{ seconds}$.
