# Allomorph

**Allomorph** is a forward electroacoustic Digital Twin simulation and neural audio modeling pipeline that transforms active, passive, and acoustic piezo bass signals into authentic digital twins of iconic **passive pickup circuits, active preamps, and acoustic double bass transducers**.

Designed for neural-capable pedalboards like the **Darkglass Anagram**, Tone3000 cloud training, and DAW plugin hosts, Allomorph produces:
1. **Forward Digital Twin Wet Audio Stems:** 48 kHz / 24-bit PCM stems synthesized via native Modified Nodal Analysis (MNA) circuit simulation and 15-parameter non-linear magnetic dynamics, calibrated to ITU-R BS.1770-4 gated loudness ($-20.50\text{ LUFS}$) with a 4x oversampled true-peak safety ceiling ($\le 0.9900 / -0.09\text{ dBFS}$).
2. **Tone3000 Upload Bundles:** Turnkey training packs partitioned by physical pickup affinity (`neck`, `bridge`, `parallel`, `direct`) with zero-scroll pedalboard naming ($\le 34$ chars) and cryptographic SHA256 manifest provenance.
3. **Neural Amp Modeler (NAM) A2 Profiles:** Dual-tier slimmable Architecture 2 neural models (`channels_3` + `channels_8`) trained directly on `(Source Digital Twin → Target Digital Twin)` stem pairs for single-block front-end deployment in **Darkglass Anagram Block 1**.
4. **Interactive Frequency Response Portal:** Polars + Altair Vega-Lite interactive comparator ([`docs/frequency_responses.html`](docs/frequency_responses.html)) visualizing analytical forward and differential transfer curves across all instruments and voicings.

---

## The Philosophy: Universal Transducer Transformation

Converting a passive or rolled-off pickup to another instrument using linear digital EQ or static Impulse Responses (IRs) hits fundamental physical limits: linearly boosting attenuated high frequencies by $+20\text{ to }+40\text{ dB}$ amplifies thermal Johnson noise and pickup hiss, and linear multiplication cannot recover harmonics extinguished in a spatial comb null ($H_{\text{src}}(f) \approx 0$).

**Allomorph solves this via direct Forward Digital Twin simulation and non-linear neural modeling:**
- **Active Sources (e.g. 18V EMG X-Series, Active Soapbars, StingRay):** Deliver a wide-bandwidth, high-headroom signal that Allomorph maps into any vintage passive RLC circuit, ToneStyler capacitive shunt, or multi-scale voicing.
- **Passive Sources (e.g. Precision, Jazz, Mustang, Rickenbacker):** Both the source bass and target voice are simulated forward from master unvoiced string excitation (`audio/input.wav`), training a NAM WaveNet A2 network directly on `(Source Twin → Target Twin)`. Rather than applying noisy linear high-frequency boosts, the non-linear WaveNet synthesizes phase-coherent upper harmonics directly from the strong fundamental and low-order string overtones.
- **Acoustic & Piezo Transducers:** Models bridge saddle force transducers ($x = 0$), maple bridge rocking resonance ($800\text{ Hz}$), and wood mass damping ($3.2\text{ kHz}$).

### Why Not Just an Impulse Response (IR)?
While Allomorph synthesizes zero-latency minimum-phase FIR filters internally for linear acoustic and circuit stages, it trains **NAM Architecture 2 (A2)** neural models for hardware deployment because physical pickups are dynamic, non-linear systems:
1. **Analog Dynamic "Give":** Captures soft-knee magnetic flux compression, Lenz velocity drag, and dynamic reluctance "quack" ($\lambda_L$) when digging in hard ($ff$ vs $pp$).
2. **Pre-Conditioning Downstream Distortion:** Sits in **Darkglass Anagram Block 1** before high-gain drives (Microtubes B7K, Vintage Ultra, Alpha·Omega), conditioning active transients with authentic passive impedance damping.
3. **Eddy Currents & Reactive Energy Storage:** Reproduces frequency-dependent pole-piece skin dispersion ($\sqrt{s}$), Foster eddy diffusion, and Dahl domain-wall hysteresis.
4. **Multi-Coil Spatial Phase Summing:** Captures compound standing-wave comb interactions and register-weighted string pull across split and dual-coil geometries.
5. **Noise-Free Harmonic Restoration:** Synthesizes missing upper harmonics non-linearly without elevating the underlying input noise floor.

---

## Modeling Architecture & Signal Chain

Allomorph simulates the complete electroacoustic path in four physical layers (see [`docs/aperture_math.md`](docs/aperture_math.md) and [`docs/circuit_theory.md`](docs/circuit_theory.md)):

```
[Master String Excitation (audio/input.wav)]
        │
        ▼
1. Spatial Aperture & Scale Continuum (src/allomorph/physics/)
   ├── 2D Cylindrical Rod / 1D Blade Sensing Apertures & Humbucker Spacing (d)
   ├── 24-Point Log Wave-Speed Continuum, Inharmonicity RBF & Saddle Stiffness
   └── Causal Multi-Pickup Arrival Delays (tau_i) & Bridge Excursion Low-Shelf
        │
        ▼
2. Branch-Wise Non-Linear Magnetic & Op-Amp Dynamics (src/allomorph/circuit/)
   ├── 15-Parameter State-Space Magnetic Feel (Lenz Sag, Pull, Quack, Dahl Hysteresis)
   └── 2-Block Active Op-Amp Headroom (18V X-Series vs 9V Classic)
        │
        ▼
3. MNA Passive RLC & Active Preamp Harness (src/allomorph/circuit/)
   ├── Coil Inductance (L), DC Resistance, Self-Capacitance & Eddy/Skin Dispersion
   ├── Coupled 2x2 Mutual Matrix, Volume/Tone Wipers & ToneStyler Shunts
   └── Active 2/3-Band Preamp Buffers, 750 pF Cable Load & 1 MΩ Receiver Input
        │
        ▼
4. Loudness Staging & Neural Pairing (src/allomorph/dsp/ & trainer/)
   ├── ITU-R BS.1770-4 Gated Loudness (-20.50 LUFS) & 4x True-Peak Ceiling (<= 0.9900)
   └── NAM A2 WaveNet Training (Source Twin -> Target Twin)
```

### Signal Flow on the Darkglass Anagram
See [`docs/anagram_workflow.md`](docs/anagram_workflow.md) for hardware input gain staging ($-6.0\text{ to }-3.0\text{ dBFS}$) and preset bank layouts:

```
[Physical Bass (Knobs 100%)] ──► [Block 1: Allomorph NAM A2] ──► [Block 2: Darkglass Preamp/Drive] ──► [Block 3: Cab IR] ──► [FOH / USB]
```

---

## Target Voice Catalog (24 Master Voicings)

Allomorph models 24 target pickup voicings, active preamps, and transducers across 9 instrument families. For full electrical parameters ($L_{\text{eq}}, R_{\text{dc}}, C_{\text{coil}}, f_r$) and circuit schematics, see [`docs/voice_catalog.md`](docs/voice_catalog.md):

| Family | Target Voice IDs | Representative Topologies & Sonic Character |
| :--- | :--- | :--- |
| **Precision** | `precision_vintage`, `precision_mids`, `precision_warm`, `precision_dub`, `precision_active` | '62 Alnico V split-coil with CTS $250\text{k}\Omega$ open harness, $22\text{nF}/47\text{nF}/100\text{nF}$ ToneStyler capacitive shunts, and Sadowsky 2-band active ceramic split-P. |
| **Jazz** | `jazz_pair_open`, `jazz_pair_mids`, `jazz_pair_active`, `jazz_bridge_growl`, `jazz_bridge_open`, `jazz_neck_warm` | 60s dual single-coils in parallel ($125\text{k}\Omega$ net load), $22\text{nF}$ vocal mids, Sadowsky 2-band active slap contour, $55\text{k}\Omega$ decoupled Jaco bridge growl, and solo pickups. |
| **StingRay** | `stingray_parallel`, `stingray_series` | Music Man bridge humbucker ($66.0\text{ mm}$ datum) with active 2-band preamp in classic parallel ($f_r = 8.5\text{ kHz}$) and high-output series ($+5.6\text{ dB}$ surge). |
| **Dingwall** | `dingwall_bridge`, `dingwall_middle`, `dingwall_parallel` | $34''\text{--}37''$ multi-scale fanned-fret angled neodymium FD3n pickups with active buffer and stainless roundwound harmonic extension. |
| **Rickenbacker** | `rickenbacker_clank`, `rickenbacker_open` | High-gain bridge single-coil with vintage $4.7\text{ nF}$ inline series high-pass clank cap, and warm open toaster neck pickup. |
| **PJ** | `pj_passive`, `pj_active` | Split-P neck + 60s J-bridge in vintage passive parallel ($250\text{k}\Omega$ harness) and modern active 2-band ceramic configuration. |
| **P∕MM** | `p_mm_parallel`, `p_mm_series` | Split-P neck + MM humbucker bridge into studio active buffer in parallel slap and series low-mid punch modes. |
| **Mudbucker** | `mudbucker_deep` | Gibson EB-style $14.4\text{ H}$ overwound series sidewinder at the neck position ($f_r = 1.2\text{ kHz}$) with massive sub-bass excursion. |
| **Upright** | `upright_piezo` | Orchestral 3/4 double bass bridge saddle piezo force transducer with $800\text{ Hz}$ maple rocking resonance and Spirocore compliance. |

---

## Quickstart & CLI Usage

Allomorph uses **`uv`** and provides a unified CLI (`allomorph`) alongside the standalone circuit runner (`allomorph-sim`):

```bash
# Install dependencies and sync environment
uv sync

# 1. Browse configured source instruments, target voices, and Tone3000 packs
uv run allomorph list

# 2. Generate interactive Polars + Altair frequency response charts & HTML portal
uv run allomorph viz

# 3. Synthesize 24-bit Forward Digital Twin wet audio stems
uv run allomorph stems generate all
uv run allomorph stems generate 30in_emg_mmtw:precision_active

# 4. Audit synthesized stems for BS.1770-4 loudness (-20.50 LUFS) & true-peak safety (<= 0.9900)
uv run allomorph audit

# 5. Package turnkey Tone3000 upload bundles (bundles/<pickup>/)
uv run allomorph pack packs

# 6. Train NAM Architecture 2 (.nam) models (native in-VRAM PyTorch or stock Lightning engine)
uv run allomorph train 30in_emg_mmtw:precision_active --engine native
uv run allomorph train pack:34in_standard_p --epochs 120 --min-epochs 100 --patience 15

# 7. Run low-level WAV SPICE circuit simulation directly
uv run allomorph-sim --voice precision_active --instrument 30in_emg_mmtw
```

---

## Documentation Index

| Document | Scope & Contents |
| :--- | :--- |
| **[`docs/architectural_guardrails.md`](docs/architectural_guardrails.md)** | Master invariant index, $C^\infty$ mathematical smoothness primitives (`cinf_smoothstep`, `alg4`, `fsqrt`), and fail-fast contracts. |
| **[`docs/aperture_math.md`](docs/aperture_math.md)** | Spatial aperture kernels (2D rod vs 1D blade), 24-point wave-speed continuum, inharmonicity RBF, saddle stiffness, and 34" placement physics. |
| **[`docs/circuit_theory.md`](docs/circuit_theory.md)** | Passive RLC networks, active preamps, Foster/Jordan/skin dispersion, Cole-Davidson dielectrics, coupled $2\times 2$ coils, and 15-parameter magnetic metallurgy. |
| **[`docs/dsp_simulation_engine.md`](docs/dsp_simulation_engine.md)** | Native Virtual Analog SIMD execution pipeline, 2x/4x oversampling, stage fusion, Numba `@njit` ODE kernels, and `allomorph-sim` CLI reference. |
| **[`docs/dsp_synthesis_engine.md`](docs/dsp_synthesis_engine.md)** | Master excitation (`audio/input.wav`), 2D modal plucks, string tilt laws, pure real-FFT homomorphic cepstrum FIRs, and BS.1770-4 LUFS leveling. |
| **[`docs/trainer/README.md`](docs/trainer/README.md)** | NAM A2 pairing theory, non-linear harmonic restoration, Native vs Stock training engines, hyperparameter defaults, and `allomorph train` / `audit`. |
| **[`docs/voice_catalog.md`](docs/voice_catalog.md)** | Engineering catalog of all 24 target pickup voicings, RLC component values, and control harnesses. |
| **[`docs/configuration_reference.md`](docs/configuration_reference.md)** | Declarative TOML schema reference for `config/instruments/`, `config/packs/`, `preamps.toml`, `scales.toml`, and `strings.toml`. |
| **[`docs/tone_packs/naming.md`](docs/tone_packs/naming.md)** | Normative bassist tone naming grammar, taxonomy dictionary, and zero-scroll character budgets ($\le 34$ chars stage / $\le 64$ chars Tone3000). |
| **[`docs/tone_packs/ranking.md`](docs/tone_packs/ranking.md) & [`catalog.md`](docs/tone_packs/catalog.md)** | Storefront voicing ranking architecture and Tone3000 pack catalog. |
| **[`docs/anagram_workflow.md`](docs/anagram_workflow.md)** | Darkglass Anagram Block 1 signal routing, input gain calibration, and stage bank organization. |
| **[`docs/clamp_and_limit_registry.md`](docs/clamp_and_limit_registry.md)** | Catalog-wide registry of empirically verified clamps, limiters, and regularization floors. |

---

## Contributing & License

We welcome contributions! All commits must be signed off (`git commit -s`) under our [Contributor License Agreement (`CLA.md`)](CLA.md). See [`CONTRIBUTING.md`](CONTRIBUTING.md) for development setup, strict `pyrefly` / `ruff` checks, and test standards, and [`AGENTS.md`](AGENTS.md) for AI agent discovery guidelines.

Licensed under the [PolyForm Noncommercial License 1.0.0](LICENSE). For commercial licensing inquiries, contact **Peter Nguyen** (<peter@phn.dev>).
