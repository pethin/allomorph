# Allomorph

**Allomorph** is an analog modeling and digital twin pipeline that transforms bass pickup and transducer signals—across active, passive, and acoustic piezo topologies—into authentic digital twin voices of iconic high-impedance **passive pickup circuits, active preamps, and acoustic instruments**.

Designed specifically for NAM-capable pedalboards like the **Darkglass Anagram**, HeadRush, hardware IR loaders, and DAW plugin hosts, Allomorph produces both:
1. **Minimum-Phase Impulse Responses (IRs):** High-precision 48 kHz / 24-bit FIR filters for fast, zero-latency pickup re-voicing on standard IR loaders.
2. **Neural Amp Modeler (NAM) Profiles:** Nano/Feather/A2 neural captures trained on native WAV SPICE circuit simulations to capture dynamic magnetic saturation, eddy-current damping, volume pot loading, and treble-bleed interactions.

---

## The Philosophy: Universal Transducer Transformation

In modern bass signal chains, trying to convert a passive bass to sound active with digital EQ/IRs runs into fundamental mathematical limits: high-frequency content has already been attenuated by the passive coil's 12 dB/octave low-pass filter. Boosting those missing frequencies elevates noise and introduces comb-filtering artifacts.

**Allomorph provides bidirectional, universal transformation:**
* **Active Sources (e.g. EMG X-Series @ 18V):** Deliver a flat, wideband (20 Hz – 20+ kHz), high-headroom, low-noise signal with near-zero inductive peaking, allowing Allomorph to carve out the exact physical and electrical transfer function of any vintage passive pickup without boosting background hiss.
* **Passive Sources (e.g. Precision, Jazz, Mustang):** Re-voiced through true differential circuit deconvolution ($H_{\text{diff}} = H_{\text{tgt}} / H_{\text{src}}$), morphing physical vintage coils into modern active buffers, hot overwound humbuckers, or alternate scale placements.
* **Acoustic & Piezo Transducers:** Bridge force transducers and body cavity resonances faithfully emulated without artificial magnetic assumptions.

---

## Why Not Just an Impulse Response (IR)?

While an Impulse Response (IR) or FIR filter can reproduce a static frequency curve, physical guitar and bass pickups and transducers are fundamentally **non-linear, dynamic, reactive electro-mechanical systems**. Relying solely on a linear IR misses the core physical behavior and tactile response of a real instrument:

1. **Linear Time-Invariance (LTI) vs. Analog Dynamic "Give":**
   * An IR is strictly linear and time-invariant: plucking pianissimo ($pp$) or digging in with aggressive slap or heavy pick strokes ($ff$) produces the identical transfer function.
   * Real passive magnetic pickups exhibit dynamic core excursion non-linearities and flux compression when strings swing close to the pole pieces. Allomorph models this dynamic non-linearity via vector soft-knee saturation ($V_{\text{sat}} \cdot \tanh(v / V_{\text{sat}})$), reproducing the $1.5\text{--}2.5\text{ dB}$ of tactile compression, bloom, and dynamic "give" experienced when digging into real copper coils.

2. **Pre-Conditioning Downstream Distortion Stages:**
   * In a digital modeler like the **Darkglass Anagram**, Allomorph sits in **Block 1**, directly feeding high-gain preamps and overdrives (Microtubes B7K, Vintage Ultra, Alpha·Omega).
   * A linear IR passes high-headroom active transients through uncompressed, causing subsequent overdrive stages to clip on artificial, brittle spikes. An Allomorph neural model pre-conditions the signal with true passive saturation and impedance damping, ensuring downstream distortion blocks saturate smoothly and musically.

3. **Eddy Currents & Time-Domain Energy Storage:**
   * Metal components (pole pieces, baseplates, covers) generate circulating eddy currents that produce frequency-dependent damping ($R_{\text{eddy}}$) and subtle phase lag during rapid string transients.
   * An IR treats this as a stationary frequency cut. A **Neural Amp Modeler (NAM)** neural network captures the dynamic time-domain energy storage and release of the complete reactive RLC network.

4. **Multi-Coil Spatial Phase Summing:**
   * Instruments with multiple coils or pickups (Jazz Bass pairs, P/J, StingRay dual coils) feature complex spatial cancellation patterns that vary with string amplitude and string displacement. NAM neural models capture compound phase interactions and harmonic cancellation across the full frequency spectrum without comb-filtering artifacts or phase smearing.

> [!NOTE]
> Allomorph *can* synthesize zero-latency minimum-phase FIR impulse responses for ultra-lightweight linear filtering. However, its flagship pipeline trains lightweight **NAM neural models** (Architecture 2 / Nano/Feather) to preserve the full dynamic touch sensitivity, bloom, and analog feel of physical passive circuits.

---

## Source Instrument Hardware Architectures (34" Standard Scale)

While Allomorph supports any active, passive, or acoustic source bass, physical transducer hardware fundamentally determines the boundary conditions of deconvolution. On a standard **34.0" scale ($863.6\text{ mm}$)**, Allomorph establishes two reference hardware architectures depending on the engineering priority:

1. **The Actual Ideal Bass: Real Harmonic Dual-Transducer Architecture (Reverse PX + MMTWX)** — Built for maximum acoustic realism, tactile dynamic feedback, reverse-split string balancing, and standalone analog versatility by sampling string standing-wave modes at their physical harmonic antinodes.
2. **The Ideal Single-Pickup Bass: The Zero-Decoupling Reference Transducer** — Built for zero cognitive overhead, foolproof patch switching on stage, zero inter-pickup phase comb notches, and zero secondary magnetic string drag.

---

### Architecture 1: The Actual Ideal Bass — Real Harmonic Dual-Transducer (34" Scale)

In physical string mechanics, transverse wave reflections off the rigid bridge saddle impose an absolute, fret-invariant spatial comb filter envelope on string displacement:
$$H_{\text{pos}}(f) = \left| \sin\left(\frac{2\pi f x}{v}\right) \right|$$
where $x$ is the physical distance from the bridge saddle and $v = \sqrt{T/\mu}$ is the string's intrinsic transverse wave speed.

Because wave speed $v$ is constant for each tuned string, the cancellation nulls occur at **fixed frequencies in Hertz across all frets**:
$$f_{\text{null}, k} = k \cdot \frac{v}{2x}, \quad k \in \{1, 2, 3, \dots\}$$

A single pickup at any fixed location $x$ enforces a single set of comb nulls. By deploying a **Reverse EMG PX Split-Coil + EMG MMTWX Dual-Mode** with active parallel blending (EMG ABCX) on a 34" scale, the instrument physically captures real spatial wave mechanics at their authentic source locations:

```
[Nut] ════════════════════════════════════════════════════════════════════════════════ [Bridge]
                                  ◄─── Reverse PX ───►      ◄─── MMTWX ───►
                                   D/G: 145.4 mm            Neck: 77.4 mm
                                   E/A: 115.6 mm            Bridge: 54.6 mm
```

#### 1. Real Spatial Comb Filtering Across the Frequency Spectrum
* **Neck Transducer (Reverse EMG PX Split @ $130.5\text{ mm}$ Centerline):**
  * **Low-Frequency Output ($\propto f \cdot x$):** Large distance from the bridge saddle yields $+7.5\text{ dB}$ higher raw fundamental output ($20\log_{10}(130/55)$) than the bridge position, providing deep, authoritative fundamental body physically at the transducer.
  * **Forward D/G Half ($x = 145.4\text{ mm}$):** Senses high strings where physical excursion is naturally small, adding warm low-mids, singing body, and long sustain without thinness.
  * **Rearward E/A Half ($x = 115.6\text{ mm}$):** First comb null sits at $307\text{ Hz}$ on low E. Senses heavy low strings closer to the bridge where mechanical excursion is controlled, tightening low-end fundamental transient attack and eliminating flub.
* **Bridge Transducer (EMG MMTWX Dual-Mode @ $66.0\text{ mm}$ Centerline):**
  * **Dual-Coil Centerline ($x = 66.0\text{ mm}$):** Pushes the first spatial comb null up to $538\text{ Hz}$ on low E (leaving the entire low-mid growl passband completely un-notched). Its physical $0.90''$ coil spacing generates the authentic $2.5\text{ kHz}$ acoustic humbucker notch directly on the string.
  * **Single-Coil Mode ($x = 54.6\text{ mm}$):** Pushes the first comb null to $651\text{ Hz}$. Senses high transverse velocity and saddle witness-point boundary stiffness ($H_{\text{saddle}}$), delivering biting 60s/70s Jazz bridge growl and percussive attack.

#### 2. Physical & Dynamic Advantages Over a Single Point
* **Minimal Deconvolution Effort ($H_{\text{diff}} \approx 0\text{ dB}$):** When targeting a Vintage P-Bass (Voice 05) or a StingRay (Voice 09), the pickup is *already in the physically authentic position*. Block 1 only applies passive RLC circuit modeling and magnetic feel—never fighting the $+8.0\text{ dB}$ Wiener soft-knee boost ceiling to reconstruct missing comb frequencies.
* **Physical Dynamic Excursion & Soft-Knee Compression:** Large string displacement over the neck pickup drives the magnetic core into natural soft-knee flux compression ($V_{\text{sat}} \cdot \tanh(v/V_{\text{sat}})$) and dynamic inductance modulation ($\lambda_L$ "quack"), responding with tactile give when digging in hard.
* **Dual Plucking Hand Anchors:** Anchoring over the PX naturally places the plucking hand in the warm, round mechanical attack zone; anchoring over the MMTWX puts the hand in the tight, fast transient recoil zone.
* **Standalone Analog Fallback:** Without digital processing, the bass natively provides 4 distinct iconic sounds (Solo Reverse P, Solo StingRay Dual, Solo J-Bridge Single, and Blended P/MM).

---

### Architecture 2: The Ideal Single-Pickup Bass — The Zero-Decoupling Reference Transducer (34" Scale)

For players prioritizing **pure software decoupling with zero cognitive overhead on stage**, the single-pickup architecture eliminates all mechanical switching:

```
[12th Fret] ◄────────────── 338.3 mm ──────────────► [Pickup Center] ◄──── 93.5 mm ────► [Bridge Saddle]
```

#### 1. The Single-Pickup Philosophy: Total Decoupling & Zero Comb Nulls
* **No Comb-Filtering Nulls to Invert:** Humbuckers with dual coils under the same string introduce physical phase cancellation notches ($f = v / 2d \approx 2.5\text{ kHz}$) that cannot be cleanly inverted in DSP without boosting noise. A single line of sensing per string provides a clean, notch-free transfer function that allows Allomorph to synthesize any target aperture or dual-coil comb filter effortlessly.
* **Total Preset Decoupling:** Eliminates the "hand-foot desync" problem. You never have to adjust physical knobs or flip coil switches to match patch changes on your pedalboard; stepping on a Darkglass Anagram footswitch transforms the tone entirely in software.
* **Zero Secondary Magnetic Drag:** With only one pickup mounted, there are no secondary magnets dampening string vibration, maximizing open-string sustain and harmonic bloom.

#### 2. Recommended Pickup: EMG PX, EMG 35AX, or EMG 35P4X (18V)
* **Narrow Aperture ($w \le 0.75''$) Stacked Coaxial or Split-Coil:** Single sensing line per string ($d = 0.0''$). The first aperture null sits above $3.7\text{ kHz}$ on low E and above $8.8\text{ kHz}$ on G.
* **Hum-Canceling Common-Mode Rejection:** Stacked dummy coil or reverse split bobbins ensure 100% hum-free performance.
* **Ceramic/Alnico X-Series Preamp @ 18V:** Ultra-wide linear bandwidth, near-zero core saturation, and $>8.5\text{V}_{\text{p-p}}$ dynamic headroom with low $2\text{ k}\Omega$ output impedance (immune to cable loading).

#### 3. Lutherie Datums: The $93.5\text{ mm}$ ($3.68''$) Canonical Intermediate Datum
On a 34.0" scale length ($863.6\text{ mm}$), **$93.5\text{ mm}$** is the exact mathematical median between a 60s Jazz Bridge ($63.5\text{ mm}$) and a standard Precision Bass ($125.0\text{ mm}$), and serves as Allomorph's **Canonical Intermediate reference datum** ($\eta_{\text{datum}} = 10.83\%$):

$$\bar{x} = \frac{63.5\text{ mm} + 125.0\text{ mm}}{2} = 94.25\text{ mm} \approx \mathbf{93.5\text{ mm}}$$

* **Exact $0.00\text{ dB}$ Spatial Tilt to Canonical Baseline:** Because the physical pickup sits directly on the datum, deconvolution into Stage 3 requires zero spatial shelving tilt ($\Delta\eta = 0.000$).
* **Symmetric Filter Bounds:** Every voice in the 23-voice catalog is reachable within $\pm 3.7\text{ dB}$ of spatial shelving tilt, never exceeding the $+8.0\text{ dB}$ Wiener soft-knee boost ceiling.

#### 4. Disadvantages & Trade-offs Relative to Real Harmonic Sampling
While operationally foolproof, the single-pickup design introduces distinct acoustic and physical compromises:
1. **Synthetic vs. Real Harmonics:** Bridge bite ($< 66\text{ mm}$) and neck warmth ($> 125\text{ mm}$) must be synthesized through DSP shelving filters ($H_{\text{tilt}}$), rather than captured from real standing-wave antinodes.
2. **No Per-String Harmonic Stagger:** A straight pickup senses all strings at the same distance, lacking the mechanical bass-tightening and treble-warming of a reverse split.
3. **Single Plucking Anchor:** Restricts physical thumb placement to a single position ($93.5\text{ mm}$).
4. **Neutral Standalone Voicing:** Without Block 1 processing, the instrument outputs a sterile intermediate tone without iconic standalone character.

---

### Head-to-Head Architectural Comparison (34" Scale)

| Architectural Metric | Architecture 1: Real Harmonic Dual-Transducer (Reverse PX + MMTWX) | Architecture 2: Single-Pickup Reference (Centroid @ $93.5\text{ mm}$) |
| :--- | :--- | :--- |
| **Harmonic Mode Sampling** | **Physical Antinodes** (Low modes @ $130.5\text{ mm}$, High modes @ $66.0\text{ mm}$) | **Single Fixed Point** (Mode 9 node @ $96\text{ mm}$ suppressed) |
| **Allomorph Filter Effort** | **Minimal** ($H_{\text{diff}} \approx 0\text{ dB}$ on P-Bass & MM targets) | **Moderate** ($\pm 3.7\text{ dB}$ synthetic shelving tilt required) |
| **String Balance (E/A vs D/G)** | **Physically Optimized** via Reverse-P stagger ($115.6\text{ mm}$ vs $145.4\text{ mm}$) | **DSP Corrected** via 24-point wave-speed continuum |
| **Right-Hand Thumb Rests** | **Dual Positions** (Deep neck warmth or tight bridge recoil) | **Single Position** ($93.5\text{ mm}$) |
| **Pedalboard Decoupling** | Requires matching pickup switch to Anagram preset | **100% Decoupled** (Never touch switches on stage) |
| **Secondary Magnetic Drag** | Slight (two active low-flux magnetic apertures) | **Absolute Zero** (Single active pickup) |
| **Standalone / Direct Sound** | **4 Iconic Pro Tones** natively (P, MM, J-Bridge, P/MM) | **1 Neutral Intermediate Tone** |
| **Ideal Player Use Case** | Pure tone aficionados, recording sessions, tactile players | Touring pedalboard players, patch-heavy live sets |

---

## Modeling Architecture

Allomorph models the complete electro-acoustic path in four distinct layers:

```
[String Vibration]
        │
        ▼
1. Magnetic Aperture & Spatial Comb Filtering
   ├── Coil Sensing Width (w): High-frequency sinc filtering across multi-string wave speeds
   └── Dual-Coil Spacing (d): Humbucker spatial phase cancellation
        │
        ▼
2. Coil RLC Network (Passive Digital Twin)
   ├── Coil Inductance (L) & DC Resistance (R_dc)
   ├── Inter-Winding Self-Capacitance (C_coil)
   └── Eddy-Current Core Damping (R_eddy) in pole pieces & blades
        │
        ▼
3. On-Instrument Control Circuit
   ├── Master Volume Potentiometer (500k audio taper)
   ├── Hybrid Treble-Bleed Network (1.0 nF || 150 kΩ + 20 kΩ)
   └── Optional Capacitive C-Switch Bank
        │
        ▼
4. Output Environment
   ├── Instrument Cable Capacitance (~750 pF for standard 15-ft cable)
   └── Receiver Input Impedance (1 MΩ || 30 pF Darkglass Anagram input stage)
```

### Scale-Length & Multi-Scale Transformation
Allomorph converts the lower tension and warm low-mid "bloom" of **30" short-scale** and **32" medium-scale** instruments into the focused, piano-like authority of full-scale and fanned-fret instruments:
* **Wave-Speed Scaling ($\kappa_v$):** Up-shifts aperture and comb-filter null frequencies by $+13.3\%$ (for 34") and $+23.3\%$ (for 37" multi-scale).
* **String Tension Filtering:** Tightens tubby $180\text{--}250\text{ Hz}$ boom while adding laser-tight sub-bass ($40\text{--}80\text{ Hz}$) and metallic Dingwall-style clank ($2.5\text{--}3.8\text{ kHz}$).
* **Spatial Placement Tracking:** Relocates pickups from short-scale bridge datums to standard 34" and 37" sweet spots.

---

### Target Voice Catalog (21 Master Configurations & Acoustic Transducers)

Allomorph includes pre-configured physical and electrical parameters for **21 distinct pickup topologies, active buffers, and transducers** including active 2-band preamps, Stellartone ToneStyler discrete capacitive switching, fanned-fret multi-scale, and upright double bass (see [`docs/voice_catalog.md`](file:///Users/peter/Projects/pethin/passivizer/docs/voice_catalog.md) for full engineering specifications):

| # | Profile ID | Pickup Type | Topology | Harness / Controls | $L_{\text{eq}}$ | $f_r$ (Peak) | Circuit & Acoustic Character |
| :- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **01** | `01_modern_jazz_active` | Modern Active Jazz | Active 2-Band | Sadowsky 2-Band ($R_{\text{in}}=1\text{M}\Omega, R_{\text{out}}=100\,\Omega$) | $1.69\text{ H}$ (isolated) | $7.8\text{ kHz}$ | Isolated 60s J-pair with Sadowsky active 2-band boost; wideband sparkle with $+4\text{ dB}$ bass/treble. |
| **02** | `02_jazz_bass_pair` | Vintage 60s J-Bass Pair | Dual Parallel | Vintage $2\times 250\text{k}\Omega$ Vol ($125\text{k}\Omega$ net), $250\text{k}\Omega$ Tone, $47\text{nF}$ | $1.69\text{ H}$ | $2.7\text{ kHz}$ | Dual narrow single-coils in parallel; 60s $92.1\text{ mm}$ ($3\frac{5}{8}''$) aperture scoop with woody resonance. |
| **02b**| `02b_jazz_bass_pair_22nf`| 60s J-Bass Pair (22nF) | Dual Parallel | Vintage $2\times 250\text{k}\Omega$ Vol, 22nF ToneStyler Pure Shunt ($R_{\text{tone}}=3.3\,\Omega$) | $1.69\text{ H}$ | $762\text{ Hz}$ | Vocal midrange honk ($762\text{ Hz}$ peak, $-3\text{ dB}$ at $1225\text{ Hz}$); pure capacitive shunt preserves Jaco bridge growl with zero wiper mud. |
| **02c**| `02c_jazz_bridge_growl_bias`| Jaco Biased J-Pair (100%/75%) | Dual Parallel Decoupled | Bridge 100% ($0\,\Omega$), Neck 75% ($55\text{k}\Omega$ wiper decoupling), $250\text{k}\Omega$ Tone | $1.69\text{ H}$ | $3.4\text{ kHz}$ | Signature Jaco Pastorius vocal bridge growl; $55\text{k}\Omega$ neck decoupling shifts notch to $550\text{--}800\text{ Hz}$ with $3.4\text{ kHz}$ bite. |
| **03** | `03_jazz_bridge_60s` | 60s J-Bass Bridge | Single Coil | Vintage $250\text{k}\Omega$ Vol, $250\text{k}\Omega$ Tone, $47\text{nF}$ | $3.60\text{ H}$ | $2.8\text{ kHz}$ | 60s bridge single-coil ($63.5\text{ mm}$ datum); focused midrange bite, authentic Jaco growl. |
| **04** | `04_modern_p_ceramic` | Modern Split P | Single Split | Boutique $500\text{k}\Omega$ Vol/Tone, $22\text{nF}$ Cap, Treble Bleed | $4.80\text{ H}$ | $2.4\text{ kHz}$ | Modern ceramic split-coil (Bartolini 8CBP); punchy attack with extended clarity from 500k harness. |
| **05** | `05_vintage_62_p_alnico` | Vintage '62 P (Tone Open)| Single Split | Vintage CTS $250\text{k}\Omega$ Vol/Tone Open, $47\text{nF}$ PIO Cap | $3.80\text{ H}$ | $2.1\text{ kHz}$ | Classic Alnico V split-coil wide open; touch-sensitive dynamic response, woody organic bloom ($4.0\text{ kHz}$ cutoff). |
| **05b**| `05b_vintage_62_p_22nf` | Vintage '62 P (22nF) | Single Split | 22nF ToneStyler Pure Shunt ($R_{\text{tone}}=3.3\,\Omega$) | $3.80\text{ H}$ | $440\text{ Hz}$ | Modern Fender spec; punchy $440\text{ Hz}$ low-mid resonant focus (+1.5 dB), $-3\text{ dB}$ cutoff at $750\text{ Hz}$. |
| **05c**| `05c_vintage_62_p_47nf` | Vintage '62 P (47nF) | Single Split | 47nF ToneStyler Pure Shunt ($R_{\text{tone}}=3.3\,\Omega$), Flatwound Heavy | $3.80\text{ H}$ | $450\text{ Hz}$ | Authentic Jamerson Motown flatwound thump; 47nF ToneStyler pure capacitive shunt with pillowy low-end warmth. |
| **05d**| `05d_vintage_50s_p_100nf`| Vintage '50s P (100nF) | Single Split | 100nF ToneStyler Pure Shunt ($R_{\text{tone}}=3.3\,\Omega$) | $3.80\text{ H}$ | Sub-bass | Original 1951–1959 Fullerton factory paper-in-oil spec; massive sub-bass shelf rolloff ($-3\text{ dB}$ at $240\text{ Hz}$, deep Motown / reggae dub thump). |
| **07** | `07_modern_pj_active` | Modern Active P/J | Active 2-Band | Sadowsky/Spector 2-Band ($R_{\text{in}}=1\text{M}\Omega, R_{\text{out}}=100\,\Omega$) | $2.06\text{ H}$ (isolated) | $7.6\text{ kHz}$ | Active 2-band boost with isolated ceramic P/J coils; punchy sub-bass fundamental, wideband sparkle, and aggressive clank. |
| **08** | `08_vintage_pj_passive` | Vintage '80s Passive P/J | Parallel Sum | Dual $250\text{k}\Omega$ Vol ($125\text{k}\Omega$ net), $250\text{k}\Omega$ Tone, $47\text{nF}$ | $1.85\text{ H}$ | $2.8\text{ kHz}$ | Vintage Alnico V split-P and 60s J-bridge in parallel; authentic '80s Fender Special / Yamaha BB thump with woody mid-punch. |
| **09** | `09_stingray_mm_parallel` | Music Man StingRay | Active MM Buffer | Music Man 2-Band ($R_{\text{in}}=1\text{M}\Omega, R_{\text{out}}=100\,\Omega$) | $1.20\text{ H}$ (isolated) | $8.5\text{ kHz}$ | Authentic active 2-band MM humbucker; cable isolation, comb notch at $2.5\text{ kHz}$, metallic clank. |
| **09b**| `09b_stingray_mm_series` | Music Man MM (Series) | Active Series Buffer | Music Man 2-Band Preamp Buffer | $4.80\text{ H}$ (isolated) | $4.1\text{ kHz}$ | Dual-coil humbucker in series with active buffer; $+5.6\text{ dB}$ series EMF surge and focused $4.1\text{ kHz}$ active resonance. |
| **10** | `10_rickenbacker_bridge_hpf` | High-Pass Bridge | Series HPF | Factory Rickenbacker $330\text{k}\Omega$ Vol/Tone, $4.7\text{nF}$ Series Cap | $3.80\text{ H}$ | $2.2\text{ kHz}$ | High-output bridge coil with vintage $4.7\text{ nF}$ series capacitor; tight high-pass cut below $150\text{ Hz}$. |
| **11** | `11_modern_pmm_active` | Modern Active P/MM | Active Buffer | Studio Active Buffer ($R_{\text{in}}=1\text{M}\Omega, R_{\text{out}}=100\,\Omega$) | $0.96\text{ H}$ (isolated) | $3.4\text{ kHz}$ | Authentic active parallel P/MM (Sandberg VM / Lakland 44-02); Split-P neck + MM parallel bridge into buffer; slap punch and growl. |
| **11b**| `11b_pmm_hybrid_series` | Modern Active P/MM (Series) | Active Series Buffer | Studio Active Buffer ($R_{\text{in}}=1\text{M}\Omega, R_{\text{out}}=100\,\Omega$) | $8.40\text{ H}$ (isolated) | $3.2\text{ kHz}$ | Split P and MM parallel humbucker wired in series before active buffer; $+5.8\text{ dB}$ inductive boost with zero cable drag. |
| **12** | `12_mudbucker_ultra_series` | Heavy Series MM | Ultra Series | Gibson $500\text{k}\Omega$ Vol/Tone, $22\text{nF}$ Cap | $14.40\text{ H}$| $1.2\text{ kHz}$ | Overwound dual-coil series humbucker; subterranean low end with natural high-frequency rolloff. |
| **13** | `13_dingwall_multiscale_bridge` | Multi-Scale MM | Angled Parallel | Dingwall Active Onboard Buffer ($R_{\text{in}}=1\text{M}\Omega, R_{\text{out}}=100\,\Omega$) | $2.30\text{ H}$ (isolated)| $7.3\text{ kHz}$ | 34"-37" fanned-fret angled bridge sweet spot ($48.0\text{ mm}$) with active buffer and stainless clank. |
| **14** | `14_upright_bridge_transducer` | Upright Transducer | Bridge Force | Direct $100\text{ M}\Omega$ Buffer, $15\text{ nF}$ Subsonic Cap | — | $4.5\text{ kHz}$ | Direct bridge force sensor (Underwood / Realist style); leaky integration, 32 Hz rumble cut, bridge compliance. Recommended with 3 Sigma AST IRs in cab block. |
| **15** | `15_neutral_character` | Neutral Character (Dynamic DI)| Character (Neutral) | Transparent Studio Buffer ($10\text{ M}\Omega \to 50\,\Omega$) | $0.00\text{ H}$ | Wideband | Preserves physical aperture and imparts only tier character (transparent bypass in Clean, organic Alnico V feel in Dynamic, overwound punch in Hot Rod). |
| **15b**| `15b_active_character` | Active Character (Active Buffer)| Character (Active) | Studio Ultra-High-Z Buffer ($10\text{ M}\Omega \to 50\,\Omega$) | $3.20\text{ H}$ (isolated) | $5.2\text{ kHz}$ | Removes passive cable loading ($750\text{ pF}$) and pot damping to restore wideband hi-fi sparkle and headroom; preserves natural pickup aperture while stacking with tier dynamics. |
| **15c**| `15c_passive_character` | Passive Character (Passive Loading)| Character (Passive) | Standard Passive Harness ($250\text{k}\Omega\text{ Vol/Tone}, 47\text{nF}, 750\text{pF}$) | $4.20\text{ H}$ | $2.8\text{ kHz}$ | Adds high-impedance passive character, resonant peak ($2.8\text{ kHz}$), $750\text{ pF}$ cable loading, and $250\text{k}\Omega$ pot damping to active basses or stacks passive tone; preserves natural aperture. |

---

## SPICE $\to$ NAM Pipeline & CLI Usage

Allomorph models acoustic aperture and scale tension in Python, executes the passive circuit digital twin directly using its native WAV SPICE simulator, and trains lightweight NAM (`.nam`) neural captures for Block 1 of the Darkglass Anagram:

### 1. Interactive Acoustic & Electrical Visualizer (`scripts/analyze_voices.py`)
Renders interactive frequency response curves in Altair (Vega-Lite), comparing all 21 target configurations against any source instrument. Outputs are organized into per-instrument standalone charts and a unified interactive portal:

```bash
# Generate interactive charts for all configured source instruments and refresh master portal:
uv run python scripts/analyze_voices.py

# Generate or refresh for a specific instrument (preserves all other instrument charts):
uv run python scripts/analyze_voices.py --instrument 32in_fretless
uv run python scripts/analyze_voices.py --instrument 30in
```
*Outputs: Master interactive portal at `docs/frequency_responses.html` (and `docs/frequency_responses/index.html`) with embedded tabbed navigation and spec breakdown, and per-instrument standalone visualizations in `docs/frequency_responses/<instrument_id>.html`.*

### 2. Native WAV SPICE Circuit Simulation (`allomorph-sim`)
Directly streams raw bass calibration audio (`audio/canonical/optimal_bass_dry.wav`) through the entire physical digital twin in a single in-memory pass:
1. **Acoustic Aperture & Placement:** De-humbucking sinc aperture filtering, spatial standing-wave comb filtering, displacement tilt ($\Delta x$), and string tension filtering.
2. **Dynamic Non-Linear Compliance:** Soft-knee saturation ($V_{\text{sat}} \cdot \tanh(v / V_{\text{sat}})$), Lenz flux sag, Dahl hysteresis, back-EMF, and dynamic reluctance quack.
3. **Passive Pickup Circuit Twin:** Exact closed-form nodal AC transfer functions, eddy-current damping, authentic volume/tone pot dividers, active preamp buffers, hybrid treble bleed, cable capacitance ($750\text{ pF}$), and pedalboard load ($1\text{ M}\Omega \parallel 30\text{ pF}$).

Allomorph features a built-in **WAV SPICE simulator** running natively on Apple Silicon (`arm64`). Accelerated by a master 21-point Lossless $C^\infty$ SIMD, Tensor & Real-FFT Optimization Suite (pure real-FFT homomorphic cepstrum synthesis, 2D continuum tensor broadcasting, branchless 3-stage `fsqrt` rail limiters, Horner dipole polynomials, and passband-constrained oversampling), the engine evaluates exact analytical nodal equations and vector non-linearities directly in memory on the audio waveform, eliminating external SPICE dependencies (such as LTspice or ngspice) and intermediate disk writes while executing in ~0.8s per voice (>1500x faster than traditional transient SPICE engines):

```bash
# Run unified WAV SPICE simulation from raw audio for a specific voice (~0.8s):
uv run allomorph-sim --voice 04_modern_p_ceramic --instrument 30in

# Simulate all 21 voices in parallel across multi-core CPU (-j / --jobs):
uv run allomorph-sim --voice all --instrument 30in -j 8

# Rapid prototyping run on first 2 seconds (96,000 samples):
uv run allomorph-sim --voice 09_stingray_mm_parallel --max-samples 96000

# Run via master pipeline (Architecture C backend targets):
uv run allomorph --stage targets --voice 04_modern_p_ceramic
```

### 3. NAM Neural Model Training (Architecture 2 / A2)
Trains a high-efficiency **NAM Architecture 2 (A2)** neural model on the input/output audio pair. A2 replaces legacy A1 models (nano/feather/standard) with a "slimmable" neural architecture designed specifically for low-power hardware like the Darkglass Anagram:

```bash
# Train NAM Architecture 2 (A2) model for Darkglass Anagram Block 1:
# The input is the raw bass calibration signal (audio/canonical/optimal_bass_dry.wav) and the target is the simulated output:
nam train audio/canonical/optimal_bass_dry.wav audio/30in_emg_mmtw/out_04_modern_p_ceramic.wav ./models/30in_emg_mmtw/04_modern_p_ceramic.nam --architecture "A2"

# Run via the automated Allomorph trainer (defaults to Architecture 2 slimmable container with goal ESR <= 0.0080):
uv run allomorph --stage train --instrument 30in --voice 04_modern_p_ceramic

# Customize goal ESR or disable early stopping:
uv run allomorph --stage train --instrument 30in --voice 04_modern_p_ceramic --goal-esr 0.0050
uv run allomorph --stage train --instrument 30in --voice 04_modern_p_ceramic --no-goal-esr --epochs 400

# Train A2-Lite channels_8 only (instead of the default slimmable container):
uv run allomorph --stage train --instrument 30in --voice 04_modern_p_ceramic --a2-lite-only
```
*(In modern versions of `neural-amp-modeler` and the official Google Colab trainer, `--architecture A2` is the default. With the optimal bass dry excitation file, Allomorph trains the full slimmable Architecture 2 container (both `channels_3` and `channels_8`) by default under the **A2 Studio Reference** standard (`--goal-esr 0.0080`, `400` max epoch safety ceiling, `--batch-size 32`, monitoring `channels_8`). To isolate the 8-channel submodel only, supply `--a2-lite-only`).*

> [!NOTE]
> **Complete Training Guide & Audio Pairings:**
> For the complete reference table of input/target audio pairings for Block 1 (Frontend), Block 2 (Target), and 1-Block Baked models, see [`docs/training.md`](docs/training.md).

### 4. Master Automation Runner (`allomorph`)
Execute the entire pipeline or specific stages with a single command:

```bash
# Run complete pipeline for 30" source instrument (canonical -> 32 frontends -> targets -> viz):
uv run allomorph --instrument 30in

# Generate the Canonical Intermediate baseline sweep:
uv run allomorph --stage canonical

# Export all 32 native frontend deconvolution IRs:
uv run allomorph --stage frontends

# Simulate 3-tier backend universal target sweeps:
uv run allomorph --stage targets --voice 09_stingray_mm_parallel

# Generate interactive Altair frequency visualizations:
uv run allomorph --stage viz

# On-demand single-block monolithic bake (directly models source instrument to target voice into a single NAM capture):
uv run allomorph --stage bake --instrument 30in --voice 04_modern_p_ceramic --train
# (By default, --stage bake uses --tier dynamic to model saturation differentially between source and target,
#  and --pickup auto to automatically resolve the mapped pickup switch position).
```

---

### Signal Flow on the Darkglass Anagram

```
[Bass: Active / Passive / Piezo]
             │
             ▼
[Block 1: Allomorph NAM Preamp]
     └── Model: "07_stingray_mm_parallel.nam" (Nano/Feather neural capture)
             │
             ▼
[Block 2: Darkglass Preamp / Drive]
     └── Microtubes B7K, Vintage Ultra, or Alpha·Omega
             │
             ▼
[Block 3: Speaker Cabinet IR Loader]
     └── Ampeg 8x10, Darkglass 4x10, or custom speaker cab impulse
             │
             ▼
[Output to FOH / Audio Interface]
```

---

## Project Structure

```
allomorph/
├── README.md                              # Project vision, theory, architecture, and CLI guide
├── docs/                                  # In-depth technical guides
│   ├── architectural_guardrails.md        # Master mathematical reference handbook & derivations
│   ├── dsp_simulation_engine.md           # Virtual Analog SIMD simulation engine & Numba JIT
│   ├── configuration_reference.md         # Complete schema & field reference for TOML configurations
│   ├── voice_catalog.md                   # Complete passive pickup technical catalog & parameters
│   ├── circuit_theory.md                  # RLC, eddy current, and cable impedance math
│   ├── aperture_math.md                   # Magnetic aperture sinc, multi-string & scale physics
│   └── anagram_workflow.md                # Darkglass Anagram Block 1 routing & gain staging
├── config/                                # Modular TOML configuration files
│   ├── instruments/                       # Source bass geometries, pickups & embedded circuits
│   │   ├── 30in_emg_mmtw.toml             # 30" active EMG MMTW dual-mode bass
│   │   ├── 32in_custom_pmm.toml           # 32" custom PX + MMTWX bass
│   │   ├── 34in_standard_p.toml           # 34" standard P-bass template
│   │   └── 34in_standard_jazz.toml        # 34" standard Jazz bass template
│   ├── preamps.toml                       # Reusable active preamp catalog (Sadowsky, StingRay, Aguilar, Dingwall)
│   ├── scales.toml                        # Scale lengths & baseline string wave speeds
│   ├── strings.toml                       # Physical string core/wrap mechanical presets
│   └── voices/                            # 23 Target voice TOMLs with embedded [circuit] tables
│       ├── 01_modern_jazz_active.toml     # Sadowsky active 2-band isolated Jazz pair
│       ├── 02_jazz_bass_pair.toml         # Dual single-coils in parallel (tone open)
│       ├── 04_modern_p_ceramic.toml       # Modern ceramic split-coil P (500k)
│       ├── 05_vintage_62_p_alnico.toml    # Vintage '62 Alnico V split-P (tone open)
│       ├── 09_stingray_mm_parallel.toml   # Music Man parallel humbucker
│       └── ...                            # 23 total declarative voice models
├── src/                                   # Core reusable library package
│   └── allomorph/
│       ├── config/                        # Modular TOML configurations & geometry
│       │   ├── scales.py                  # Scale length & wave-speed loader
│       │   ├── strings.py                 # String mechanics presets loader
│       │   ├── voices.py                  # Voice registry & alias resolver
│       │   ├── instruments.py             # Source instrument loader & cache
│       │   └── geometry.py                # Pickup coils & aperture geometry resolution
│       ├── naming.py                      # UI slugs, tier prefixes, and CLI resolution
│       ├── dsp.py                         # Minimum-phase FIR synthesis and 24-bit WAV I/O
│       ├── physics/                       # Physical acoustic & spatial modeling subpackage
│       │   ├── strings.py                 # String mechanics, dispersion, wave continuum
│       │   ├── aperture.py                # Sinc & Bessel aperture integrals, saddle stiffness
│       │   ├── deconvolution.py           # Transducer electrical deconvolution biquads
│       │   └── prefilter.py               # Minimum-phase FIR prefilter synthesis
│       ├── circuit/                       # Native WAV SPICE circuit simulation subpackage
│       │   ├── parser.py                  # SPICE netlist tokenizer & CircuitModel
│       │   ├── solver.py                  # Analytical nodal RLC matrix solver & AC curves
│       │   ├── saturation.py              # State-space non-linear saturation & Numba kernels
│       │   ├── audio.py                   # Vectorized FFT convolution & 24-bit audio buffers
│       │   ├── simulation.py              # Audio simulation orchestration & batch workers
│       │   └── staging.py                 # Architecture C two-stage runner & allomorph-sim CLI
│       ├── visualizer/                    # Polars + Altair frequency visualization library
│       │   ├── dataframe.py               # Polars data modeling & continuum transfer curves
│       │   ├── charts.py                  # Interactive Altair visualization builders
│       │   └── portal.py                  # Responsive dark-mode HTML portal generator
│       ├── pipeline/                      # Multi-stage automation & batch orchestration
│       │   ├── stages.py                  # Visualization, prefilter, simulation, training stages
│       │   ├── batch.py                   # Concurrency pool & ProcessPoolExecutor runner
│       │   └── cli.py                     # Allomorph CLI argument parsing & workflow dispatcher
│       └── cli.py                         # Master CLI entrypoint delegation for `allomorph`
├── scripts/                               # Workflow utilities & CLI entrypoints
│   ├── analyze_voices.py                  # Thin delegating CLI wrapper for allomorph.visualizer
│   ├── train_nam.py                       # Local NAM A2 PyTorch/MPS GPU trainer
│   └── generate_tone3000_artwork.py       # Tone3000 storefront artwork generator
├── tests/                                 # Hierarchical pytest test suite (304 tests)
│   ├── circuit/                           # SPICE netlists, nodal RLC solving, ODE saturation, simulation
│   └── physics/                           # Aperture sinc filters, string mechanics, dispersion, FIR synthesis
└── models/                                # Exported .nam neural models
```

---

## Roadmap

### Completed Milestones
- [x] **Electro-Acoustic Physical Modeling:** Magnetic aperture sinc filtering, spatial comb nulls, scale-length wave-speed scaling ($30''/32'' \to 34''/37''$), 2D rod apertures, saddle boundary layer stiffness, longitudinal clank, and differential string tension modeling.
- [x] **Native WAV SPICE Simulator:** High-performance Apple Silicon engine (`allomorph-sim`) solving analytical nodal RLC equations, Foster 2-stage core eddy diffusion, Dahl magnetic domain-wall pinning hysteresis, asymmetric magnet saturation compliance, sub-audible 8 Hz DC blocking, passive RLC Johnson noise dither, and automatic output level normalization based on input sweep dBFS at >1500x speed.
- [x] **Lossless $C^\infty$ SIMD, Tensor & Real-FFT Optimization Suite:** Master 21-point mathematical vectorization accelerating FIR synthesis via pure real-FFT homomorphic cepstrum, 2D continuum tensor broadcasting, branchless 3-stage `fsqrt` rail limiting, passband-constrained oversampling, and analytical impedance Jacobians.
- [x] **21 Voice Profiles & Transducers:** Modern active 2-band Jazz, vintage single-coil, split-coil, series/parallel dual-coils, active Music Man, fanned multi-scale, upright double-bass bridge piezo force transducers, and flat dynamic twins.
- [x] **Interactive Visualization Portal:** Polars + Altair frequency response portal with spec sheets and per-instrument interactive charts (`docs/frequency_responses.html`).
- [x] **Automated NAM Training Pipeline:** End-to-end Architecture 2 (A2) neural model training targeting Darkglass Anagram Block 1.
- [x] **Automated Test Suite:** Comprehensive 304-test pytest verification covering physical filters, nodal transfer functions, real-FFT homomorphic cepstrum DSP, 2D tensor continuum, ODE saturation, audio simulation, and architectural guardrails.

### Upcoming Objectives
- [ ] **Interactive A/B Audio Auditioning CLI:** Terminal and real-time audio auditioning tool (`scripts/preview_voices.py`) with seamless dry-to-wet switching, looping bass riffs, and instantaneous A/B comparison across pickup voices before neural training or pedalboard export.
- [ ] **Hardware Reference Calibration:** Dry-DI spectral matching and A/B verification against physical vintage instruments (1962 P-Bass, 1975 Jazz Bass, 1979 StingRay).
- [ ] **In-Browser Audio Player:** Interactive audio preview player embedded directly into the Altair documentation portal.
- [ ] **Anagram Marketplace Native Block:** Develop a dedicated, all-in-one "Allomorph" custom block for the Darkglass Anagram Marketplace (`marketplace.anagram.shop`), featuring rotary voice switching across all 21 pickup configurations, automatic gain normalization, and interactive volume/cable load controls in a single native Block 1 module.

---

## Contributing

We welcome community contributions, netlists, and optimizations! Because Allomorph uses a dual-licensing model, all contributors must agree to the [Contributor License Agreement](CLA.md) via standard commit sign-off (`git commit -s`).

Please see [`CONTRIBUTING.md`](CONTRIBUTING.md) for full development setup, coding guidelines, and pull request instructions.

---

## License

This project and its distributed assets are licensed under the [PolyForm Noncommercial License 1.0.0](LICENSE).

### Scope & Permissions
- **Permitted Uses:** Free to use, study, modify, and distribute for personal study, experimentation, sound design, and noncommercial music production.
- **Coverage:** This license applies to all source code, SPICE netlists, configuration schemas, compiled/trained neural models (`.nam`), and synthesized impulse responses (`.wav`) generated by or distributed with Allomorph.
- **Commercial Restrictions:** Commercial use, sale, bundling into commercial plugins/pedalboards, or monetization of the software, neural profiles, or impulse responses is strictly prohibited without prior written permission and a commercial license from the author.
- **Commercial Licensing Inquiries:** Contact **Peter Nguyen** (<peter@phn.dev>).

