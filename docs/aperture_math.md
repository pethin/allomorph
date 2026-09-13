# Magnetic Aperture & Spatial Comb-Filtering Math

This document details the spatial, mechanical, and string-vibration physics modeled in **Allomorph** to recreate physical pickup geometry.

---

## 1. Magnetic Aperture Width ($w$)

A magnetic pickup does not sample the vibrating string at an infinitesimal point. Instead, its permanent magnets and pole pieces create a sensing field with a finite spatial aperture $w$ along the string's axis.

```
                  ◄────── w ──────►
                  ┌───────────────┐
String: ══════════│═══════════════│═══════════
                  └───────────────┘
                     Pickup Pole
```

### The Sinc Comb-Filter Effect
Because the pickup integrates string motion across width $w$, wavelengths shorter than $w$ cancel out across the pole. In the frequency domain, assuming a string wave speed $v$, the transfer function is modeled by the unnormalized sinc function:

$$H_{\text{aperture}}(f) = \left| \frac{\sin\left(\frac{\pi f w}{v}\right)}{\frac{\pi f w}{v}} \right| = \left| \text{sinc}\left( \frac{f \cdot w}{v} \right) \right|$$

#### First Aperture Null:
$$f_{\text{null}} = \frac{v}{w}$$

* **Narrow Aperture (Jazz Bass single-coil, $w \approx 0.75''$):** The first null sits higher up in the frequency spectrum, preserving harmonic transients, "snap," and pick attack.
* **Wide Aperture (Precision Bass / Music Man, $w \approx 1.00''\text{ to }1.50''$):** The null pulls down into the upper treble ($4\text{--}6\text{ kHz}$), creating a naturally warmer, thicker, low-pass characteristic.

---

## 2. Multi-String Wave Speed Problem & The Allomorph Solution

In simple scripts like `precifier`, wave speed is assumed constant based on the open E string:
$$v_E = 2 \cdot L \cdot f_0 = 2 \cdot (34 \times 0.0254) \cdot 41.20 \approx 71.07\text{ m/s}$$

### The Multi-String Reality
Wave speed is governed by string tension ($T$) and linear mass density ($\mu$):
$$v = \sqrt{\frac{T}{\mu}} = 2 \cdot L \cdot f_{\text{open}}$$

For a standard 34" 4-string bass (standard gauge .045 – .105):

| String | Open Frequency ($f_0$) | Wave Speed ($v$) | Aperture Null ($w=1.0''$) |
| :--- | :--- | :--- | :--- |
| **E1** | $41.20\text{ Hz}$ | $71.07\text{ m/s}$ | $2,798\text{ Hz}$ |
| **A1** | $55.00\text{ Hz}$ | $94.93\text{ m/s}$ | $3,737\text{ Hz}$ |
| **D2** | $73.42\text{ Hz}$ | $126.73\text{ m/s}$ | $4,989\text{ Hz}$ |
| **G2** | $98.00\text{ Hz}$ | $169.15\text{ m/s}$ | $6,659\text{ Hz}$ |
| *(Low B)* | $30.87\text{ Hz}$ | $53.28\text{ m/s}$ | $2,098\text{ Hz}$ |

> [!WARNING]
> If an IR uses a single wave speed ($v_E = 71\text{ m/s}$), it forces a deep notch at $2.8\text{ kHz}$. While correct for the open E string, this creates an artificial, hollow notch on the D and G strings where the real notch is well above $5\text{ kHz}$.

### The Continuous Wave-Speed Continuum Formulation
Rather than constraining models to fixed 4-string standard tunings ($E, A, D, G$), Allomorph employs a continuous, log-spaced wave-speed continuum $v(f_0) = 2 \cdot L(f_0) \cdot f_0$ spanning the entire physical operating register of the electric bass ($f_0 \in [30.87\text{ Hz}, 100.00\text{ Hz}]$, from Low B through High G):

$$H_{\text{composite}}(f) = \frac{1}{N} \sum_{i=1}^N \left| \text{sinc}\left(\frac{f \cdot w}{v_{\text{disp}}(f_{0,i})}\right) \right|, \quad N = 24$$

where each continuum point incorporates:
1. **Multi-Scale Fanned Fretboard Continuum Interpolation:**
   On standard single-scale instruments, $L(f_0) = L_{\text{inst}}$.
   On multi-scale instruments (e.g. $34''\text{--}37''$ Dingwall Combustion/NG, $32''\text{--}35''$ Dingwall SP1), vibrating scale length $L(f_0)$ smoothly and logarithmically interpolates from the longest scale at the lowest register down to the shortest scale at the highest register:
   $$t(f_0) = \frac{\log_2(f_0) - \log_2(f_{\text{min}})}{\log_2(f_{\text{max}}) - \log_2(f_{\text{min}})}$$
   $$L(f_0) = L_{\text{max}} - t(f_0) \cdot (L_{\text{max}} - L_{\text{min}})$$
   $$v_0(f_0) = 2 \cdot L(f_0) \cdot f_0$$
   where $f_{\text{min}} = 30.87\text{ Hz}$ (Low B) has scale $L_{\text{max}}$ ($37.0'' = 0.9398\text{ m}$ on NG; $35.0'' = 0.8890\text{ m}$ on SP1) and $f_{\text{max}} = 100.00\text{ Hz}$ (High G) has scale $L_{\text{min}}$ ($34.0'' = 0.8636\text{ m}$ on NG; $32.0'' = 0.8128\text{ m}$ on SP1).
2. **Dynamic Inharmonicity Dispersion ($v_{\text{disp}}(f)$):**
   High-frequency wave speed expands with flexural string stiffness:
   $$v_{\text{disp}}(f) = v_0(f_0) \sqrt{1 + B_s(f_0) \frac{(f / f_0)^2}{1 + (f / 3500\text{ Hz})^2}}$$
   where the string stiffness parameter $B_s(f_0)$ is continuously interpolated in log-frequency space across empirical anchor datums:
   $$\mathbf{f}_{\text{anchors}} = [27.50, 30.87, 41.20, 55.00, 73.42, 98.00, 130.81, 196.00]\text{ Hz}$$
   $$\mathbf{B}_{s,\text{anchors}} = [2.8, 2.5, 2.0, 1.2, 0.6, 0.3, 0.15, 0.08] \times 10^{-5}$$
3. **Geometric Register Half Routing:**
   Split-coil pickups (like the Precision Bass) dynamically evaluate lower-register continuum points ($i < N/2$) on the forward bass coil half and upper-register points ($i \ge N/2$) on the rearward treble coil half, completely independent of note names, tunings, or string gauges.
4. **Tuning and Gauge Invariance:**
   Seamlessly accounts for Standard, Drop D, Drop C, C Standard, D Standard, Drop A, 5-string (Low B), and 6-string setups without manual reconfiguration or retuning.
5. **2D SIMD Tensor Broadcasting:**
   To eliminate scalar iteration over 24 register points and thousands of frequency bins, Allomorph broadcasts the continuum into a 2D tensor matrix ($\mathbf{V}_{\text{disp}} \in \mathbb{R}^{K \times M}$ where $K=24$ and $M$ is the number of frequency bins):
   $$\mathbf{V}_{\text{disp}} = \mathbf{v}_0 \odot \sqrt{1 + \mathbf{B}_s \odot \frac{(\mathbf{F} \oslash \mathbf{f}_0)^2}{1 + (\mathbf{F} / 3500\text{ Hz})^2}}$$
   where $\mathbf{F} \in \mathbb{R}^{1 \times M}$ and $\mathbf{f}_0, \mathbf{v}_0, \mathbf{B}_s \in \mathbb{R}^{K \times 1}$. Aperture sinc factors, Bessel disc integrals, and standing-wave comb envelopes are evaluated across all registers simultaneously via vectorized NumPy array operations.

This eliminates discrete localized comb teeth and guarantees smooth, physically authentic spatial filtering across any instrument configuration.

---

## 3. Dual-Coil Humbucker Spatial Comb-Filtering ($d$)

In multi-coil pickups such as the **Bartolini MM42CBJD3 quad-coil** or vintage StingRay humbuckers, two coil centers are separated by distance $d$ along the string (typically $d \approx 0.75'' = 19.05\text{ mm}$):

```
        Coil 1 (Front)      Coil 2 (Rear)
             ┌───┐               ┌───┐
String: ═════│═══│═══════════════│═══│═════
             └───┘               └───┘
               ◄──────── d ───────►
```

When connected in phase (whether Parallel or Series), the summed signal exhibits spatial comb filtering:

$$H_{\text{comb}}(f) = \left| \cos\left( \frac{\pi f d}{v} \right) \right|$$

#### First Humbucker Notch:
$$f_{\text{notch}} = \frac{v}{2d}$$

For $d = 0.75''$ ($0.01905\text{ m}$) at average wave speed $v \approx 95\text{ m/s}$:
$$f_{\text{notch}} \approx \frac{95}{2 \times 0.01905} \approx 2,493\text{ Hz}$$

This spatial notch around $2.5\text{ kHz}$ is the acoustic fingerprint of a Music Man humbucker—it creates the mid-scoop that gives the StingRay its open, aggressive "growl" and slap clarity.

#### Physical Spatial Cross-Coherence Decay ($f > \frac{v}{d}$)
In an idealized 1D string model, the cosine comb pattern repeats indefinitely at odd harmonics ($f = 3 f_{\text{notch}}, 5 f_{\text{notch}}, \dots$). However, on physical wound bass strings, transverse vibration across dual pole pieces becomes diffuse and incoherent once the acoustic wavelength becomes comparable to or shorter than the coil spacing ($\lambda \le d$):
* **Coherent Regime ($\lambda > 2d$, $f < \frac{v}{2d}$):** Phasic wave interference dominates. The fundamental acoustic comb notch ($f_{\text{notch}}$) and the constructive rise ($f_{\text{peak}} = \frac{v}{d}$) are 100% preserved ($\gamma = 1.0$).
* **Wavelength-Dependent Transition ($\frac{v}{d} \le f \le 1.8 \frac{v}{d}$):** As frequency passes the fundamental constructive peak ($f_{\text{start}} = v/d$), the coherence decay transitions smoothly from coherent phase sum ($P_{\text{coh}} = |\sum w_i H_i|^2$) to incoherent power summation ($P_{\text{incoh}} = \sum w_i^2 |H_i|^2$):
  $$\gamma(f, v) = \frac{1}{2} \left[ 1 + \cos\left( \pi \cdot \text{clip}\left( \frac{f - f_{\text{start}}}{0.8 f_{\text{start}}}, 0, 1 \right) \right) \right]$$
  $$|H_{\text{blend}}(f)| = \sqrt{\gamma P_{\text{coh}} + (1 - \gamma) P_{\text{incoh}}}$$
* **Result:** Secondary harmonic nulls (such as the unphysical E-string notch at $5.6\text{ kHz}$) are naturally eliminated, producing a smooth, organic high-frequency response while strictly preserving the authentic low-mid humbucker scoop.

---

## 4. Scale-Length Transformation & Multi-Scale Physics (30"/32" $\to$ 34" / 37")

When translating a **30" short-scale** or **32" medium-scale** source bass into standard **34"** or **34"–37" multi-scale (Dingwall-style)** tones, Allomorph models three physical phenomena:

### A. Wave-Speed Scaling Ratio ($\kappa_v$)
Wave speed scales directly with vibrating length for any given pitch:
$$\kappa_v = \frac{L_{\text{target}}}{L_{\text{source}}}$$

* **30" to 34" Conversion:** $\kappa_v = \frac{34}{30} = 1.133\ (+13.3\%)$
* **32" to 34" Conversion:** $\kappa_v = \frac{34}{32} = 1.0625\ (+6.25\%)$
* **30" to 37" Multi-Scale Conversion:** $\kappa_v = \frac{37}{30} = 1.233\ (+23.3\%)$

As wave speed increases, the physical aperture and comb-filter null frequencies shift upward:
$$f_{\text{null, target}} = f_{\text{null, source}} \cdot \kappa_v$$
This lifts the frequency ceiling of the pickup, extending harmonic transient definition and high-register "snap."

### B. String Tension & Low-Mid De-Mudding ($H_{\text{tension}}$)
Short-scale (30") and medium-scale (32") instruments have lower string tension, which produces:
1. Higher mechanical displacement at the fundamental (warm, loose "tubby" bloom in the $180\text{--}250\text{ Hz}$ range).
2. Slightly softer high-harmonic transient attack.

In contrast, full-scale 34" and 37" multi-scale basses exhibit massive tension ($\sim 42\text{--}48\text{ lbs}$ on low B/E strings), producing:
* **Laser-tight, piano-like sub-bass** ($40\text{--}80\text{ Hz}$).
* **Controlled, clean low-mids** without muddiness.
* **Aggressive, percussive metallic clank** ($2.5\text{--}3.8\text{ kHz}$) that stays coherent under heavy Darkglass drive engines.

Allomorph models this acoustic transformation with a multi-band tension transfer filter:
* **For 34" Conversion:** $-2.0\text{ dB}$ dip @ $210\text{ Hz}$ ($Q=1.2$) $+$ $+1.8\text{ dB}$ high shelf @ $2.8\text{ kHz}$.
* **For Multi-Scale (Dingwall) Conversion:** $+1.5\text{ dB}$ sub focus @ $75\text{ Hz}$, $-3.5\text{ dB}$ de-mud @ $220\text{ Hz}$, $+3.5\text{ dB}$ metallic clank peak @ $3.2\text{ kHz}$, and $+2.0\text{ dB}$ top-end air @ $5\text{ kHz}$.

### C. Angled Multi-Scale Pickup Geometry
On fanned-fret instruments like the Dingwall NG2/NG3, pickups are mounted parallel to the fanned bridge saddles. This ensures that the sensing point relative to the scale line ($x / L$) remains uniform across all strings, eliminating the flubby low-end of straight pickups on low B and E strings while maintaining smooth treble on the G string. Allomorph compensates for this geometric alignment across string channels.

### D. Scale-Normalized Spatial Bridge Proximity Tilt ($\Delta\eta$)
String standing-wave vibrational modes obey $y_n(x) \propto \sin(n\pi x / L) = \sin(n\pi \eta)$, where $\eta = x / L$ is the fractional distance along the vibrating string length from the bridge saddle.
For the fundamental vibrational mode ($n=1$), $\sin(\pi \eta) \approx \pi \eta$ for small $\eta \ll 1$. Thus, the fundamental displacement amplitude scales directly proportional to distance from the bridge:
$$y_1(\eta) \propto \eta$$
At high frequencies ($f \ge 1\text{--}2\text{ kHz}$), the standing wave RMS envelope is constant ($1/\sqrt{2}$) everywhere along the string. Therefore, bridge proximity physically affects fundamental excursion, but does **not** alter high-frequency standing-wave amplitude.

#### Standing-Wave Displacement Ratio Low-Shelf Model ($H_{\text{pos}}$):
To model physical excursion scaling accurately without injecting artificial high-frequency boost or cut, Allomorph applies a **single low-shelf filter** governed by the physical logarithmic distance ratio:

$$\eta_{\text{tgt}} = \frac{x_{\text{tgt}}}{L_{\text{tgt}}}, \quad \eta_{\text{src}} = \frac{x_{\text{src}}}{L_{\text{src}}}$$

$$\Delta G = 20 \log_{10}\left(\frac{\eta_{\text{tgt}}}{\eta_{\text{src}}}\right)$$

$$\Delta G_{\text{soft}} = 8.0 \cdot \tanh\left(\frac{\Delta G}{8.0}\right)$$

$$g_0 = 10^{\Delta G_{\text{soft}} / 20.0}$$

$$H_{\text{pos}}(f) = \sqrt{\frac{g_0^2 + (f / 220\text{ Hz})^2}{1 + (f / 220\text{ Hz})^2}}$$

* **DC Fundamental Excursion ($f \to 0$):** $H_{\text{pos}}(0) = g_0 = 10^{\Delta G_{\text{soft}} / 20.0}$, precisely matching the physical fundamental standing wave displacement ratio.
* **High-Frequency Invariance ($f \gg 220\text{ Hz}$):** $H_{\text{pos}}(\infty) \equiv 1.0000$ ($0.00\text{ dB}$ identity). Zero artificial treble counter-tilt or harsh clank injection.
* **2-Block Decoupled Architecture Identity:** Because logarithmic distance ratios are strictly additive:
  $$\Delta G_{\text{B1}} + \Delta G_{\text{B2}} = 20\log_{10}\left(\frac{\eta_{\text{can}}}{\eta_{\text{src}}}\right) + 20\log_{10}\left(\frac{\eta_{\text{tgt}}}{\eta_{\text{can}}}\right) \equiv 20\log_{10}\left(\frac{\eta_{\text{tgt}}}{\eta_{\text{src}}}\right) = \Delta G_{\text{direct}}$$
  the two-stage decoupled pipeline satisfies the spatial scaling identity bit-exact.
* **Proportional Sweet Spot Invariance:** Identical proportional locations across scales (e.g. 32" MM @ $62.2\text{ mm} \implies \eta = 7.65\%$ vs 34" MM @ $66.0\text{ mm} \implies \eta = 7.64\%$) have $\eta_{\text{tgt}} / \eta_{\text{src}} \approx 1.0$ and receive **exact zero spurious scaling** ($0.00\text{ dB}$ flat).

#### Proportional Scale Tension Snap ($H_{\text{tension}}$):
When converting from shorter scales to longer scales ($L_{\text{src}} < L_{\text{tgt}}$), Allomorph applies a $C^\infty$ infinitely differentiable softplus tension snap:
$$\Delta L = L_{\text{tgt}} - L_{\text{src}}$$
$$\Delta L_{\text{soft}} = \frac{1}{2} \ln(1 + e^{2 \Delta L})$$
$$\text{snap}_{\text{dB}} = 3.5 \cdot \tanh\left(\frac{1.8 \Delta L_{\text{soft}}}{4.0 \times 3.5}\right)$$
$$H_{\text{tension}}(f) = \sqrt{\frac{1 + 10^{\text{snap}_{\text{dB}} / 10.0} \cdot (f / 2800\text{ Hz})^2}{1 + (f / 2800\text{ Hz})^2}}$$

Yields $+1.8\text{ dB}$ for 30" short scale, $+0.9\text{ dB}$ for 32" medium scale, and $0.0\text{ dB}$ for standard 34" scale, restoring the tight piano-like high-frequency snap of higher string tension.

---

## 5. Active Pickup Electrical Resonance & Wiener Deconvolution

Active EMG pickups incorporate internal low-impedance coils loaded directly into an onboard differential op-amp buffer. Unlike passive pickups whose resonance is dictated by external guitar cables and amplifier inputs, an active pickup's electrical resonance is determined internally:

$$|H_{\text{elec}}(f)| = \frac{1}{\sqrt{\left(1 - \left(\frac{f}{f_r}\right)^2\right)^2 + \left(\frac{f}{Q f_r}\right)^2}}$$

### Official EMG Hardware Datums

| Pickup Model | Mode | Magnet Type | Resonant Frequency ($f_r$) | Typical $Q$ | Output Level (String / Peak) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **EMG MMTW / MMTWX** | Dual-Coil | Ceramic / Steel | **$2,500\text{ Hz}$** ($2.50\text{ kHz}$) | $1.35$ | $2.0\text{V} / 4.5\text{V}$ |
| **EMG MMTW / MMTWX** | Single-Coil (L2+L3) | Ceramic / Steel | **$3,500\text{ Hz}$** ($3.50\text{ kHz}$) | $1.40$ | $1.0\text{V} / 3.0\text{V}$ |
| **EMG P-X** | Split-Coil | Ceramic | **$3,200\text{ Hz}$** ($3.20\text{ kHz}$) | $1.40$ | $2.0\text{V} / 8.5\text{V}$ |
| **EMG PA-X** | Split-Coil | Alnico V | **$2,610\text{ Hz}$** ($2.61\text{ kHz}$) | $1.35$ | $2.0\text{V} / 8.5\text{V}$ |
| **EMG P-CS-X** | Split-Coil | Ceramic / Steel | **$2,610\text{ Hz}$** ($2.61\text{ kHz}$) | $1.35$ | $2.0\text{V} / 8.5\text{V}$ |

### Active Buffered Blends (EMG ABCX)
Because the EMG ABCX active blend potentiometer buffers each pickup input with dedicated op-amp stages prior to summing, there is zero passive mutual loading. The compound response is the exact linear weighted superposition:
$$H_{\text{blend, elec}}(f) = \frac{\sum_i w_i \cdot H_{\text{elec}, i}(f)}{\sum_i w_i}$$

### Biquad Anti-Resonance Equalizer ($H_{\text{anti}}$)
To eliminate the internal active resonant peak without causing high-frequency noise explosion or distorting the low-end gain structure during FIR normalization, Allomorph uses a 2nd-order biquad anti-resonance notch filter ($Q_{\text{target}} = 1.00$):

$$|H_{\text{anti}}(f)| = \frac{\sqrt{\left(1 - \left(\frac{f}{f_r}\right)^2\right)^2 + \left(\frac{f}{Q_{\text{src}} \cdot f_r}\right)^2}}{\sqrt{\left(1 - \left(\frac{f}{f_r}\right)^2\right)^2 + \left(\frac{f}{Q_{\text{target}} \cdot f_r}\right)^2}}$$

* **At DC / Sub-Bass ($f \to 0$):** Exactly $1.0$ ($0.0\text{ dB}$), keeping the fundamental low-end energy 100% unaltered.
* **At Resonance ($f = f_r$):** The response evaluates to $Q_{\text{target}} / Q_{\text{src}} = 1.0 / Q_{\text{src}}$, attenuating the $+2.6\text{--}3.0\text{ dB}$ internal bump by exactly $-2.6\text{--}3.0\text{ dB}$ to achieve a critically flat $0.0\text{ dB}$ net response.
* **At Upper Treble ($f \gg f_r$):** Both numerator and denominator scale as $(f/f_r)^2$, causing the ratio to smoothly and asymptotically return to $1.0$ ($0.0\text{ dB}$).
* **Max Gain Bounded at $\le 1.0$ ($0.0\text{ dB}$):** Unlike blind Wiener inversion (which attempts to boost upper frequencies by $+14\text{ dB}$, squashing low-end headroom by $10\text{ dB}$ during normalization), the biquad anti-resonance filter is strictly a surgical cut filter. It prevents noise amplification and preserves exact bass gain balance throughout the SPICE and NAM pipeline.

---

## 6. Upright Acoustic Bridge Force Transducer Physics (`sensor_type = "bridge_force"`)

Unlike magnetic pickups that sense string velocity across a spatial aperture ($H(x, w)$), an upright acoustic bridge transducer (e.g., Fishman Full Circle, David Gage Realist, Underwood piezo) senses mechanical downforce and shear rocking torque transmitted directly through the wooden bridge foot into the soundboard:

### A. Regularized Spatial De-Combing ($H_{\text{decomb}}$)
Magnetic pickups impart spatial comb-filtering nulls $H_{\text{comb}}(f) = \left|\sin\left(\frac{2\pi f x}{v}\right)\right|$ governed by distance $x$ from the bridge. In contrast, an acoustic bridge pickup sits directly at the bridge termination point ($x \approx 0\text{ mm}$), where no spatial comb cancellations occur within the audible audio band.

To deconvolve the magnetic comb filter of the source instrument without introducing infinite gain at the null points or upper-frequency noise flare, Allomorph applies regularized spatial inversion:
$$H_{\text{decomb}}(f) = \frac{H_{\text{src, acoustic}}(f)}{H_{\text{src, acoustic}}^2(f) + \epsilon_{\text{reg}}}, \quad \epsilon_{\text{reg}} = 0.08$$
The decombed response is normalized relative to its median value across the $100\text{--}1,000\text{ Hz}$ core passband, restoring smooth low-register string dynamics.

### B. Soundboard & Bridge Wood Damping ($H_{\text{damp}}$)
Carved spruce and maple double bass tops absorb string vibration rapidly above the mid-treble register. Allomorph models acoustic wood dissipation with a 2nd-order critically damped low-pass filter ($Q = 0.707$):
$$|H_{\text{damp}}(f)| = \frac{1}{\sqrt{\left(1 - \left(\frac{f}{f_d}\right)^2\right)^2 + 2\left(\frac{f}{f_d}\right)^2}}, \quad f_d = 4,200\text{ Hz}$$
This smoothly attenuates electric fret click, metallic string whistle, and upper electromagnetic hash, imparting a warm, woody acoustic decay.

### C. Subsonic Stage Rumble Cut with Cepstral Regularization ($H_{\text{sub}}$)
Stage handling, bow scrapes, and floor vibrations produce strong sub-audible excursions. Allomorph applies a high-pass filter at $32\text{ Hz}$ with a bounded $-16.5\text{ dB}$ floor:
$$H_{\text{sub}}(f) = \max\left(\frac{f}{\sqrt{f^2 + f_{\text{sub}}^2}}, 0.15\right), \quad f_{\text{sub}} = 32\text{ Hz}$$
The $0.15$ lower bound ensures that the discrete log-magnitude spectrum $\ln |H(f)|$ does not plunge to $-\infty$ at DC ($f=0$). This eliminates unphysical cepstral Gibbs ringing that would otherwise notch the $30\text{--}40\text{ Hz}$ low B fundamental during minimum-phase FIR synthesis.

### D. Leaky Velocity-to-Force Tilt Integrator ($H_{\text{tilt}}$)
Piezoelectric crystals generate voltage proportional to applied stress/strain (the integral of string displacement/velocity). Due to finite mechanical bridge compliance and transducer charge leakage, pure $1/f$ integration transitions into flat velocity transfer at very low frequencies:
$$H_{\text{tilt}}(f) = \frac{\sqrt{1 + \left(\frac{f}{250\text{ Hz}}\right)^2}}{\sqrt{1 + \left(\frac{f}{70\text{ Hz}}\right)^2}}$$
Normalized by peak gain, this imparts a gentle $+6\text{ dB/octave}$ mechanical force transition between $70\text{ Hz}$ and $250\text{ Hz}$, delivering the signature percussive "thump" of a plucked acoustic bass.

### E. Resonant Body Bloom ($41.5''$ 3/4 Double Bass)
A 3/4 double bass features a $41.5''$ ($105.4\text{ cm}$) vibrating string length and a massive resonant air cavity. Allomorph synthesizes this acoustic body bloom with:
$$H_{\text{bloom}}(f) = \frac{\sqrt{g_{\text{bloom}}^2 + \left(\frac{f}{100\text{ Hz}}\right)^2}}{\sqrt{1 + \left(\frac{f}{100\text{ Hz}}\right)^2}}, \quad g_{\text{bloom}} = 10^{\Delta \text{bloom} / 20.0}$$
where $\Delta \text{bloom} = \text{bloom}_{\text{target}} - \text{bloom}_{\text{source}}$, delivering the deep, resonant low-end bloom characteristic of a full-size upright acoustic instrument.

---

## 7. Dual-Sided String Physics & Differential Mechanical Transfer

Allomorph models string behavior as a **differential transfer function** between the physical strings on the player's source instrument ($S_{\text{src}}$) and the authentic goal strings of the target voicing ($S_{\text{tgt}}$):

$$H_{\text{string\_transfer}}(f) = \frac{H_{\text{string, target}}(f)}{H_{\text{string, source}}(f)}$$

### A. Anti-Double-Damping Spectral Deconvolution
When an electric bass is strung with flatwounds (such as **La Bella Low Tension Flats** on a 32" fretless), the physical strings already roll off high-frequency harmonics above $2.8\text{ kHz}$. If a static acoustic upright low-pass filter ($f_d \approx 3.8\text{--}4.2\text{ kHz}$) is applied directly, the tone suffers from double-damping:
1. **Fused Differential String Damping Ratio:**
   Rather than evaluating separate reciprocal square roots and performing numerical division, Allomorph computes the exact analytical ratio in a single pass:
   $$\frac{H_{\text{damp, tgt}}(f)}{H_{\text{damp, src}}(f)} = \sqrt{\frac{1 + \left(\frac{f}{f_{d,\text{src}}}\right)^{2 n_{\text{src}}}}{1 + \left(\frac{f}{f_{d,\text{tgt}}}\right)^{2 n_{\text{tgt}}}}}$$
   This guarantees strictly non-zero positivity for subsequent decibel conversions and eliminates redundant floating-point divisions.
2. **Differential Anti-Double-Damping Ratio with Bidirectional Soft-Knee Saturation:**
   $$r_{\text{db}} = 20 \log_{10}\left(\frac{H_{\text{damp, tgt}}(f)}{H_{\text{damp, src}}(f)}\right)$$
   $$r_{\text{soft\_db}} = \begin{cases} g_{\text{max}} \cdot \tanh\left(\frac{r_{\text{db}}}{g_{\text{max}}}\right), & r_{\text{db}} > 0.0 \\ g_{\text{min}} \cdot \tanh\left(\frac{r_{\text{db}}}{g_{\text{min}}}\right), & r_{\text{db}} \le 0.0 \end{cases}$$
   where $g_{\text{max}} = +8.0\text{ dB}$ and $g_{\text{min}} = -36.0\text{ dB}$.
   $$H_{\text{damp\_ratio}}(f) = 10^{r_{\text{soft\_db}} / 20.0}$$
3. **Differential String Cavity Bloom ($H_{\text{bloom}}$):**
   $$\Delta\text{bloom}_{\text{dB}} = \text{bloom}_{\text{tgt}} - \text{bloom}_{\text{src}}$$
   $$g_{\text{bloom}} = 10^{\Delta\text{bloom}_{\text{dB}} / 20.0}$$
   $$H_{\text{bloom}}(f) = \sqrt{\frac{g_{\text{bloom}}^2 + \left(\frac{f}{90.0\text{ Hz}}\right)^2}{1 + \left(\frac{f}{90.0\text{ Hz}}\right)^2}}$$
   $$H_{\text{string\_transfer}}(f) = H_{\text{damp\_ratio}}(f) \cdot H_{\text{bloom}}(f)$$

This automatically preserves the natural woody clarity and fingerboard mwah of flatwounds on fretless basses, while still applying full acoustic damping when fed by clanky roundwound strings.

### B. Dynamic Bridge Compliance & Excursion Scaling ($V_{\text{sat}}$)
Low-tension strings (e.g. La Bella LTF $\sim 132\text{ lbs}$) exhibit larger physical displacement under pizzicato plucking than high-tension strings ($\sim 160\text{--}190\text{ lbs}$). Allomorph dynamically scales the soft-knee bridge compliance saturation threshold:
$$V_{\text{sat, eff}} = \frac{V_{\text{sat, base}}}{\text{pluck\_excursion\_factor}_{\text{src}}}$$
For the 32" Fretless ($1.25\times$ excursion), $V_{\text{sat}}$ scales from $0.42\text{ V}$ down to $0.336\text{ V}$, faithfully capturing the increased mechanical bridge rocking and natural acoustic compression.

---

## 8. 2D Cylindrical Rod vs 1D Blade Sensing Apertures

Pickup magnetic pole pieces feature two fundamentally distinct spatial sensing geometries that filter transverse string vibrations differently:

```
    A. 2D Cylindrical Rod Pole                     B. 1D Continuous Bar Blade
         (Jazz, Precision, MM)                          (Active EMG, Dual-Rails)
            ┌─────────┐                                      ┌───────────────┐
   String: ═│════●════│═════                        String: ═│═══════════════│═════
            └─────────┘                                      └───────────────┘
          Airy / Bessel J1                                    Rectangular Sinc
```

### The Unified Geometric Sensing Aperture Kernel
Allomorph unifies both circular disc and rectangular slit geometries into a single parameterized $C^\infty$ SIMD kernel:
$$H_{\text{aperture}}(f, v; \beta, w) = \frac{1}{\sqrt{1 + \beta \cdot \left(\frac{2\pi (w/2) f}{v}\right)^2}}$$

### A. 2D Cylindrical Rod Poles (`pole_type = "rod"`)
Cylindrical Alnico or steel rod magnets (radius $r_p = w_m / 2$) integrate string vibration over a circular 2D disc. Setting geometric parameter $\beta = 0.25$ evaluates the circular 2D Airy disc sensitivity ($J_1(k r_p) / (k r_p)$):
$$k = \frac{2\pi f}{v_{\text{disp}}(f)}, \quad H_{\text{rod}}(f) = \frac{1}{\sqrt{1 + 0.25 \cdot (k \cdot r_p)^2}}$$

* **Acoustic Character:** Gentler high-frequency rolloff ($6\text{ dB/octave}$ asymptote) without sharp cancellation nulls in the audible passband, preserving pick snap, vowel-like articulation, and touch-sensitive harmonic bite.

### B. 1D Continuous Bar Blade Sensors (`pole_type = "blade"`)
Bar magnets or steel blades span continuously under the strings with rectangular spatial aperture width $w_m$. Setting $\beta = 1/3$ evaluates 1D rectangular slit integration ($r_p = w_m / 2 \implies 2\pi r_p = \pi w_m$):
$$H_{\text{blade}}(f) = \frac{1}{\sqrt{1 + \frac{1}{3} \cdot \left(\frac{\pi w_m f}{v_{\text{disp}}(f)}\right)^2}}$$

* **Acoustic Character:** Sharper high-frequency suppression than rod poles, smoothing out high-register harshness and delivering the ultra-consistent, modern active pickup character.
* **Identity Preservation:** Both formulations evaluate to exact $1.0000$ ($0.00\text{ dB}$) at DC ($f=0$) and evaluate to bit-exact $0.00\text{ dB}$ flat response on identity transformations.

---

## 9. Bridge Saddle Witness-Point Boundary Layer Stiffness ($H_{\text{saddle}}$)

Real bass strings have finite flexural bending stiffness ($E \cdot I > 0$). At the bridge saddle termination point ($x = 0$), string displacement and slope are mechanically constrained, creating an exponential boundary layer:

$$l_b \approx \sqrt{B_s} \cdot L \approx 2.0\text{--}3.5\text{ mm}$$

For pickups located extremely close to the bridge saddle ($x < 0.075\text{ m} = 75\text{ mm}$, such as 60s/70s Jazz bridge pickups at $63.5\text{ mm}$, StingRay bridge coils at $50.8\text{ mm}$, or Dingwall angled bridge sweet spots at $48.0\text{ mm}$), the mechanical boundary layer suppresses extreme high-frequency string modes:

$$\text{ratio} = \text{clip}\left(\frac{x}{0.075\text{ m}}, 0.0, 1.0\right)$$
$$\text{shelf}_{\text{dB}} = -4.0 \cdot (1.0 - \text{ratio})$$
$$g_{\text{saddle}} = 10^{\text{shelf}_{\text{dB}} / 20.0}$$
$$H_{\text{saddle}}(f, x) = \sqrt{\frac{1.0 + g_{\text{saddle}}^2 \cdot \left(\frac{f}{4500\text{ Hz}}\right)^2}{1.0 + \left(\frac{f}{4500\text{ Hz}}\right)^2}}$$

Evaluated differentially:
$$H_{\text{saddle, eff}}(f) = \min\left( \frac{H_{\text{saddle, tgt}}(f)}{\max(H_{\text{saddle, src}}(f), 10^{-6})}, 1.0 \right)$$

* **Acoustic Consequence:** Softens brittle, piercing ultrasonic pick scratch and fret clatter ($> 7\text{ kHz}$) on bridge-side pickups without dulling the punchy midrange growl ($1.5\text{--}3.5\text{ kHz}$).
* Evaluates to exact $1.0000$ ($0.00\text{ dB}$) when $x \ge 75\text{ mm}$ or when source geometry matches target.

---

## 10. Longitudinal Core Compression Waves & Percussive Clank ($H_{\text{long}}$)

Plucking a wound bass string excites two distinct acoustic wave modes:
1. **Transverse Shear Waves:** The primary musical pitch ($v_T = \sqrt{T/\mu} \approx 60\text{--}170\text{ m/s}$).
2. **Longitudinal Compression Waves:** Compression-dilation pulses propagating through the solid steel core wire with longitudinal speed:
   $$c_L = \sqrt{\frac{E}{\rho_{\text{steel}}}} \approx 5,100\text{ m/s}$$

The longitudinal compression wave reflects back and forth between the bridge saddle and nut/fret, generating an instantaneous resonant metallic clank at:

$$f_L = \frac{c_L}{2 L} \approx 2.7\text{--}3.3\text{ kHz}, \quad Q_L = 8.0$$

When the target voicing strings (such as Dingwall stainless-steel roundwounds) have higher longitudinal coupling than the source instrument strings ($\Delta k_{\text{long}} = \max(k_{\text{long, tgt}} - k_{\text{long, src}}, 0) > 0$):

$$H_{\text{long}}(f) = 1.0 + \Delta k_{\text{long}} \cdot \frac{f / f_L}{Q_L \sqrt{\left(1 - \left(\frac{f}{f_L}\right)^2\right)^2 + \left(\frac{f}{Q_L f_L}\right)^2}} \cdot e^{-\left(\frac{f}{6000\text{ Hz}}\right)^2}$$

* **Musical Feel:** Reproduces the distinct metallic "piano clank" that cuts through heavy Darkglass drive engines without adding sterile high-frequency EQ boost.
* Returns exact $1.0000$ ($0.00\text{ dB}$) when source strings match target strings.

---

## 11. Diffuse Mechanical Body-Pickup Microphonic Coupling ($H_{\text{body}}$)

Unlike modern active pickups which are vacuum-encapsulated in dense epoxy resin, vintage passive pickups (such as 1960s Fender Alnico split-coils and single-coils) are unpotted or lightly wax-potted. Acoustic vibrations from the wooden instrument body travel into the pickup bobbins, vibrating the copper windings within the magnetic field:

$$\Delta k_{\text{body}} = \max(k_{\text{body, tgt}} - k_{\text{body, src}}, 0.0)$$

Where $k_{\text{body}} = 0.08$ for Alnico V; $0.10$ for Alnico II; $0.03$ for Ceramic; $0.00$ for epoxy-potted Active pickups:

$$f_b = 6200.0\text{ Hz}, \quad Q_b = 1.8, \quad f_{\text{damp}} = 9500.0\text{ Hz}$$
$$f_n = \frac{f}{f_b}$$
$$\text{res}(f) = \frac{f_n}{Q_b \sqrt{\left(1 - f_n^2\right)^2 + \left(\frac{f_n}{Q_b}\right)^2}}$$
$$H_{\text{body}}(f) = 1.0 + \Delta k_{\text{body}} \cdot \text{res}(f) \cdot e^{-\left(\frac{f}{f_{\text{damp}}}\right)^2}$$

* **Acoustic Consequence:** Adds authentic woody mechanical air and organic body bloom in the $5.5\text{--}7.5\text{ kHz}$ register, imparting vintage realism to sterile active signals.

---

## 12. Multi-Pickup Spatial Propagation Delays & Causal Sample Shifting

In multi-pickup instruments (Jazz Bass pairs, P/J, P/MM), transverse string waves take a physical propagation time $\tau_i$ to travel between pickup positions:

$$\tau_i = \frac{x_{\text{max}} - x_i}{\bar{c}}, \quad \bar{c} = 2 \cdot L \cdot \bar{f}_0 \approx 115.6\text{ m/s}$$

### The Gibbs Truncation Ripple Problem with Circular FFT Rotation:
Applying this delay in the frequency domain via circular phase rotation ($H(f) \cdot e^{-j 2\pi f \tau_i}$) on minimum-phase FIR filters concentrates the impulse peak at tap 0. Because continuous-time sinc interpolation wraps the negative-time non-causal sinc tail around to the end of the circular buffer, slicing the buffer back to $N$ taps discards this wrapped tail, convolving the frequency spectrum with a Dirichlet kernel and generating periodic Gibbs truncation ripples ($\Delta f = 1/\tau_i \approx 1.2\text{ kHz}$) across the high frequencies ($8\text{--}20\text{ kHz}$).

### The Allomorph Causal Discrete Solution:
To eliminate Gibbs truncation ripples with $100\%$ mathematical rigor, Allomorph applies spatial propagation delays strictly via **causal integer sample shifts**:

$$\text{delay\_samples} = \left\lfloor \tau_i \cdot f_s + 0.5 \right\rfloor$$
$$\text{fir}_{\text{delayed}} = [0]^{\text{delay\_samples}} + \text{fir}[:N - \text{delay\_samples}]$$

This guarantees:
1. Zero high-frequency truncation ripple artifacts in the differential transfer functions.
2. Exact causal impulse response propagation.
3. Authentic inter-pickup acoustic phase comb cancellation at $600\text{--}800\text{ Hz}$ when blending neck and bridge pickups in parallel.

---

## 13. Real Spatial Comb Mechanics, Excursion Physics & 34" Transducer Geometries

To evaluate pickup placement with true mathematical and physical rigor, the interaction between vibrating strings and magnetic pole pieces must be analyzed across three physical dimensions: **stationary spatial comb filtering**, **low-frequency velocity tilt**, and **amplitude-dependent dynamic excursion non-linearity**.

---

### A. The Fretted String Reality: Fret-Invariant Spatial Comb Filtering

A pervasive myth in guitar lutherie claims that pickups are positioned to cancel specific musical harmonics on the open string (e.g., placing a P-bass pickup at $125\text{ mm}$ to cancel the 7th harmonic at $L/7 = 123.4\text{ mm}$). 

In physical wave mechanics, this static node explanation is a misconception:
1. **The Moving Node Paradox:** When a string is fretted at fret $F$, the vibrating string length shortens to $L_{\text{eff}} = L \cdot 2^{-F/12}$. While pickup distance $x$ from the bridge saddle remains fixed, the harmonic node locations ($x_{\text{node}} = k L_{\text{eff}} / n$) move with every fret. At the 5th fret, the 7th harmonic node has moved to $92.4\text{ mm}$; at the 12th fret, it sits at $61.7\text{ mm}$. A static pickup does **not** permanently cancel any musical harmonic across all notes.
2. **The Fret-Invariant Spatial Comb Filter:** What **does** remain strictly invariant across all frets is the **spatial comb filter frequency response** $H_{\text{pos}}(f)$. 
   
Transverse wave propagation speed on a given string is determined strictly by string tension $T$ and linear mass density $\mu$:
$$v = \sqrt{\frac{T}{\mu}} = 2 \cdot L_{\text{open}} \cdot f_{\text{open}}$$

Because the bridge saddle acts as a rigid termination ($y(0) = 0$), the superposition of incident and reflected waves at distance $x$ from the saddle imposes an absolute spatial comb filter envelope on string displacement:
$$H_{\text{pos}}(f) = \left| \sin\left(\frac{2\pi f x}{v}\right) \right|$$

The spatial cancellation nulls occur at fixed frequencies in Hertz:
$$f_{\text{null}, k} = k \cdot \frac{v}{2x}, \quad k \in \{1, 2, 3, \dots\}$$

Because wave speed $v$ is an intrinsic physical property of each tuned string, **these cancellation frequencies in Hertz are identical whether the note is played open or fretted anywhere up the neck**:

| Transducer Position ($x$) | Low E ($v_E = 71.1\text{ m/s}$) 1st Null | A String ($v_A = 95.0\text{ m/s}$) 1st Null | D String ($v_D = 126.8\text{ m/s}$) 1st Null | High G ($v_G = 169.3\text{ m/s}$) 1st Null |
| :--- | :---: | :---: | :---: | :---: |
| **Bridge Single-Coil ($54.6\text{ mm}$)** | **$651\text{ Hz}$** | $870\text{ Hz}$ | $1,161\text{ Hz}$ | $1,550\text{ Hz}$ |
| **StingRay MM Center ($66.0\text{ mm}$)** | **$538\text{ Hz}$** | $720\text{ Hz}$ | $961\text{ Hz}$ | $1,283\text{ Hz}$ |
| **Single Centroid Datum ($93.5\text{ mm}$)** | **$380\text{ Hz}$** | $508\text{ Hz}$ | $678\text{ Hz}$ | $905\text{ Hz}$ |
| **Reverse-P: E/A Half ($115.6\text{ mm}$)** | **$307\text{ Hz}$** | $411\text{ Hz}$ | — | — |
| **Standard P-Bass ($125.0\text{ mm}$)** | **$284\text{ Hz}$** | $380\text{ Hz}$ | $507\text{ Hz}$ | $677\text{ Hz}$ |
| **Reverse-P: D/G Half ($145.4\text{ mm}$)** | — | — | $436\text{ Hz}$ | **$582\text{ Hz}$** |

#### The Deconvolution Implication:
If a physical bass possesses only a single pickup at the **$93.5\text{ mm}$ median**, the instrument has an acoustic null on the low E string at **$380\text{ Hz}$**. Recreating a StingRay tone (whose first null sits much higher at $538\text{ Hz}$ and has massive constructive energy at $380\text{ Hz}$) forces the deconvolution filter to boost the missing $380\text{ Hz}$ content. In Allomorph, Wiener noise regularization clamps maximum boost to $+8.0\text{ dB}$, meaning an inverted single pickup will always be an approximation of the physical comb spectrum.

---

### B. Low-Frequency Velocity Tilt & Fundamental Scaling ($\propto f \cdot x$)

For low frequencies where the spatial argument is small ($\frac{2\pi f x}{v} \ll 1$):
$$\sin\left(\frac{2\pi f x}{v}\right) \approx \frac{2\pi f x}{v} \propto f \cdot x$$

This first-order Taylor expansion yields two foundational acoustic behaviors:
1. **Raw Fundamental Output Scales Directly with $x$:**
   Low-frequency fundamental voltage scales linearly with distance $x$ from the bridge. Moving from a bridge position ($x = 55\text{ mm}$) to a neck position ($x = 130\text{ mm}$) yields a physical gain increase of:
   $$\Delta G_{\text{fund}} = 20 \log_{10}\left(\frac{130\text{ mm}}{55\text{ mm}}\right) \approx \mathbf{+7.5\text{ dB}}$$
   This $+7.5\text{ dB}$ fundamental fullness is physically present in the string's motion at the neck position; it is not an EQ artifact.
2. **High-Pass Spectral Derivative Tilt ($+6\text{ dB/octave}$):**
   Because $\sin\left(\frac{2\pi f x}{v}\right) \propto f$ at small $x$, bridge-side pickups act as physical differentiators (velocity sensors), introducing a natural $+6\text{ dB/octave}$ high-pass spectral tilt that attenuates the sub-fundamental while accentuating high harmonics and transient pick attack. 
   At the neck position ($x = 130\text{--}145\text{ mm}$), the linear approximation breaks down above $80\text{--}100\text{ Hz}$, delivering a flat, massive fundamental passband.

---

### C. Amplitude-Dependent String Excursion & Magnetic Non-Linear Saturation

The maximum mechanical excursion of a plucked string forms a triangular profile tapering to zero at the bridge saddle witness point ($x = 0$):
$$y_{\text{string}}(x) \approx y_{\text{pluck}} \cdot \frac{x}{x_{\text{pluck}}}$$

This excursion gradient creates fundamentally different magnetic flux compression regimes:
* **Neck Transducer ($x = 130.5\text{ mm}$):** The large physical string displacement $\Delta y$ modulates magnetic flux deeply into the non-linear curvature of the pickup's magnetic circuit. This generates dynamic soft-knee compression:
  $$V_{\text{out}}(t) = V_{\text{sat}} \cdot \tanh\left(\frac{v(t)}{V_{\text{sat}}}\right)$$
  as well as displacement-modulated reluctance inductance shifts ($\lambda_L$ "dynamic quack"). When the bassist digs in aggressively ($ff$), the neck pickup physically compresses and blooms with tactile give.
* **Bridge Transducer ($x = 54.6\text{--}66.0\text{ mm}$):** String displacement is physically constrained by bridge proximity, while transverse string acceleration and angular velocity are high. The pickup operates in an almost purely linear magnetic regime, producing sharp, uncompressed, metallic transient clank.
* **Single Centroid Sensor ($x = 93.5\text{ mm}$):** Experiences only a static, intermediate excursion level, physically incapable of delivering the deep non-linear bloom of the neck position or the extreme linear transient spikes of the bridge position.

---

### D. Reverse-P Mechanical Stagger Physics on 34" Scale

Bass strings have vastly different mechanical properties across registers:
* Low E strings feature large linear mass density ($\mu \approx 0.018\text{ kg/m}$) and wide physical excursion. Samping E/A closer to the bridge (**$115.6\text{ mm}$**) restricts mechanical excursion, tightening low-end fundamental transient attack and eliminating flub.
* High G strings feature low linear mass ($\mu \approx 0.003\text{ kg/m}$) and high bending stiffness. Samping D/G further from the bridge (**$145.4\text{ mm}$**) captures larger mechanical excursion, providing singing warmth, fundamental weight, and sustain.

A single straight pickup or soapbar senses all strings at an identical distance $x$, lacking this physical register-dependent mechanical balance.

---

### E. Architectural Synthesis: Real Harmonics Dual-Transducer vs. Single-Pickup Reference

| Architectural Criterion | Architecture 1: Real Harmonic Dual-Transducer (Reverse PX + MMTWX) | Architecture 2: Single-Pickup Reference (Centroid @ $93.5\text{ mm}$) |
| :--- | :--- | :--- |
| **Spatial Comb Filtering** | **Physical Ground Truth:** Comb nulls physically located at $284\text{ Hz}$ (P) and $538\text{ Hz}$ (MM) | **DSP Synthesis:** Fixed null at $380\text{ Hz}$; requires Wiener deconvolution boost |
| **Fundamental Energy** | **Authentic Physical $+7.5\text{ dB}$ Excursion** at neck pickup | **Synthesized Shelving Tilt** ($H_{\text{tilt}}$) |
| **Non-Linear Core Saturation** | **Dynamic Physical Bloom** on neck pickup; linear velocity spikes on bridge | **Uniform Intermediate Compression** across all registers |
| **Per-String Mechanical Balance** | **Reverse-P Physical Stagger** ($115.6\text{ mm}$ E/A vs $145.4\text{ mm}$ D/G) | **Flat Uniform Distance;** DSP continuum correction |
| **Plucking Hand Feedback** | **Dual Physical Anchors** (pliable neck zone or stiff bridge recoil) | **Single Physical Anchor** ($93.5\text{ mm}$) |
| **Live Stage Operation** | Requires matching pickup switch to Anagram target preset | **100% Decoupled;** Zero switch-mismatch cognitive load |
| **Secondary Magnetic Drag** | Minimal active drag (two low-flux active apertures) | **Absolute Zero Drag** (Single active pickup) |
| **Standalone Analog Tone** | **4 Pro Iconic Sounds Natively** (P, MM, J-Bridge, P/MM) | **1 Neutral Intermediate Tone** without DSP |
| **Optimal Use Case** | Pure tone recording, studio sessions, tactile players | Patch-heavy live touring, zero-distraction pedalboard setups |



