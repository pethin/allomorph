# Circuit Theory & Electrical Modeling

This document details the electrical equations, component parameters, and physical phenomena modeled in **Allomorph** to recreate authentic passive bass pickups and vintage/modern active preamps from active EMG X-Series signals.

---

## 1. The Passive Pickup Equivalent Network

A passive magnetic pickup is not merely a voltage source; it is a complex, high-impedance RLC resonant transducer. The standard linear equivalent circuit consists of:

$$V_{\text{out}}(s) = V_{\text{ind}}(s) \cdot H_{\text{elec}}(s)$$

Where:
* $V_{\text{ind}}$ is the voltage induced across the coils by the vibrating string ($V_{\text{ind}} \propto \frac{d\Phi}{dt}$).
* $H_{\text{elec}}(s)$ is the transfer function of the pickup loaded by the guitar's internal circuitry, cable, and amplifier.

```
                  ┌───────── R_eddy ────────┐
                  │                         │
[V_ind] ───[ L_coil ]───[ R_dc ]──┬─────────┴───┬─── [Lug 3: Volume In]
                                  │             │
                              [ C_coil ]    [ C-Switch ]
                                  │             │
                                 GND           GND
```

### Key Parameters:

1. **Coil Inductance ($L_{\text{coil}}$):**
   * Typically $1.6\text{ to }14.4\text{ Henries}$ for bass pickups.
   * Directly governs the frequency of the resonant peak ($f_r$). Higher inductance shifts the peak down into the low-mids.
   * Series wiring quadruples inductance relative to single coils ($L_{\text{ser}} \approx 4 \times L_{\text{single}}$), while parallel wiring halves it ($L_{\text{par}} \approx \frac{1}{2} L_{\text{single}}$).

2. **DC Resistance ($R_{\text{dc}}$):**
   * Resistance of thousands of turns of AWG 42 or 43 copper wire (typically $3.6\text{ k}\Omega\text{ to }28\text{ k}\Omega$).
   * Determines the baseline damping of the circuit and limits the maximum quality factor ($Q$).

3. **Inter-Winding Self-Capacitance ($C_{\text{coil}}$):**
   * Stray capacitance between adjacent coil turns, typically $50\text{ to }150\text{ pF}$.
   * Minor compared to cable capacitance in passive circuits, but sets the upper ceiling for unloaded resonance and active buffer isolation.

4. **Core Losses & Eddy Currents ($R_{\text{eddy}}$):**
   * Moving magnetic fields induce circular eddy currents in conductive pole pieces, steel baseplates, and magnets.
   * Modeled as a parallel resistance ($R_{\text{eddy}} \approx 60\text{ k}\Omega\text{ to }250\text{ k}\Omega$) across the coil.
   * This creates frequency-dependent damping: upper harmonics roll off more smoothly than an ideal 2nd-order filter, eliminating harsh, synthetic high-end resonance.

5. **Dynamic Core Compliance & String Excursion Saturation ($B_{\text{comp}}$):**
   * High-amplitude pick and slap transients push magnetic string coupling into non-linear excursion ($d\Phi/dx$).
   * Modeled in SPICE via displacement-domain pre/de-emphasis with soft-knee saturation:
     $$V_{\text{dyn}}(t) = V_{\text{sat}} \cdot \tanh\left(\frac{V(t)}{V_{\text{sat}}}\right)$$
   * For analog buffer power rail limiting, the branchless 3-stage `fsqrt` algebraic rail limiter ($p=8$) is applied:
     $$f(x) = \frac{x}{\left(1 + \left(\frac{x}{V_{\text{sat}}}\right)^8\right)^{1/8}}$$
     guaranteeing smooth $C^\infty$ saturation without digital clipping or sign branching.
   * For light to medium playing ($< 0.2\text{V}$), the response is $98\text{--}100\%$ linear.
   * On heavy pluck spikes ($> 0.4\text{V}$), the smooth saturation provides $1.5\text{--}2.5\text{ dB}$ of analog soft-knee dynamic give, capturing authentic passive pickup compression.

---

## 2. Onboard Control Harnesses & Loading Topologies

Allomorph does not assume a generic volume/tone harness for all instruments. Each voice utilizes its authentic manufacturer and era-specific harness:

### A. Active Preamp Buffer Topologies (`01_modern_jazz_active`, `07_modern_pj_active`, `09_stingray_mm_parallel`, `09b_stingray_mm_series`, `11_modern_pmm_active`, `11b_pmm_hybrid_series`)
In instruments equipped with active onboard preamps or active buffers (e.g., Sadowsky NYC 2-band, Spector 2-band, Music Man 2-band, Sandberg/Glockenklang, Lakland LH3, EMG ABCX), an internal discrete JFET or op-amp buffer stage directly interfaces with the pickup coils:

```
[Coils] ── Lug 3 (Preamp In) ──► [Buffer Op-Amp: Zin = 1Meg || 25pF] ──► [Active 2-Band EQ] ──► [R_out: 100Ω] ──► [Cable + Anagram]
```

1. **Cable Capacitance Isolation:**
   * In a traditional passive bass, the $750\text{ pF}$ instrument cable load capacitance directly shunts the high-impedance pickup coils, dragging the resonant peak down into the $2.5\text{--}3.5\text{ kHz}$ range.
   * An active preamp presents a high input impedance ($R_{\text{preamp\_in}} = 1.0\text{ M}\Omega$, $C_{\text{preamp\_in}} = 25\text{ pF}$). The coils resonate solely against $C_{\text{coil}} + C_{\text{in}} \approx 75\text{--}205\text{ pF}$, shifting the raw electrical resonant peak into the ultra-clarity $3.4\text{--}8.5\text{ kHz}$ region.
2. **Low-Impedance Cable Driver ($R_{\text{out}} = 100\,\Omega$):**
   * The low output impedance ($100\,\Omega$) drives long instrument cables and downstream pedalboards without high-frequency attenuation:
     $$f_{\text{cable\_cutoff}} = \frac{1}{2\pi \cdot 100 \cdot 780\text{ pF}} \approx 2.04\text{ MHz}$$
3. **Active Preamp & Buffer Voicings:**
   * **Sadowsky 2-Band (Jazz & P/J):** $+4.0\text{ dB}$ Bass boost ($40\text{ Hz}$ shelf) and $+4.0\text{ dB}$ Treble boost ($4.0\text{ kHz}$ shelf).
   * **Music Man StingRay 2-Band:** $+1.8\text{ dB}$ Bass boost ($50\text{ Hz}$ shelf) and $+2.2\text{ dB}$ Treble boost ($4\text{--}7\text{ kHz}$ shelf).
   * **Active Parallel P/MM (`11_modern_pmm_active`):** Split-P neck ($4.80\text{ H}$) and MM bridge parallel humbucker ($1.20\text{ H}$) summed in active parallel ($L_{\text{par}} = 0.96\text{ H}$) into a high-headroom transparent buffer stage; delivers punchy, articulate slap growl with resonant clarity at $3.4\text{ kHz}$ and zero cable drag.
   * **Active Series P/MM (`11b_pmm_hybrid_series`):** Split-P neck ($4.80\text{ H}$) and MM bridge parallel humbucker ($3.60\text{ H}$) wired in series before the active buffer stage; generates massive $+5.8\text{ dB}$ series boost and authoritative low-mid punch with resonant clarity at $3.2\text{ kHz}$ and complete isolation from cable capacitance.
   * **Series Coil Open-Circuit EMF Surge (`09b_stingray_mm_series`):** Switching the two MM coils into series doubles the open-circuit induced EMF ($V_{\text{out}} = e_1 + e_2 \approx 2e$), generating $+5.6\text{ dB}$ of authoritative passband gain with an authentic $4.1\text{ kHz}$ upper-mid resonance.

### B. Vintage Dual-Volume Harness (`02_jazz_bass_pair`, `02b_jazz_bass_pair_22nf`, `08_vintage_pj_passive`)
* Standard Jazz Basses and passive P/Js utilize two separate $250\text{k}\Omega$ volume pots wired in parallel.
* At $100\%$ volume, the two pots act as a combined resistive load:
  $$R_{\text{vol\_net}} = 250\text{ k}\Omega \parallel 250\text{ k}\Omega = 125\text{ k}\Omega$$
* This heavy $125\text{ k}\Omega$ loading naturally damps the $Q$ factor of the pickup coils, producing the warm, woody, organic low-mid bloom characteristic of vintage 1960s Jazz Basses and 1980s P/Js.
* **Tone Wide Open (`02_jazz_bass_pair`):** Full $250\text{ k}\Omega$ wiper series resistance isolates the $47\text{ nF}$ capacitor, leaving the loaded peak at $2.7\text{ kHz}$.
* **ToneStyler 22nF Detent (`02b_jazz_bass_pair_22nf`):** Replaces simulated potentiometer wiper resistance with a Stellartone ToneStyler pure capacitive shunt ($R_{\text{tone}} = 3.3\,\Omega$, $C_{\text{tone}} = 22\text{ nF}$). Because there is zero series wiper resistance to damp the tank circuit, the high-$Q$ resonant bump is preserved directly at $762\text{ Hz}$ (+4.8 dB above a rolled pot), preserving the vocal Jaco bridge burp while cutting out harsh treble clank and pick transients.
* **Spatial Acoustic Propagation Delay:** Transverse string wave takes $\tau = (x_{\text{neck}} - x_{\text{bridge}}) / \bar{c_s} \approx 0.81\text{ ms}$ to travel between pickups, creating the iconic hollow acoustic phase comb cancellation at $600\text{--}800\text{ Hz}$.

### C. Vintage Split-P ToneStyler Progression (`05`, `05b`, `05c`, `05d`)
* Classic 1962 Fender Precision specification:
  * Volume: CTS $250\text{ k}\Omega$ Audio Pot
  * Net parallel pot load: $250\text{k} \parallel 250\text{k} = 125\text{ k}\Omega$ (unloaded) / with $1\text{ M}\Omega$ receiver: $111\text{ k}\Omega$.
* **Tone Pot Rolling vs. Stellartone ToneStyler Switching:**
  * **Traditional Pot Rolling:** Turning a standard tone knob inserts variable wiper resistance ($R_{\text{tone}} \approx 10\text{--}100\text{ k}\Omega$) in series with the tone cap. This heavily over-damps the circuit ($Q < 0.5$), flattening the resonant peak into a muddy shelf.
  * **Stellartone ToneStyler Pure Switching:** Connects discrete precision capacitors directly to ground ($R_{\text{tone}} \approx 3.3\,\Omega$). The undamped resonant peak ($Q \approx 1.2\text{--}1.8$) is retained as the resonant frequency glides down through the spectrum:
    $$f_r = \frac{1}{2\pi \sqrt{L \cdot (C_{\text{coil}} + C_{\text{cable}} + C_{\text{tone}})}}$$
* **The 4-Position ToneStyler Progression:**
  * **Tone Open (`05_vintage_62_p_alnico`):** CTS $250\text{k}\Omega$ tone pot wide open. Resonant peak at $2128\text{ Hz}$, $-3\text{ dB}$ cutoff at $3985\text{ Hz}$ (open, woody, articulate vintage growl).
  * **22nF ToneStyler (`05b_vintage_62_p_22nf`):** Pure $22\text{ nF}$ shunt (Modern Fender spec). Resonant peak at $440\text{ Hz}$ (+1.5 dB), $-3\text{ dB}$ cutoff at $750\text{ Hz}$ (punchy low-mid focus, eliminates fret clatter while retaining punch).
  * **47nF ToneStyler (`05c_vintage_62_p_47nf`):** Pure $47\text{ nF}$ shunt with heavy flatwound damping (Golden '60s Motown / Jamerson spec). Resonant peak at $450\text{ Hz}$ (+1.2 dB), steep rolloff above $800\text{ Hz}$ (pillowy low-mid bloom, authentic Motown thump).
  * **100nF ToneStyler (`05d_vintage_50s_p_100nf`):** Pure $100\text{ nF}$ (0.1µF) shunt (Original 1951–1959 Fullerton factory spec). Sub-bass shelf, $-3\text{ dB}$ cutoff at $240\text{ Hz}$ (deep Motown / reggae dub thump).

### D. Modern Boutique 500k Harness with Treble Bleed (`04_modern_p_ceramic`)
* Modern ceramic split-coils use $500\text{ k}\Omega$ pots to maintain high-frequency extension:
  * Volume: $500\text{ k}\Omega$ Audio Pot
  * Tone: $500\text{ k}\Omega$ Audio Pot with $22\text{ nF}$ Orange Drop capacitor (higher cutoff than $47\text{ nF}$, preserving punchy midrange bite)
  * **Hybrid Treble Bleed Network:**
    $$Z_{\text{tb}}(s) = R_{\text{ser}} + \frac{R_{\text{par}}}{1 + s \cdot R_{\text{par}} \cdot C_{\text{tb}}}$$
    where $C_{\text{tb}} = 1000\text{ pF}$, $R_{\text{par}} = 150\text{ k}\Omega$, $R_{\text{ser}} = 20\text{ k}\Omega$.
    Maintains crisp pick transient definition when backing off volume without thinness.

### E. Factory Rickenbacker 330k Harness with Series HPF (`10_rickenbacker_bridge_hpf`)
* Authentic Rickenbacker 4001/4003 circuit:
  * Volume: $330\text{ k}\Omega$ Pot
  * Tone: $330\text{ k}\Omega$ Pot with $47\text{ nF}$ capacitor
  * **Vintage Push-Pull Series HPF:** In vintage mode, a $4.7\text{ nF}$ capacitor is inserted in series between the bridge pickup hot lead and the volume control:
    $$f_c = \frac{1}{2\pi \cdot R_{\text{load}} \cdot C_{\text{series}}} \approx 150\text{ Hz}$$
    Rolls off sub-bass rumble while sharpening the aggressive $1.5\text{--}2.5\text{ kHz}$ bridge bite.

### F. High-Inductance 500k Harnesses (`12_mudbucker_ultra_series`)
* Ultra-high-inductance passive pickups ($L > 10\text{ H}$, such as the Gibson Mudbucker @ $14.4\text{ H}$) require $500\text{ k}\Omega$ volume and tone pots with a $22\text{ nF}$ capacitor to prevent complete treble cutoff, producing a deep, dark subterranean fundamental.

---

## 3. Output Cable & Receiver Loading

A passive instrument cannot be modeled without including cable and receiver impedances:

### Instrument Cable Capacitance ($C_{\text{cable}}$)
A standard $15\text{--}20\text{ ft}$ quality instrument cable exhibits $30\text{ to }50\text{ pF per foot}$:
$$C_{\text{cable}} \approx 750\text{ pF}$$

This capacitance appears in parallel with the pickup's self-capacitance and receiver input capacitance:
$$C_{\text{tot}} \approx C_{\text{coil}} + C_{\text{cable}} + C_{\text{receiver}}$$

### Receiver Input Load
The Darkglass Anagram hardware input stage presents:
$$R_{\text{receiver}} = 1.0\text{ M}\Omega, \quad C_{\text{receiver}} \approx 30\text{ pF}$$

### Instrument Cable Dielectric Dissipation Loss ($\tan \delta$)
Real-world instrument cables insulated with vintage PVC or rubber exhibit a frequency-dependent dielectric loss factor (dissipation factor $\tan \delta \approx 0.025$). This manifests as a frequency-proportional shunt conductance in parallel with the cable capacitance:
$$G_{\text{cable}}(\omega) = \omega \cdot C_{\text{cable}} \cdot \tan \delta$$
$$Y_{\text{cable}}(s) = s C_{\text{cable}} + \omega C_{\text{cable}} \tan \delta$$

At DC ($\omega = 0$), $G_{\text{cable}} = 0$, guaranteeing strictly zero DC attenuation ($0.00\text{ dB}$ transfer ratio) in adherence to Architectural Guardrail §5.9. Across the resonant peak ($f_r \approx 2\text{--}3\text{ kHz}$), $G_{\text{cable}}$ introduces a gentle $0.1\text{--}0.3\text{ dB}$ softening of high-Q peaks, preventing artificial metallic harshness while preserving genuine high-frequency clarity. Active buffered pickups isolate the coils from this loss network entirely.

All Allomorph passive SPICE netlists incorporate this complete load network to guarantee zero tonal discrepancy between simulated DAW pipelines and hardware pedalboard operation.

---

## 4. Acoustic Bridge Force Transducers (`14_upright_bridge_transducer`)

Unlike magnetic pickups whose output is induced via Faraday's Law across an inductive coil ($V \propto d\Phi/dt$), an acoustic bridge transducer (e.g. Underwood, David Gage Realist) operates through the **piezoelectric effect**, generating electrical charge from mechanical shear stress within the bridge:

```
[V_piezo] ───[ R_dc: 50Ω ]──┬───[ C_rick: 15nF ]───┬─── [Direct Buffer Out]
                            │                      │
                        [ C_sensor: 1.2nF ]     [ Zin = 100Meg || 750pF ]
                            │                      │
                           GND                    GND
```

1. **Pure Capacitive Sensor ($L \approx 1\ \mu\text{H}, C_{\text{sensor}} = 1.2\text{ nF}$):**
   * Negligible inductance eliminates RLC peaking; electrical response is purely capacitive.
2. **Direct High-Z Buffer ($100\text{ M}\Omega$):**
   * Transducer bypasses guitar volume/tone pots into a transparent high-impedance buffer.
3. **Subsonic Rumble Decoupling ($C_{\text{series}} = 15\text{ nF}$):**
   * Decouples subsonic mechanical rumble ($f_c \approx 10.6\text{ Hz}$ into $1\text{ M}\Omega$).
4. **Dynamic Bridge Rocking Compliance ($B_{\text{comp}}$):**
   * Non-linear mechanical rocking under pizzicato attack modeled via:
     $$V_{\text{dyn}}(t) = V_{\text{sat}} \cdot \tanh\left(\frac{V(t)}{V_{\text{sat}}}\right)$$
     with $V_{\text{sat}} = 0.42\text{ V}$ (roundwounds) scaled to $0.336\text{ V}$ on 32" fretless flatwounds.
5. **Acoustic Body & Soundboard Radiation (3 Sigma AST IRs):**
   * The bridge transducer model synthesizes the raw mechanical shear stress and force at the saddle. To radiate this bridge force through a resonant double-bass spruce soundboard and cavity, the model is designed and **recommended to be paired with 3 Sigma Audio Upright Bass AST (Acoustic Sound Technology) IRs** downstream in Block 3 (or DAW IR host). Specifically, use the **`Acoustic Upright Standard`** folder (identified by the **`AST`** file tag), which 3 Sigma designed specifically for acoustic upright basses with standard piezo bridge pickups.

---

## 5. Passive-to-Passive Modeling & Differential SPICE Engine

When transforming a passive source bass (such as a stock Fender Precision Bass or Jazz Bass) into a target passive voice:

```
Active Source (EMG):    [Flat ~20 kHz] ──► [H_target(s)] ──► [Target Sound]
Passive Source:         [V_string] ──► [H_source(s)] ──► [Track] ──► [H_diff(s)] ──► [Target Sound]
```

### The Regularized Differential Transfer Function
To prevent double-filtering (which would cause a disastrous $-24\text{ dB/octave}$ cutoff), Allomorph uses an analytical differential transfer function:

$$|H_{\text{diff}}(f)| = \frac{|H_{\text{target}}(f)| \cdot |H_{\text{source}}(f)|}{|H_{\text{source}}(f)|^2 + \epsilon^2}$$

* $\epsilon = 0.05$ is the **Wiener regularization floor**, preventing division by near-zero stopband values.
* **High-Frequency Clamping:** Above $4.5\text{ kHz}$, maximum boost is clamped to $+6.0\text{ dB}$ relative to $1\text{ kHz}$ reference gain, preventing amplification of passive coil noise and audio interface hiss.
* **True Identity Verification:** When the source instrument equals the target voice (e.g., standard Precision Bass into `05_vintage_62_p_alnico`, or standard Jazz Bass into `02_jazz_bass_pair`), $H_{\text{diff}}$ evaluates to an exact flat line ($\Delta < 0.10\text{ dB}$ across $40\text{--}4500\text{ Hz}$).
* **Dynamic Saturation Bypass:** When `electronics = "passive"`, forward $\tanh$ saturation is automatically bypassed, preserving authentic uncompressed dynamics.

---

## 6. Fractional Core Eddy-Current Diffusion (Foster Ladder)

Traditional guitar circuit simulations treat pickup coil inductance as a fixed, frequency-independent parameter ($s L$). In physical pickups, the high electrical conductivity of solid metallic pole pieces (e.g. Alnico V cylinder magnets or mild steel pole screws) causes eddy currents to circulate within the core as frequency rises.

This magnetic skin effect expels magnetic flux from the core's center into the outer air, causing:
1. An effective reduction in inductance $L(f)$ above $1\text{--}3\text{ kHz}$ (typically $6\text{--}10\%$ drop for conductive Alnico alloys).
2. Frequency-dependent resistive eddy damping ($R_{\text{eddy}} \propto \sqrt{f}$).

Allomorph models this phenomenon using a physical **Foster 2-stage ladder network**:

```
           ┌─── L_core ───┐
──[ L_inf ]┴─── R_core ───┴──
```

$$Z_L(s) = s L_{\infty} + \frac{s L_{\text{core}} R_{\text{core}}}{s L_{\text{core}} + R_{\text{core}}}$$

Where:
* $L_{\infty} = (1 - k_{\text{core}}) L_0$ is the high-frequency asymptote where core flux is expelled.
* $L_{\text{core}} = k_{\text{core}} L_0$ is the core inductance subject to eddy suppression.
* $R_{\text{core}} = 2\pi f_{\text{core}} L_{\text{core}}$ is the eddy loss resistance tuned to the pole piece skin-depth transition frequency ($f_{\text{core}}$).

---

## 7. Dahl Magnetic Domain-Wall Pinning Hysteresis Model

Ferromagnetic core materials exhibit touch-sensitive sustain bloom and micro-hysteresis due to the pinning of magnetic domain walls at grain boundaries and material defects.

Allomorph simulates rate-independent minor hysteresis loops in the string displacement domain at $96\text{ kHz}$ oversampling:

$$\Delta[n] = |x[n] - z[n-1]|$$
$$\text{coupling}[n] = \frac{\Delta[n]}{\Delta[n] + r}$$
$$z[n] = z[n-1] + \Delta x[n] \cdot \text{coupling}[n]$$
$$x_{\text{hyst}}[n] = (1 - \eta_{\text{hyst}}) x[n] + \eta_{\text{hyst}} z[n]$$

Where:
* $z[n]$ represents the internal pinned magnetic polarization state.
* $r = 0.06$ is the domain-wall unpinning threshold.
* $\eta_{\text{hyst}}$ is the material-specific hysteresis coupling factor ($0.06$ for Alnico V; $0.09$ for Alnico II; $0.02$ for Ceramic; $0.01$ for Neodymium; $0.00$ for Piezo).
* This provides warm sustain bloom and subtle phase delay without introducing DC offset.

---

## 8. Higher-Order Dipole Proximity & Dynamic Lenz-Law Core Flux Sag

Physical guitar pickups deviate from ideal linear transducers in two critical dynamic regimes during aggressive fingerstyle, slap, or pick attacks:

### A. Higher-Order Magnetic Dipole Proximity Stiffening ($\alpha_3$)
The magnetic field $B(z)$ above a cylindrical pole piece decays nonlinearly with air-gap distance $z$ as a magnetic dipole:
$$B(z) \propto \frac{\mu_0 m}{2\pi (z^2 + R^2)^{3/2}}$$

Expanding the string excursion $x(t)$ about its resting equilibrium distance $z_0$ in a Taylor series yields both quadratic (even) and cubic (odd) proximity stiffening terms:
$$v_{\text{geom}}(t) = x_{\text{disp}}(t) + \alpha \cdot x_{\text{disp}}(t)^2 + \alpha_3 \cdot x_{\text{disp}}(t)^3$$

* $\alpha$ (quadratic asymmetry) generates even-order (2nd harmonic) musical warmth and octave bloom.
* $\alpha_3$ (cubic expansion) generates touch-sensitive 3rd-harmonic punch and tactile low-mid compression when digging in hard on low strings.
* On lower-coercivity magnets (Alnico II $\alpha_3 = 0.14$, Alnico V $\alpha_3 = 0.10$), the string modulates the magnetic field more deeply, yielding rich harmonic crunch. High-coercivity magnets (Ceramic $\alpha_3 = 0.04$, Neodymium $\alpha_3 = 0.02$) remain significantly tighter and cleaner; non-magnetic Piezo ($\alpha_3 = 0.00$) is purely linear.

### B. Dynamic Lenz-Law Core Flux Sag ($k_{\text{sag}}$)
Under explosive pluck transients, the rapid rate of change of magnetic flux ($d\Phi/dt$) induces counter-electromotive eddy currents within conductive magnet alloys according to Lenz's Law. These circulating counter-currents momentarily oppose and depress the net core flux during the initial pluck attack:

Allomorph models this using a dual-time-constant envelope detector coupled to a 1-pole register crossover ($f_c = 750\text{ Hz}$):
$$\tau_{\text{att}} = 6\text{ ms} \implies \alpha_{\text{att}} = 1 - e^{-1 / (f_s \cdot \tau_{\text{att}})}$$
$$\tau_{\text{rel}} = 45\text{ ms} \implies \alpha_{\text{rel}} = 1 - e^{-1 / (f_s \cdot \tau_{\text{rel}})}$$

The envelope $\text{env}[n]$ tracks the absolute peak excursion:
$$\text{env}[n] = \begin{cases} \text{env}[n-1] + \alpha_{\text{att}} (|x[n]| - \text{env}[n-1]), & \text{if } |x[n]| > \text{env}[n-1] \\ \text{env}[n-1] + \alpha_{\text{rel}} (|x[n]| - \text{env}[n-1]), & \text{otherwise} \end{cases}$$

When $\text{env}[n] > V_{\text{sat}}$, the excess excursion ratio drives dynamic core drag:
$$\text{excess}[n] = \text{clip}\left(\frac{\text{env}[n] - V_{\text{sat}}}{V_{\text{sat}}}, 0, 1\right)$$
$$\text{drag}_{\text{high}}[n] = 1.0 - (k_{\text{sag}} + \text{eddy\_factor} + \text{pull\_damping} + \text{stein\_damping} + \text{emf\_damping}) \cdot \text{excess}[n]$$
$$\text{drag}_{\text{low}}[n] = 1.0 - (0.25 \cdot k_{\text{sag}} + 0.50 \cdot \text{pull\_damping}) \cdot \text{excess}[n]$$

This separates the deep fundamental low-end energy from high-frequency string clank, providing organic pick compression on forte attacks that blooms into sustained notes without harsh clipping or flat-topping.

---

## 9. Solid Pole Eddy Skin-Effect Fractional Dispersion ($Z_{\text{skin}}(s)$)

In pickups with solid conducting metallic pole pieces (such as Alnico V rod magnets in Fender Jazz and Precision basses), circular eddy currents inside the cylindrical magnet volume force high-frequency magnetic flux to the cylinder periphery. This classical skin effect increases the series AC impedance of the inductive coil branch proportionally to $\sqrt{\omega}$:

$$Z_{\text{skin}}(s) = R_{\text{dc}} \cdot k_{\text{skin}} \cdot \left( \sqrt{1 + \frac{s}{\omega_{\text{skin}}}} - 1 \right), \quad \omega_{\text{skin}} = 2\pi f_{\text{skin}}$$

```
[V_ind] ───[ L_coil ]───[ R_dc ]───[ Z_skin(s) ]──┬── [Lug 3: Volume In]
                                                  │
                                              [ C_coil ]
                                                  │
                                                 GND
```

### Physical Properties:
1. **Exact DC Transparency:** At DC ($s = 0$), $Z_{\text{skin}}(0) \equiv 0$. The measured DC resistance $R_{\text{dc}}$ is preserved bit-exact, guaranteeing exact $0.00\text{ dB}$ DC transfer matching across identity transformations.
2. **High-Frequency Asymptote:** At high frequencies ($f \gg f_{\text{skin}}$):
   $$Z_{\text{skin}}(s) \approx R_{\text{dc}} \cdot k_{\text{skin}} \cdot \sqrt{\frac{s}{\omega_{\text{skin}}}} = R_{\text{dc}} \cdot k_{\text{skin}} \cdot \sqrt{\frac{\omega}{2 \omega_{\text{skin}}}} (1 + j)$$
   This introduces a natural $45^\circ$ phase angle and a $\sqrt{\omega}$ series loss that gently softens top-end hash without inducing artificial resonant peaking or Gibbs truncation ripples.
3. **Metallurgy Mapping:** $k_{\text{skin}} = 0.08, f_{\text{skin}} = 3200\text{ Hz}$ for Alnico V; $k_{\text{skin}} = 0.10, f_{\text{skin}} = 2200\text{ Hz}$ for Alnico II; $k_{\text{skin}} = 0.02$ for high-resistivity Ceramic ferrite; $k_{\text{skin}} = 0.00$ for insulating Active/Piezo.

---

## 10. Complex Magnetic Permeability Dispersion ($\mu^*(\omega)$ Jordan After-Effect)

Ferromagnetic cores do not possess constant magnetic permeability across audio frequencies. Due to microscopic magnetic domain-wall inertia and the Jordan magnetic after-effect, magnetic permeability exhibits a logarithmic frequency dispersion with causal imaginary losses:

$$\mu_{\text{rel}}(s) = 1.0 - \chi_{\mu} \ln\left(1.0 + \frac{s}{\omega_{\mu}}\right), \quad \omega_{\mu} = 2\pi \cdot 1200.0\text{ rad/s}$$

The complete core impedance formulation combines Foster eddy diffusion, complex permeability, and solid skin dispersion:

$$Z_L(s) = \mu_{\text{rel}}(s) \cdot \left[ s L_{\infty} + \frac{s L_{\text{core}} R_{\text{core}}}{s L_{\text{core}} + R_{\text{core}}} \right] + Z_{\text{skin}}(s)$$

Where:
* $\chi_{\mu}$ is the Jordan core relaxation susceptibility ($0.035$ for Alnico V, $0.050$ for Alnico II, $0.010$ for Ceramic, $0.005$ for Neodymium).
* This provides a subtle, natural softening of high-$Q$ electrical resonance peaks by introducing causal phase delay and loss directly inside the inductor core.

### Analytical Core Impedance Jacobians ($\nabla_\theta Z_L(s)$)
For gradient-based parameter calibration and sensitivity sweeps, the exact closed-form Jacobians with respect to all primary physical core parameters are evaluated in closed form without numerical perturbation:
$$\frac{\partial Z_L}{\partial L_0} = s \cdot \mu_{\text{rel}}(s) \cdot \left[ (1 - k_{\text{core}}) + \frac{k_{\text{core}} R_{\text{core}}^2}{(s L_{\text{core}} + R_{\text{core}})^2} \right]$$
$$\frac{\partial Z_L}{\partial k_{\text{core}}} = s L_0 \cdot \mu_{\text{rel}}(s) \cdot \left[ -1 + \frac{R_{\text{core}} (2 s L_{\text{core}} + R_{\text{core}})}{(s L_{\text{core}} + R_{\text{core}})^2} \right]$$
$$\frac{\partial Z_L}{\partial f_{\text{core}}} = \mu_{\text{rel}}(s) \cdot \frac{2\pi s^2 L_{\text{core}}^3}{(s L_{\text{core}} + R_{\text{core}})^2}$$
$$\frac{\partial Z_L}{\partial k_{\text{skin}}} = R_{\text{dc}} \cdot \left( \sqrt{1 + \frac{s}{\omega_{\text{skin}}}} - 1 \right)$$
$$\frac{\partial Z_L}{\partial f_{\text{skin}}} = -R_{\text{dc}} \cdot k_{\text{skin}} \cdot \frac{\pi s}{\omega_{\text{skin}}^2 \sqrt{1 + \frac{s}{\omega_{\text{skin}}}}}$$

---

## 11. Fractional-Order Dielectric Absorption (Cole-Davidson Relaxation)

Real-world instrument cables and tone capacitors do not behave as ideal pure capacitors ($1 / sC$). Real polymer and paper-in-oil (PIO) dielectrics exhibit fractional-order relaxation described by the **Cole-Davidson model**:

$$Y_C(s) = s \cdot C \cdot \left(\frac{\omega}{\omega_0}\right)^{\alpha - 1} \cdot e^{j (\alpha - 1) \frac{\pi}{2}}$$

Where:
* $\omega_0 = 2\pi \cdot 1000.0\text{ rad/s}$ is the $1\text{ kHz}$ calibration anchor.
* $\alpha_{\text{cable}} = 0.994$ models dielectric loss in $15\text{--}20\text{ ft}$ PVC/rubber shielded instrument cables.
* $\alpha_{\text{tone}} = 0.988$ models dielectric dissipation in vintage paper-in-oil and polyester tone capacitors.
* **Vectorized Pre-Factored Formulation:** Rather than raising complex frequency arrays to fractional powers, the solver isolates the complex phase into a scalar pre-factor $\kappa = C \cdot e^{j (\alpha - 1) \pi / 2}$, evaluating $Y_C(s) = \kappa \cdot s \cdot s_{\text{norm}}^{\alpha - 1}$ with cached real power vectors ($s_{\text{norm}}^{\alpha - 1}$).
* **Acoustic Consequence:** As tone knobs are rolled down, the dielectric loss prevents the circuit from forming an artificial, piercing high-$Q$ peak, delivering the warm, musical low-pass contour characteristic of vintage passive instruments.

---

## 12. Distributed Inter-Winding Transmission Line Admittance

A high-inductance pickup coil ($> 5,000\text{ turns}$ of fine copper wire) is a distributed transmission line with distributed inter-winding capacitance and series resistance. Allomorph replaces lumped parallel capacitance ($s C_{\text{coil}}$) with the hyperbolic transmission factor:

$$\gamma_{\text{dist}} = k_{\text{dist}} \sqrt{\frac{s}{\omega_{\text{dist}}}}, \quad \omega_{\text{dist}} = 2\pi \cdot 10000.0\text{ rad/s}$$
$$Y_{\text{coil}}(s) = (s C_{\text{coil}} + G_{\text{coil}}) \cdot \frac{\tanh(\gamma_{\text{dist}})}{\gamma_{\text{dist}}}$$

Where:
* $k_{\text{dist}} \in [0.10, 0.20]$ controls the distributed transmission depth ($0.18$ for Alnico V, $0.20$ for Alnico II, $0.12$ for Ceramic).
* $G_{\text{coil}} = \omega C_{\text{coil}} \tan\delta_{\text{coil}}$ ($\tan\delta_{\text{coil}} = 0.025$) models inter-layer enamel insulation dissipation.
* For $\gamma_{\text{dist}} \to 0$, $\frac{\tanh(\gamma)}{\gamma} \to 1.0$, recovering the classic lumped model. At frequencies above $5\text{ kHz}$, the hyperbolic factor dampens unphysical high-frequency parasitic resonances.

---

## 13. Coupled $2\times 2$ Mutual Inductive & Capacitive Nodal Matrix

In dual-pickup instruments (such as Jazz Bass pairs, P/J configurations, and Music Man dual coils), two coils in close physical proximity experience mutual magnetic flux coupling ($M$) and electrostatic inter-coil capacitive coupling ($C_m$):

$$M = k_m \sqrt{L_{\text{neck}} \cdot L_{\text{bridge}}}, \quad Z_m = s M, \quad Y_m = s C_m$$

The complete coupled parallel system is solved in closed form via nodal matrix analysis:

$$\Delta_Z = Z_n Z_b - Z_m^2$$
$$H_{n \to 2}(s) = \frac{Z_b - Z_m}{\Delta_Z(Y_{\text{eff}2} + Y_m) + Z_n + Z_b - 2 Z_m}$$
$$H_{b \to 2}(s) = \frac{Z_n - Z_m}{\Delta_Z(Y_{\text{eff}2} + Y_m) + Z_n + Z_b - 2 Z_m}$$

Where:
* $Z_n = R_{\text{dc},n} + Z_{L,n}$ and $Z_b = R_{\text{dc},b} + Z_{L,b}$ are the branch series impedances.
* $Y_{\text{eff}2} = Y_{c,n} + Y_{c,b} + Y_{\text{tone}} + Y_{\text{load}}$ is the total shunt admittance at the blend node.
* $k_m \in [0.03, 0.08]$ represents inter-pickup mutual inductance (higher in close dual-coil humbuckers like the StingRay; lower in spaced Jazz Bass pairs).

---

## 14. Interactive Potentiometer Wiper Division & Cable Loading ($P_{\text{vol}}, P_{\text{tone}}$)

Unlike simplified digital models that apply a static volume multiplier, Allomorph dynamically solves the complete resistive/capacitive wiper divider:

```
[Pickup Coils] ── Lug 3 (Vol In) ──[ R_top ]──┬── Lug 2 (Vol Out) ──► [Cable + Anagram]
                                              │
                                           [ R_bot ]
                                              │
                                             GND
```

1. **Volume Pot Wiper Splitting ($P_{\text{vol}} \in [0, 1]$):**
   $$R_{\text{top}} = R_{\text{vol\_total}} \cdot (1 - P_{\text{vol}}), \quad R_{\text{bot}} = R_{\text{vol\_total}} \cdot P_{\text{vol}}$$
   * When $P_{\text{vol}} = 1.0$: $R_{\text{top}} \approx 0\,\Omega$, $R_{\text{bot}} = R_{\text{vol}}$, reproducing full-output netlist defaults.
   * When $P_{\text{vol}} < 1.0$: $R_{\text{top}}$ inserts series resistance between the pickup coils and the cable, while $R_{\text{bot}}$ attenuates signal to ground. The series resistance interacts directly with cable capacitance ($C_{\text{cable}} = 750\text{ pF}$), shifting the resonant peak downward and producing the classic high-end roll-off experienced when backing off passive volume knobs.
2. **Treble-Bleed Network Interaction:**
   When equipped with a treble bleed network (e.g. `04_modern_p_ceramic`), the series impedance $Z_{23}$ becomes:
   $$Z_{\text{tb}}(s) = R_{\text{tb,ser}} + \frac{R_{\text{tb,par}}}{1 + s \cdot R_{\text{tb,par}} \cdot C_{\text{tb}}}$$
   $$Z_{23,\text{pot}}(s) = \frac{R_{\text{top}} \cdot Z_{\text{tb}}(s)}{R_{\text{top}} + Z_{\text{tb}}(s)}$$
   which bypasses high-frequency transients around $R_{\text{top}}$ to maintain pick definition at lower volumes.
3. **Tone Pot Wiper Splitting ($P_{\text{tone}} \in [0, 1]$):**
   $$R_{\text{tone}} = R_{\text{tone\_total}} \cdot P_{\text{tone}}$$
   * At $P_{\text{tone}} = 1.0$: maximum series resistance ($250\text{ k}\Omega\text{ to }500\text{ k}\Omega$) isolates the tone capacitor.
   * As $P_{\text{tone}} \to 0.0$: series resistance vanishes, connecting $C_{\text{tone}}$ directly to ground and rolling off high frequencies into the heavy low-pass shelf.
4. **Logarithmic Audio Taper Denominator Pre-computation:**
   When evaluating logarithmic audio potentiometer rotation $R(\theta) = R_{\text{total}} \cdot \frac{e^{\alpha \theta} - 1}{e^\alpha - 1}$:
   * **Audio 10% Taper ($\alpha = \ln(81)$):** The denominator simplifies to $e^{\ln(81)} - 1 = 81.0 - 1.0 = 80.0$ bit-exact.
   * **Audio 15% Taper ($\alpha = 3.4689$):** The denominator evaluates to $\text{expm1}(3.4689)$ precomputed once.
   This replaces runtime transcendental power functions with fast scalar multiplications in inner parameter evaluation loops.

---

## 15. Conformal Geometric Clearance Asymmetry ($\kappa_{\text{geom}}$)

As a vibrating bass string approaches a magnet pole piece, the magnetic field intensity increases asymptotically following a rational conformal mapping rather than symmetric polynomial expansion:

$$x_{\text{geom}}[n] = \frac{x_{\text{disp}}[n]}{1.0 - \kappa_{\text{geom}} \cdot \tanh\left(\frac{x_{\text{disp}}[n]}{V_{\text{sat}}}\right)}$$

Where:
* $\kappa_{\text{geom}}$ controls the proximity divergence factor ($0.22$ for Alnico V, $0.25$ for Alnico II, $0.15$ for Ceramic, $0.10$ for Neodymium).
* When string moves toward the pole ($x > 0$), denominator shrinks toward $1 - \kappa_{\text{geom}}$, creating asymmetrical growl, pick snap, and tactile resistance.
* When string moves away ($x < 0$), denominator smoothly expands toward $1 + \kappa_{\text{geom}}$, avoiding harsh clipping and preserving natural acoustic decay.

---

## 16. Dynamic Steinmetz AC Core Loss ($k_{\text{stein}}$)

Under forte pluck attacks, the rapid rate of magnetic flux change generates dynamic hysteresis and microscopic eddy losses proportional to the Steinmetz power law ($P_{\text{loss}} \propto B^{1.6} \cdot f$):

$$\text{flux\_rate}[n] = |x_{\text{high}}[n] - x_{\text{high}}[n-1]| \cdot 7.639437$$
$$\text{stein\_damping}[n] = k_{\text{stein}} \cdot \text{excess}[n] \cdot \left(\frac{\text{flux\_rate}[n]}{V_{\text{sat}}}\right)^{0.6}$$

Where:
* $k_{\text{stein}} = 0.035$ for Alnico V; $0.045$ for Alnico II; $0.015$ for Ceramic; $0.008$ for Neodymium.
* This loss dynamically damps harsh, metallic high-frequency transients on the instantaneous pluck attack while leaving sustained notes open and airy.

---

## 17. Dynamic Reluctance Inductance Modulation ($\lambda_L$) & Back-EMF String Braking ($k_{\text{emf}}$)

### A. Dynamic Reluctance Modulation ($\lambda_L$ "Vowel Quack")
Plucking a ferromagnetic steel string near a magnetic pickup pole momentarily compresses the magnetic air gap, decreasing magnetic reluctance ($\mathcal{R}_{\text{gap}} \propto g$) and dynamically increasing instantaneous coil inductance:
$$L(t) = L_0 \cdot \left(1 + \Delta\lambda_L \cdot \text{excess}[n] \cdot \tanh\left(\frac{|x[n]|}{V_{\text{sat}}}\right)\right)$$

In the recursive state formulation:
$$\text{ind\_mod}[n] = -\Delta\lambda_L \cdot \text{excess}[n] \cdot \tanh\left(\frac{|x[n]|}{V_{\text{sat}}}\right) \cdot (x_{\text{high}}[n] - x_{\text{high}}[n-1])$$

This momentarily depresses the resonant frequency $f_r(t)$ downward by $5\text{--}10\%$ during high-amplitude pluck transients before gliding back to rest, producing the organic, vocal "quack" and tactile feel characteristic of passive Alnico split-coils.

### B. Electromechanical Back-EMF String Braking ($k_{\text{emf}}$)
Current circulating through the pickup coil produces an opposing magnetic field that exerts a physical Lorentz braking force back on the vibrating string ($\mathbf{F}_{\text{EMF}} \propto i_{\text{coil}}(t)$):
$$\text{emf\_damping}[n] = \Delta k_{\text{emf}} \cdot \text{excess}[n] \cdot \tanh\left(\frac{|x_{\text{high}}[n]|}{V_{\text{sat}}}\right)$$

This provides instantaneous damping on extreme high-frequency string clank without deadening the fundamental pitch.

---

## 18. Nonlinear Magnetic String Pull Dynamics & Attack Pitch Sag ($k_{\text{pull}}, w_{\text{reg}}$)

Physical pickups exert a static magnetic attraction on ferromagnetic strings ("Stratitis" / Alnico string pull). During strong plucks, this magnetic attraction increases nonlinearly with displacement, causing:
1. Localized velocity damping on positive excursions.
2. Attack pitch sag (transient frequency drop during the first $20\text{--}40\text{ ms}$).

Allomorph dynamically weights string pull by the register excursion ratio:

$$w_{\text{reg}} = 0.70 + 0.60 \cdot \frac{|x_{\text{low}}[n]|}{\max(|x_{\text{low}}[n]| + |x_{\text{high}}[n]|, 10^{-6})}$$
$$\text{pull\_damping} = k_{\text{pull}} \cdot w_{\text{reg}} \cdot \text{excess}[n] \cdot \tanh\left(\frac{\max(x[n], 0)}{V_{\text{sat}}}\right)$$
$$\text{pitch\_sag} = -k_{\text{pull}} \cdot w_{\text{reg}} \cdot \text{excess}[n] \cdot (x_{\text{high}}[n] - x_{\text{high}}[n-1])$$

Thick lower strings ($E_1, B_0$) experience up to $1.3\times$ pull damping and transient pitch sag, while upper strings sustain freely ($0.7\times$).

---

## 19. Excursion-Dependent Dynamic Touch Spectral Tilt ($\tau_{\text{touch}}$)

When plucking lightly, the string motion is smooth and dominated by low-order harmonics. When digging in hard, non-linear string-fret contact and magnetic shear inject high-frequency harmonic energy into the signal:

$$H_{\text{hp}}(s) = \frac{s}{s + 2\pi \cdot 400.0}$$
$$\text{touch\_mod} = \tau_{\text{touch}} \cdot \tanh\left(\frac{|x_{\text{disp}}|}{V_{\text{sat}}}\right) \cdot x_{\text{disp,hp}}$$
$$x_{\text{disp}} = x_{\text{disp}} + \text{touch\_mod}$$

Where $\tau_{\text{touch}} \in [0.03, 0.05]$ dynamically brightens the attack proportional to plucking force, imparting touch-sensitive expressive dynamics.

---

## 20. Dynamic Core Inductance Curvature Wobble ($\beta_{\text{curv}}$) & 2D Elliptical Orbit Bloom ($\kappa_{\text{orbit}}$)

### A. Core Inductance Curvature Wobble ($\beta_{\text{curv}}$)
$$\text{wobble} = \beta_{\text{curv}} \cdot \tanh\left(\frac{x[n]^2}{V_{\text{sat}}^2}\right) \cdot (x_{\text{high}}[n] - x_{\text{high}}[n-1])$$
Models instantaneous resonant wobble as high string amplitude modulates core magnetization state.

### B. 2D Elliptical String Orbit Precession Bloom ($\kappa_{\text{orbit}}$)
Plucked strings oscillate in 2D elliptical orbital planes rather than simple 1D transverse lines. This orbital precession modulates pickup proximity at twice the fundamental frequency ($2f_0$). Allomorph simulates this via analytic Hilbert transform quadrature projection:
$$x_{\text{quad}} = x \cdot \mathcal{H}\{x\}$$
$$x_{\text{out}} = x + \kappa_{\text{orbit}} \cdot \tanh\left(\frac{|x|}{V_{\text{sat}}}\right) \cdot x_{\text{quad}}$$
Generates warm, organic second-harmonic octave bloom without introducing DC offset.

---

## 21. Transient Magnetic Slew-Rate Soft Limiting ($f_{\text{slew}}$)

Domain-wall displacement within ferromagnetic pole pieces cannot occur instantaneously. Barkhausen jumps and eddy damping impose a physical slew-rate limit on magnetic polarization:

$$\Delta x_{\text{max}} = \frac{2\pi f_{\text{slew}} V_{\text{sat}}}{f_s}, \quad f_{\text{slew}} = 16000.0\text{ Hz}$$
$$\Delta x_{\text{slew}}[n] = \Delta x_{\text{max}} \cdot \tanh\left(\frac{x[n] - x_{\text{slewed}}[n-1]}{\Delta x_{\text{max}}}\right)$$
$$x_{\text{slewed}}[n] = x_{\text{slewed}}[n-1] + \Delta x_{\text{slew}}[n]$$

Smoothly caps ultrasonic flux acceleration without hard-clipping harshness.

---

## 22. Passive RLC-Shaped Johnson-Nyquist Thermal Noise Dither ($-108\text{ dBFS}$)

Real high-impedance passive pickups generate thermal Johnson noise ($e_n = \sqrt{4 k_B T R \Delta f}$) shaped by the passive RLC resonant network. To prevent neural network zero-gating pops and activation chatter on hardware pedalboard DACs (Darkglass Anagram), Allomorph injects calibrated $-108\text{ dBFS}$ colored thermal dither:

1. White Gaussian noise is generated: $w[n] \sim \mathcal{N}(0, 1)$.
2. Convolved with a 512-tap minimum-phase FIR shaped to the target pickup's specific RLC impedance curve.
3. Normalized to target RMS:
   $$\text{target\_rms} = 10^{-108.0 / 20.0} \approx 3.98 \times 10^{-6}\ (-108\text{ dBFS})$$
4. Bypassed on small signals ($\le 0.10$ peak) and identity transformations to preserve 100% test linearity.

---

## 23. Sub-Audible DC-Blocking Filter (8.0 Hz) & Gibbs Truncation Ripple Prevention

Non-linear quadratic asymmetry terms ($v + \alpha v^2$) naturally generate a small positive DC offset. Left unaddressed, DC bias shifts downstream neural network operating points and produces large thump pops when feeding high-gain Darkglass drive stages (Microtubes B7K, Alpha·Omega).

Allomorph applies a specialized sub-audible high-pass filter ($f_c = 8.0\text{ Hz}$):
* Suppresses DC offset by $>140\text{ dB}$ ($< 10^{-10}$ DC mean).
* Transparent across the entire bass playing register: $< 0.28\text{ dB}$ attenuation at Low B ($30.87\text{ Hz}$) and $< 0.15\text{ dB}$ at Low E ($41.20\text{ Hz}$).
* **Gibbs Truncation Prevention:** Active preamps must feature flat, finite DC transmission ($H_{\text{preamp}}(0) \ge 1.0$). Sub-audible AC-coupling differentiators ($s / (s + \omega_{\text{sub}})$) are strictly excluded from active preamp transfer models, eliminating periodic Gibbs truncation ripples ($\Delta f = f_s / N = 23.4\text{ Hz}$) between $20\text{ Hz}$ and $300\text{ Hz}$.
* **Homomorphic Real-Cepstrum FIR Synthesis Tail Windowing:** In discrete causal FIR generation (`synthesize_minimum_phase_fir`), the trailing 15% of taps are tapered with the canonical real-analytic mollifier $w(t) = 1.0 - S_\infty(t)$ (`cinf_smoothstep`). Because all boundary derivatives vanish identically at the window onset and buffer end ($t=1$), algebraic $O(1/n^2)$ boundary truncation leakage is eliminated.

---

## 24. Differential Non-Linear Metallurgy Softening Matrix

When converting between different pickup metallurgies (e.g. Active EMG $\to$ Vintage Alnico V, or Ceramic $\to$ Alnico II), Allomorph evaluates dynamic softening **differentially**:

$$\Delta\alpha = \max(\alpha_{\text{tgt}} - \alpha_{\text{src}}, 0), \quad \Delta\alpha_3 = \max(\alpha_{3,\text{tgt}} - \alpha_{3,\text{src}}, 0)$$
$$\Delta\eta_{\text{hyst}} = \max(\eta_{\text{tgt}} - \eta_{\text{src}}, 0), \quad \Delta k_{\text{sag}} = \max(k_{\text{sag,tgt}} - k_{\text{sag,src}}, 0)$$
$$\Delta k_{\text{eddy}} = \max(k_{\text{eddy,tgt}} - k_{\text{eddy,src}}, 0), \quad \Delta\kappa_{\text{orbit}} = \max(\kappa_{\text{orbit,tgt}} - \kappa_{\text{orbit,src}}, 0)$$
$$\Delta\beta_{\text{curv}} = \max(\beta_{\text{curv,tgt}} - \beta_{\text{curv,src}}, 0), \quad \Delta k_{\text{pull}} = \max(k_{\text{pull,tgt}} - k_{\text{pull,src}}, 0)$$
$$\Delta\tau_{\text{touch}} = \max(\tau_{\text{touch,tgt}} - \tau_{\text{touch,src}}, 0), \quad \Delta\kappa_{\text{geom}} = \max(\kappa_{\text{geom,tgt}} - \kappa_{\text{geom,src}}, 0)$$
$$\Delta k_{\text{stein}} = \max(k_{\text{stein,tgt}} - k_{\text{stein,src}}, 0), \quad \Delta k_{\text{emf}} = \max(k_{\text{emf,tgt}} - k_{\text{emf,src}}, 0)$$
$$\Delta\lambda_L = \max(\lambda_{L,\text{tgt}} - \lambda_{L,\text{src}}, 0)$$

### Effective Saturation Ceiling ($V_{\text{sat,eff}}$):
$$V_{\text{sat,eff}} = \begin{cases} V_{\text{sat,tgt}} & \text{if active source} \\ \frac{V_{\text{sat,tgt}}}{1.0 - \min\left(0.85, \frac{V_{\text{sat,tgt}}}{V_{\text{sat,src}}}\right) + 0.15} & \text{if passive source with } V_{\text{sat,tgt}} < V_{\text{sat,src}} \\ 10.0 & \text{otherwise (bypassed)} \end{cases}$$

### $C^\infty$ Softplus Dynamic Lenz Velocity Drag:
In `_lenz_velocity_drag_core`, core saturation excess is evaluated via an infinitely differentiable ($C^\infty$) softplus excess with smooth soft-knee saturation, eliminating piecewise conditional slope kinks (`if e > vsat:`):
$$\text{excess}_{\text{raw}} = \frac{1}{\alpha} \ln(1 + e^{\alpha(e - V_{\text{sat}})}) = \frac{1}{\alpha} \text{logaddexp}(0, \alpha(e - V_{\text{sat}})), \quad \alpha = 16.0$$
$$\text{excess} = \text{excess}_{\text{raw}} - \left(\text{excess}_{\text{raw}} - 0.5 \cdot \tanh\left(\frac{\text{excess}_{\text{raw}}}{0.5}\right)\right)$$
This preserves small-signal linearity ($< 10^{-4}$ excess below threshold) while maintaining continuous derivatives across the saturation boundary in Numba-compiled ODE state loops.

This guarantees that:
1. Active sources (which exhibit zero passive core compression) receive full, authentic target saturation.
2. Passive sources being converted to stiffer targets (e.g. Alnico V into Ceramic) bypass saturation, preventing unphysical double-compression.
3. Identical instruments evaluate to bit-exact linear unity.

---

## 25. Comprehensive Magnet Metallurgy Reference Table

The following table summarizes all 15 electrical, magnetic, and dynamic parameters across the standard pickup metallurgy profiles implemented in Allomorph:

| Metallurgy Parameter | Symbol | Alnico V | Alnico II | Ceramic | Hybrid | Neodymium | Piezo | Active Buffer | Physical Phenomenon |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Eddy Loss Fraction** | $k_{\text{core}}$ | $0.08$ | $0.10$ | $0.02$ | $0.05$ | $0.01$ | $0.00$ | $0.00$ | Foster 2-stage ladder inductance reduction |
| **Core Frequency (Hz)** | $f_{\text{core}}$ | $2500$ | $1800$ | $6500$ | $4500$ | $8500$ | $0$ | $0$ | Eddy loss pole-piece skin-depth transition |
| **Skin Effect Fraction** | $k_{\text{skin}}$ | $0.08$ | $0.10$ | $0.02$ | $0.05$ | $0.02$ | $0.00$ | $0.00$ | Cylindrical pole piece series AC resistance |
| **Skin Frequency (Hz)** | $f_{\text{skin}}$ | $3200$ | $2200$ | $6500$ | $4500$ | $8000$ | $0$ | $0$ | $\sqrt{\omega}$ skin resistance transition frequency |
| **Reluctance Quack** | $\lambda_L$ | $0.05$ | $0.06$ | $0.02$ | $0.03$ | $0.01$ | $0.00$ | $0.00$ | Dynamic air-gap inductance modulation |
| **Back-EMF Braking** | $k_{\text{emf}}$ | $0.04$ | $0.05$ | $0.02$ | $0.03$ | $0.01$ | $0.00$ | $0.00$ | Coil current Lorentz force clank damping |
| **Dahl Hysteresis** | $\eta_{\text{hyst}}$ | $0.06$ | $0.09$ | $0.02$ | $0.04$ | $0.01$ | $0.00$ | $0.00$ | Magnetic domain-wall pinning sustain bloom |
| **Quadratic Asymmetry**| $\alpha$ | $0.26$ | $0.32$ | $0.12$ | $0.18$ | $0.08$ | $0.00$ | $0.00$ | 2nd-harmonic dipole expansion warmth |
| **Cubic Proximity** | $\alpha_3$ | $0.10$ | $0.14$ | $0.04$ | $0.07$ | $0.02$ | $0.00$ | $0.00$ | 3rd-harmonic tactile low-mid punch |
| **Lenz Flux Sag** | $k_{\text{sag}}$ | $0.08$ | $0.12$ | $0.03$ | $0.05$ | $0.01$ | $0.00$ | $0.00$ | Dynamic counter-EMF core flux depression |
| **Saturation Voltage** | $V_{\text{sat}}$ | $0.50\text{V}$ | $0.45\text{V}$ | $0.70\text{V}$ | $0.60\text{V}$ | $0.90\text{V}$ | $1.00\text{V}$ | $1.20\text{V}$ | Soft-knee analog dynamic compression floor |
| **Eddy De-Qing** | $k_{\text{eddy}}$ | $0.16$ | $0.20$ | $0.03$ | $0.08$ | $0.01$ | $0.00$ | $0.00$ | Flux-rate high-frequency damping on plucks |
| **Orbit Bloom (2f0)** | $\kappa_{\text{orbit}}$ | $0.06$ | $0.08$ | $0.02$ | $0.04$ | $0.01$ | $0.00$ | $0.00$ | 2D elliptical string precession bloom |
| **Microphonic Body** | $k_{\text{body}}$ | $0.08$ | $0.10$ | $0.03$ | $0.05$ | $0.01$ | $0.00$ | $0.00$ | Mechanical wood-to-coil vibration coupling |
| **Curvature Wobble** | $\beta_{\text{curv}}$ | $0.030$ | $0.040$ | $0.010$ | $0.020$ | $0.005$ | $0.000$ | $0.000$ | Excursion-dependent resonant wobble |
| **Pole Pull Drag** | $k_{\text{pull}}$ | $0.040$ | $0.060$ | $0.015$ | $0.025$ | $0.010$ | $0.000$ | $0.000$ | Localized magnetic pull and attack pitch sag |
| **Touch Spectral Tilt**| $\tau_{\text{touch}}$| $0.050$ | $0.060$ | $0.025$ | $0.035$ | $0.015$ | $0.000$ | $0.000$ | Excursion-modulated high-register brightness |
| **Jordan Dispersion** | $\chi_{\mu}$ | $0.035$ | $0.050$ | $0.010$ | $0.020$ | $0.005$ | $0.000$ | $0.000$ | Complex magnetic permeability relaxation |
| **Distributed Winding**| $k_{\text{dist}}$ | $0.18$ | $0.20$ | $0.12$ | $0.15$ | $0.10$ | $0.00$ | $0.00$ | Transmission line hyperbolic attenuation |
| **Clearance Asymmetry**| $\kappa_{\text{geom}}$ | $0.22$ | $0.25$ | $0.15$ | $0.18$ | $0.10$ | $0.00$ | $0.00$ | Conformal rational air-gap growl |
| **Steinmetz Core Loss** | $k_{\text{stein}}$ | $0.035$ | $0.045$ | $0.015$ | $0.025$ | $0.008$ | $0.000$ | $0.000$ | Flux-rate dynamic AC hysteresis core damping |

---

## 26. Differential Circuit Deconvolution Headroom & $C^\infty$ High-Frequency Shelving

When evaluating differential circuit transfer functions ($H_{\text{diff}} = H_{\text{target}} / H_{\text{source}}$), passive pickups with high coil inductance transforming into wideband reference topologies (such as the Canonical Intermediate circuit) generate authentic electrical resonance peaks up to $+6.77\text{ dB}$ (e.g. passive split-P).

To accommodate genuine electroacoustic resonance without premature compression while bounding digital gain under Guardrail 5.3.6:
1. **Thresholded Headroom Expansion (`smooth_soft_knee_db`):**
   Circuit deconvolution headroom is bounded via strictly $C^\infty$ thresholded soft-knee saturation:
   $$h_{\text{soft\_db}} = \text{smooth\_soft\_knee\_db}(h_{\text{db}}, \text{thresh}=5.5\text{ dB}, \text{ceiling}=8.0\text{ dB}, \alpha=2.0)$$
   Passband responses below $+5.5\text{ dB}$ remain 100% linear, allowing passive split-P coils to authentically reach $+6.77\text{ dB}$ ($< 0.14\text{ dB}$ deviation from theoretical resonance).
2. **$C^\infty$ Real-Analytic High-Frequency Shelving Taper:**
   Above $8\text{ kHz}$, out-of-band circuit transfer inversions are smoothly shelved toward $0.00\text{ dB}$ at $20\text{ kHz}$ using the canonical real-analytic mollifier:
   $$w(f) = 1.0 - S_\infty\left(\frac{f - 8000\text{ Hz}}{12000\text{ Hz}}\right)$$
   $$H_{\text{diff,final}}(f) = 1.0 + w(f) \cdot (H_{\text{diff}}(f) - 1.0)$$
   Because all boundary derivatives of $S_\infty$ vanish at $8\text{ kHz}$ and $20\text{ kHz}$, this shelves high-frequency noise without injecting spectral slope kinks.



