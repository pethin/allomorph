# Allomorph Master Voice Catalog (Target Voicings & Transducers)

This catalog details the physical parameters, equivalent RLC circuit values, acoustic apertures, control harnesses, and MNA-verified electrical characteristics for the **Allomorph Digital Twin Profiles**.

---

## Quick Reference Summary

| # | Profile ID | Canonical Voicing | Pickup Architecture | Topology | Harness / Controls | $L_{\text{eq}}$ | $R_{\text{dc}}$ | $f_r$ (MNA Peak) | Acoustic & Circuit Character |
| :- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **01** | `jazz_pair_active` | `34in_standard_jazz:pair_active` | Modern Active Jazz | Active 2-Band | Sadowsky 2-Band Preamp ($R_{\text{in}}=1\text{M}\Omega, C_{\text{in}}=150\text{pF}, R_{\text{out}}=100\,\Omega$) | $1.74\text{ H}$ | $3.67\text{ k}\Omega$ | $5.44\text{ kHz}$ ($Q=1.06$) | Sadowsky-style active 2-band boost ($+3.5\text{ dB}$ @ $60\text{ Hz}$ & $3.5\text{ kHz}$) with 60s J-pair ($92.1\text{ mm}$ spacing), wideband hi-fi sparkle, and punchy low-end weight. |
| **02** | `jazz_pair_open` | `34in_standard_jazz:pair_open` | Vintage 60s J-Bass Pair | Dual Parallel | Vintage 60s $2\times 250\text{k}\Omega$ Vol ($125\text{k}\Omega$ net), $250\text{k}\Omega$ Tone, $47\text{nF}$ Cap | $1.74\text{ H}$ | $3.67\text{ k}\Omega$ | $2.82\text{ kHz}$ ($Q=0.92$) | Dual narrow single-coils in parallel ($3.20\text{ H}$ neck, $3.80\text{ H}$ bridge); authentic 60s $92.1\text{ mm}$ ($3\frac{5}{8}''$) aperture scoop with natural woody low-mid resonance. |
| **02b**| `jazz_pair_mids` | `34in_standard_jazz:pair_mids` | 60s J-Bass Pair (22nF ToneStyler) | Dual Parallel | Vintage 60s $2\times 250\text{k}\Omega$ Vol, 22nF ToneStyler Shunt ($R_{\text{tone}}=3.3\,\Omega$) | $1.74\text{ H}$ | $3.67\text{ k}\Omega$ | $727.9\text{ Hz}$ ($Q=1.12$) | Vocal midrange honk ($727.9\text{ Hz}$ resonant peak); low-ESR capacitive shunt preserves punchy midrange peak with zero wiper mud. |
| **02c**| `jazz_bridge_growl` | `34in_standard_jazz:bridge_growl` | 60s J-Bass Pair (Jaco Bias) | Dual Parallel | Vintage dual $250\text{k}\Omega$ Vol (Bridge 100%, Neck @ 99% / $8.80\text{ k}\Omega$ series isolation), $250\text{k}\Omega$ Tone | $1.74\text{ H}$ | $3.67\text{ k}\Omega$ | $2.62\text{ kHz}$ ($Q=0.88$) | Classic Jaco Pastorius fingerstyle vocal bridge growl; neck volume backed off slightly adds $8.80\text{ k}\Omega$ series wiper resistance ($-6.43\text{ dB}$ neck attenuation, $67.7\% / 32.3\%$ bridge/neck blend), shifting comb scoop into the vocal midrange while keeping low-end body. |
| **03** | `jazz_bridge_open` | `34in_standard_jazz:bridge_open` | 60s J-Bass Bridge | Single Coil | Vintage $2\times 250\text{k}\Omega$ Vol (Neck @ 0%), $250\text{k}\Omega$ Tone, $47\text{nF}$ Cap | $3.80\text{ H}$ | $7.50\text{ k}\Omega$ | Overdamped | Narrow single-coil placed in 60s bridge position ($63.5\text{ mm}$ from bridge); focused midrange growl with articulate transient snap. |
| **04** | `precision_active` | `34in_standard_pj:neck_active` | Fender Ultra II Active P | Single Split | Fender Ultra II 18V 3-Band Preamp ($R_{\text{in}}=1\text{M}\Omega, C_{\text{in}}=100\text{pF}, R_{\text{out}}=100\,\Omega$), $250\text{k}\Omega$ Passive Tone | $4.00\text{ H}$ | $10.50\text{ k}\Omega$ | $3.99\text{ kHz}$ ($Q=0.93$) | Ultra II Noiseless Vintage P split-coil ($2\times 2.0\text{ H}$ series) with 18V 3-band active preamp isolating coils from cable capacitance; extended clarity and punchy low-mids. |
| **05** | `precision_vintage` | `34in_standard_p:vintage_open` | Vintage '62 P (Tone Open) | Single Split | Vintage 1962 CTS $250\text{k}\Omega$ Vol, $250\text{k}\Omega$ Tone Open, $47\text{nF}$ Cap | $3.80\text{ H}$ | $10.50\text{ k}\Omega$ | $2.06\text{ kHz}$ ($Q=0.95$) | Alnico V split-coil wide open ($2\times 1.90\text{ H}$ series); touch-sensitive dynamic response and woody organic bloom. |
| **05b**| `precision_mids` | `34in_standard_p:vintage_mids` | Vintage '62 P (22nF ToneStyler) | Single Split | 22nF ToneStyler Shunt ($R_{\text{tone}}=3.3\,\Omega$) | $3.80\text{ H}$ | $10.50\text{ k}\Omega$ | $418.1\text{ Hz}$ ($Q=1.00$) | Punchy $418.1\text{ Hz}$ low-mid resonant focus; eliminates fret clatter while retaining authoritative midrange punch. |
| **05c**| `precision_warm` | `34in_standard_p:vintage_warm` | Vintage '62 P (Motown Warm) | Single Split | $47\text{nF}$ Damped Tone Roll-Off ($R_{\text{tone}}=15\text{k}\Omega$), Heavy Flatwounds | $3.80\text{ H}$ | $10.50\text{ k}\Omega$ | Overdamped | Authentic Jamerson Motown flatwound thump (`flatwound_vintage_heavy`, $1.8\text{ kHz}$ wrap damping) with smooth damped $47\text{ nF}$ tone roll-off. |
| **05d**| `precision_dub` | `34in_standard_p:slab_dub` | Vintage '50s P (100nF Dub) | Single Split | $100\text{nF}$ Damped Tone Roll-Off ($R_{\text{tone}}=18\text{k}\Omega$), Heavy Flatwounds | $3.80\text{ H}$ | $10.50\text{ k}\Omega$ | Overdamped | Original 1950s Fullerton $0.1\,\mu\text{F}$ paper-in-oil spec with heavy flatwounds; deep sub-bass reggae/dub thump. |
| **07** | `pj_active` | `34in_standard_pj:pair_active` | Fender Ultra II Active P/J | Active 3-Band | Fender Ultra II 18V 3-Band Preamp ($R_{\text{in}}=1\text{M}\Omega, C_{\text{in}}=100\text{pF}, R_{\text{out}}=100\,\Omega$), $250\text{k}\Omega$ MN Pan & Tone | $1.78\text{ H}$ | $4.60\text{ k}\Omega$ | $3.66\text{ kHz}$ ($Q=0.83$) | Fender American Ultra II active P/J combining Ultra II Noiseless split-P ($4.00\text{ H}$) and Noiseless bridge J ($3.20\text{ H}$) through the 18V 3-band active preamp. |
| **08** | `pj_passive` | `34in_standard_pj:pair_parallel` | Vintage '80s Passive P/J | Parallel Sum | Dual $250\text{k}\Omega$ Vol ($125\text{k}\Omega$ net), $250\text{k}\Omega$ Tone, $47\text{nF}$ Cap | $1.90\text{ H}$ | $4.38\text{ k}\Omega$ | $2.66\text{ kHz}$ ($Q=0.90$) | Vintage Alnico V split-P ($3.80\text{ H}, 10.50\text{ k}\Omega$) and 60s J-bridge ($3.80\text{ H}, 7.50\text{ k}\Omega$) in parallel; authentic '80s Fender Special / Yamaha BB punch. |
| **09** | `stingray_parallel` | `34in_active_stingray:parallel_classic` | MM Humbucker Active | Active 2-Band | Music Man 2-Band Preamp ($R_{\text{in}}=1\text{M}\Omega, C_{\text{in}}=850\text{pF}, R_{\text{out}}=100\,\Omega$) | $1.34\text{ H}$ | $2.20\text{ k}\Omega$ | $4.03\text{ kHz}$ ($Q=1.18$) | Dual-coil parallel humbucker ($2.40\text{ H} \parallel 2.40\text{ H}, k=0.12$) into Music Man 2-band preamp ($+1.8\text{ dB}$ @ $80\text{ Hz}$, $+2.2\text{ dB}$ @ $4\text{ kHz}$); iconic slap clank and mid scoop. |
| **09b**| `stingray_series` | `34in_active_stingray:series_punch` | MM Series Humbucker Active | Active 2-Band | Music Man 2-Band Preamp ($R_{\text{in}}=1\text{M}\Omega, C_{\text{in}}=850\text{pF}, R_{\text{out}}=100\,\Omega$) | $5.38\text{ H}$ | $8.80\text{ k}\Omega$ | $2.19\text{ kHz}$ ($Q=1.47$) | Dual-coil series humbucker ($2\times 2.40\text{ H}, k=0.12$) into Music Man 2-band preamp; $+6.0\text{ dB}$ series inductive surge and punchy $2.19\text{ kHz}$ rock bark. |
| **10** | `rickenbacker_clank` | `33in_rickenbacker_4003:bridge_clank` | 4003 Bridge HPF | Series HPF | Factory Rickenbacker $330\text{k}\Omega$ Vol, $330\text{k}\Omega$ Tone, $4.7\text{nF}$ Series HPF | $3.80\text{ H}$ | $8.40\text{ k}\Omega$ | $2.28\text{ kHz}$ ($Q=0.64$) | High-output ceramic single-coil at $108.0\text{ mm}$ with vintage $4.7\text{ nF}$ series HPF; rolls off sub-bass below $150\text{ Hz}$ for aggressive pick clang. |
| **11** | `p_mm_parallel` | `34in_active_pmm:pmm_parallel` | Modern Active P/MM | Active 3-Band | Lakland LH-3 Preamp ($R_{\text{in}}=1\text{M}\Omega, C_{\text{in}}=470\text{pF}, R_{\text{out}}=100\,\Omega$), Buffered Blend | $4.80 / 1.61\text{ H}$ | $9.50 / 2.20\text{ k}\Omega$ | $2.85\text{ kHz}$ ($Q=1.11$) | Lakland 44-02 / Sandberg VM4 active P/MM; Split-P neck ($4.80\text{ H}$) and parallel MM bridge ($1.61\text{ H}$) isolated by dual-buffered active blend (`buffered = true`) into LH-3 3-band preamp. |
| **11b**| `p_mm_series` | `34in_active_pmm:pmm_series` | Modern Active P/MM (Series) | Active 3-Band | Lakland LH-3 Preamp ($R_{\text{in}}=1\text{M}\Omega, C_{\text{in}}=470\text{pF}, R_{\text{out}}=100\,\Omega$), Buffered Blend | $4.80 / 6.45\text{ H}$ | $9.50 / 8.80\text{ k}\Omega$ | $2.46\text{ kHz}$ ($Q=1.09$) | Lakland 44-02 / Sandberg VM4 with bridge humbucker in series ($6.45\text{ H}$) isolated by dual-buffered active blend into LH-3 3-band preamp; muscular mid-forward authority. |
| **12** | `mudbucker_deep` | `30in_gibson_eb0:neck_deep` | Overwound Series | Ultra Series | Gibson $500\text{k}\Omega$ Vol, $500\text{k}\Omega$ Tone, $22\text{nF}$ Cap | $18.0\text{ H}$ ($k=0.25$) | $27.90\text{ k}\Omega$ | $1.05\text{ kHz}$ ($Q=0.98$) | Extreme dual-coil series Sidewinder ($2\times 7.20\text{ H}, k=0.25$) at $266.8\text{ mm}$ on $30.5''$ scale; subterranean low-end focus and steep natural top-end roll-off. |
| **13** | `dingwall_bridge` | `37in_multiscale_dingwall:bridge_clank` | Multi-Scale FD3n | Angled Series | Dingwall / Glockenklang 3-Band Preamp ($R_{\text{in}}=0.5\text{M}\Omega, C_{\text{in}}=100\text{pF}, R_{\text{out}}=1\text{k}\Omega$) | $2.30\text{ H}$ | $4.40\text{ k}\Omega$ | $6.62\text{ kHz}$ ($Q=1.40$) | $34''\text{--}37''$ fanned-fret bridge position ($48.0\text{ mm}$) with high-tension stainless wave speeds and active buffered FD-3n neodymium sparkle. |
| **14** | `upright_piezo` | `41in_upright_bass:bridge_piezo` | Upright Bridge Piezo | Transducer | Direct Bridge Force Sensor (`sensor_type = "bridge_force"`) | — | — | $800.0\text{ Hz}$ ($Q=1.80$) | Authentic $41.25''$ 3/4 double bass bridge-wing piezo force sensor; captures $800\text{ Hz}$ maple bridge rocking resonance, $3.2\text{ kHz}$ wood mass roll-off, and Spirocore damping. |

---

## Detailed Profile Specifications

### 01. `jazz_pair_active` (`34in_standard_jazz:pair_active` — Modern Active Jazz Bass Pair)
* **Archetype:** Sadowsky NYC Active Jazz Bass (Dual Single-Coil with 2-Band Boost-Only Preamp)
* **Pickup Architecture:** Dual parallel Alnico V single coils (Neck: $L = 3.20\text{ H}, R_{\text{dc}} = 7.20\text{ k}\Omega$ @ $155.6\text{ mm}$; Bridge: $L = 3.80\text{ H}, R_{\text{dc}} = 7.50\text{ k}\Omega$ @ $63.5\text{ mm}$; $92.1\text{ mm}$ spacing)
* **Control Harness:** Active harness with $250\text{k}\Omega$ master volume, $250\text{k}\Omega$ MN blend pot at center detent (`0.5`), and Sadowsky 2-band boost-only preamp (`sadowsky_2band`, $R_{\text{in}} = 1.0\text{ M}\Omega, C_{\text{in}} = 150\text{ pF}, R_{\text{out}} = 100\,\Omega$) providing $+3.5\text{ dB}$ @ $60\text{ Hz}$ low shelf and $+3.5\text{ dB}$ @ $3.5\text{ kHz}$ high shelf.
* **Electrical Parameters:** Parallel $L_{\text{eq}} = 1.74\text{ H}, R_{\text{dc}} = 3.67\text{ k}\Omega$. The preamp buffer isolates the coils from external cable capacitance, placing the compound loaded resonant peak at **$5,441.0\text{ Hz}$** ($Q = 1.06$).
* **Acoustic Character:** Modern NYC active slap sparkle, tight extended sub-bass weight, and scooped midrange clarity.

### 02. `jazz_pair_open` (`34in_standard_jazz:pair_open` — Vintage 60s Jazz Bass Pair)
* **Archetype:** Standard 1960s Fender Jazz Bass Dual Single-Coil Pair (Tone Wide Open)
* **Pickup Architecture:** Dual parallel Alnico V single coils (Neck: $3.20\text{ H}, 7.20\text{ k}\Omega$ @ $155.6\text{ mm}$; Bridge: $3.80\text{ H}, 7.50\text{ k}\Omega$ @ $63.5\text{ mm}$)
* **Control Harness:** Authentic vintage dual $250\text{k}\Omega$ volume pots in parallel ($125\text{k}\Omega$ net load), single $250\text{k}\Omega$ tone pot wide open, $47\text{ nF}$ paper-in-oil capacitor, $750\text{ pF}$ cable.
* **Electrical Parameters:** Combined $L_{\text{eq}} = 1.74\text{ H}, R_{\text{dc}} = 3.67\text{ k}\Omega$. MNA loaded resonance sits at **$2,817.2\text{ Hz}$** ($Q = 0.92$).
* **Acoustic Character:** Symmetrical dual-aperture phase cancellation produces a hollow mid-scoop around $1\text{ kHz}$ with classic woody low-mid bloom and open top-end.

### 02b. `jazz_pair_mids` (`34in_standard_jazz:pair_mids` — Vintage 60s Jazz Bass Pair with 22nF ToneStyler)
* **Archetype:** 1960s Fender Jazz Bass Dual Single-Coil Pair with Stellartone ToneStyler ($22\text{ nF}$ Pure Capacitive Shunt)
* **Control Harness:** Dual $250\text{k}\Omega$ volume pots wide open, $22\text{ nF}$ capacitor shunt to ground ($R_{\text{tone}} = 3.3\,\Omega$ lead ESR, zero wiper resistance damping).
* **Electrical Parameters:** Combined $L_{\text{eq}} = 1.74\text{ H}, R_{\text{dc}} = 3.67\text{ k}\Omega$. Undamped resonant peak at **$727.9\text{ Hz}$** ($Q = 1.12$).
* **Acoustic Character:** Eliminating variable potentiometer wiper resistance preserves a vocal $+2.5\text{ dB}$ resonant bump at $727.9\text{ Hz}$ while filtering out upper-frequency fret clank.

### 02c. `jazz_bridge_growl` (`34in_standard_jazz:bridge_growl` — 60s Jazz Bass Pair, Jaco Bridge Bias)
* **Archetype:** 1960s Fender Jazz Bass Dual Single-Coil Pair with Bridge-Biased Volume Pots (`bridge_vol = 1.0`, `neck_vol = 0.99`)
* **Control Harness:** Standard Fender Jazz VVT harness where each pickup feeds the outer lug of its $250\text{k}\Omega$ audio-taper volume pot and both wipers tie to the shared output bus. Backing `neck_vol` off slightly to `0.99` inserts $R_{\text{top,n}} = 8.80\text{ k}\Omega$ of series wiper resistance between the neck pickup and the shared output bus ($R_{\text{bot,n}} = 241.20\text{ k}\Omega$).
* **Electrical & Acoustic Physics:** Because the output bus is loaded by the low impedance of the bridge pickup ($L = 3.80\text{ H}, R_{\text{dc}} = 7.50\text{ k}\Omega$), the $8.80\text{ k}\Omega$ series resistance attenuates the neck pickup by $-6.43\text{ dB}$ relative to the bridge at $100\text{ Hz}$ (normalized MNA pickup weights: $67.7\%$ bridge / $32.3\%$ neck, effective acoustic center $x_{\text{tgt}} = 93.2\text{ mm}$, loaded $f_r = 2,620.3\text{ Hz}, Q = 0.88$). This asymmetric summing softens the deep $1\text{ kHz}$ phase cancellation notch into a vocal midrange growl while retaining low-end body.

### 03. `jazz_bridge_open` (`34in_standard_jazz:bridge_open` — 60s Jazz Bass Bridge Single-Coil)
* **Archetype:** 1960s Fender Jazz Bass Solo Bridge Single Coil
* **Coil Model:** Alnico V single coil ($w = 0.75''$) positioned at $63.5\text{ mm}$ ($2.5''$) from the bridge saddle.
* **Electrical Parameters:** $L = 3.80\text{ H}, R_{\text{dc}} = 7.50\text{ k}\Omega, R_{\text{eddy}} = 120\text{ k}\Omega, C_{\text{coil}} = 70\text{ pF}$. Loaded by the VVT passive harness and $750\text{ pF}$ cable, the passband response is smooth and slightly overdamped ($f_r = \text{None (overdamped)}$; un-damped open-bus coil resonance $\sim 2.46\text{ kHz}$).

### 04. `precision_active` (`34in_standard_pj:neck_active` — Fender American Ultra II Precision Active)
* **Archetype:** Fender American Ultra II Precision Bass (Active Solo P with 18V 3-Band Preamp)
* **Coil Model:** Ultra II Noiseless Vintage P split-coil (`ultra2_split_p`, $w = 1.00''$, E/A at $139.0\text{ mm}$, D/G at $111.0\text{ mm}$, centerline $125.0\text{ mm}$, Alnico V)
* **Control Harness:** Fender American Ultra II active harness (`active` in `config/instruments/34in_standard_pj.toml`): $250\text{k}\Omega$ MN pan pot (`pan = 0.0` soloing neck P), global $250\text{k}\Omega$ passive tone pot wide open ($47\text{ nF}$ cap) ahead of the `fender_ultra2_3band` 18V 3-band active preamp ($R_{\text{in}} = 1.0\text{ M}\Omega, C_{\text{in}} = 100\text{ pF}, R_{\text{out}} = 100\,\Omega$, EQ controls at `0.5` flat center detent).
* **Electrical Parameters:** Series split-coil $L_{\text{eq}} = 4.00\text{ H}$ ($2\times 2.00\text{ H}$), $R_{\text{dc}} = 10.50\text{ k}\Omega$ ($2\times 5.25\text{ k}\Omega$), $R_{\text{eddy}} = 180\text{ k}\Omega$ ($2\times 90\text{ k}\Omega$), $C_{\text{coil}} = 35\text{ pF}$ net series. MNA loaded resonant peak at **$3,985.1\text{ Hz}$** ($Q = 0.93$).

### 05. `precision_vintage` (`34in_standard_p:vintage_open` — Vintage '62 Split-Coil Alnico V, Tone Open)
* **Archetype:** 1962 Fender Precision Bass with Alnico V Pole Pieces (Tone Wide Open)
* **Electrical Parameters:** Series split-coil $L_{\text{eq}} = 3.80\text{ H}$ ($2\times 1.90\text{ H}$), $R_{\text{dc}} = 10.50\text{ k}\Omega$ ($2\times 5.25\text{ k}\Omega$), $R_{\text{eddy}} = 180\text{ k}\Omega$, $C_{\text{coil}} = 60\text{ pF}$ net series, CTS $250\text{k}\Omega$ volume and tone pots wide open ($47\text{ nF}$ cap, $750\text{ pF}$ cable). MNA loaded resonant peak at **$2,061.7\text{ Hz}$** ($Q = 0.95$).

### 05b. `precision_mids` (`34in_standard_p:vintage_mids` — Vintage '62 Split-P, 22nF ToneStyler)
* **Archetype:** 1962 Fender Precision Bass with $22\text{ nF}$ ToneStyler Shunt ($R_{\text{tone}} = 3.3\,\Omega$)
* **Electrical Parameters:** $L_{\text{eq}} = 3.80\text{ H}, R_{\text{dc}} = 10.50\text{ k}\Omega, C_{\text{tone}} = 22\text{ nF}, R_{\text{tone}} = 3.3\,\Omega$. MNA loaded resonant peak at **$418.1\text{ Hz}$** ($Q = 1.00$).

### 05c. `precision_warm` (`34in_standard_p:vintage_warm` — Vintage '62 Split-P Motown Flatwound Warm)
* **Archetype:** 1962 Fender Precision Bass with Damped $47\text{ nF}$ Tone Roll-Off & Jamerson Heavy Flatwounds (`flatwound_vintage_heavy`)
* **Electrical Parameters:** $L_{\text{eq}} = 3.80\text{ H}, R_{\text{dc}} = 10.50\text{ k}\Omega, C_{\text{tone}} = 47\text{ nF}, R_{\text{tone}} = 15,000\,\Omega$. Overdamped low-pass roll-off ($f_r = \text{None (overdamped)}$) paired with $1,800\text{ Hz}$ order-$2.0$ flatwound wrap damping.

### 05d. `precision_dub` (`34in_standard_p:slab_dub` — Vintage '50s Split-P 100nF Dub)
* **Archetype:** 1950s Fender Precision Bass with $100\text{ nF}$ Tone Roll-Off & Heavy Flatwounds (`flatwound_vintage_heavy`)
* **Electrical Parameters:** $L_{\text{eq}} = 3.80\text{ H}, R_{\text{dc}} = 10.50\text{ k}\Omega, C_{\text{tone}} = 100\text{ nF}, R_{\text{tone}} = 18,000\,\Omega$. Overdamped sub-bass roll-off ($f_r = \text{None (overdamped)}$).

### 07. `pj_active` (`34in_standard_pj:pair_active` — Fender American Ultra II Active P/J)
* **Archetype:** Fender American Ultra II Precision Bass Active P/J (18V 3-Band Preamp)
* **Pickup Architecture:** Dual Ultra II Noiseless Alnico V pickups blended at center detent (`pan = 0.5`):
  * **Neck Split-P (`ultra2_split_p`):** $L = 4.00\text{ H}$ ($2\times 2.00\text{ H}$ series), $R_{\text{dc}} = 10.50\text{ k}\Omega$ (E/A @ $139.0\text{ mm}$, D/G @ $111.0\text{ mm}$)
  * **Bridge Single-J (`ultra2_bridge_j`):** $L = 3.20\text{ H}, R_{\text{dc}} = 8.20\text{ k}\Omega$ (@ $63.5\text{ mm}$)
* **Control Harness:** Fender American Ultra II active harness: $250\text{k}\Omega$ MN pan pot at `0.5`, global $250\text{k}\Omega$ passive tone pot wide open ($47\text{ nF}$ cap), and `fender_ultra2_3band` 18V 3-band active preamp ($R_{\text{in}} = 1.0\text{ M}\Omega, C_{\text{in}} = 100\text{ pF}, R_{\text{out}} = 100\,\Omega$, flat center detent).
* **Electrical Parameters:** Parallel $L_{\text{eq}} = 1.78\text{ H}, R_{\text{dc}} = 4.60\text{ k}\Omega$. MNA loaded compound resonant peak at **$3,658.7\text{ Hz}$** ($Q = 0.83$), MNA-weighted acoustic center $x_{\text{tgt}} = 107.1\text{ mm}$.

### 08. `pj_passive` (`34in_standard_pj:pair_parallel` — Vintage '80s Passive P/J Bass)
* **Archetype:** 1980s Fender Precision Special / Yamaha BB3000 Passive P/J
* **Pickup Architecture:** Vintage '62 Alnico V Split-P ($L = 3.80\text{ H}, R_{\text{dc}} = 10.50\text{ k}\Omega$) + Vintage '60s Alnico V Bridge J ($L = 3.80\text{ H}, R_{\text{dc}} = 7.50\text{ k}\Omega$) in parallel ($L_{\text{eq}} = 1.90\text{ H}, R_{\text{dc}} = 4.38\text{ k}\Omega$).
* **Electrical Parameters:** Dual $250\text{k}\Omega$ volume pots and $250\text{k}\Omega$ tone pot wide open ($47\text{ nF}$ cap, $750\text{ pF}$ cable). MNA loaded compound resonant peak at **$2,661.7\text{ Hz}$** ($Q = 0.90$), MNA-weighted acoustic center $x_{\text{tgt}} = 100.4\text{ mm}$.

### 09. `stingray_parallel` (`34in_active_stingray:parallel_classic` — Music Man Active Dual-Coil Parallel)
* **Archetype:** Music Man StingRay 4-String Active Dual-Coil Humbucker (Parallel)
* **Electrical Parameters:** Two Alnico V coils ($L = 2.40\text{ H}, R_{\text{dc}} = 4.40\text{ k}\Omega$ each @ $77.1\text{ mm}$ and $54.9\text{ mm}$, centerline $66.0\text{ mm}$) wired in parallel with $k_{\text{mutual}} = 0.12$ ($L_{\text{eq}} = 1.34\text{ H}, R_{\text{dc}} = 2.20\text{ k}\Omega$) into `stingray_2band` ($R_{\text{in}} = 1.0\text{ M}\Omega, C_{\text{in}} = 850\text{ pF}, R_{\text{out}} = 100\,\Omega$, $+1.8\text{ dB}$ @ $80\text{ Hz}$, $+2.2\text{ dB}$ @ $4.0\text{ kHz}$). MNA loaded resonant peak at **$4,032.9\text{ Hz}$** ($Q = 1.18$).

### 09b. `stingray_series` (`34in_active_stingray:series_punch` — Music Man Active Dual-Coil Series)
* **Archetype:** Music Man StingRay 5 / Sterling Series Humbucker
* **Electrical Parameters:** Two Alnico V coils ($L = 2.40\text{ H}, R_{\text{dc}} = 4.40\text{ k}\Omega$ each) wired in series with $k_{\text{mutual}} = 0.12$ ($L_{\text{eq}} = 5.38\text{ H}, R_{\text{dc}} = 8.80\text{ k}\Omega$) into `stingray_2band`. MNA loaded resonant peak at **$2,185.0\text{ Hz}$** ($Q = 1.47$) with $+6.0\text{ dB}$ series inductive voltage doubling.

### 10. `rickenbacker_clank` (`33in_rickenbacker_4003:bridge_clank` — Rickenbacker 4003 Bridge with 4.7nF HPF)
* **Archetype:** Rickenbacker 4001/4003 Bridge Pickup with Vintage $4.7\text{ nF}$ Series High-Pass Capacitor
* **Electrical Parameters:** Ceramic single-coil ($L = 3.80\text{ H}, R_{\text{dc}} = 8.40\text{ k}\Omega$) at $108.0\text{ mm}$ on $33.25''$ scale ($844.55\text{ mm}$), factory $330\text{k}\Omega$ pots, $4.7\text{ nF}$ series capacitor. MNA loaded resonant peak at **$2,282.2\text{ Hz}$** ($Q = 0.64$).

### 11. `p_mm_parallel` (`34in_active_pmm:pmm_parallel` — Modern Active P/MM Bass, Parallel Bridge)
* **Archetype:** Lakland 44-02 / Sandberg California VM4 Active P/MM (Parallel Bridge Mode)
* **Pickup Architecture & Control Harness:** Split-P neck ($2\times 2.40\text{ H} = 4.80\text{ H}, R_{\text{dc}} = 9.50\text{ k}\Omega$ @ $125.0\text{ mm}$) and MM bridge in parallel ($2.88\text{ H} \parallel 2.88\text{ H}, k = 0.12 \implies L_{\text{eq}} = 1.61\text{ H}, R_{\text{dc}} = 2.20\text{ k}\Omega$ @ $66.0\text{ mm}$) feeding a dual-buffered active blend pot (`taper = "mn_blend", buffered = true`) and `lakland_lh3` 3-band preamp ($R_{\text{in}} = 1.0\text{ M}\Omega, C_{\text{in}} = 470\text{ pF}, R_{\text{out}} = 100\,\Omega$).
* **Electrical Parameters:** Dual FET input buffers prevent coil-to-coil loading (individual pickup peaks at $2,504.1\text{ Hz}$ neck and $4,773.1\text{ Hz}$ bridge; compound MNA peak at **$2,848.0\text{ Hz}$**, $Q = 1.11$, weighted acoustic center $x_{\text{tgt}} = 104.9\text{ mm}$).

### 11b. `p_mm_series` (`34in_active_pmm:pmm_series` — Modern Active P/MM Bass, Series Bridge)
* **Archetype:** Lakland 44-02 / Sandberg California VM4 Active P/MM (Series Bridge Mode)
* **Electrical Parameters:** Split-P neck ($4.80\text{ H}, 9.50\text{ k}\Omega$) and MM bridge in series ($2\times 2.88\text{ H}, k = 0.12 \implies L_{\text{eq}} = 6.45\text{ H}, R_{\text{dc}} = 8.80\text{ k}\Omega$) isolated by the dual-buffered active blend pot into `lakland_lh3`. Compound MNA resonant peak at **$2,461.1\text{ Hz}$** ($Q = 1.09$, weighted acoustic center $x_{\text{tgt}} = 95.4\text{ mm}$).

### 12. `mudbucker_deep` (`30in_gibson_eb0:neck_deep` — Gibson EB-0 Sidewinder Mudbucker)
* **Archetype:** 1960s Gibson EB-0 Sidewinder Dual-Coil Series Humbucker ($30.5'' = 774.7\text{ mm}$ Scale)
* **Electrical Parameters:** Two overwound ceramic coils ($L = 7.20\text{ H}, R_{\text{dc}} = 13.95\text{ k}\Omega$ each) in series with mutual coupling $k_{\text{mutual}} = 0.25$ ($L_{\text{eq}} = 18.0\text{ H}$ coupled / $14.40\text{ H}$ uncoupled, $R_{\text{dc}} = 27.90\text{ k}\Omega$) at $266.8\text{ mm}$ from bridge, $500\text{k}\Omega$ volume/tone pots ($22\text{ nF}$ cap). MNA loaded resonant peak at **$1,053.0\text{ Hz}$** ($Q = 0.98$).

### 13. `dingwall_bridge` (`37in_multiscale_dingwall:bridge_clank` — Fanned-Fret Multi-Scale Bridge)
* **Archetype:** Dingwall Combustion / NG3 $34''\text{--}37''$ Multi-Scale Angled Bridge FD-3n
* **Electrical Parameters:** Neodymium reverse split-coil humbucker ($2\times 1.15\text{ H} = 2.30\text{ H}$ series, $R_{\text{dc}} = 4.40\text{ k}\Omega$ @ $48.0\text{ mm}$ centerline) into `dingwall_active` Glockenklang 3-band preamp ($R_{\text{in}} = 0.5\text{ M}\Omega, C_{\text{in}} = 100\text{ pF}, R_{\text{out}} = 1.0\text{ k}\Omega$, flat center detent). MNA loaded resonant peak at **$6,616.9\text{ Hz}$** ($Q = 1.40$).

### 14. `upright_piezo` (`41in_upright_bass:bridge_piezo` — Upright Piezo Bridge Transducer)
* **Archetype:** Direct Bridge Force Transducer on 3/4 Orchestral Double Bass ($41.25'' = 1,047.75\text{ mm}$ Scale)
* **Target String Archetype:** Thomastik-Infeld Spirocore 3/4 Double Bass spiral rope-core strings (`double_bass_spirocore`, $265\text{ lbs}$ tension, $f_d = 3,200\text{ Hz}$, order $2.0$).
* **Sensor & Bridge Mechanics:** Direct bridge saddle force sensor (`sensor_type = "bridge_force"`, $x = 5.0\text{ mm}$) with in-plane maple bridge rocking resonance ($f_r = 800.0\text{ Hz}, Q = 1.8$, $+2.5\text{ dB}$ peak), $3,200\text{ Hz}$ wood mass roll-off, $8.0\text{ Hz}$ sub-audible DC block, and PZT compressive strain saturation ($V_{\text{sat}} = 0.42\text{ V}, \alpha_p = 0.15$).

---

## Geometry & Scale-Normalized Displacement Mapping (30" EMG MM $\to$ Target Datums)

When pairing a **30" short-scale bass ($L_{\text{src}} = 30.0'' = 762.0\text{ mm}, x_{\text{src}} = 77.5\text{ mm}, \eta_{\text{src}} = 10.17\%$)** with target voicings, standing-wave bridge proximity is governed by scale-normalized fractional coordinates ($\eta = x / L$):

| Profile ID | Canonical Voicing | Target $x_{\text{tgt}}$ | Target Scale | Fractional $\eta_{\text{tgt}}$ | MNA Peak $f_r$ ($Q$) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **`jazz_pair_active`** | `34in_standard_jazz:pair_active` | $111.3\text{ mm}$ | $34.0''$ | $12.89\%$ | $5,441.0\text{ Hz}$ ($1.06$) |
| **`jazz_pair_open`** | `34in_standard_jazz:pair_open` | $111.3\text{ mm}$ | $34.0''$ | $12.89\%$ | $2,817.2\text{ Hz}$ ($0.92$) |
| **`jazz_pair_mids`** | `34in_standard_jazz:pair_mids` | $111.3\text{ mm}$ | $34.0''$ | $12.89\%$ | $727.9\text{ Hz}$ ($1.12$) |
| **`jazz_bridge_growl`** | `34in_standard_jazz:bridge_growl` | $93.2\text{ mm}$ | $34.0''$ | $10.79\%$ | $2,620.3\text{ Hz}$ ($0.88$) |
| **`jazz_bridge_open`** | `34in_standard_jazz:bridge_open` | $63.5\text{ mm}$ | $34.0''$ | $7.35\%$ | Overdamped |
| **`precision_active`** | `34in_standard_pj:neck_active` | $125.0\text{ mm}$ | $34.0''$ | $14.47\%$ | $3,985.1\text{ Hz}$ ($0.93$) |
| **`precision_vintage`** | `34in_standard_p:vintage_open` | $125.0\text{ mm}$ | $34.0''$ | $14.47\%$ | $2,061.7\text{ Hz}$ ($0.95$) |
| **`precision_mids`** | `34in_standard_p:vintage_mids` | $125.0\text{ mm}$ | $34.0''$ | $14.47\%$ | $418.1\text{ Hz}$ ($1.00$) |
| **`precision_warm`** | `34in_standard_p:vintage_warm` | $125.0\text{ mm}$ | $34.0''$ | $14.47\%$ | Overdamped |
| **`precision_dub`** | `34in_standard_p:slab_dub` | $125.0\text{ mm}$ | $34.0''$ | $14.47\%$ | Overdamped |
| **`pj_active`** | `34in_standard_pj:pair_active` | $107.1\text{ mm}$ | $34.0''$ | $12.40\%$ | $3,658.7\text{ Hz}$ ($0.83$) |
| **`pj_passive`** | `34in_standard_pj:pair_parallel` | $100.4\text{ mm}$ | $34.0''$ | $11.63\%$ | $2,661.7\text{ Hz}$ ($0.90$) |
| **`stingray_parallel`** | `34in_active_stingray:parallel_classic` | $66.0\text{ mm}$ | $34.0''$ | $7.64\%$ | $4,032.9\text{ Hz}$ ($1.18$) |
| **`stingray_series`** | `34in_active_stingray:series_punch` | $66.0\text{ mm}$ | $34.0''$ | $7.64\%$ | $2,185.0\text{ Hz}$ ($1.47$) |
| **`rickenbacker_clank`** | `33in_rickenbacker_4003:bridge_clank` | $108.0\text{ mm}$ | $33.25''$ | $12.79\%$ | $2,282.2\text{ Hz}$ ($0.64$) |
| **`p_mm_parallel`** | `34in_active_pmm:pmm_parallel` | $104.9\text{ mm}$ | $34.0''$ | $12.15\%$ | $2,848.0\text{ Hz}$ ($1.11$) |
| **`p_mm_series`** | `34in_active_pmm:pmm_series` | $95.4\text{ mm}$ | $34.0''$ | $11.05\%$ | $2,461.1\text{ Hz}$ ($1.09$) |
| **`mudbucker_deep`** | `30in_gibson_eb0:neck_deep` | $266.8\text{ mm}$ | $30.5''$ | $34.44\%$ | $1,053.0\text{ Hz}$ ($0.98$) |
| **`dingwall_bridge`** | `37in_multiscale_dingwall:bridge_clank` | $48.0\text{ mm}$ | $37.0''$ | $5.11\%$ | $6,616.9\text{ Hz}$ ($1.40$) |
| **`upright_piezo`** | `41in_upright_bass:bridge_piezo` | $5.0\text{ mm}$ | $41.25''$ | $0.48\%$ | $800.0\text{ Hz}$ ($1.80$) |
