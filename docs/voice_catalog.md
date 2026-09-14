# Allomorph Master Voice Catalog (Target Voicings & Transducers)

This catalog details the physical parameters, equivalent RLC circuit values, acoustic apertures, control harnesses, and electrical characteristics for the **Allomorph Digital Twin Profiles**.

---

## Quick Reference Summary

| # | Profile ID | Pickup Architecture | Topology | Harness / Controls | $L_{\text{eq}}$ | $R_{\text{dc}}$ | $f_r$ (Peak) | Acoustic & Circuit Character |
| :- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **01** | `jazz_pair_active` | Modern Active Jazz | Active 2-Band | Sadowsky 2-Band Preamp ($R_{\text{in}}=1\text{M}\Omega, R_{\text{out}}=100\,\Omega$) | $1.69\text{ H}$ (isolated) | $3.69\text{ k}\Omega$ | $7.8\text{ kHz}$ | Sadowsky-style active 2-band boost with isolated 60s J-pair ($92.1\text{ mm}$ aperture scoop), wideband hi-fi sparkle, punchy active bass and treble shelving. |
| **02** | `jazz_pair_open` | Vintage 60s J-Bass Pair | Dual Parallel | Vintage 60s $2\times 250\text{k}\Omega$ Vol ($125\text{k}\Omega$ net), $250\text{k}\Omega$ Tone, $47\text{nF}$ Cap | $1.69\text{ H}$ | $3.69\text{ k}\Omega$ | $2.7\text{ kHz}$ | Dual narrow single-coils in parallel; authentic 60s $92.1\text{ mm}$ ($3\frac{5}{8}''$) aperture scoop with natural woody low-mid resonance. |
| **02b**| `jazz_pair_mids` | 60s J-Bass Pair (22nF ToneStyler)| Dual Parallel | Vintage 60s $2\times 250\text{k}\Omega$ Vol, 22nF ToneStyler Pure Shunt ($R_{\text{tone}}=3.3\,\Omega$) | $1.69\text{ H}$ | $3.69\text{ k}\Omega$ | $762\text{ Hz}$ | Vocal midrange honk ($762\text{ Hz}$ resonant peak, $-3\text{ dB}$ at $1225\text{ Hz}$); pure capacitive shunt preserves punchy Jaco bridge growl with zero wiper mud. |
| **02c**| `jazz_bridge_growl` | 60s J-Bass Pair (Jaco Bias)| Dual Parallel | Vintage dual $250\text{k}\Omega$ Vol (Bridge 100%, Neck 75% decoupled), $250\text{k}\Omega$ Tone | $1.69\text{ H}$ | $3.69\text{ k}\Omega$ | $2.8\text{ kHz}$ | Classic Jaco Pastorius fingerstyle vocal bridge growl; neck volume backed off to 75% adds $55\text{ k}\Omega$ series wiper isolation, shifting comb scoop to vocal $550\text{--}800\text{ Hz}$ while keeping punchy low-end body. |
| **03** | `jazz_bridge_open` | 60s J-Bass Bridge | Single Coil | Vintage $250\text{k}\Omega$ Vol, $250\text{k}\Omega$ Tone, $47\text{nF}$ Cap | $3.60\text{ H}$ | $7.80\text{ k}\Omega$ | $2.8\text{ kHz}$ | Narrow single-coil placed in 60s bridge position ($63.5\text{ mm}$ from bridge); focused midrange growl with articulate transient snap. |
| **04** | `precision_active` | Modern Active Split P | Single Split | Sadowsky 2-Band Active Preamp ($R_{\text{in}}=1\text{M}\Omega, R_{\text{out}}=100\,\Omega$), $500\text{k}\Omega$ Vol | $4.80\text{ H}$ (isolated) | $9.50\text{ k}\Omega$ | $4.8\text{ kHz}$ | Ceramic split-coil with active 2-band preamp buffer isolating coils from cable loading; extended high-end air, punchy low-mids, and wide dynamic headroom. |
| **05** | `precision_vintage` | Vintage '62 P (Tone Open)| Single Split | Vintage 1962 CTS $250\text{k}\Omega$ Vol, $250\text{k}\Omega$ Tone Open, $47\text{nF}$ Cap | $3.80\text{ H}$ | $10.50\text{ k}\Omega$ | $2.1\text{ kHz}$ | Alnico V split-coil wide open; touch-sensitive dynamic response, woody organic bloom ($4.0\text{ kHz}$ cutoff). |
| **05b**| `precision_mids` | Vintage '62 P (22nF ToneStyler)| Single Split | 22nF ToneStyler Pure Shunt ($R_{\text{tone}}=3.3\,\Omega$) | $3.80\text{ H}$ | $10.50\text{ k}\Omega$ | $440\text{ Hz}$ | Modern Fender spec; punchy $440\text{ Hz}$ low-mid resonant focus (+1.5 dB), $-3\text{ dB}$ cutoff at $750\text{ Hz}$, eliminates fret clatter while retaining punch. |
| **05c**| `precision_warm` | Vintage '62 P (47nF ToneStyler)| Single Split | 47nF ToneStyler Pure Shunt ($R_{\text{tone}}=3.3\,\Omega$), Flatwound Heavy | $3.80\text{ H}$ | $10.50\text{ k}\Omega$ | $450\text{ Hz}$ | Authentic Jamerson Motown flatwound thump with 47nF ToneStyler pure capacitive shunt; low-tension bloom and deep pillowy warmth. |
| **05d**| `precision_dub` | Vintage '50s P (100nF ToneStyler)| Single Split | 100nF ToneStyler Pure Shunt ($R_{\text{tone}}=3.3\,\Omega$) | $3.80\text{ H}$ | $10.50\text{ k}\Omega$ | Sub-bass | Original 1951–1959 Fullerton factory paper-in-oil spec; massive sub-bass shelf rolloff ($-3\text{ dB}$ at $240\text{ Hz}$, deep Motown / reggae dub thump). |
| **07** | `pj_active` | Modern Active P/J | Active 2-Band | Sadowsky/Spector 2-Band Preamp ($R_{\text{in}}=1\text{M}\Omega, R_{\text{out}}=100\,\Omega$) | $2.06\text{ H}$ (isolated) | $4.28\text{ k}\Omega$ | $7.6\text{ kHz}$ | Active 2-band boost with isolated ceramic P/J coils ($139/111\text{ mm}$ P + $63.5\text{ mm}$ J); punchy sub-bass fundamental, wideband sparkle, and aggressive bridge clank. |
| **08** | `pj_passive` | Vintage '80s Passive P/J| Parallel Sum | Dual $250\text{k}\Omega$ Vol ($125\text{k}\Omega$ net), $250\text{k}\Omega$ Tone, $47\text{nF}$ Cap | $1.85\text{ H}$ | $4.48\text{ k}\Omega$ | $2.8\text{ kHz}$ | Vintage Alnico V split-P and 60s J-bridge in parallel; authentic '80s Fender Special / Yamaha BB thump with woody mid-punch. |
| **09** | `stingray_parallel` | MM Humbucker Active | Active 2-Band | Music Man 2-Band Preamp ($R_{\text{in}}=1\text{M}\Omega, R_{\text{out}}=100\,\Omega$) | $1.20\text{ H}$ (isolated) | $2.20\text{ k}\Omega$ | $8.5\text{ kHz}$ | Dual-coil parallel humbucker with authentic active MM preamp buffer isolating coils from cable loading; comb notch at $2.5\text{ kHz}$ and clank peak at $8.5\text{ kHz}$. |
| **09b**| `stingray_series` | MM Series Humbucker Active | Active 2-Band | Music Man 2-Band Preamp ($R_{\text{in}}=1\text{M}\Omega, R_{\text{out}}=100\,\Omega$) | $4.80\text{ H}$ (isolated) | $8.80\text{ k}\Omega$ | $4.1\text{ kHz}$ | Dual-coil series humbucker with active 2-band buffer; physical $4:1$ impedance scaling ($L_{\text{ser}}=4 L_{\text{par}}$), $+5.6\text{ dB}$ open-circuit EMF surge, and focused $4.1\text{ kHz}$ active series resonance (StingRay 5 / Sterling series switch). |
| **10** | `rickenbacker_clank` | 4003 Bridge HPF | Series HPF | Factory Rickenbacker $330\text{k}\Omega$ Vol, $330\text{k}\Omega$ Tone, $4.7\text{nF}$ Series HPF | $3.80\text{ H}$ | $8.40\text{ k}\Omega$ | $2.2\text{ kHz}$ | High-output single-coil with vintage $4.7\text{ nF}$ series HPF; removes low-end mud below $150\text{ Hz}$, delivering aggressive pick grit and clang. |
| **11** | `p_mm_parallel` | Modern Active P/MM | Active Buffer | Studio Active Buffer ($R_{\text{in}}=1\text{M}\Omega, R_{\text{out}}=100\,\Omega$) | $0.96\text{ H}$ (isolated) | $1.79\text{ k}\Omega$ | $3.4\text{ kHz}$ | Authentic active parallel P/MM (Sandberg California VM / Lakland 44-02); Split-P neck and MM parallel bridge into high-Z buffer with zero cable drag; articulate punch and modern slap growl. |
| **11b**| `p_mm_series` | Modern Active P/MM (Series)| Active Buffer | Studio Active Buffer ($R_{\text{in}}=1\text{M}\Omega, R_{\text{out}}=100\,\Omega$) | $8.40\text{ H}$ (isolated) | $16.90\text{ k}\Omega$ | $3.2\text{ kHz}$ | Split P and MM parallel humbucker wired in series before active buffer; $+5.8\text{ dB}$ inductive boost with punchy authority and zero cable drag. |
| **12** | `mudbucker_deep` | Overwound Series| Ultra Series | Gibson $500\text{k}\Omega$ Vol, $500\text{k}\Omega$ Tone, $22\text{nF}$ Cap | $14.40\text{ H}$| $27.90\text{ k}\Omega$ | $1.2\text{ kHz}$ | Extreme dual-coil series network; subterranean low-end focus, steep natural top-end rolloff. |
| **13** | `dingwall_bridge` | Multi-Scale MM | Angled Parallel | Dingwall Active Onboard Buffer ($1\text{ M}\Omega \parallel 25\text{ pF}$, low-Z out) | $2.30\text{ H}$ (isolated)| $4.40\text{ k}\Omega$ | $7.3\text{ kHz}$ | 34"-37" fanned-fret bridge position ($48.0\text{ mm}$) with high-tension wave speeds and active buffered FD3 parallel dual-coil sparkle. |
| **14** | `upright_acoustic` | Upright Bridge Piezo | Transducer | High-Z Piezo Buffer ($R_{\text{in}}=10\text{ M}\Omega$) | $0.00\text{ H}$ | $10.0\text{ M}\Omega$ | Woody Acoustic | Authentic double bass bridge-wing piezo sensor; captures body wood resonance, bow bite, and organic finger thump. |
| **15** | `studio_direct` | Studio Direct (Dynamic DI) | Studio (Direct) | Transparent ($0.00\text{ dB}$ flat linear transfer) | $0.00\text{ H}$ | $50\,\Omega$ | Flat (0 dB) | Preserves physical pickup aperture and imparts organic dynamic DI feel and non-linear Alnico compliance. |
| **15b**| `studio_active` | Studio Active (Modern Active Buffer) | Studio (Active) | Studio Active Buffer ($R_{\text{in}}=1\text{M}\Omega, R_{\text{out}}=100\,\Omega$) | $3.20\text{ H}$ (isolated) | $7.20\text{ k}\Omega$ | $5.2\text{ kHz}$ | Removes passive cable loading ($750\text{ pF}$) and pot damping to restore wideband hi-fi sparkle and headroom; preserves natural pickup aperture with clean active headroom. |
| **15c**| `studio_passive` | Studio Passive (Passive Loading) | Studio (Passive) | Standard Passive Harness ($250\text{k}\Omega\text{ Vol/Tone}, 47\text{nF}, 750\text{pF}$) | $4.20\text{ H}$ | $8.50\text{ k}\Omega$ | $2.8\text{ kHz}$ | Adds high-impedance passive impedance dynamics, resonant peak ($2.8\text{ kHz}$), $750\text{ pF}$ cable loading, and $250\text{k}\Omega$ pot damping to active basses or stacks passive tone; preserves natural aperture. |


---

## Detailed Profile Specifications

### 01. `jazz_pair_active` (Modern Active Jazz Bass Pair)
* **Archetype:** Sadowsky / Modern NYC Active Jazz Bass (Dual Single-Coil with 2-Band Boost Preamp)
* **Pickup Architecture:** Dual parallel single coils (Neck: $3.20\text{ H}$, Bridge: $3.60\text{ H}$ in 60s spacing, $92.1\text{ mm}$ coil spacing)
* **Control Harness:** Sadowsky-style active 2-band preamp buffer ($R_{\text{in}} = 1.0\text{ M}\Omega$, $C_{\text{in}} = 25\text{ pF}$, low-impedance $R_{\text{out}} = 100\,\Omega$). Preamp active EQ provides $+4.0\text{ dB}$ bass boost ($40\text{ Hz}$ shelf) and $+4.0\text{ dB}$ treble boost ($4.0\text{ kHz}$ shelf).
* **Electrical Parameters:** The active buffer completely isolates the high-impedance pickup coils from cable capacitance ($750\text{ pF}$). The coils resonate with only $C_{\text{coil}} + C_{\text{in}} \approx 170\text{ pF}$, shifting the raw coil resonance up to $\sim 7.8\text{ kHz}$.
* **Acoustic Character:** Modern hi-fi sparkle, punchy extended low-end weight, tight transient response, and transparent scooped midrange.
* **32" Bass Setting:** ABCX Blend at **Center Detent (50/50)** (MMTWX in Single-Coil Mode - Push/Pull Down).

### 02. `jazz_pair_open` (Vintage 60s Jazz Bass Pair)
* **Archetype:** Standard 1960s Fender Jazz Bass Dual Single-Coil Pair (Tone Wide Open)
* **Pickup Architecture:** Dual parallel single coils (Neck: $3.20\text{ H}$; Bridge: $3.60\text{ H}$) spaced $92.1\text{ mm}$ ($3\frac{5}{8}''$) apart ($155.6\text{ mm}$ neck, $63.5\text{ mm}$ bridge)
* **Control Harness:** Authentic vintage dual $250\text{k}\Omega$ volume pots in parallel ($125\text{k}\Omega$ net load resistance), single $250\text{k}\Omega$ tone pot, $47\text{ nF}$ capacitor, no treble bleed.
* **Electrical Parameters:** Dual parallel SPICE branches; combined $L_{\text{eq}} = 1.69\text{ H}$, $R_{\text{dc}} = 3.69\text{ k}\Omega$, $R_{\text{eddy}} = 65\text{ k}\Omega$, $C_{\text{coil}} = 145\text{ pF}$. When loaded by $125\text{k}\Omega$ volume, $250\text{k}\Omega$ tone, and $750\text{ pF}$ cable, the composite loaded resonance sits at $2.7\text{ kHz}$.
* **Acoustic Character:** Symmetrical dual-aperture phase cancellation produces a hollow mid-scoop at $1\text{ kHz}$ with classic woody low-mid resonance and open high-end.
* **32" Bass Setting:** ABCX Blend at **Center Detent (50/50)** (MMTWX in Single-Coil Mode - Push/Pull Down).

### 02b. `jazz_pair_mids` (Vintage 60s Jazz Bass Pair - 22nF ToneStyler)
* **Archetype:** Standard 1960s Fender Jazz Bass Dual Single-Coil Pair with Stellartone ToneStyler (22nF Pure Capacitive Shunt)
* **Pickup Architecture:** Dual parallel single coils (Neck: $3.20\text{ H}$; Bridge: $3.60\text{ H}$) spaced $92.1\text{ mm}$ ($3\frac{5}{8}''$) apart ($155.6\text{ mm}$ neck, $63.5\text{ mm}$ bridge)
* **Control Harness:** Authentic vintage dual $250\text{k}\Omega$ volume pots in parallel ($125\text{k}\Omega$ net load), Stellartone ToneStyler rotary switch selecting a precision $22\text{ nF}$ capacitor shunt to ground ($R_{\text{tone}} = 3.3\,\Omega$ switch/lead ESR, zero variable wiper damping).
* **Electrical Parameters:** Combined $L_{\text{eq}} = 1.69\text{ H}$, $R_{\text{dc}} = 3.69\text{ k}\Omega$, $R_{\text{tone}} = 3.3\,\Omega$, $C_{\text{tone}} = 22\text{ nF}$. Undamped resonant peak at **$762\text{ Hz}$**, $-3\text{ dB}$ cutoff at **$1225\text{ Hz}$**.
* **Acoustic Character:** The iconic Jaco Pastorius / Marcus Miller fingerstyle vocal bridge burp: by eliminating variable potentiometer wiper resistance, the high-$Q$ resonant bump (+4.8 dB above standard 50% pot) is preserved directly in the vocal $760\text{ Hz}$ register, filtering out fret click and pick noise while maintaining commanding midrange articulation and punch.
* **32" Bass Setting:** ABCX Blend at **Center Detent (50/50)** (MMTWX in Single-Coil Mode - Push/Pull Down).

### 02c. `jazz_bridge_growl` (60s Jazz Bass Pair - Bridge Growl Bias / Jaco Setting)
* **Archetype:** 1960s Fender Jazz Bass Dual Single-Coil Pair with Bridge-Biased Decoupled Volume Pots (Bridge @ 100%, Neck @ 75%)
* **Pickup Architecture:** Dual parallel single coils (Neck: $3.20\text{ H}$; Bridge: $3.60\text{ H}$) spaced $92.1\text{ mm}$ ($3\frac{5}{8}''$) apart ($155.6\text{ mm}$ neck, $63.5\text{ mm}$ bridge)
* **Control Harness:** Authentic vintage dual $250\text{k}\Omega$ audio volume pots wired in independent/decoupled configuration with master $250\text{k}\Omega$ tone pot ($47\text{ nF}$ paper-in-oil cap). Bridge volume at 100% ($R_{\text{wiper,b}} = 0\,\Omega$), Neck volume backed off to ~75% ($R_{\text{wiper,n}} = 55\text{ k}\Omega$, $R_{\text{bot,n}} = 195\text{ k}\Omega$).
* **Electrical & Acoustic Physics:** Backing off the neck volume introduces $55\text{ k}\Omega$ of series resistance on the neck branch, isolating its resonant circuit and attenuating neck output by ~2.5 dB. This asymmetric summing shifts the phase cancellation comb notch from the traditional 1 kHz scoop into the vocal $550\text{--}800\text{ Hz}$ frequency zone, while preserving the punchy low-end body of both pickups working together.
* **Acoustic Character:** The signature Jaco Pastorius / fingerstyle vocal bridge growl: punchier and fuller than the solo bridge pickup, but with sharper, more vocal midrange articulation and burp than both pickups wide open.
* **32" Bass Setting:** ABCX Blend at **~65% Bridge / 35% Neck** (MMTWX in Single-Coil Mode - Push/Pull Down).

### 03. `jazz_bridge_open` (60s Jazz Bass Bridge Single-Coil)
* **Archetype:** 1960s Fender Jazz Bass Bridge Single Coil
* **Coil Model:** Single coil ($w = 0.75''$) positioned $2.5''$ ($63.5\text{ mm}$) from the bridge
* **Control Harness:** Authentic $250\text{k}\Omega$ volume pot, $250\text{k}\Omega$ tone pot, $47\text{ nF}$ tone capacitor, no treble bleed.
* **Electrical Parameters:** $L = 3.60\text{ H}$, $R_{\text{dc}} = 7.80\text{ k}\Omega$, $R_{\text{eddy}} = 125\text{ k}\Omega$, $C_{\text{coil}} = 70\text{ pF}$. Loaded $f_r = 2.8\text{ kHz}$.
* **Acoustic Character:** Focused harmonic bite centered in the $800\text{ Hz}\text{--}1.6\text{ kHz}$ range with solid, articulate low-end punch. Accentuates fingerstyle definition and classic Jaco growl.
* **32" Bass Setting:** ABCX Blend **100% Bridge** (MMTWX in Single-Coil Mode - Push/Pull Down).

### 04. `precision_active` (Modern Active Split-Coil Ceramic)
* **Archetype:** Modern Active Ceramic Split-Coil P with Sadowsky 2-Band Preamp
* **Coil Model:** Ceramic split-coil ($w = 1.00''$, E/A at $139.0\text{ mm}$, D/G at $111.0\text{ mm}$)
* **Control Harness:** Active buffer isolating coils ($R_{\text{in}} = 1\text{ M}\Omega$), Sadowsky 2-band boost circuit, $500\text{k}\Omega$ master volume.
* **Electrical Parameters:** $L = 4.80\text{ H}$, $R_{\text{dc}} = 9.50\text{ k}\Omega$, $R_{\text{eddy}} = 110\text{ k}\Omega$, $C_{\text{coil}} = 80\text{ pF}$, $V_{\text{sat}} = 0.40\text{ V}$. Loaded $f_r = 4.8\text{ kHz}$.
* **Acoustic Character:** Hi-fi punch and immediate pick attack with transparent low-mids ($300\text{--}800\text{ Hz}$) and extended high-frequency clarity. The active buffer eliminates cable capacitive loading, producing crisp ceramic definition without passive treble roll-off.
* **32" Bass Setting:** ABCX Blend **100% Neck (PX/PCSX)**.

### 05. `precision_vintage` (Vintage '62 Split-Coil Alnico V - Tone Open)
* **Archetype:** 1962 Fender Precision with Alnico V Pole Pieces (Tone Wide Open)
* **Coil Model:** Alnico V split-coil ($w = 1.00''$)
* **Control Harness:** Vintage 1962 CTS $250\text{k}\Omega$ volume pot, $250\text{k}\Omega$ tone pot wide open, $47\text{ nF}$ paper-in-oil tone capacitor, no treble bleed.
* **Electrical Parameters:** $L = 3.80\text{ H}$, $R_{\text{dc}} = 10.50\text{ k}\Omega$, $R_{\text{eddy}} = 180\text{ k}\Omega$, $C_{\text{coil}} = 60\text{ pF}$. Loaded $f_r = 2.1\text{ kHz}$, $-3\text{ dB}$ cutoff at $4.0\text{ kHz}$.
* **Acoustic Character:** Lower core eddy-current damping yields an open, touch-sensitive, uncompressed response with organic woody upper-mid bloom and complete harmonic extension.
* **32" Bass Setting:** ABCX Blend **100% Neck (PX/PCSX)**.

### 05b. `precision_mids` (Vintage '62 Split-P - 22nF ToneStyler)
* **Archetype:** 1962 Fender Precision with Stellartone ToneStyler (22nF Pure Capacitive Shunt)
* **Coil Model:** Alnico V split-coil ($w = 1.00''$)
* **Control Harness:** Vintage 1962 CTS $250\text{k}\Omega$ volume pot, Stellartone ToneStyler rotary switch selecting a precision $22\text{ nF}$ capacitor shunt to ground ($R_{\text{tone}} = 3.3\,\Omega$, zero series wiper damping).
* **Electrical Parameters:** $L = 3.80\text{ H}$, $R_{\text{dc}} = 10.50\text{ k}\Omega$, $R_{\text{tone}} = 3.3\,\Omega$, $C_{\text{tone}} = 22\text{ nF}$. Undamped resonant peak at **$440\text{ Hz}$** (+1.5 dB), $-3\text{ dB}$ cutoff at **$750\text{ Hz}$**.
* **Acoustic Character:** Modern Fender tone capacitor spec: eliminates fret clatter and pick click while preserving high-$Q$ punch in the punchy $440\text{ Hz}$ low-mid register.
* **32" Bass Setting:** ABCX Blend **100% Neck (PX/PCSX)**.

### 05c. `precision_warm` (Vintage '62 Split-P - 47nF ToneStyler Motown)
* **Archetype:** Vintage 1960s Precision Bass with Stellartone ToneStyler 47nF Pure Capacitive Shunt & Heavy Flatwounds (Jamerson / Motown style)
* **Coil Model:** Alnico V split-coil ($w = 1.00''$)
* **Control Harness:** Vintage CTS $250\text{k}\Omega$ volume pot, Stellartone ToneStyler rotary switch selecting a precision $47\text{ nF}$ paper-in-oil capacitor shunt to ground ($R_{\text{tone}} = 3.3\,\Omega$ contact/lead ESR, zero series wiper resistance damping).
* **Electrical Parameters:** $L = 3.80\text{ H}$, $R_{\text{dc}} = 10.50\text{ k}\Omega$, $R_{\text{eddy}} = 180\text{ k}\Omega$, $C_{\text{coil}} = 60\text{ pF}$, $R_{\text{tone}} = 3.3\,\Omega$, $C_{\text{tone}} = 47\text{ nF}$. Loaded resonant peak at **$450\text{ Hz}$** (+1.2 dB), steep rolloff above $800\text{ Hz}$.
* **Acoustic Character:** Removes fret clatter and transient click above $800\text{ Hz}$ while preserving a punchy resonant low-mid bump at $450\text{ Hz}$. Paired with heavy flatwound string acoustic damping, delivers iconic pillowy Motown thump.
* **32" Bass Setting:** ABCX Blend **100% Neck (PX/PCSX)**.

### 05d. `precision_dub` (Vintage '50s Split-P - 100nF ToneStyler)
* **Archetype:** Early 1950s Fender Precision with Stellartone ToneStyler (100nF / 0.1µF Pure Capacitive Shunt)
* **Coil Model:** Alnico V split-coil ($w = 1.00''$)
* **Control Harness:** Vintage 1962 CTS $250\text{k}\Omega$ volume pot, Stellartone ToneStyler rotary switch selecting a precision $100\text{ nF}$ (0.1µF) paper-in-oil / ceramic disc capacitor shunt to ground ($R_{\text{tone}} = 3.3\,\Omega$).
* **Electrical Parameters:** $L = 3.80\text{ H}$, $R_{\text{dc}} = 10.50\text{ k}\Omega$, $R_{\text{tone}} = 3.3\,\Omega$, $C_{\text{tone}} = 100\text{ nF}$. Sub-bass shelf, $-3\text{ dB}$ cutoff at **$240\text{ Hz}$**.
* **Acoustic Character:** Original 1951–1959 Fullerton factory 0.1µF spec: massive low-pass filter concentrating all energy into the sub-bass register, yielding deep Motown / reggae dub thump.
* **32" Bass Setting:** ABCX Blend **100% Neck (PX/PCSX)**.

*(Note: Slot 06 is currently reserved / unassigned.)*

### 07. `pj_active` (Modern Active P/J Bass)
* **Archetype:** Sadowsky P/J / Spector NS-2 Active 2-Band P/J
* **Pickup Architecture:** Dual parallel pickups:
  * **Precision Neck:** Modern ceramic split-coil Bartolini 8CBP ($L = 4.80\text{ H}$, E/A @ $139\text{ mm}$, D/G @ $111\text{ mm}$)
  * **Jazz Bridge:** Modern ceramic single-coil Bartolini 9CBJS1 ($L = 3.60\text{ H}$, @ $63.5\text{ mm}$)
* **Control Harness:** Sadowsky/Spector active 2-band preamp buffer ($R_{\text{in}} = 1.0\text{ M}\Omega$, $C_{\text{in}} = 25\text{ pF}$, low-impedance $R_{\text{out}} = 100\,\Omega$). Active EQ provides $+4.0\text{ dB}$ bass boost ($40\text{ Hz}$ shelf) and $+4.0\text{ dB}$ treble boost ($4.0\text{ kHz}$ shelf).
* **Electrical Parameters:** Active buffer isolates both pickups from cable capacitance ($750\text{ pF}$). Raw ceramic resonance sits in the high-clarity $7.6\text{ kHz}$ region.
* **Acoustic Character:** Thunderous modern low-end weight with zero muddiness, balanced against articulate, clanky bridge bite. Cuts through dense modern pop, gospel, and metal mixes.
* **32" Bass Setting:** ABCX Blend at **Center Detent (50/50)** (MMTWX in Single-Coil Mode - Push/Pull Down).

### 08. `pj_passive` (Vintage '80s Passive P/J Bass)
* **Archetype:** 1980s Fender Jazz Bass Special / Yamaha BB3000 Style Passive P/J
* **Pickup Architecture:** Dual parallel pickups:
  * **Precision Neck:** Vintage '62 Alnico V split-coil ($L = 3.80\text{ H}, R_{\text{dc}} = 10.50\text{ k}\Omega$, E/A @ $139\text{ mm}$, D/G @ $111\text{ mm}$)
  * **Jazz Bridge:** Vintage 60s Alnico V single-coil ($L = 3.60\text{ H}, R_{\text{dc}} = 7.80\text{ k}\Omega$, @ $63.5\text{ mm}$)
* **Control Harness:** Authentic passive dual volume harness ($2\times 250\text{k}\Omega$ pots in parallel = $125\text{k}\Omega$ net load), $250\text{k}\Omega$ tone pot, $47\text{ nF}$ capacitor, no treble bleed.
* **Electrical Parameters:** Dual parallel SPICE branches; composite $L_{\text{eq}} = 1.85\text{ H}$, $R_{\text{dc}} = 4.48\text{ k}\Omega$, $R_{\text{eddy}} = 74\text{ k}\Omega$, $C_{\text{coil}} = 130\text{ pF}$. Loaded $f_r = 2.8\text{ kHz}$.
* **Acoustic Character:** Classic '80s rock/punk/funk thump (Duff McKagan / Michael Anthony style). Warm, woody, mid-forward punch with gentle high-end roll-off.
* **32" Bass Setting:** ABCX Blend at **Center Detent (50/50)** (MMTWX in Single-Coil Mode - Push/Pull Down).

### 09. `stingray_parallel` (Music Man Active Dual-Coil Parallel)
* **Archetype:** Music Man StingRay 4-String Active Dual-Coil Humbucker
* **Coil Model:** Dual-coil humbucker ($w = 1.50''$, $d = 0.75''$) wired in parallel
* **Control Harness:** Authentic Music Man active 2-band op-amp preamp buffer ($R_{\text{in}} = 1.0\text{ M}\Omega$, $C_{\text{in}} = 25\text{ pF}$, low-impedance $R_{\text{out}} = 100\,\Omega$). Active EQ provides $+1.8\text{ dB}$ bass boost ($50\text{ Hz}$ shelf) and $+2.2\text{ dB}$ treble boost ($4\text{--}7\text{ kHz}$ shelf).
* **Electrical Parameters:** The active buffer isolates the coils from the $750\text{ pF}$ instrument cable. Parallel equivalent $L = 1.20\text{ H}$, $R_{\text{dc}} = 2.20\text{ k}\Omega$, $R_{\text{eddy}} = 75\text{ k}\Omega$, $C_{\text{coil}} = 180\text{ pF}$. Isolated from cable capacitance, peak resonance sits at $8.5\text{ kHz}$.
* **Acoustic Character:** Dual-coil phase comb cancellation notch at $2.5\text{ kHz}$ combined with authoritative active bass punch and signature metallic treble clank.
* **32" Bass Setting:** ABCX Blend **100% Bridge** (MMTWX in Dual-Coil Mode - Push/Pull Up).

### 09b. `stingray_series` (Music Man Active Dual-Coil Series)
* **Archetype:** Music Man StingRay 5 / Sterling 4-String Series Humbucker (3-Way Switch Position 1)
* **Coil Model:** Dual-coil humbucker ($w = 1.50''$, $d = 0.75''$) wired in series
* **Control Harness:** Authentic Music Man active 2-band op-amp preamp buffer ($R_{\text{in}} = 1.0\text{ M}\Omega$, $C_{\text{in}} = 25\text{ pF}$, low-impedance $R_{\text{out}} = 100\,\Omega$). Active EQ provides $+1.8\text{ dB}$ bass boost ($50\text{ Hz}$ shelf) and $+2.2\text{ dB}$ treble shelf ($4\text{--}7\text{ kHz}$).
* **Electrical Parameters:** Two coils wired in series satisfying physical $4:1$ impedance scaling; combined $L = 4.80\text{ H}$ ($4\times 1.20\text{ H}$), $R_{\text{dc}} = 8.80\text{ k}\Omega$ ($4\times 2.20\text{ k}\Omega$), $R_{\text{eddy}} = 150\text{ k}\Omega$, $C_{\text{coil}} = 210\text{ pF}$. Isolated from cable capacitance, the authentic active series resonance sits at $4.1\text{ kHz}$.
* **Acoustic Character:** The series coil connection adds $+5.6\text{ dB}$ open-circuit voltage doubling, filling the midrange with commanding $4.1\text{ kHz}$ bark while naturally softening extreme $>7\text{ kHz}$ treble clank. The definitive punchy, cut-through-the-mix voice favored for heavy rock, slap thumb punch, and modern gospel.
* **32" Bass Setting:** ABCX Blend **100% Bridge** (MMTWX in Dual-Coil Mode - Push/Pull Up).

### 10. `rickenbacker_clank` (Rickenbacker 4003 Bridge with 4.7nF HPF)
* **Archetype:** Rickenbacker 4001/4003 Bridge Pickup with Vintage High-Pass Push-Pull
* **Coil Model:** High-output single coil with $4.7\text{ nF}$ series capacitor
* **Control Harness:** Factory Rickenbacker $330\text{k}\Omega$ volume pot, $330\text{k}\Omega$ tone pot, $47\text{ nF}$ tone cap, $4.7\text{ nF}$ vintage series capacitor.
* **Electrical Parameters:** $L = 3.80\text{ H}$, $R_{\text{dc}} = 8.40\text{ k}\Omega$, $C_{\text{series}} = 4.7\text{ nF}$, $C_{\text{coil}} = 90\text{ pF}$. Loaded $f_r = 2.2\text{ kHz}$.
* **Acoustic Character:** The series capacitor acts as a high-pass filter, rolling off sub-bass below $150\text{ Hz}$ while focusing midrange punch ($1.5\text{--}2.5\text{ kHz}$). Produces an aggressive, gritty pick attack.
* **32" Bass Setting:** ABCX Blend **100% Bridge** (MMTWX in Single-Coil Mode - Push/Pull Down).

### 11. `p_mm_parallel` (Modern Active P/MM Bass)
* **Archetype:** Sandberg California VM4/VM5 / Lakland 44-02 Style Active Parallel P/MM
* **Pickup Architecture:** Dual parallel pickups:
  * **Precision Neck:** Split-coil Bartolini 8CBP ($L = 4.80\text{ H}, R_{\text{dc}} = 9.50\text{ k}\Omega$)
  * **Music Man Bridge:** Dual-coil parallel humbucker ($L = 1.20\text{ H}, R_{\text{dc}} = 2.20\text{ k}\Omega$)
* **Control Harness:** Onboard active buffer stage ($R_{\text{in}} = 1\text{ M}\Omega, C_{\text{in}} = 25\text{ pF}$, low-Z output driver $R_{\text{out}} = 100\,\Omega$).
* **Electrical Parameters:** Active parallel summation: $L_{\text{par}} = 0.96\text{ H}$, $R_{\text{dc,par}} = 1.79\text{ k}\Omega$. Loaded $f_r = 3.4\text{ kHz}$.
* **Acoustic Character:** Zero cable capacitive loss on the coils. Delivers woody Split-P low-end authority combined with the laser-focused attack, wideband punch, and metallic growl of the Music Man sweet spot. The quintessential modern active slap and fingerstyle tone.
* **32" Bass Setting:** ABCX Blend at **Center Detent (50/50)** (MMTWX in Dual-Coil Mode - Push/Pull Up).

### 11b. `p_mm_series` (Modern Active P/MM Bass - Series)
* **Archetype:** Modern Active P/MM in Series Mode (Lakland 44-02 / Sandberg California VM Series Switch)
* **Pickup Architecture:** Dual series pickups:
  * **Precision Neck:** Split-coil Bartolini 8CBP ($L = 4.80\text{ H}, R_{\text{dc}} = 9.50\text{ k}\Omega$)
  * **Music Man Bridge:** Dual-coil parallel humbucker ($L = 3.60\text{ H}, R_{\text{dc}} = 7.40\text{ k}\Omega$)
* **Control Harness:** Onboard active buffer stage ($R_{\text{in}} = 1\text{ M}\Omega, C_{\text{in}} = 25\text{ pF}$, low-Z output driver $R_{\text{out}} = 100\,\Omega$).
* **Electrical Parameters:** Active series summation: $L_{\text{ser}} = 8.40\text{ H}$, $R_{\text{dc}} = 16.90\text{ k}\Omega$. Buffered $f_r \approx 3.2\text{ kHz}$.
* **Acoustic Character:** Series wiring before active buffer delivers massive $+5.8\text{ dB}$ signal surge with commanding low-mid authority while preserving crisp transient attack through zero cable loading.
* **32" Bass Setting:** ABCX Blend at **Center Detent (50/50)** (MMTWX in Dual-Coil Mode - Push/Pull Up).

### 12. `mudbucker_deep` (Ultra-High Inductance Overwound Series)
* **Archetype:** Gibson EB-0 Mudbucker / Overwound Series Dual-Coil
* **Coil Model:** Extreme dual-coil overwound series network
* **Physical Placement:** Mounted directly against the 20th-fret neck heel ($266.8\text{ mm}$ from bridge / $165.0\text{ mm}$ forward from the 12th fret datum).
* **Control Harness:** Gibson-spec $500\text{k}\Omega$ volume pot, $500\text{k}\Omega$ tone pot, $22\text{ nF}$ tone capacitor.
* **Electrical Parameters:** $L = 14.40\text{ H}$, $R_{\text{dc}} = 27.90\text{ k}\Omega$, $R_{\text{eddy}} = 450\text{ k}\Omega$, $C_{\text{coil}} = 80\text{ pF}$. Loaded $f_r = 1.2\text{ kHz}$.
* **Acoustic Character:** Resonant peak pulls down to $1.2\text{ kHz}$, naturally rolling off high frequencies with steep second-order attenuation. Deep, subterranean low end with zero high-frequency fizz, ideal for heavy fuzz and saturated drive stages.
* **32" Bass Setting:** ABCX Blend **100% Neck (PX/PCSX)**.

### 13. `dingwall_bridge` (Fanned-Fret Multi-Scale Bridge)
* **Archetype:** Dingwall NG Multi-Scale Angled Bridge Position
* **Coil Model:** Dual-coil humbucker in parallel positioned $48.0\text{ mm}$ from bridge
* **Control Harness:** Dingwall Combustion/NG active onboard buffer ($1\text{ M}\Omega \parallel 25\text{ pF}$ input, $100\,\Omega$ low-Z output driver isolating coils from cable capacitance).
* **Electrical Parameters:** $L = 2.30\text{ H}$, $R_{\text{dc}} = 4.40\text{ k}\Omega$, $R_{\text{eddy}} = 150\text{ k}\Omega$, $C_{\text{coil}} = 120\text{ pF}$. Active buffered $f_r = 7.3\text{ kHz}$.
* **Acoustic Character:** 34"-37" fanned-fret wave-speed scaling with high string tension, angled bridge sweet spot ($48.0\text{ mm}$), active buffered FD3 dual-coil parallel sparkle ($7.3\text{ kHz}$), and stainless-steel string harmonic extension.
* **32" Bass Setting:** ABCX Blend **100% Bridge** (MMTWX in Dual-Coil Mode - Push/Pull Up).

### 14. `upright_acoustic` (Upright Acoustic Bridge Transducer)
* **Archetype:** Direct Bridge Force Transducer on 3/4 Double Bass (41.5" Scale, Underwood / David Gage Realist style)
* **Target String Archetype:** 3/4 Double Bass spiral rope-core strings (`double_bass_spirocore`, $265\text{ lbs}$ tension, coupled spruce soundboard damping)
* **Sensor Model:** Direct bridge saddle force sensor ($x = 5.0\text{ mm}$) with velocity-to-force leaky integration ($+6\text{ dB/oct}$ from $70\text{ Hz}$ to $250\text{ Hz}$)
* **Dynamic Compliance:** Soft-knee bridge rocking saturation ($V_{\text{sat}} \cdot \tanh(v / V_{\text{sat}})$), dynamically scaled for lower-tension strings ($V_{\text{sat}} = 0.336\text{ V}$ on 32" fretless with La Bella LTF vs $0.42\text{ V}$ baseline)
* **Anti-Double-Damping:** Automatically matches source string damping ($f_{\text{damp, src}}$) to target double-bass damping ($f_{\text{damp, tgt}}$). When evaluated from flatwound instruments (La Bella LTF), the filter avoids double-muffling the high frequencies while preserving woody bridge bite.
* **Electrical Parameters:** Pure capacitive piezo sensor ($C_{\text{piezo}} = 1.2\text{ nF}$), direct $100\text{ M}\Omega$ buffer, $15\text{ nF}$ subsonic rumble decoupling @ $32\text{ Hz}$
* **Acoustic Character:** Eliminates magnetic pickup comb filtering; delivers smooth, woody double-bass pizzicato bloom with deep sub-bass body resonance.
* **Downstream Acoustic Soundboard Radiation (Recommended AST IRs):** Upright Piezo synthesizes the authentic mechanical force directly at the bridge saddle. To radiate this raw transducer signal into realistic 3/4 double bass acoustic body and soundboard resonance, it is **strongly recommended to pair this model with 3 Sigma Audio Upright Bass AST (Acoustic Sound Technology) IRs** in Block 3 (Cab IR loader) or your DAW host. Specifically, select impulses from the **`Acoustic Upright Standard`** folder (identified by the **`AST`** file tag), which 3 Sigma Audio designed specifically for acoustic upright basses with a standard piezo pickup.
* **32" Fretless Setting:** ABCX Blend to **Upright Blend** (85% PCSX + 15% MMTWX Single - Push/Pull Down: J Mode). Pair with **3 Sigma Upright Bass AST IR** in Anagram Block 3.

### 15. `studio_direct` (Studio Direct - Dynamic Studio DI)
* **Archetype:** Physical Aperture Preservation / Dynamic Studio DI
* **Design Rationale:** For bass players who want to preserve their instrument's exact physical pickup aperture and add organic Alnico V dynamic feel.
* **Signal Flow & Anagram Routing:**
  Loaded into **Block 1 (NAM Preamp)**. Engages authentic Alnico V compliance ($V_{\text{sat}} = 0.50$), Dahl hysteresis ($\eta = 0.06$), Lenz velocity drag and attack pitch sag ($k_{\text{sag}} = 0.08, k_{\text{pull}} = 0.04$), dynamic eddy current de-Qing ($k_{\text{eddy}} = 0.16$), and back-EMF string braking ($k_{\text{emf}} = 0.04$).
* **Pickup & Aperture Model:** Preserves the instrument's natural physical pickup placement and aperture ($H_{\text{prefilter}}(f) \equiv 1.0$).

### 15b. `studio_active` (Studio Active - Modern Active Buffer)
* **Archetype:** Studio Pure High-Impedance Active Buffer Twin (Cable Isolation & Wideband Sparkle)
* **Design Rationale:** For passive bass players who want the crystalline sparkle, ultra-fast transient punch, and wideband extension of an active bass without synthetic shelving boosts. Can also be used on active basses to provide extra treble air.
* **Signal Flow & Anagram Routing:**
  Loaded into **Block 1 (NAM Preamp)**. Preserves physical pickup aperture ($H_{\text{prefilter}}(f) \equiv 1.0$) and applies the active buffer circuit ($L=3.2\text{ H}$, active buffer, zero cable capacitance) with wideband active headroom.
* **Electrical Physics:** Simulates an onboard $1\text{ M}\Omega$ active buffer stage ($R_{\text{in}} = 1.0\text{ M}\Omega, C_{\text{in}} = 25\text{ pF}, R_{\text{out}} = 100\,\Omega$). Deconvolves heavy $750\text{ pF}$ instrument cable loading and $250\text{k}\Omega$ potentiometer damping, raising effective resonance into the air band ($5.2\text{ kHz}$) and restoring crystalline pick articulation.
* **Dynamic Headroom:** Clean active headroom ($V_{\text{sat}} = 1.20$), fast uncompressed attack.

### 15c. `studio_passive` (Studio Passive - High-Impedance Loading)
* **Archetype:** High-Impedance Passive Pickup & Cable Loading Digital Twin
* **Design Rationale:** Adds authentic high-impedance passive impedance dynamics (RLC resonant peak ~2.8 kHz, $750\text{ pF}$ cable capacitance loading, and $250\text{k}\Omega$ pot damping) to active basses. Can also be used on passive basses to stack vintage warmth and rolled-off highs.
* **Signal Flow & Anagram Routing:**
  Loaded into **Block 1 (NAM Preamp)**. Preserves physical pickup aperture ($H_{\text{prefilter}}(f) \equiv 1.0$) and applies the passive RLC harness ($L=4.2\text{ H}, R_{\text{dc}}=8.5\text{ k}\Omega, C_{\text{cable}}=750\text{ pF}, R_{\text{vol}}=250\text{ k}\Omega$) with passive dynamic loading.
* **Electrical Physics:** Simulates classic passive pickup loading: introduces a warm mid-resonance peak at $2.8\text{ kHz}$ ($Q \approx 1.4$) followed by authentic high-frequency cable roll-off ($-15\text{ dB}$ at $10\text{ kHz}$).
* **Dynamic Character:** Alnico V vintage compliance ($V_{\text{sat}} = 0.50$), magnetic hysteresis, and pick bloom.

---

## Geometry & Scale-Normalized Displacement Mapping (30" EMG MM $\to$ Target Datums)

When running Passivizer from a **30" short-scale bass with a single 18V EMG MM pickup**, the software compensates for spatial aperture, standing-wave bridge proximity, and physical scale differences using **scale-normalized fractional coordinates** ($\eta = x / L$) rather than raw millimeter subtractions (adhering strictly to Architectural Guardrail §1.5):

### Physical Datums:
* **Source Instrument (30" Short Scale Bass):**
  * Vibrating Scale Length: $L_{\text{src}} = 30.0'' = 762.0\text{ mm}$
  * 12th Fret Datum: $381.0\text{ mm}$ from nut
  * Pickup Center Datum: $303.5\text{ mm}$ from 12th fret toward bridge
  * **Pickup Center from Bridge Saddle ($x_{\text{src}}$):** $381.0 - 303.5 = \mathbf{77.5\text{ mm}}\ (3.051'')$
  * **Source Fractional Position ($\eta_{\text{src}}$):** $\eta_{\text{src}} = \frac{77.5\text{ mm}}{762.0\text{ mm}} \approx \mathbf{0.1017}\ (10.17\%)$
  * Pickup Architecture: Fixed Dual-Coil Humbucker ($w = 1.50''$, $d = 0.75''$)

* **Target Reference Instruments:**
  * **Standard 34" Scale ($L = 863.6\text{ mm}$):** Standard Fender ($125.0\text{ mm}$ P, $155.6\text{ mm}$ 60s J neck, $63.5\text{ mm}$ 60s J bridge), Music Man ($66.0\text{ mm}$ StingRay), and Rickenbacker ($50.8\text{ mm}$ 4003 bridge).
  * **37" Multi-Scale ($L = 939.8\text{ mm}$):** Dingwall NG angled bridge sweet spot ($48.0\text{ mm}$).
  * **41.5" Upright ($L = 1054.1\text{ mm}$):** 3/4 Double Bass acoustic bridge transducer ($5.0\text{ mm}$).

### Scale-Normalized Displacement & Proximity Tilt Formulation:
$$\eta_{\text{tgt}} = \frac{x_{\text{tgt}}}{L_{\text{tgt}}}, \quad \Delta\eta = \eta_{\text{tgt}} - \eta_{\text{src}}$$
$$\Delta x_{\text{norm\_in}} = \Delta\eta \times 34.0'', \quad \text{tilt}_{\text{dB}} = \Delta x_{\text{norm\_in}} \times 1.5\text{ dB/in}$$

| Profile ID | Target Location | Target $x$ | Target Scale | Fractional $\eta_{\text{tgt}}$ | Norm Offset $\Delta x_{\text{norm\_in}}$ | Proximity Tilt & Acoustic Compensation |
| :--- | :--- | :--- | :--- | :---: | :---: | :--- |
| **`jazz_pair_active`** | 60s J-Pair Center | $109.6\text{ mm}$ | $34.0''$ | $12.69\%$ | $+0.86''$ | $+1.3\text{ dB}$ forward tilt, de-humbuck, active 2-band boost |
| **`jazz_pair_open`** | 60s J-Pair Center | $109.6\text{ mm}$ | $34.0''$ | $12.69\%$ | $+0.86''$ | $+1.3\text{ dB}$ forward tilt, dual-volume $125\text{k}\Omega$ loading |
| **`jazz_pair_mids`** | 60s J-Pair Center | $109.6\text{ mm}$ | $34.0''$ | $12.69\%$ | $+0.86''$ | $+1.3\text{ dB}$ forward tilt, 22nF ToneStyler undamped peak |
| **`jazz_bridge_growl`** | Bridge-Biased J-Pair | $86.5\text{ mm}$ | $34.0''$ | $10.02\%$ | $-0.05''$ | $-0.1\text{ dB}$ neutral tilt, 75% bridge / 25% neck Jaco growl |
| **`jazz_bridge_open`** | 60s J Bridge Single | $63.5\text{ mm}$ | $34.0''$ | $7.35\%$ | $-0.96''$ | $-1.4\text{ dB}$ bridge bite tilt, narrow single-coil aperture |
| **`precision_active`** | Split-P Centerline | $125.0\text{ mm}$ | $34.0''$ | $14.47\%$ | $+1.46''$ | $+2.2\text{ dB}$ forward tilt, active 2-band preamp buffer |
| **`precision_vintage`** | Split-P Centerline | $125.0\text{ mm}$ | $34.0''$ | $14.47\%$ | $+1.46''$ | $+2.2\text{ dB}$ forward tilt, vintage 250k Alnico V bloom |
| **`precision_mids`** | Split-P Centerline | $125.0\text{ mm}$ | $34.0''$ | $14.47\%$ | $+1.46''$ | $+2.2\text{ dB}$ forward tilt, 22nF ToneStyler 440 Hz punch |
| **`precision_warm`** | Split-P Centerline | $125.0\text{ mm}$ | $34.0''$ | $14.47\%$ | $+1.46''$ | $+2.2\text{ dB}$ forward tilt, 47nF Motown flatwound thump |
| **`precision_dub`** | Split-P Centerline | $125.0\text{ mm}$ | $34.0''$ | $14.47\%$ | $+1.46''$ | $+2.2\text{ dB}$ forward tilt, 100nF deep reggae sub-thump |
| **`pj_active`** | P/J Parallel Center | $94.3\text{ mm}$ | $34.0''$ | $10.91\%$ | $+0.25''$ | $+0.4\text{ dB}$ forward tilt, active 2-band boost, shimmer |
| **`pj_passive`** | P/J Parallel Center | $94.3\text{ mm}$ | $34.0''$ | $10.91\%$ | $+0.25''$ | $+0.4\text{ dB}$ forward tilt, vintage dual-volume warmth |
| **`stingray_parallel`** | StingRay Centerline| $66.0\text{ mm}$ | $34.0''$ | $7.64\%$ | $-0.86''$ | $-1.3\text{ dB}$ bridge bite, active 2-band boost, comb notch |
| **`stingray_series`** | StingRay Centerline| $66.0\text{ mm}$ | $34.0''$ | $7.64\%$ | $-0.86''$ | $-1.3\text{ dB}$ bridge bite, $+4.5\text{ dB}$ series inductive surge |
| **`rickenbacker_clank`** | 4003 Bridge Coil | $50.8\text{ mm}$ | $34.0''$ | $5.88\%$ | $-1.46''$ | $-2.2\text{ dB}$ bite tilt, 4.7nF series HPF clank bite |
| **`p_mm_parallel`** | P + MM Parallel Center| $95.5\text{ mm}$ | $34.0''$ | $11.06\%$ | $+0.30''$ | $+0.5\text{ dB}$ forward tilt, active buffer cable isolation, slap punch |
| **`p_mm_series`** | P + MM Series Center| $95.5\text{ mm}$ | $34.0''$ | $11.06\%$ | $+0.30''$ | $+0.5\text{ dB}$ forward tilt, active series surge and buffer clarity |
| **`mudbucker_deep`** | Sidewinder Center | $95.5\text{ mm}$ | $34.0''$ | $11.06\%$ | $+0.30''$ | Extreme $14.4\text{ H}$ series foundation, $1.25''$ aperture |
| **`dingwall_bridge`** | Angled Sweet Spot | $48.0\text{ mm}$ | $37.0''$ | $5.11\%$ | $-1.72''$ | $-2.6\text{ dB}$ bite tilt, multiscale continuum clank |
| **`studio_direct`** | Preserved Datum | Source $x$ | Source $L$ | Source $\eta$ | $0.00''$ | Bit-exact $0.00\text{ dB}$ linear transfer, preserves dynamic feel |
| **`studio_active`** | Preserved Datum | Source $x$ | Source $L$ | Source $\eta$ | $0.00''$ | Preserved aperture, active buffer cable deconvolution |
| **`studio_passive`** | Preserved Datum | Source $x$ | Source $L$ | Source $\eta$ | $0.00''$ | Preserved aperture, passive RLC cable loading ($750\text{ pF}$) |



