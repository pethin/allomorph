# Allomorph Storefront Voicing Ranking Architecture

This document defines the ranking and ordering system governing the digital twin voicings in **Allomorph Tone Packs** on [Tone3000](https://tone3000.com).

---

## 1. Core Principles

The Allomorph ranking architecture is built on four fundamental tenets designed to maximize user engagement, storefront clarity, and musical value:

### 1.1 Inverse-Availability Principle (Native Identity Exclusion)
When a player browses or loads a tone pack for their physical bass, the primary value proposition is **transformational capacity**—giving the player sounds they **cannot** physically achieve with their stock instrument.
- **Rule:** 1:1 base-instrument identity duplicates ($H_{\text{diff}} \equiv 1.0$) where the target pickup geometry and circuit match the player's stock instrument (producing a straight-wire bypass with zero difference between dry and wet) are **excluded** from the pack. Players do not need a neural network to emulate the exact physical pickup and circuit already installed in their guitar.
- **Example (Precision Bass):** Excludes `Precision Vintage` (since the player's bass already possesses this exact Alnico V split-coil CTS 250k circuit). Retains modified ToneStyler capacitive shunts (Precision Mids, Precision Warm, Precision Dub) and modern ceramic 500k configurations which deliver distinct electrical curves.
- **Example (Jazz Bass):** Excludes `Jazz Pair Open` and `Jazz Bridge Open` (matches stock dual single-coils and bridge single-coil).
- **Example (P/J Bass):** Excludes `PJ Passive`, `Precision Vintage`, and `Jazz Bridge Open` (matches the player's native 3-way switch positions).
- **Example (Active StingRay):** Excludes `StingRay Parallel` (matches the stock active bridge humbucker in parallel).

### 1.2 Popularity & Transformative Demand Weighting
Among the non-native/transformative voicings, tones are ranked according to **real-world musical demand and recording ubiquity**:
- **Tier 1 (Universal Workhorses):** Jazz Bass dual-single scooped slap and active Sadowsky NYC 2-band; Music Man StingRay active parallel bite and series punch.
- **Tier 2 (Modern Boutique & Rock Legends):** Lakland/Sandberg P/MM active hybrids, Rickenbacker 4003 prog-rock clank with HPF, Dingwall multi-scale modern metal clank.
- **Tier 3 (Specialty & Historical Archetypes):** Gibson EB-0 Mudbucker ultra-low series humbucker.
- **Tier 4 (Acoustic Transducers):** Upright Acoustic Double Bass bridge force sensor simulation.

### 1.3 Strict Modular Topology Separation
Storefront categories must accurately reflect analog electrical topologies and physical coil arrangements. Arbitrary "catch-all" groupings are strictly forbidden:
- **Music Man StingRay Family:** Pure 2-band active dual-coil humbuckers placed at the bridge sweet spot are strictly isolated from hybrid models.
- **P/MM Modern Hybrids:** Compound dual-pickup configurations (Split-P neck + Music Man bridge) form their own dedicated hybrid category.
- **Progressive & Classic Rock Legends:** Historic and modern rock clank archetypes (Rickenbacker 4003, Dingwall, Gibson Mudbucker) are unified under a dedicated rock category.
- **Acoustic Transducers:** Piezo force sensor simulations are cleanly isolated from magnetic pickup guitars.

### 1.4 Terminal Studio DI & Buffer Anchor
The final positions in every pack are reserved for studio impedance and loading transformations:
- **Studio Active (Modern Studio Active Buffer):** Deconvolves $750\text{ pF}$ cable capacitance and pot damping from the user's physical pickups, shifting resonance into the hi-fi air band ($7.5\text{--}9.0\text{ kHz}$) with $1\text{M}\Omega$ input impedance.
- **Studio Passive (High-Impedance Passive Pickup & Cable Loading):** Simulates an authentic high-impedance passive RLC network ($L = 4.2\text{ H}, f_r = 2.8\text{ kHz}, Q = 1.4$) loaded by a standard $250\text{k}\Omega$ CTS volume/tone harness and $750\text{ pF}$ cable capacitance. On active instruments (StingRay, Active Soapbar), it replaces sterile onboard active buffering with organic vintage passive dynamics, woody low-mid body, and rolled-off highs.
- **Ordering Rule (Inverse-Availability):** For passive instrument editions (`standard_precision_bass`, `standard_jazz_bass`, `standard_pj_bass`, `mustang_pj_bass`), Studio Active is ranked before Studio Passive because passive players seeking an active buffer prioritize the active transformation. Conversely, for active instrument editions (`active_emg_bass`, `active_stingray_bass`, `preamp_soapbar_bass`), Studio Passive is ranked before Studio Active, as transforming an active instrument into an organic high-impedance passive RLC network provides the primary transformative value of the product ("Passivizer").

---

## 2. Edition-Specific Ranking Hierarchies

### 2.1 Standard P/J Bass Edition (19 Voicings)
*Source Instrument:* Physical Split-P neck + Single-coil J bridge ($34''$ scale).  
*Native Exclusions:* `PJ Passive`, `Precision Vintage`, and `Jazz Bridge Open` ($H_{\text{diff}} \equiv 1.0$).

| Rank Range | Sonic Family | Rationale / Musical Character |
| :--- | :--- | :--- |
| **01–04** | **Jazz Bass Family** | #1 most requested non-native sound: Modern Jazz Active, 60s Jazz Pair, 60s Jazz 22nF, Jaco Bridge Growl. |
| **05–06** | **Music Man StingRay Family** | Active humbucker punch, hollow $2.5\text{ kHz}$ mid-scoop, and series bark. |
| **07–08** | **P/MM Modern Hybrids** | Modern boutique session tone (Sandberg California / Lakland 44-02) combining Split-P weight with Music Man bridge authority. |
| **09–11** | **Progressive & Classic Rock Legends** | Aggressive pick clank and sub-bass weight: Rickenbacker 4003 ($4.7\text{nF}$ HPF), Dingwall FD3 ($7.3\text{ kHz}$ peak), and Gibson EB-0 Mudbucker ($14.4\text{ H}$). |
| **12** | **Acoustic Transducers** | Bridge piezo force transducer on $41.5''$ upright scale with velocity-to-force leaky integration. |
| **13–16** | **Precision Bass & Tone Shaper Family** | Modern Ceramic Split-P 500k, and ToneStyler shunts ($22\text{nF}$, $47\text{nF}$ flatwound, $100\text{nF}$ Fullerton). |
| **17** | **P/J Bass Family** | Modern Active P/J Bass (+4 dB @ 40 Hz & 4 kHz). |
| **18–19** | **Studio Buffers & Dynamics** | Modern Studio Active Buffer ($1\text{M}\Omega$ zero-loading) and Studio Passive ($4.2\text{ H}$ RLC loading). |

---

### 2.2 Mustang P/J Bass Edition (22 Voicings)
*Source Instrument:* Short-scale ($30''$) physical Split-P neck + Single-coil J bridge.  
*Native Exclusions:* None (all $34''$ targets require scale-length tension and spatial deconvolution from the $30''$ body).

| Rank Range | Sonic Family | Rationale / Musical Character |
| :--- | :--- | :--- |
| **01–05** | **Jazz Bass Family** | Reconstructs full $34''$ wide-aperture dual-single comb scoop and Sadowsky NYC 2-band slap from short scale. |
| **06–07** | **Music Man StingRay Family** | High-demand active humbucker punch, hollow $2.5\text{ kHz}$ mid-scoop, and series bark. |
| **08–09** | **P/MM Modern Hybrids** | Split-P + Music Man bridge dual-coil in parallel and series for modern boutique punch. |
| **10–12** | **Progressive & Classic Rock Legends** | Rickenbacker 4003 ($4.7\text{nF}$ HPF), Dingwall multi-scale ($7.3\text{ kHz}$ peak), and Gibson Mudbucker ($14.4\text{ H}$). |
| **13** | **Acoustic Transducers** | Bridge piezo force transducer on $41.5''$ upright scale with velocity-to-force leaky integration. |
| **14–18** | **Precision Bass & Tone Shaper Family** | Full $34''$ scale Vintage '62 CTS, Modern Ceramic 500k, and ToneStyler shunts ($22\text{nF}$, $47\text{nF}$, $100\text{nF}$). |
| **19–20** | **P/J Bass Family** | Modern Active and Vintage '80s Passive P/J in standard $34''$ scale tension. |
| **21–22** | **Studio Buffers & Dynamics** | Modern Studio Active Buffer ($1\text{M}\Omega$ zero-loading) and Studio Passive ($4.2\text{ H}$ RLC loading). |

---

### 2.3 Standard Jazz Bass Edition (20 Voicings)
*Source Instrument:* Dual single-coils in 60s spacing ($155.6\text{ mm}$ neck, $63.5\text{ mm}$ bridge, $34''$ scale).  
*Native Exclusions:* `Jazz Pair Open` and `Jazz Bridge Open` ($H_{\text{diff}} \equiv 1.0$).

| Rank Range | Sonic Family | Rationale / Musical Character |
| :--- | :--- | :--- |
| **01–05** | **Precision Bass & Tone Shaper Family** | #1 most requested transformation for Jazz Bass players: Vintage '62 Alnico V, Modern Ceramic 500k, and ToneStyler shunts ($22\text{nF}$, $47\text{nF}$, $100\text{nF}$). |
| **06–07** | **Music Man StingRay Family** | Active 2-band parallel and series bridge humbucker punch, $2.5\text{ kHz}$ scoop, and signature metallic slap clank. |
| **08–09** | **P/J Bass Family** | Modern active and vintage '80s passive P/J configurations combining P-bass body with bridge single-coil cut. |
| **10–11** | **P/MM Modern Hybrids** | Split P + Music Man bridge dual-coil in parallel and series for modern boutique versatility. |
| **12–14** | **Progressive & Classic Rock Legends** | Rickenbacker 4003 ($4.7\text{nF}$ HPF pick clank), Dingwall multi-scale ($7.3\text{ kHz}$ peak), and Gibson Mudbucker ($14.4\text{ H}$). |
| **15** | **Acoustic Transducers** | Bridge piezo force sensor on $41.5''$ upright scale with velocity-to-force leaky integration. |
| **16–18** | **Jazz Bass Family** | Re-EQed NYC active 2-band, ToneStyler $22\text{nF}$ vocal mid bump, and Jaco decoupled growl. |
| **19–20** | **Studio Buffers & Dynamics** | Studio $1\text{M}\Omega$ active buffer and Studio Passive ($4.2\text{ H}$ RLC loading). |

---

### 2.4 Standard Precision Bass Edition (21 Voicings)
*Source Instrument:* Single split-coil P-Bass centered at $125.0\text{ mm}$ ($34''$ scale).  
*Native Exclusions:* `Precision Vintage` ($H_{\text{diff}} \equiv 1.0$).

| Rank Range | Sonic Family | Rationale / Musical Character |
| :--- | :--- | :--- |
| **01–05** | **Jazz Bass Family** | Top transformative capability: converts a single split-coil into dual-aperture 60s single-coils with wideband slap sparkle and Jaco bridge growl. |
| **06–07** | **Music Man StingRay Family** | Active 2-band parallel and series bridge humbucker clank and punch. |
| **08–09** | **P/J Bass Family** | Modern active and vintage '80s passive P/J combinations providing dual-pickup growl and articulation. |
| **10–11** | **P/MM Modern Hybrids** | Split P + Music Man bridge dual-coil in parallel and series for modern boutique versatility and punch. |
| **12–14** | **Progressive & Classic Rock Legends** | Rickenbacker 4003 ($4.7\text{nF}$ HPF), Dingwall multi-scale ($7.3\text{ kHz}$ peak), and Gibson Mudbucker ($14.4\text{ H}$). |
| **15** | **Acoustic Transducers** | Bridge piezo force sensor on $41.5''$ upright scale with velocity-to-force leaky integration. |
| **16–19** | **Precision Bass & Tone Shaper Family** | Modern Ceramic 500k and ToneStyler shunts ($22\text{nF}$ punch, $47\text{nF}$ flatwound Motown thump, $100\text{nF}$ Fullerton dub). |
| **20–21** | **Studio Buffers & Dynamics** | Studio $1\text{M}\Omega$ active buffer (cable deconvolution) and Studio Passive ($4.2\text{ H}$ RLC loading). |

---

### 2.5 Preamp Soapbar Bass Edition (22 Voicings)
*Source Instrument:* Modern 34" preamp bass with dual passive Alnico dual-coil soapbars ($135.0\text{ mm}$ neck, $55.0\text{ mm}$ bridge, exposed round pole pieces) and 3-band active EQ buffer. Universally calibrated for 4, 5, and 6-string instruments.  
*Native Exclusions:* None (all target models deconvolve cylindrical pole soapbar aperture and active buffering).

| Rank Range | Sonic Family | Rationale / Musical Character |
| :--- | :--- | :--- |
| **01–05** | **Precision Bass & Tone Shaper Family** | #1 most requested transformation for preamp soapbar players: warm, organic Alnico V split-coil body and ToneStyler flatwound thump. |
| **06–10** | **Jazz Bass Family** | Vintage 1960s dual single-coils ($1\text{ kHz}$ phase-cancellation scoop and Jaco decoupled bridge growl). |
| **11–12** | **Music Man StingRay Family** | Sweet-spot humbucker clank, $2.5\text{ kHz}$ hollow mid-scoop, and series bark. |
| **13–14** | **P/MM Modern Hybrids** | Sandberg California / Lakland 44-02 style Split-P + Music Man parallel and series authority. |
| **15–17** | **Progressive & Classic Rock Legends** | Rickenbacker 4003 ($4.7\text{nF}$ HPF), Dingwall multi-scale ($7.3\text{ kHz}$ sparkle), and Gibson Mudbucker ($14.4\text{ H}$). |
| **18** | **Acoustic Transducers** | Bridge piezo force sensor on $41.5''$ upright scale with velocity-to-force leaky integration. |
| **19–20** | **P/J Bass Family** | Modern active and vintage '80s passive P/J configurations. |
| **21–22** | **Studio Buffers & Dynamics** | Studio Passive ($4.2\text{ H}$ RLC passive network replacing sterile active highs) and Studio $1\text{M}\Omega$ active buffer. |

---

### 2.6 Active StingRay Bass Edition (21 Voicings)
*Source Instrument:* 34" active Music Man StingRay / Sterling Ray34 / Ray35 (4 and 5-string) with single bridge sweet-spot dual-coil humbucker ($66.0\text{ mm}$) and active 2-band or 3-band preamp. Universally calibrated down to low B ($30.87\text{ Hz}$).  
*Native Exclusions:* `StingRay Parallel` ($H_{\text{diff}} \equiv 1.0$).

| Rank Range | Sonic Family | Rationale / Musical Character |
| :--- | :--- | :--- |
| **01–05** | **Precision Bass & Tone Shaper Family** | #1 missing sound for StingRay players: warm low-mid thump of a neck split-coil ($125\text{ mm}$) and flatwound Motown thump. |
| **06–10** | **Jazz Bass Family** | Dual single-coil wide-aperture $1\text{ kHz}$ comb scoop, Marcus Miller slap sparkle, and Jaco bridge bite. |
| **11–12** | **P/MM Modern Hybrids** | Combines virtual P-neck with the player's physical StingRay bridge humbucker in parallel and series. |
| **13–14** | **P/J Bass Family** | Modern active and vintage '80s passive P/J configurations combining P-neck weight with bridge clarity. |
| **15–17** | **Progressive & Classic Rock Legends** | Rickenbacker 4003 ($4.7\text{nF}$ HPF), Dingwall multi-scale ($7.3\text{ kHz}$ peak), and Gibson Mudbucker ($14.4\text{ H}$). |
| **18** | **Acoustic Transducers** | Bridge piezo force sensor on $41.5''$ upright scale with velocity-to-force leaky integration. |
| **19** | **Music Man StingRay Family** | Active 2-Band Series bark (+2.5 dB gain, forward 2.1 kHz mids). |
| **20–21** | **Studio Buffers & Dynamics** | Studio Passive ($4.2\text{ H}$ RLC passive network restoring vintage passive dynamics) and Studio $1\text{M}\Omega$ active buffer. |

---

### 2.7 Active EMG Bass Edition (22 Voicings)
*Source Instrument:* 34" active EMG soapbar bass (Spector / ESP LTD / Schecter) with dual EMG ceramic dual-blade soapbars ($135.0\text{ mm}$ neck, $55.0\text{ mm}$ bridge) and internal low-noise buffers calibrated under Option C ($f_r = 4.15\text{ kHz}, Q = 1.40$). Universally calibrated for 4, 5, and 6-string instruments.  
*Native Exclusions:* None (all target models deconvolve active EMG dual-blade aperture and active buffer resonance).

| Rank Range | Sonic Family | Rationale / Musical Character |
| :--- | :--- | :--- |
| **01–05** | **Precision Bass & Tone Shaper Family** | #1 most requested transformation for active EMG players: warm, organic Alnico V split-coil body and ToneStyler flatwound thump. |
| **06–10** | **Jazz Bass Family** | Vintage 1960s dual single-coils ($1\text{ kHz}$ phase-cancellation scoop and Jaco decoupled bridge growl). |
| **11–12** | **Music Man StingRay Family** | Sweet-spot humbucker clank, $2.5\text{ kHz}$ hollow mid-scoop, and series bark. |
| **13–14** | **P/MM Modern Hybrids** | Sandberg California / Lakland 44-02 style Split-P + Music Man parallel and series authority. |
| **15–17** | **Progressive & Classic Rock Legends** | Rickenbacker 4003 ($4.7\text{nF}$ HPF), Dingwall multi-scale ($7.3\text{ kHz}$ sparkle), and Gibson Mudbucker ($14.4\text{ H}$). |
| **18** | **Acoustic Transducers** | Bridge piezo force sensor on $41.5''$ upright scale with velocity-to-force leaky integration. |
| **19–20** | **P/J Bass Family** | Modern active and vintage '80s passive P/J configurations. |
| **21–22** | **Studio Buffers & Dynamics** | Studio Passive ($4.2\text{ H}$ RLC passive network replacing active EMG highs) and Studio $1\text{M}\Omega$ active buffer. |

---

## 3. Governance & Quality Checklist

When modifying or generating new Tone3000 pack listings:
1. **Verify Native Identity Exclusion:** Confirm that $1:1$ identical base-instrument configurations ($H_{\text{diff}} \equiv 1.0$) are omitted.
2. **Preserve Distinct Families:** Confirm StingRay is never merged with P/J, and P/MM is never merged with Acoustic Transducers.
3. **Check Character Limits:** Ensure the total listing length strictly remains under 10,000 characters.
4. **Preserve Bracketed Tags:** Every multi-pickup listing must contain `[Parallel]`, `[Neck]`, or `[Bridge]` tags.
5. **Active Calibration Integrity:** Active editions must direct players to set onboard active EQs flat at center detents.
