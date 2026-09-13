# Tone3000 Tone Pack Catalog

This directory contains the standardized storefront product descriptions for **Allomorph Tone Packs** on [Tone3000](https://tone3000.com), engineered for Neural Amp Modeler (NAM) and the Darkglass Anagram pedalboard.

Each pack contains up to 22 precision digital twin voicings calibrated via true differential circuit deconvolution ($H_{\text{diff}} = H_{\text{tgt}} / H_{\text{src}}$) to transform a specific physical bass guitar into iconic vintage, modern active, heavy multi-scale, and acoustic bass topologies.

---

## Pack Catalog & Storefront Listings

| Pack Edition | Source Instrument Calibration | Recommended Knobs / Switches | Profile Document | Production Artwork (SVG / JPG) |
| :--- | :--- | :--- | :--- | :--- |
| **Standard Precision Bass Edition** | 34" Standard Fender P-Bass (Passive Split-P, 21 Voicings) | Vol 100%, Tone 100% | [`standard_precision_bass.txt`](standard_precision_bass.txt) | [`allomorph_standard_precision_bass.svg`](../assets/allomorph_standard_precision_bass.svg) &bull; [JPG](../assets/allomorph_standard_precision_bass.jpg) |
| **Standard Jazz Bass Edition** | 34" Standard Fender Jazz Bass (Passive Single-Coil Pair, 20 Voicings) | Vol 100%, Tone 100%, Pickup configs: `[Parallel]`, `[Neck]`, `[Bridge]` | [`standard_jazz_bass.txt`](standard_jazz_bass.txt) | [`allomorph_standard_jazz_bass.svg`](../assets/allomorph_standard_jazz_bass.svg) &bull; [JPG](../assets/allomorph_standard_jazz_bass.jpg) |
| **Standard P/J Bass Edition** | 34" Standard Fender P/J Bass (Split-P Neck + J Bridge, 19 Voicings) | Vol 100%, Tone 100%, Pickup configs: `[Parallel]`, `[Neck]`, `[Bridge]` | [`standard_pj_bass.txt`](standard_pj_bass.txt) | [`allomorph_standard_pj_bass.svg`](../assets/allomorph_standard_pj_bass.svg) &bull; [JPG](../assets/allomorph_standard_pj_bass.jpg) |
| **Mustang P/J Bass Edition** | 30" Short-Scale Fender Mustang P/J Bass (22 Voicings) | Vol 100%, Tone 100%, Pickup configs: `[Parallel]`, `[Neck]`, `[Bridge]` | [`mustang_pj_bass.txt`](mustang_pj_bass.txt) | [`allomorph_mustang_pj_bass.svg`](../assets/allomorph_mustang_pj_bass.svg) &bull; [JPG](../assets/allomorph_mustang_pj_bass.jpg) |
| **Preamp Soapbar Bass Edition** | 34" Preamp Dual-Soapbar Bass (Passive Alnico Coils + Active EQ, 22 Voicings) | Active EQ Flat (Center Detents), Blend configs: `[Parallel]`, `[Neck]`, `[Bridge]` | [`preamp_soapbar_bass.txt`](preamp_soapbar_bass.txt) | [`allomorph_preamp_soapbar_bass.svg`](../assets/allomorph_preamp_soapbar_bass.svg) &bull; [JPG](../assets/allomorph_preamp_soapbar_bass.jpg) |
| **Active StingRay Bass Edition** | 34" Active Music Man StingRay / Sterling Ray34/Ray35 (4/5-String, 21 Voicings) | Active EQ Flat (Center Detents), Switch in Parallel (if 3-way) | [`active_stingray_bass.txt`](active_stingray_bass.txt) | [`allomorph_active_stingray_bass.svg`](../assets/allomorph_active_stingray_bass.svg) &bull; [JPG](../assets/allomorph_active_stingray_bass.jpg) |
| **Active EMG Bass Edition** | 34" Active EMG Soapbar Bass (Option C Baseline, 22 Voicings) | Active EQ Flat (Center Detents), Blend configs: `[Parallel]`, `[Neck]`, `[Bridge]` | [`active_emg_bass.txt`](active_emg_bass.txt) | [`allomorph_active_emg_bass.svg`](../assets/allomorph_active_emg_bass.svg) &bull; [JPG](../assets/allomorph_active_emg_bass.jpg) |

---

## Storefront Engineering Constraints

All pack descriptions in this directory strictly adhere to Tone3000 platform guidelines:
1. **Character Limit Compliance:** Tone3000 enforces a strict maximum length of 10,000 characters per listing. All 7 packs stay safely under this ceiling.
2. **Pickup Configuration Tags:** Multi-pickup editions (Jazz, P/J, Mustang P/J, Active Soapbar) include bracketed physical selector tags on every voicing (`[Parallel]`, `[Neck]`, `[Bridge]`) so users know exactly how to set their instrument switches for optimal acoustic cancellation matching.
3. **Concise Voicing Summaries:** Each target voicing is distilled into 1–2 punchy, informative sentences highlighting real-world musical feel, mix behavior, and sonic character.
4. **Independent Legal Disclaimers:** Trademark disclaimers and non-commercial/commercial user rights are explicitly codified per listing.
5. **Native Identity Exclusion:** Base-instrument 1:1 identical duplicates ($H_{\text{diff}} \equiv 1.0$) are omitted, ensuring every model in the pack provides transformational capacity (see [`ranking_system.md`](ranking_system.md)).
6. **Tone Naming & Display Character Budgets:** Every model filename strictly adheres to the 64-character Tone3000 upload limit and 34-character Darkglass Anagram pedalboard screen target (see master guide [`../../docs/tone_naming_guidelines.md`](../../docs/tone_naming_guidelines.md)).

---

## Storefront Voice & Bassist-Centric Copywriting Guidelines

When writing or updating Tone3000 storefront descriptions, always translate complex analog modeling math into musical language that working bassists immediately understand and connect with:

### 1. The Cardinal Rule
**Never describe the mathematics when you can describe the musical tone, feel, and mix placement.**
Bassists browse Tone3000 to find sounds that inspire them, solve gigging problems, or give them iconic tones they cannot physically get with their stock instrument. Focus on:
- **Iconic Players & Eras:** James Jamerson, Pino Palladino, Jaco Pastorius, Marcus Miller, Chris Squire, Geddy Lee, NYC session slap, '60s Motown, '80s hard rock.
- **Mix Placement & EQ Feel:** Vocal midrange growl, hollow slap scoop, pillowy sub-bass thump, cutting pick clank, woody low-mids, glassy hi-fi air.
- **Playing Styles:** Slap & pop, 16th-note staccato fingerstyle, drop-tuned metal, reggae/dub, acoustic jazz gigs.

### 2. Jargon Translation Guide

| Avoid Engineering Mumbo-Jumbo | Use Bassist-Friendly Language |
| :--- | :--- |
| *True differential circuit deconvolution ($H_{\text{diff}} = H_{\text{tgt}} / H_{\text{src}}$)* | Transforms your instrument's sound right at the pickups in real time |
| *Aperture geometry & spatial string comb-filtering* | Pickup placement, sweet spots, and harmonic phase interactions |
| *Bessel $J_1(x)/x$ cylindrical pole transfer* | Round pole-piece punch and dynamic transient attack |
| *High-impedance coil eddy currents & skin-effect damping* | Warm, woody low-mid body and smooth vintage top-end bloom |
| *Decoupled dual volumes with 55k neck wiper resistance* | Rolling the neck volume back just a touch for singing Jaco bridge growl |
| *22nF pure capacitive shunt with undamped vocal peak* | Vocal midrange punch that cuts finger clatter and keeps notes articulate |
| *47nF paper-in-oil cap with flatwound damping* | Classic Jamerson/Pino Motown flatwound thump with fat, pillowy sub-bass |
| *0.1µF Fullerton factory spec with sub-bass shelf rolloff* | Ultra-deep vintage '50s dub spec for massive, earth-shaking low end |
| *Sadowsky-style 2-band buffer with cable isolation* | The quintessential NYC active slap tone: massive lows and crisp, glassy highs |
| *Series wiring with +5.8 dB inductive surge* | Hotter, muscular series punch with wall-shaking low-mid growl and sustain |
| *4.7nF series high-pass capacitor prog-rock clank* | Classic prog-rock clank (Chris Squire/Geddy Lee): tight lows with aggressive pick grit |
| *FD3 34"–37" multi-scale fanned fret active sparkle* | Modern progressive metal tone: razor-sharp pick attack and piano-like low-B clarity |
| *14.4H overwound neck-heel sidewinder Mudbucker* | The Gibson Mudbucker: dark, colossal vintage bass wall of sound |
| *Bridge piezo force sensor with leaky integration* | Acoustic upright double bass: woody resonance, body thump, and organic acoustic feel |
| *Studio 1MΩ active buffer (750 pF cable deconvolution)* | Studio Active Buffer: opens up wide-open hi-fi highs, fast transients, and airy attack |
| *4.2H RLC high-Z network replacing active buffering* | Vintage Passive Character: turns an active bass into a warm, organic vintage passive instrument |

---

## Digital Twin Voice Architecture & Ranking

Every pack provides comprehensive coverage across eight modular sonic families, ordered according to the [Tone Ranking Architecture](ranking_system.md):

1. **Jazz Bass Family:**
   - **Modern Jazz Active** (Modern Active Jazz Bass NYC 2-Band Preamp with Cable Isolation)
   - **60s Jazz Pair** (Vintage '60s Jazz Bass Pair, Dual Parallel 250k Harness with 1 kHz Scoop)
   - **60s Jazz 22nF** ('60s Jazz Bass Pair, 22nF ToneStyler Vocal Mid Bump @ 762 Hz)
   - **Jaco Bridge Growl** ('60s Jazz Bridge Growl, Jaco Bias 55k Wiper Decoupling)
   - **60s Jazz Bridge** ('60s Jazz Bridge Single-Coil, Articulate 63.5mm Datum Punch)
2. **Music Man StingRay Family:**
   - **StingRay Parallel** (Music Man StingRay Active 2-Band Parallel, Classic Hollow Mid-Scoop & Clank)
   - **StingRay Series** (Music Man StingRay Active 2-Band Series, Aggressive 2.1 kHz Bark & +2.5 dB EMF)
3. **P/MM Modern Hybrids:**
   - **Modern P/MM Active** (Modern Active P/MM Bass, Sandberg/Lakland Parallel Split + MM Sweet Spot)
   - **P/MM Hybrid Series** (P/MM Hybrid Series Sum, Massive 7.2H Inductive Surge & +5.8 dB Punch)
4. **Progressive & Classic Rock Legends:**
   - **Rickenbacker 4003** (Rickenbacker 4003 Bridge, Vintage 4.7nF Series HPF Clank)
   - **Dingwall FD3** (Dingwall Fanned Multi-Scale Bridge, 34"–37" High Wave-Speed Sparkle @ 7.3 kHz)
   - **EB-0 Mudbucker** (Gibson Mudbucker Ultra Series, 14.4H Neck-Heel Sidewinder Sub-Bass)
5. **Acoustic Transducers:**
   - **Upright Piezo** (Upright Acoustic Double Bass, Underwood/Realist Bridge Force Sensor with Leaky Integration — recommended with 3 Sigma Audio "Acoustic Upright Standard" AST IRs)
6. **Precision Bass & Tone Shaper Family:**
   - **Vintage 62 P** (Vintage '62 P-Bass Alnico V, CTS 250k Tone Open)
   - **Modern P Ceramic** (Modern Ceramic Split-P, Boutique 500k with Hybrid Treble Bleed)
   - **Vintage 62 P 22nF** (Vintage '62 Split-P, 22nF ToneStyler Punch @ 440 Hz)
   - **Vintage 62 P 47nF** (Vintage '62 Split-P Motown Thump, 47nF ToneStyler Flatwound Damping)
   - **Vintage 50s P 100nF** (Vintage '50s Split-P Deep Dub, 100nF ToneStyler Fullerton 0.1µF Spec)
7. **P/J Bass Family:**
   - **Modern PJ Active** (Modern Active P/J Bass, Active 2-Band Boost @ 40 Hz & 4 kHz)
   - **Vintage PJ Passive** (Vintage '80s Passive P/J Bass, Duff McKagan / BB3000 Parallel Sum)
8. **Studio Buffers & Dynamics:**
   - **Active Character** (Modern Studio Active Buffer, Zero Cable Loading / 1M Impedance Twin @ 7.5–9.0 kHz)
   - **Passive Character** (High-Impedance Passive Pickup & Cable Loading: 4.2H RLC, 750pF Cable, 250k Pots)

---

## Recommended Darkglass Anagram Signal Flow

```
[Physical Bass]
      │  (Volume 100%, Tone 100%)
      ▼
[Block 1: Allomorph NAM Preamp]  ◄── Tone3000 Pickup Twin
      │  (Impedance matching, coil deconvolution, aperture reshaping)
      ▼
[Block 2: Preamp / Overdrive]   ◄── Microtubes B7K, Vintage Ultra, SVT, etc.
      │  (Distorts authentic pickup resonance peaks)
      ▼
[Block 3: Cab IR Loader]        ◄── 8x10, 4x10, 1x15 Speaker Cab Impulse (or 3 Sigma AST IRs for Upright)
      │
      ▼
[FOH / Interface / DAW]
```
