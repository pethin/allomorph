# Tone3000 Tone Pack Catalog

This document details the standardized storefront product configurations for **Allomorph Tone Packs** on [Tone3000](https://tone3000.com), engineered for Neural Amp Modeler (NAM) and the Darkglass Anagram pedalboard.

Each pack contains 23–26 precision Forward Digital Twin voicings paired directly from the source instrument's simulated stem (`Source Digital Twin → NAM Block 1 → Target Digital Twin`) to transform a specific physical bass guitar into iconic vintage, modern active, heavy multi-scale, and acoustic bass topologies.

Tone pack metadata and bespoke storefront product copy are defined in `config/packs/*.toml`, while canonical license terms and trademark notices are defined in `config/legal.toml`.

---

## Pack Catalog & Storefront Listings

| Pack Edition | Source Instrument Calibration | Recommended Knobs / Switches | Pack Configuration | Production Artwork (SVG / JPG) |
| :--- | :--- | :--- | :--- | :--- |
| **Standard Precision Bass Edition** | 34" Standard Fender P-Bass (Passive Split-P, 25 Voicings) | Vol 100%, Tone 100% | [`34in_standard_p.toml`](../../config/packs/34in_standard_p.toml) | [`allomorph_standard_precision_bass.svg`](../../tone3000/assets/allomorph_standard_precision_bass.svg) &bull; [JPG](../../tone3000/assets/allomorph_standard_precision_bass.jpg) |
| **Standard Jazz Bass Edition** | 34" Standard Fender Jazz Bass (Passive Single-Coil Pair, 23 Voicings) | Vol 100%, Tone 100%, Pickup configs: `[Parallel]`, `[Neck]`, `[Bridge]` | [`34in_standard_jazz.toml`](../../config/packs/34in_standard_jazz.toml) | [`allomorph_standard_jazz_bass.svg`](../../tone3000/assets/allomorph_standard_jazz_bass.svg) &bull; [JPG](../../tone3000/assets/allomorph_standard_jazz_bass.jpg) |
| **Standard P/J Bass Edition** | 34" Standard Fender P/J Bass (Split-P Neck + J Bridge, 23 Voicings) | Vol 100%, Tone 100%, Pickup configs: `[Parallel]`, `[Neck]`, `[Bridge]` | [`34in_standard_pj.toml`](../../config/packs/34in_standard_pj.toml) | [`allomorph_standard_pj_bass.svg`](../../tone3000/assets/allomorph_standard_pj_bass.svg) &bull; [JPG](../../tone3000/assets/allomorph_standard_pj_bass.jpg) |
| **Mustang P/J Bass Edition** | 30" Short-Scale Fender Mustang P/J Bass (26 Voicings) | Vol 100%, Tone 100%, Pickup configs: `[Parallel]`, `[Neck]`, `[Bridge]` | [`30in_mustang_pj.toml`](../../config/packs/30in_mustang_pj.toml) | [`allomorph_mustang_pj_bass.svg`](../../tone3000/assets/allomorph_mustang_pj_bass.svg) &bull; [JPG](../../tone3000/assets/allomorph_mustang_pj_bass.jpg) |
| **Preamp Soapbar Bass Edition** | 34" Preamp Dual-Soapbar Bass (Passive Alnico Coils + Active EQ, 26 Voicings) | Active EQ Flat (Center Detents), Blend configs: `[Parallel]`, `[Neck]`, `[Bridge]` | [`34in_preamp_soapbar.toml`](../../config/packs/34in_preamp_soapbar.toml) | [`allomorph_preamp_soapbar_bass.svg`](../../tone3000/assets/allomorph_preamp_soapbar_bass.svg) &bull; [JPG](../../tone3000/assets/allomorph_preamp_soapbar_bass.jpg) |
| **Active StingRay Bass Edition** | 34" Active Music Man StingRay / Sterling Ray34/Ray35 (4/5-String, 25 Voicings) | Active EQ Flat (Center Detents), Switch in Parallel (if 3-way) | [`34in_active_stingray.toml`](../../config/packs/34in_active_stingray.toml) | [`allomorph_active_stingray_bass.svg`](../../tone3000/assets/allomorph_active_stingray_bass.svg) &bull; [JPG](../../tone3000/assets/allomorph_active_stingray_bass.jpg) |
| **Active EMG Bass Edition** | 34" Active EMG Soapbar Bass (Option C Baseline, 26 Voicings) | Active EQ Flat (Center Detents), Blend configs: `[Parallel]`, `[Neck]`, `[Bridge]` | [`34in_active_emg.toml`](../../config/packs/34in_active_emg.toml) | [`allomorph_active_emg_bass.svg`](../../tone3000/assets/allomorph_active_emg_bass.svg) &bull; [JPG](../../tone3000/assets/allomorph_active_emg_bass.jpg) |

---

## Storefront Engineering Constraints

All pack descriptions strictly adhere to Tone3000 platform guidelines:
1. **Character Limit Compliance:** Tone3000 enforces a strict maximum length of 10,000 characters per listing. All 7 packs stay safely under this ceiling.
2. **Pickup Configuration Tags:** Multi-pickup editions (Jazz, P/J, Mustang P/J, Active Soapbar) include bracketed physical selector tags on every voicing (`[Parallel]`, `[Neck]`, `[Bridge]`) so users know exactly how to set their instrument switches for optimal acoustic cancellation matching.
3. **Concise Voicing Summaries:** Each target voicing is distilled into 1–2 punchy, informative sentences highlighting real-world musical feel, mix behavior, and sonic character.
4. **Independent Legal Disclaimers:** Trademark disclaimers and non-commercial/commercial user rights are centrally codified in `config/legal.toml` and appended to each exported listing.
5. **Native Identity Exclusion:** Base-instrument 1:1 identical duplicates (where source and target voicings are identical) are omitted via `is_identity_voicing`, ensuring every model in the pack provides transformational capacity (see [`ranking.md`](ranking.md)).
6. **Tone Naming & Display Character Budgets:** Every model filename strictly adheres to the 64-character Tone3000 upload limit and 34-character Darkglass Anagram pedalboard screen target (see master guide [`naming.md`](naming.md)).

---

## Storefront Voice & Bassist-Centric Copywriting Guidelines

When writing or updating Tone3000 storefront descriptions in `config/packs/*.toml`, always translate complex analog modeling math into musical language that working bassists immediately understand and connect with:

### 1. The Cardinal Rule
**Never describe the mathematics when you can describe the musical tone, feel, and mix placement.**
Bassists browse Tone3000 to find sounds that inspire them, solve gigging problems, or give them iconic tones they cannot physically get with their stock instrument. Focus on:
- **Iconic Players & Eras:** James Jamerson, Pino Palladino, Jaco Pastorius, Marcus Miller, Chris Squire, Geddy Lee, NYC session slap, '60s Motown, '80s hard rock.
- **Mix Placement & EQ Feel:** Vocal midrange growl, hollow slap scoop, pillowy sub-bass thump, cutting pick clank, woody low-mids, glassy hi-fi air.
- **Playing Styles:** Slap & pop, 16th-note staccato fingerstyle, drop-tuned metal, reggae/dub, acoustic jazz gigs.

### 2. Jargon Translation Guide

| Avoid Engineering Mumbo-Jumbo | Use Bassist-Friendly Language |
| :--- | :--- |
| *Forward Digital Twin source-to-target neural mapping* | Transforms your instrument's sound right at the pickups in real time |
| *Aperture geometry & spatial string comb-filtering* | Pickup placement, sweet spots, and harmonic phase interactions |
| *Bessel $J_1(x)/x$ cylindrical pole transfer* | Round pole-piece punch and dynamic transient attack |
| *High-impedance coil eddy currents & skin-effect damping* | Warm, woody low-mid body and smooth vintage top-end bloom |
| *Decoupled dual volumes with 8.8k neck wiper resistance* | Rolling the neck volume back just a touch for singing Jaco bridge growl |
| *1951 single-coil Alnico V 2.4H slab pickup* | Raw, uncompressed '51 single-coil Precision growl and open top-end snap |
| *47nF paper-in-oil cap with flatwound damping* | Classic Jamerson/Pino Motown flatwound thump with fat, pillowy sub-bass |
| *0.1µF Fullerton factory spec with sub-bass shelf rolloff* | Ultra-deep vintage '50s dub spec for massive, earth-shaking low end |
| *Sadowsky-style 2-band buffer with cable isolation* | The quintessential NYC active slap tone: massive lows and crisp, glassy highs |
| *Series wiring with +6.0 dB inductive surge* | Hotter, muscular series punch with wall-shaking low-mid growl and sustain |
| *4.7nF series high-pass capacitor prog-rock clank* | Classic prog-rock clank (Chris Squire/Geddy Lee): tight lows with aggressive pick grit |
| *FD3 34"–37" multi-scale fanned fret active sparkle* | Modern progressive metal tone: razor-sharp pick attack and piano-like low-B clarity |
| *18.0H coupled overwound neck-heel sidewinder Mudbucker* | The Gibson Mudbucker: dark, colossal vintage bass wall of sound |
| *2.5H shunt inductor + 22nF series cap Baritone filter* | Gibson EB-2/EB-3 Baritone choke filter: cuts sub-mud for snarling 60s rock bark |
| *Bridge piezo force sensor with leaky integration* | Acoustic upright double bass: woody resonance, body thump, and organic acoustic feel |

---

## Digital Twin Voice Architecture & Ranking

Every pack provides comprehensive coverage across seven modular sonic families (up to 26 target voicings), ordered according to the [Tone Ranking Architecture](ranking.md):

1. **Jazz Bass Family:**
   - **Jazz Pair Active** (Modern Active Jazz Bass NYC 2-Band Preamp with Cable Isolation)
   - **Jazz Pair Open** (Vintage '60s Jazz Bass Pair, Dual Parallel 250k Harness with 640 Hz Scoop)
   - **Jazz Bridge Growl** ('60s Jazz Bridge Growl, Jaco Bias 8.8k Wiper Decoupling)
   - **Jazz Bridge Open** ('60s Jazz Bridge Single-Coil, Articulate 63.5mm Datum Punch)
   - **Jazz Bridge 70s** ('70s Jazz Bridge Single-Coil, Closer 53.3mm Datum Bite — Geddy Lee / Marcus Miller)
   - **Jazz Neck Warm** ('60s Jazz Neck Single-Coil, Deep Woody Warmth & Round Low-End Body)
2. **Music Man StingRay Family:**
   - **StingRay Parallel** (Music Man StingRay Active 2-Band Parallel, Classic Hollow Mid-Scoop & Clank)
   - **StingRay Series** (Music Man StingRay Active 2-Band Series, Aggressive 2.19 kHz Bark & +6.0 dB EMF)
3. **P/MM Modern Hybrids:**
   - **P∕MM Parallel** (Modern Boutique Session Hybrid, Lakland LH-3 Active Preamp & Bridge MM Parallel Coils)
   - **P∕MM Series** (Modern Active P/MM Hybrid, Lakland LH-3 Active Preamp & Bridge MM Series Coils)
4. **Progressive & Classic Rock Legends:**
   - **Rickenbacker Clank** (Rickenbacker 4003 Bridge, Vintage 4.7nF Series HPF Clank)
   - **Rickenbacker Vintage** (Rickenbacker 4001/4003 Dual Pair, Full-Range Neck + 4.7nF HPF Bridge)
   - **Rickenbacker Neck** (Rickenbacker 4003 Solo Neck, 210mm 24th-Fret Harmonic Node Bloom)
   - **Dingwall Bridge** (Dingwall Fanned Multi-Scale Bridge, 34"–37" High Wave-Speed Sparkle @ 6.6 kHz)
   - **Dingwall Middle** (Dingwall Fanned Multi-Scale Middle, Punchy Low-Mid Grunt & Growl)
   - **Dingwall Parallel** (Dingwall Fanned Multi-Scale Dual Coils in Parallel, Scooped Aggression)
   - **Mudbucker Deep** (Gibson Mudbucker Ultra Series, 18.0H Coupled Sidewinder Sub-Bass)
   - **Mudbucker Baritone** (Gibson EB-2/EB-3 Baritone 2.5H Choke + 22nF Cap Midrange Rock Snarl)
5. **Acoustic Transducers:**
   - **Upright Piezo** (Upright Acoustic Double Bass, Bridge Force Sensor — recommended with 3 Sigma Audio "Acoustic Upright Standard" AST IRs)
6. **Precision Bass & Tone Shaper Family:**
   - **Precision Vintage** (Vintage '62 P-Bass Alnico V, CTS 250k Tone Open)
   - **Precision 51 Slab** (Vintage '51 Single-Coil Precision Pickup on Roundwounds, Raw 2.48 kHz Snap)
   - **Precision Active** (Fender American Ultra II Active Split-P, 18V 3-Band Preamp)
   - **Precision Warm** (Vintage '62 Split-P Motown Thump, Damped 47nF Roll-Off & Heavy Flatwounds)
   - **Precision Dub** (Vintage '50s Split-P Deep Dub, 100nF Fullerton 0.1µF Spec & Heavy Flatwounds)
7. **P/J Bass Family:**
   - **PJ Active** (Fender American Ultra II Active P/J, 18V 3-Band Preamp)
   - **PJ Passive** (Vintage '80s Passive P/J Bass, Duff McKagan / BB3000 Parallel Sum)

---

## Recommended Darkglass Anagram Signal Flow

```
[Physical Bass]
      │  (Volume 100%, Tone 100%)
      ▼
[Block 1: Allomorph NAM Preamp]  ◄── Tone3000 Pickup Twin
      │  (Forward Digital Twin source-to-target transformation)
      ▼
[Block 2: Preamp / Overdrive]   ◄── Microtubes B7K, Vintage Ultra, SVT, etc.
      │  (Distorts authentic pickup resonance peaks)
      ▼
[Block 3: Cab IR Loader]        ◄── 8x10, 4x10, 1x15 Speaker Cab Impulse (or 3 Sigma AST IRs for Upright)
      │
      ▼
[FOH / Interface / DAW]
```
