# Tone3000 Tone Pack Catalog

This document details the standardized storefront product configurations for **Allomorph Tone Packs** on [Tone3000](https://tone3000.com), engineered for Neural Amp Modeler (NAM) and the Darkglass Anagram pedalboard.

Each pack contains 25–35 precision Forward Digital Twin voicings paired directly from the source instrument's simulated stem (`Source Digital Twin → NAM Block 1 → Target Digital Twin`) to transform a specific physical bass guitar into iconic vintage passive, modern active, semi-hollow/hollow-body, heavy multi-scale, acoustic, and rig-calibrated Studio Super bass topologies.

Tone pack metadata and bespoke storefront product copy are defined in `config/packs/*.toml`, while canonical license terms and trademark notices are defined in `config/legal.toml`.

---

## Pack Catalog & Storefront Listings

| Pack Edition | Source Instrument Calibration | Recommended Knobs / Switches | Pack Configuration | Production Artwork (SVG / JPG) |
| :--- | :--- | :--- | :--- | :--- |
| **Standard Precision Bass Edition** | 34" Standard Fender P-Bass (Passive Split-P, 35 Voicings) | Vol 100%, Tone 100% | [`34in_standard_p.toml`](../../config/packs/34in_standard_p.toml) | [`allomorph_standard_precision_bass.svg`](../../tone3000/assets/allomorph_standard_precision_bass.svg) &bull; [JPG](../../tone3000/assets/allomorph_standard_precision_bass.jpg) |
| **Standard Jazz Bass Edition** | 34" Standard Fender Jazz Bass (Passive Single-Coil Pair, 25 Voicings) | Neck Vol 100%, Bridge Vol 100%, Tone 100% (Both Pickups in Parallel) | [`34in_standard_jazz.toml`](../../config/packs/34in_standard_jazz.toml) | [`allomorph_standard_jazz_bass.svg`](../../tone3000/assets/allomorph_standard_jazz_bass.svg) &bull; [JPG](../../tone3000/assets/allomorph_standard_jazz_bass.jpg) |
| **Standard P/J Bass Edition** | 34" Standard Fender P/J Bass (Split-P Neck + J Bridge, 25 Voicings) | Neck Vol 100%, Bridge Vol 100%, Tone 100% (Both Pickups in Parallel) | [`34in_standard_pj.toml`](../../config/packs/34in_standard_pj.toml) | [`allomorph_standard_pj_bass.svg`](../../tone3000/assets/allomorph_standard_pj_bass.svg) &bull; [JPG](../../tone3000/assets/allomorph_standard_pj_bass.jpg) |
| **Mustang P/J Bass Edition** | 30" Short-Scale Fender Mustang P/J Bass (26 Voicings) | 3-Way Switch in Middle (Both Pickups in Parallel), Vol 100%, Tone 100% | [`30in_mustang_pj.toml`](../../config/packs/30in_mustang_pj.toml) | [`allomorph_mustang_pj_bass.svg`](../../tone3000/assets/allomorph_mustang_pj_bass.svg) &bull; [JPG](../../tone3000/assets/allomorph_mustang_pj_bass.jpg) |
| **Preamp Soapbar Bass Edition** | 34" Preamp Dual-Soapbar Bass (Passive Alnico Coils + Active EQ, 26 Voicings) | Blend at Center Detent (Parallel), Vol 100%, Active EQ Flat (Center Detents) | [`34in_preamp_soapbar.toml`](../../config/packs/34in_preamp_soapbar.toml) | [`allomorph_preamp_soapbar_bass.svg`](../../tone3000/assets/allomorph_preamp_soapbar_bass.svg) &bull; [JPG](../../tone3000/assets/allomorph_preamp_soapbar_bass.jpg) |
| **Active StingRay Bass Edition** | 34" Active Music Man StingRay / Sterling Ray34/Ray35 (4/5-String, 25 Voicings) | Active EQ Flat (Center Detents), Switch in Parallel (if 3-way) | [`34in_active_stingray.toml`](../../config/packs/34in_active_stingray.toml) | [`allomorph_active_stingray_bass.svg`](../../tone3000/assets/allomorph_active_stingray_bass.svg) &bull; [JPG](../../tone3000/assets/allomorph_active_stingray_bass.jpg) |
| **Active EMG Bass Edition** | 34" Active EMG Soapbar Bass (Option C Baseline, 26 Voicings) | Blend at Center Detent (Parallel), Vol 100%, Active EQ Flat (Center Detents) | [`34in_active_emg.toml`](../../config/packs/34in_active_emg.toml) | [`allomorph_active_emg_bass.svg`](../../tone3000/assets/allomorph_active_emg_bass.svg) &bull; [JPG](../../tone3000/assets/allomorph_active_emg_bass.jpg) |

---

## Storefront Engineering Constraints

All pack descriptions strictly adhere to Tone3000 platform guidelines:
1. **Character Limit Compliance:** Tone3000 enforces a strict maximum length of 10,000 characters per listing. All 7 packs stay safely under this ceiling.
2. **Unified Parallel Setup for Dual-Pickup Editions:** All dual-pickup editions (Jazz, P/J, Mustang P/J, Preamp Soapbar, Active EMG) are calibrated from a single unified parallel input (both pickups 100% / blend at center detent), preserving physical RWRP hum-cancellation on Jazz basses and eliminating manual pickup-switch changes between presets on stage.
3. **Single-Sentence Concise Voicing Summaries:** Every numbered voicing description in `config/packs/*.toml` must be a single concise sentence focusing on musical style and tonal character (plus expected amp + cab + mic pairing for rig-calibrated Studio Super voicings), omitting filler adjectives, millimeter/inductance measurements, and multi-sentence explanations.
4. **Independent Legal Disclaimers:** Trademark disclaimers and non-commercial/commercial user rights are centrally codified in `config/legal.toml` and appended to each exported listing.
5. **Native Identity Exclusion:** Base-instrument 1:1 identical duplicates (where source and target voicings are identical) are omitted via `is_identity_voicing`, ensuring every model in the pack provides transformational capacity (see [`ranking.md`](ranking.md)).
6. **Utilitarian & Mix-Ready Selection:** Every storefront voicing delivers non-invertible electroacoustic physics (or SNR-preserving neural bandwidth extension) and sits mix-ready (or is easily EQed to mix-ready) without thin solo-bridge fundamental loss or redundant passive RC low-pass variations (see [`ranking.md` §1.4](ranking.md#14-utilitarian--mix-ready-storefront-selection-principle)).
7. **Tone Naming & Display Character Budgets:** Every model filename strictly adheres to the 64-character Tone3000 upload limit and 34-character Darkglass Anagram pedalboard screen target (see master guide [`naming.md`](naming.md)).

---

## Storefront Voice & Bassist-Centric Copywriting Guidelines

When writing or updating Tone3000 storefront descriptions in `config/packs/*.toml`, always translate complex analog modeling math into musical language that working bassists immediately understand and connect with:

### 1. The Cardinal Rule
**Never describe the mathematics when you can describe the musical tone, feel, and mix placement.**
Bassists browse Tone3000 to find sounds that inspire them, solve gigging problems, or give them iconic tones they cannot physically get with their stock instrument. Focus on:
- **Iconic Players & Eras:** James Jamerson, Pino Palladino, Jaco Pastorius, Marcus Miller, Paul McCartney, Chris Squire, Geddy Lee, NYC session slap, '60s Motown, '80s hard rock.
- **Mix Placement & EQ Feel:** Vocal midrange growl, hollow slap scoop, pillowy sub-bass thump, cutting pick clank, woody semi-hollow/hollow bloom, glassy hi-fi air.
- **Playing Styles:** Slap & pop, 16th-note staccato fingerstyle, drop-tuned metal, British invasion Beatle-bass thump, acoustic jazz gigs.

### 2. The Concise Single-Sentence Rule
Keep every numbered tone description in `config/packs/*.toml` to **one tight, scan-friendly sentence**:
- **Cut filler and measurements:** Strip conversational filler (*"The quintessential..."*, *"heard on thousands of hit records"*), physical dimensions (*"53.3mm bridge-spaced"*, *"210 mm"*), and raw component values (*"15.0H inductor + 20nF capacitor"*).
- **Format:** `[Archetype / Era]: [concise tonal character & mix feel].`
- **Rig-Calibrated Voicings (`Studio Super`):** Make it explicit that the model is a Block 1 instrument front-end and append the required Darkglass Anagram signal chain in parentheses: `(requires Anagram blocks: [Amp Block] + [Cab Block] + [Mic Block] [Mic Position])`.
- **Examples:**
  - ❌ *Verbose:* `"The quintessential NYC '70s active slap tone: tight, massive sub-bass and crisp, glassy highs with a 53.3mm bridge-spaced midrange scoop that pops right out of a dense mix."`
  - ✅ *Concise:* `"NYC '70s active slap tone: tight, massive sub-bass and crisp, glassy highs with a midrange scoop."`
  - ✅ *Rig-Calibrated (`Studio Super`):* `"Punchy '80s/'90s alt-rock and funk front-end with tight lows and snappy upper-mid presence (requires Anagram blocks: Kallinen Kruukkeri + SWIRL 4x10 + Dynamic 421 Cone On-Axis)."`

### 3. Jargon Translation Guide

| Avoid Engineering Mumbo-Jumbo | Use Bassist-Friendly Language |
| :--- | :--- |
| *Forward Digital Twin source-to-target neural mapping* | Transforms your instrument's sound right at the pickups in real time |
| *Aperture geometry & spatial string comb-filtering* | Pickup placement, sweet spots, and harmonic phase interactions |
| *Bessel $J_1(x)/x$ cylindrical pole transfer* | Round pole-piece punch and dynamic transient attack |
| *High-impedance coil eddy currents & skin-effect damping* | Warm, woody low-mid body and smooth vintage top-end bloom |
| *Decoupled dual volumes with 8.8k neck wiper resistance* | Rolling the neck volume back just a touch for singing Jaco bridge growl |
| *1951 single-coil Alnico V 2.8H slab pickup* | Raw, uncompressed '51 single-coil Precision growl and open top-end snap |
| *47nF paper-in-oil cap with flatwound damping* | Classic Jamerson/Pino Motown flatwound thump with fat, pillowy sub-bass |
| *Sadowsky-style 2-band buffer with '70s bridge spacing* | NYC '70s active slap tone: tight, massive sub-bass and crisp, glassy highs with a midrange scoop |
| *Series wiring with +6.0 dB inductive surge* | Hotter, muscular series punch with wall-shaking low-mid growl and sustain |
| *4.7nF series high-pass capacitor prog-rock clank* | Classic prog-rock bridge clank: tight, lean lows with biting pick attack and upper-mid grit |
| *FD3 34"–37" multi-scale fanned fret active sparkle* | Modern progressive metal bridge tone: razor-sharp pick clank and piano-like low-B definition |
| *18.0H coupled overwound neck-heel sidewinder Mudbucker* | Solid-body Gibson EB-0 Mudbucker: a dark, colossal wall of vintage sub-bass rumble |
| *Semi-hollow / hollow-archtop admittance & sustain sink* | Authentic hollow-body wood bloom and fast, thumpy sustain decay |
| *Bridge piezo force sensor with hollow sustain sink* | Acoustic upright double bass: woody pizzicato decay and organic finger feel |

---

## Digital Twin Voice Architecture & Ranking

Every pack provides comprehensive coverage across nine core modular sonic families (plus the 10 Darkglass Anagram rig-calibrated Studio Super voicings in the Standard Precision Bass Edition), ordered according to the [Tone Ranking Architecture](ranking.md):

1. **Jazz Bass Family:**
   - **Jazz Pair Active** (Modern Active '70s-Spacing Jazz Bass, Sadowsky NYC 2-Band Preamp with Cable Isolation)
   - **Jazz Pair Open** (Vintage '60s Jazz Bass Pair, Dual Parallel 250k Harness with $63.5\text{ mm}$ Bridge & 640 Hz Scoop)
   - **Jazz Pair Series** (Fender S-1 / 4-Way Series Jazz Pair, $7.0\text{ H}$ Humbucker Midrange Roar)
   - **Jazz Bridge Growl** ('60s Jazz Bridge Growl, Jaco Bias 25.5k Wiper Decoupling @ `neck_vol = 0.97`)
   - **Jazz Neck Warm** ('60s Jazz Neck Single-Coil, Deep Woody Warmth & Round Low-End Body)
2. **Music Man StingRay Family:**
   - **StingRay Parallel** (Music Man StingRay Active 2-Band Parallel, Classic Hollow Mid-Scoop & Clank)
   - **StingRay Series** (Music Man StingRay Active 2-Band Series, Aggressive 2.19 kHz Bark & +6.0 dB EMF)
3. **P/MM Modern Hybrids:**
   - **P∕MM Parallel** (Modern Boutique Session Hybrid, Lakland LH-3 Active Preamp & Bridge MM Parallel Coils)
   - **P∕MM Series** (Modern Active P/MM Hybrid, Lakland LH-3 Active Preamp & Bridge MM Series Coils)
4. **Rickenbacker 4001/4003 Family:**
   - **Rickenbacker Clank** (Rickenbacker 4003 Bridge, Vintage 4.7nF Series HPF Clank)
   - **Rickenbacker Vintage** (Rickenbacker 4001/4003 Dual Pair, Full-Range Neck + 4.7nF HPF Bridge)
   - **Rickenbacker Neck** (Rickenbacker 4003 Solo Neck Single-Coil at 24th-Fret Harmonic Node, 2.34 kHz Glassy Roundwound Peak)
5. **Dingwall Multi-Scale Family:**
   - **Dingwall Bridge** (Dingwall Fanned Multi-Scale Bridge, 34"–37" High Wave-Speed Sparkle @ 6.6 kHz)
   - **Dingwall Parallel** (Dingwall Fanned Multi-Scale Bridge + Middle in Parallel, Scooped Aggression)
   - **Dingwall Series** (Dingwall Fanned Multi-Scale Bridge + Middle in Series, Rotary Pos 3 Midrange Roar)
6. **Gibson Mudbucker & Hollow-Body Family:**
   - **Mudbucker Deep** (Solid-Body Gibson EB-0 Mudbucker Ultra Series, 18.0H Coupled Sidewinder Sub-Bass)
   - **Mudbucker Hollow** (Semi-Hollow 1958 Gibson EB-2 Mudbucker, 165 Hz Maple Archtop Cavity Bloom & Woody Decay)
   - **Mudbucker Baritone** (Semi-Hollow Gibson EB-2 Baritone Choke + Cap Midrange Rock Snarl)
   - **Hofner Violin Warm** (1962 Höfner 500/1 Fully Hollow Violin Bass, Neck Staple Humbucker & Flatwound Beatle Thump)
7. **Acoustic Transducers:**
   - **Upright Piezo** (Upright Acoustic Double Bass, Bridge Force Sensor + Hollow-Archtop Sustain Decay — recommended with 3 Sigma Audio "Acoustic Upright Standard" AST IRs)
8. **Precision Bass & Super P Family:**
   - **Precision Vintage** (Vintage '62 P-Bass Alnico V, CTS 250k Tone Open)
   - **Precision Active** (Fender American Ultra II Active Split-P, 18V 3-Band Preamp)
   - **Dingwall Super P** (Dingwall SP1 32"–35" Multi-Scale Passive Neodymium Dual-P Split-Coil, 3.05 kHz Presence Peak)
   - **Precision Warm** (Vintage '62 Split-P Motown Thump, Damped 47nF Roll-Off & Heavy Flatwounds)
9. **P/J Bass Family:**
   - **PJ Active** (Fender American Ultra II Active P/J, 18V 3-Band Preamp)
   - **PJ Passive** (Vintage '80s Passive P/J Bass, Duff McKagan / BB3000 Parallel Sum)
10. **Studio Super Family (Darkglass Anagram Rig-Calibrated):**
   - **Studio Super SVT Classic** (Peggy Classic + Vintage Peggy 8x10 + Dynamic 421 Cone On-Axis)
   - **Studio Super SVT Blueline** (Peggy Bass + Peggy 8x10 + Dynamic 421 Cap Off-Axis)
   - **Studio Super B15** (Peggy FlipTop + Peggy Fliptop 1x15 + Condenser 67 Cone On-Axis)
   - **Studio Super Bassman** (Super California + California Bass 4x10 + Condenser 67 Cone On-Axis)
   - **Studio Super GK 800RB** (Kallinen Kruukkeri + SWIRL 4x10 + Dynamic 421 Cone On-Axis)
   - **Studio Super Mesa Block** (Modern ISOBRICK750 OD + Modern Bass 4x10 + Dynamic 421 Cone On-Axis)
   - **Studio Super Orange** (Fruity Horror Bass 500 + Fruity Bass 4x10 + Dynamic 7 Cone On-Axis)
   - **Studio Super SWR Goliath** (Swirl-900 + SWIRL 4x10 + Condenser 103 Cap Off-Axis)
   - **Studio Super Markbass** (Black and Yellow + Black & Yellow 4x10 + Condenser 414 Cap Off-Axis)
   - **Studio Super EBS TD650** (SWE TC650 + Darkglass Neo 4x10 + Dynamic 7 Cap Off-Axis)

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
