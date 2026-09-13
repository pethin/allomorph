# Allomorph - Tone Naming Guidelines for Bassists & Tone3000 Pack Architecture

This document establishes the official tone naming standard across Allomorph. All agents, automated workflows, and human contributors must adhere strictly to these guidelines when naming target voices (`tone_name`), trained Neural Amp Modeler models (`.nam`), baked audio stems (`.wav`), and Tone3000 storefront catalog entries.

---

## 1. Core Principles: The Working Bassist's Mindset

Bassists browse tone libraries, dial in presets, and switch sounds on stage with a very specific mental model. They do not think in terms of circuit transfer functions, Bessel roots, or electrical component values. They think in terms of:

1. **Bass Family & Heritage:** Precision, Jazz, StingRay, PJ, P/MM, Rickenbacker, Mudbucker, Dingwall, Upright.
2. **Coil Selection & Wiring:** Dual single-coil pair, solo bridge pickup, parallel humbucker, series humbucker.
3. **Musical Sonic Profiles & Tone Control States:**
   - An open tone control with natural air and bite (`Open`)
   - A vocal mid-range push that cuts through guitars (`Mids`)
   - Classic rolled-off Motown flatwound warmth (`Warm`)
   - Deep 1950s reggae/dub sub-bass thump (`Dub`)
   - Active 2-band preamp slap scoop (`Active`)
   - Bridge-biased singing fretless growl (`Growl`)
   - Aggressive high-pass pick bite (`Clank`)
4. **Physical Instrument Controls:** Exact physical switch settings (`[Neck]`, `[Bridge]`, `[Parallel]`, `[Series]`, `[Split]`) required on their own bass to match the digital twin.

### The Golden Rule of Tone Naming
> **Never name a tone after raw electrical components, uninformative capacitor values, or arbitrary artist nicknames when you can describe its instrument family, pickup configuration, and musical tonal contour.**
> 
> A working bassist glancing down at a pedalboard screen on a dark stage needs to know immediately **what bass sound they are playing** and **where their physical bass switch should be set**.

---

## 2. The First-Principles Tone Name Grammar

Every tone name across Allomorph follows a strict, predictable 2-to-3 token grammar:

$$\textbf{Tone Name} = \texttt{[Family]} \ + \ \left[\texttt{Configuration/Position}\right] \ + \ \texttt{[Voicing Modifier]}$$

When combined with source instrument physical switch positions and versioning, the full hardware and storefront model name is:

$$\textbf{Full Model Name} = \underbrace{\texttt{Tone Name}}_{\text{9--18 chars}} \ + \ \underbrace{\texttt{[Pickup Position]}}_{\text{Optional: 6--10 chars}} \ + \ \underbrace{\texttt{ v[dsp].[inst].[voice]}}_{\text{7 chars}}$$

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│  Full Model Name Anatomy: [Tone Name] [Pickup Position] v[dsp].[inst].[voice]    │
├──────────────────────────────────────┬───────────────────────────────────────────┤
│  Darkglass Anagram Screen Viewport   │  Tone3000 Upload Ceiling                  │
│  Target: <= 34 characters (Static)   │  Hard Limit: <= 64 characters (Strict)    │
│  100% Zero-Scroll achieved on stage  │  Upload rejected by Tone3000 if > 64      │
└──────────────────────────────────────┴───────────────────────────────────────────┘
```

---

## 3. Permitted Vocabulary & Taxonomy Dictionary

To prevent ad-hoc drift, agents and contributors must strictly draw tokens from the approved taxonomy dictionary:

### 3.1 Family Tokens (`[Family]`)
Identifies the bass instrument/pickup archetype:

| Family Token | Target Instrument / Pickup Architecture | Notes |
| :--- | :--- | :--- |
| `Jazz` | Fender Jazz Bass dual single-coils (60s spacing) | Never use `J-Bass` or `60s Jazz` as family token |
| `Precision` | Fender Precision Bass split-coil | Never use `P-Bass` or `Vintage 62 P` |
| `StingRay` | Music Man StingRay bridge dual-coil humbucker | Never use `MusicMan` or `MM` alone |
| `PJ` | Precision split-neck + Jazz bridge single-coil | Standard abbreviation without slash |
| `P∕MM` | Precision split-neck + Music Man bridge humbucker | Uses Unicode Division Slash (`∕`, `\u2215`) |
| `Rickenbacker` | Rickenbacker 4001/4003 bridge single-coil | Never abbreviate to `Rick` |
| `Mudbucker` | Gibson EB-0 sidewinder neck humbucker | Iconic bassist term for the 14.4H neck pickup |
| `Dingwall` | Multiscale fanned-fret angled dual-coil bridge | Represents progressive multiscale clarity |
| `Upright` | Double bass acoustic bridge piezo force transducer | Replaces generic `Piezo` |
| `Studio` | Aperture-preserving DI / impedance character presets | Replaces generic `Character` |
| `Canonical` | Internal wideband reference baseline | Excluded from storefront packs |

### 3.2 Configuration Tokens (`[Configuration]`)
Specifies the active coil topology or pickup selection on the modeled target instrument:

| Configuration Token | Meaning | When to Use |
| :--- | :--- | :--- |
| `Pair` | Both pickups engaged in parallel | Used for dual-pickup basses (`Jazz Pair`) |
| `Bridge` | Solo bridge pickup | Used when isolating the bridge coil (`Jazz Bridge`, `Dingwall Bridge`) |
| `Neck` | Solo neck pickup | Used when isolating the neck coil (`Mudbucker Neck`) |
| `Parallel` | Dual-coil humbucker wired in parallel | Used for dual-coil models (`StingRay Parallel`, `P∕MM Parallel`) |
| `Series` | Dual-coil humbucker wired in series | Used for high-output series models (`StingRay Series`, `P∕MM Series`) |
| *(Omitted)* | Inherent single-pickup architecture | Inherent for `Precision` (always split-coil), `Upright` (always bridge piezo), and `Studio` |

### 3.3 Voicing Modifier Tokens (`[Voicing Modifier]`)
Describes the musical tone contour or circuit state. Replaces all raw electronic component values (`22nF`, `47nF`, `100nF`) and artist nicknames with musician terms:

| Modifier Token | Musical Sonic Function | Replaced Legacy Terms | Analog / Physical Basis |
| :--- | :--- | :--- | :--- |
| `Open` | Wide-open tone control with full harmonic air | `Pair`, `60s` | CTS 250k/500k pot wide open |
| `Mids` | Vocal midrange push; cuts through guitars | `22nF`, `ToneStyler 22nF` | Shunt capacitor rolls off highs, boosting 440–760 Hz |
| `Warm` | Classic rolled-off flatwound / Motown thump | `47nF`, `ToneStyler 47nF` | 47nF shunt capacitor creates smooth, woody low-mids |
| `Dub` | Ultra-deep sub-bass roll-off; reggae/dub | `100nF`, `ToneStyler 100nF` | 100nF shunt capacitor removes all top-end for massive sub |
| `Vintage` | Alnico V vintage winding; warm compression | `Vintage 62`, `Alnico` | Alnico V split-coil with authentic 250k CTS harness |
| `Modern` | Ceramic magnet / high output; crisp bite | `Ceramic`, `Modern P` | High-inductance ceramic coil with 500k harness |
| `Active` | Onboard active 2-band preamp EQ (slap scoop) | `Active Buffer`, `Modern Jazz` | Active 2-band EQ with boosted lows/highs & scooped mids |
| `Passive` | Passive high-impedance RLC loading feel | `Vintage PJ`, `Passive Character` | Authentic passive RLC network with 250k pots & 750pF cable |
| `Growl` | Bridge-biased blend for singing fretless tone | `Jaco Bridge Growl` | Decoupled volume pot blend (100% bridge, 75% neck) |
| `Clank` | Aggressive pick bite and high-pass clank | `4.7nF HPF`, `4003` | In-line series 4.7nF capacitor rolling off sub-bass |
| `Deep` | Maximum low-end sidewinder sub resonance | `Ultra Series` | 14.4H high-inductance sidewinder coil |
| `Acoustic` | Wood body resonance & bridge force response | `Piezo`, `Transducer` | Bridge force sensor simulation on 41.5" upright scale |
| `Direct` | Pure, transparent studio DI transmission | `Neutral Character` | Transparent bypass preserving source aperture |

---

## 4. Hardware Constraints & Character Budgets

Every Allomorph tone name must satisfy two physical and software thresholds:

### 1. Darkglass Anagram Screen Viewport ($\le 34$ Characters)
- **Viewport:** The Darkglass Anagram pedalboard LCD block displays **34 characters** without scrolling.
- **Goal:** Achieve **100% Zero-Scroll** on stage across all production voices.
- Under our systematic grammar, the longest name in the catalog (`Rickenbacker Clank [Bridge] v2.1.1`) is exactly **34 characters**, ensuring zero marquee scrolling across the entire library!

### 2. Tone3000 Upload Ceiling ($\le 64$ Characters)
- **Ceiling:** The Tone3000 cloud uploader strictly rejects model basenames exceeding **64 characters** (excluding `.nam`).
- **Enforcement:** Programmatically enforced in `get_t3k_basename()` in `src/allomorph/naming.py`, which raises a diagnostic `ValueError` if violated.
- With our grammar, all model basenames range between **20 and 34 characters**, providing a massive safety margin.

---

## 5. Complete 24-Voice Catalog Reference Table

Below is the definitive catalog of all 24 Allomorph target voices under the systematic naming system, showing character lengths and screen viewport status:

| Voice ID | Declarative `tone_name` | Source Pos Tag | Full Model Basename (`.nam`) | Chars | Screen Status | Musical Personality / Bassist Context |
| :--- | :--- | :---: | :--- | :---: | :--- | :--- |
| `00_canonical_intermediate` | `Canonical Datum` | — | `Canonical Datum v2.1.1` | 22 | **Zero-Scroll** | 34" @ 93.5mm reference baseline |
| `01_modern_jazz_active` | `Jazz Pair Active` | `[Parallel]` | `Jazz Pair Active [Parallel] v2.1.1` | 33 | **Zero-Scroll** | Active 2-band EQ slap scoop; scooped mids |
| `02_jazz_bass_pair` | `Jazz Pair Open` | `[Parallel]` | `Jazz Pair Open [Parallel] v2.1.1` | 31 | **Zero-Scroll** | Classic 60s dual single-coils, tone wide open |
| `02b_jazz_bass_pair_22nf` | `Jazz Pair Mids` | `[Parallel]` | `Jazz Pair Mids [Parallel] v2.1.1` | 31 | **Zero-Scroll** | Vocal mid-honk; smooth top with punchy mids |
| `02c_jazz_bridge_growl_bias` | `Jazz Bridge Growl` | `[Bridge]` | `Jazz Bridge Growl [Bridge] v2.1.1` | 33 | **Zero-Scroll** | Singing fretless bridge burp & vocal growl |
| `03_jazz_bridge_60s` | `Jazz Bridge Open` | `[Bridge]` | `Jazz Bridge Open [Bridge] v2.1.1` | 33 | **Zero-Scroll** | Tight, biting 60s bridge single-coil, wide open |
| `04_modern_p_ceramic` | `Precision Modern` | `[Split]` | `Precision Modern [Split] v2.1.1` | 31 | **Zero-Scroll** | High-output ceramic split-P with 500k pots |
| `05_vintage_62_p_alnico` | `Precision Vintage` | `[Split]` | `Precision Vintage [Split] v2.1.1` | 32 | **Zero-Scroll** | Classic '62 Alnico V split-P, CTS 250k open |
| `05b_vintage_62_p_22nf` | `Precision Mids` | `[Split]` | `Precision Mids [Split] v2.1.1` | 29 | **Zero-Scroll** | Cutting vocal P-bass with articulate punch |
| `05c_vintage_62_p_47nf` | `Precision Warm` | `[Split]` | `Precision Warm [Split] v2.1.1` | 29 | **Zero-Scroll** | Classic Motown / flatwound woody thump |
| `05d_vintage_50s_p_100nf` | `Precision Dub` | `[Split]` | `Precision Dub [Split] v2.1.1` | 28 | **Zero-Scroll** | Deep 1950s sub-bass dub thump |
| `07_modern_pj_active` | `PJ Active` | `[Parallel]` | `PJ Active [Parallel] v2.1.1` | 28 | **Zero-Scroll** | Split-P + J-bridge with active 2-band preamp |
| `08_vintage_pj_passive` | `PJ Passive` | `[Parallel]` | `PJ Passive [Parallel] v2.1.1` | 29 | **Zero-Scroll** | Organic P-thump with J-bridge articulation |
| `09_stingray_mm_parallel` | `StingRay Parallel` | `[Bridge]` | `StingRay Parallel [Bridge] v2.1.1` | 33 | **Zero-Scroll** | Classic 2-band active Music Man authority |
| `09b_stingray_mm_series` | `StingRay Series` | `[Bridge]` | `StingRay Series [Bridge] v2.1.1` | 31 | **Zero-Scroll** | Muscular, mid-forward high-output MM punch |
| `10_rickenbacker_bridge_hpf` | `Rickenbacker Clank` | `[Bridge]` | `Rickenbacker Clank [Bridge] v2.1.1` | 34 | **Zero-Scroll** | Classic 4003 bridge coil with high-pass clank |
| `11_modern_pmm_active` | `P∕MM Parallel` | `[Parallel]` | `P∕MM Parallel [Parallel] v2.1.1` | 31 | **Zero-Scroll** | Split-P + MM bridge in parallel with active buffer |
| `11b_pmm_hybrid_series` | `P∕MM Series` | `[Series]` | `P∕MM Series [Series] v2.1.1` | 27 | **Zero-Scroll** | Split-P + MM bridge in series with active buffer |
| `12_mudbucker_ultra_series` | `Mudbucker Deep` | `[Neck]` | `Mudbucker Deep [Neck] v2.1.1` | 29 | **Zero-Scroll** | Massive 14.4H sidewinder neck sub-bass rumble |
| `13_dingwall_multiscale_bridge` | `Dingwall Bridge` | `[Bridge]` | `Dingwall Bridge [Bridge] v2.1.1` | 32 | **Zero-Scroll** | High-tension fanned-fret progressive clarity |
| `14_upright_bridge_transducer` | `Upright Acoustic` | — | `Upright Acoustic v2.1.1` | 23 | **Zero-Scroll** | Woody double-bass piezo bridge transducer |
| `15_neutral_character` | `Studio Direct` | — | `Studio Direct v2.1.1` | 20 | **Zero-Scroll** | Pure acoustic aperture; transparent studio DI |
| `15b_active_character` | `Studio Active` | — | `Studio Active v2.1.1` | 20 | **Zero-Scroll** | 1MΩ wideband active buffer (restores sparkle) |
| `15c_passive_character` | `Studio Passive` | — | `Studio Passive v2.1.1` | 21 | **Zero-Scroll** | Organic high-Z passive pickup and cable load |

---

## 6. Filesystem Safety & Character Sanitization

To ensure model files are compatible with macOS, Linux, and Windows filesystems, Allomorph applies specific sanitization rules in `get_t3k_basename()`:

1. **Path Separator Replacement (`/` and `\`):**
   - Slashes in tone names (such as `P/MM`) must never become filesystem subdirectories.
   - Slashes are automatically replaced with the **Unicode Division Slash (`∕`, `\u2215`)**:
     ```python
     name = name.replace("/", "\u2215").replace("\\", "\u2215")
     ```
   - Result: `P/MM Parallel` $\to$ `P∕MM Parallel` (renders as a clean visual slash without triggering directory nesting).
2. **Quotation Marks:**
   - Avoid unescaped ASCII double quotes (`"`).
3. **Prohibited Characters:**
   - Never use `:`, `*`, `?`, `<`, `>`, or `|`, as these are illegal in Windows filenames.

---

## 7. Do's and Don'ts for Agents & Contributors

### DO:
- **DO follow the 3-token grammar:** `[Family] [Configuration] [Voicing Modifier]`.
- **DO use permitted vocabulary:** Draw tokens strictly from Section 3.
- **DO verify character lengths:** Ensure the complete basename with version tag is $\le 34$ characters for zero scroll on stage (and never exceed 64 chars).
- **DO use standard physical switch tags for multi-pickup source basses:** `[Neck]`, `[Bridge]`, `[Parallel]`, `[Series]`, `[Split]`.
- **DO verify with automated tests:** Run `uv run pytest tests/tone3000/test_tone3000.py tests/pipeline/test_staging.py` after any voice modification.

### DO NOT:
- **DO NOT include electrical component values in tone names:**
  - ❌ `Vintage 62 P 22nF`
  - ✅ `Precision Mids`
  - ❌ `60s Jazz 47nF`
  - ✅ `Jazz Pair Warm`
  - ❌ `Vintage 50s P 100nF`
  - ✅ `Precision Dub`
- **DO NOT use artist nicknames:**
  - ❌ `Jaco Bridge Growl`
  - ✅ `Jazz Bridge Growl`
- **DO NOT include the source instrument name in `tone_name`:**
  - ❌ `34in Standard P Precision Vintage v2.1.1.nam` (43 chars, marquee scrolls)
  - ✅ `Precision Vintage [Split] v2.1.1.nam` (32 chars, zero-scroll)
- **DO NOT invent arbitrary abbreviations:**
  - ❌ `P-Bss Vntg [Splt] v2.1.1.nam` (cryptic and unprofessional)
  - ✅ `Precision Vintage [Split] v2.1.1.nam`
- **DO NOT mix and match word order:**
  - ❌ `Active Modern Jazz Pair`
  - ✅ `Jazz Pair Active`
