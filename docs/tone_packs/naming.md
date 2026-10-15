# Allomorph - Tone Naming Guidelines for Bassists & Tone3000 Pack Architecture

This document establishes the official tone naming standard across Allomorph. All agents, automated workflows, and human contributors must adhere strictly to these guidelines when naming target voices (`tone_name`), trained Neural Amp Modeler models (`.nam`), target audio stems (`.wav`), and Tone3000 storefront catalog entries.

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

### 2.1 Standard Single-Instrument Grammar

Every standard tone name across Allomorph follows a strict, predictable 2-to-3 token grammar:

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

### 2.2 Multi-Source Collection Grammar (`Source [Voicing] → Target`)

For specialized multi-source tone packs (e.g. Fretless-to-Upright collections where multiple source instruments target a common voice), individual bundle targets declare a bespoke custom name in the pack TOML configuration:

```toml
[[bundles]]
name = "fretless_jazz_bridge"
source = "34in_fretless_jazz:bridge_growl"
targets = [
    { target = "41in_upright_bass:bridge_piezo", name = "Fretless J [Bridge] → Upright" },
]
```

#### Anatomy of a Multi-Source Target Name:

$$\textbf{Custom Model Name} = \texttt{[Source Bass]} \ + \ \texttt{[Source Voicing/Tag]} \ + \ \texttt{" → "} \ + \ \texttt{[Target Voicing]} \ + \ \texttt{ v[dsp].[inst].[voice]}$$

- **Arrow Sanitization:** Authors may specify `->` or `-->` in configuration files; the Allomorph naming pipeline automatically sanitizes ASCII arrows into the standardized Unicode Rightwards Arrow (`→`, `\u2192`).
- **Character Budget:** Even with source identification, multi-source names must strive for zero-scroll on the Darkglass Anagram screen ($\le 34$ characters) and strictly obey the Tone3000 hard ceiling ($\le 64$ characters).

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

### 3.2 Configuration Tokens (`[Configuration]`)
Specifies the active coil topology or pickup selection on the modeled target instrument:

| Configuration Token | Meaning | When to Use |
| :--- | :--- | :--- |
| `Pair` | Both pickups engaged in parallel | Used for dual-pickup basses (`Jazz Pair`) |
| `Bridge` | Solo bridge pickup | Used when isolating the bridge coil (`Jazz Bridge`, `Dingwall Bridge`) |
| `Neck` | Solo neck pickup | Used when isolating the neck coil (`Mudbucker Neck`) |
| `Parallel` | Dual-coil humbucker wired in parallel | Used for dual-coil models (`StingRay Parallel`, `P∕MM Parallel`) |
| `Series` | Dual-coil humbucker wired in series | Used for high-output series models (`StingRay Series`, `P∕MM Series`) |
| *(Omitted)* | Inherent single-pickup architecture | Inherent for `Precision` (always split-coil) and `Upright` (always bridge piezo) |

### 3.3 Voicing Modifier Tokens (`[Voicing Modifier]`)
Describes the musical tone contour or circuit state. Replaces all raw electronic component values (`22nF`, `47nF`, `100nF`) and artist nicknames with musician terms:

| Modifier Token | Musical Sonic Function | Replaced Legacy Terms | Analog / Physical Basis |
| :--- | :--- | :--- | :--- |
| `Open` | Wide-open tone control with full harmonic air | `Pair`, `60s` | CTS 250k/500k pot wide open |
| `70s` | 1970s closer-to-bridge pickup placement ($53.3\text{ mm}$) | `70s Spacing` | 1970s Fender $0.4''$ bridge-shifted single-coil datum |
| `51 Slab` | 1951–1956 single-coil Precision pickup | `Tele P`, `51 P` | 2.4H Alnico V single-coil on roundwound strings |
| `Warm` | Classic rolled-off flatwound / Motown thump | `47nF`, `ToneStyler 47nF` | 47nF shunt capacitor creates smooth, woody low-mids |
| `Dub` | Ultra-deep sub-bass roll-off; reggae/dub | `100nF`, `ToneStyler 100nF` | 100nF shunt capacitor removes all top-end for massive sub |
| `Vintage` | Alnico V vintage winding or 4001 cap-in pair | `Vintage 62`, `4001 Pair` | Alnico V split-coil or 4001 dual-pickup with 4.7nF bridge HPF |
| `Modern` | Ceramic magnet / high output; crisp bite | `Ceramic`, `Modern P` | High-inductance ceramic coil with 500k harness |
| `Active` | Onboard active 2-band/3-band preamp EQ | `Active Buffer`, `Modern Jazz` | Active preamp with boosted lows/highs & isolated coils |
| `Passive` | Passive high-impedance RLC loading feel | `Vintage PJ` | Authentic passive RLC network with 250k pots & 750pF cable |
| `Growl` | Bridge-biased blend for singing fretless tone | `Jaco Bridge Growl` | Decoupled volume pot blend (100% bridge, 99% neck) |
| `Clank` | Aggressive pick bite and high-pass clank | `4.7nF HPF`, `4003` | In-line series 4.7nF capacitor rolling off sub-bass |
| `Deep` | Maximum low-end sidewinder sub resonance | `Ultra Series` | 14.4H high-inductance sidewinder coil |
| `Baritone` | Gibson 2.5H choke + 22nF high-pass mid snarl | `EB-2 Baritone`, `Choke` | 2.5H shunt inductor + 22nF series cap cutting sub-bass mud |
| `Piezo` | Bridge force response & rocking compliance | `Transducer`, `Wing` | Bridge force sensor simulation on 41.25" upright scale |
| `Direct` | Pure, transparent studio DI transmission | `DI`, `Studio DI` | Transparent bypass preserving source aperture |

---

## 4. Hardware Constraints & Character Budgets

Every Allomorph tone name must satisfy two physical and software thresholds:

### 1. Darkglass Anagram Screen Viewport ($\le 34$ Characters)
- **Viewport:** The Darkglass Anagram pedalboard LCD block displays **34 characters** without scrolling.
- **Goal:** Achieve **100% Zero-Scroll** on stage across all production voices.
- Under our systematic grammar, the longest unversioned display name in the storefront pack catalog (`Rickenbacker Vintage [Parallel]`) is **31 characters**, ensuring zero marquee scrolling on stage!

### 2. Tone3000 Upload Ceiling ($\le 64$ Characters)
- **Ceiling:** The Tone3000 cloud uploader strictly rejects model basenames exceeding **64 characters** (excluding `.nam`).
- **Enforcement:** Programmatically enforced in `get_t3k_basename()` in `src/allomorph/naming.py`, which raises a diagnostic `ValueError` if violated.
- With our grammar, all versioned `.nam` model basenames range between **20 and 38 characters** (max **38 chars** for `Rickenbacker Vintage [Parallel] v7.1.1`), providing a massive safety margin below the 64-character ceiling.

---

## 5. Complete 26-Voicing Storefront Target Reference Table

Below is the definitive table of the 26 curated storefront target voicings across the 16 catalog instruments, showing character lengths and screen viewport status:

| Target (`<inst_id>:<voicing_id>`) | Declarative `tone_name` | Source Pos Tag | Full Model Basename (`.nam`) | Unversioned / Versioned Chars | Screen Status | Musical Personality / Bassist Context |
| :--- | :--- | :---: | :--- | :---: | :--- | :--- |
| `34in_standard_jazz:pair_active` | `Jazz Pair Active` | `[Parallel]` | `Jazz Pair Active [Parallel] v7.1.1` | 27 / 34 | **Zero-Scroll** | Active 2-band EQ slap scoop; scooped mids |
| `34in_standard_jazz:pair_open` | `Jazz Pair Open` | `[Parallel]` | `Jazz Pair Open [Parallel] v7.1.1` | 25 / 32 | **Zero-Scroll** | Classic 60s dual single-coils, tone wide open |
| `34in_standard_jazz:bridge_growl` | `Jazz Bridge Growl` | `[Bridge]` | `Jazz Bridge Growl [Bridge] v7.1.1` | 26 / 33 | **Zero-Scroll** | Singing Jaco bridge burp & vocal growl |
| `34in_standard_jazz:bridge_open` | `Jazz Bridge Open` | `[Bridge]` | `Jazz Bridge Open [Bridge] v7.1.1` | 25 / 32 | **Zero-Scroll** | Tight, biting 60s bridge single-coil ($63.5\text{ mm}$) |
| `34in_standard_jazz:bridge_70s` | `Jazz Bridge 70s` | `[Bridge]` | `Jazz Bridge 70s [Bridge] v7.1.1` | 24 / 31 | **Zero-Scroll** | 70s bridge single-coil ($53.3\text{ mm}$); Geddy/Marcus bite |
| `34in_standard_jazz:neck_warm` | `Jazz Neck Warm` | `[Neck]` | `Jazz Neck Warm [Neck] v7.1.1` | 21 / 28 | **Zero-Scroll** | Warm, woody 60s jazz neck single-coil bloom |
| `34in_standard_p:vintage_open` | `Precision Vintage` | `[Neck]` | `Precision Vintage [Neck] v7.1.1` | 24 / 31 | **Zero-Scroll** | Classic '62 Alnico V split-P, CTS 250k open |
| `34in_standard_p:slab_51` | `Precision 51 Slab` | `[Neck]` | `Precision 51 Slab [Neck] v7.1.1` | 24 / 31 | **Zero-Scroll** | 1951 single-coil Precision on roundwounds; raw snap |
| `34in_standard_pj:neck_active` | `Precision Active` | `[Neck]` | `Precision Active [Neck] v7.1.1` | 23 / 30 | **Zero-Scroll** | Fender Ultra II active split-P with 18V 3-band preamp |
| `34in_standard_p:vintage_warm` | `Precision Warm` | `[Neck]` | `Precision Warm [Neck] v7.1.1` | 21 / 28 | **Zero-Scroll** | Classic Motown / flatwound woody thump |
| `34in_standard_p:slab_dub` | `Precision Dub` | `[Neck]` | `Precision Dub [Neck] v7.1.1` | 20 / 27 | **Zero-Scroll** | Deep 1950s sub-bass dub thump |
| `34in_standard_pj:pair_active` | `PJ Active` | `[Parallel]` | `PJ Active [Parallel] v7.1.1` | 20 / 27 | **Zero-Scroll** | Ultra II Split-P + J-bridge with 18V 3-band preamp |
| `34in_standard_pj:pair_parallel` | `PJ Passive` | `[Parallel]` | `PJ Passive [Parallel] v7.1.1` | 21 / 28 | **Zero-Scroll** | Organic P-thump with J-bridge articulation |
| `34in_active_stingray:parallel_classic` | `StingRay Parallel` | `[Bridge]` | `StingRay Parallel [Bridge] v7.1.1` | 26 / 33 | **Zero-Scroll** | Classic 2-band active Music Man authority |
| `34in_active_stingray:series_punch` | `StingRay Series` | `[Bridge]` | `StingRay Series [Bridge] v7.1.1` | 24 / 31 | **Zero-Scroll** | Muscular, mid-forward high-output MM series bark |
| `33in_rickenbacker_4003:bridge_clank` | `Rickenbacker Clank` | `[Bridge]` | `Rickenbacker Clank [Bridge] v7.1.1` | 27 / 34 | **Zero-Scroll** | Classic 4003 bridge coil with 4.7nF high-pass clank |
| `33in_rickenbacker_4003:pair_vintage` | `Rickenbacker Vintage` | `[Parallel]` | `Rickenbacker Vintage [Parallel] v7.1.1` | 31 / 38 | **Zero-Scroll** | Vintage 4001 dual pair with 4.7nF bridge HPF |
| `33in_rickenbacker_4003:neck_warm` | `Rickenbacker Neck` | `[Neck]` | `Rickenbacker Neck [Neck] v7.1.1` | 24 / 31 | **Zero-Scroll** | Solo 210mm neck single-coil at 24th-fret node |
| `34in_active_pmm:pmm_parallel` | `P∕MM Parallel` | `[Parallel]` | `P∕MM Parallel [Parallel] v7.1.1` | 24 / 31 | **Zero-Scroll** | Split-P + MM bridge (parallel coils) with LH-3 preamp |
| `34in_active_pmm:pmm_series` | `P∕MM Series` | `[Parallel]` | `P∕MM Series [Parallel] v7.1.1` | 22 / 29 | **Zero-Scroll** | Split-P + MM bridge (series coils) with LH-3 preamp |
| `30in_gibson_eb0:neck_deep` | `Mudbucker Deep` | `[Neck]` | `Mudbucker Deep [Neck] v7.1.1` | 21 / 28 | **Zero-Scroll** | Massive 18.0H coupled sidewinder neck sub-bass rumble |
| `30in_gibson_eb0:neck_baritone` | `Mudbucker Baritone` | `[Neck]` | `Mudbucker Baritone [Neck] v7.1.1` | 25 / 32 | **Zero-Scroll** | Gibson 2.5H choke + 22nF series cap midrange rock bark |
| `37in_multiscale_dingwall:bridge_clank` | `Dingwall Bridge` | `[Bridge]` | `Dingwall Bridge [Bridge] v7.1.1` | 24 / 31 | **Zero-Scroll** | High-tension fanned-fret progressive clarity |
| `37in_multiscale_dingwall:middle_growl` | `Dingwall Middle` | `[Neck]` | `Dingwall Middle [Neck] v7.1.1` | 22 / 29 | **Zero-Scroll** | Focused aggressive low-mid punch & vocal growl |
| `37in_multiscale_dingwall:pair_parallel` | `Dingwall Parallel` | `[Parallel]` | `Dingwall Parallel [Parallel] v7.1.1` | 28 / 35 | **Zero-Scroll** | Modern scooped aggression, ultra-fast transients |
| `41in_upright_bass:bridge_piezo` | `Upright Piezo` | `[Neck]` | `Upright Piezo [Neck] v7.1.1` | 20 / 27 | **Zero-Scroll** | Woody double-bass piezo bridge force transducer |

---

## 6. Filesystem Safety & Character Sanitization

To ensure model files are compatible with macOS, Linux, and Windows filesystems, Allomorph applies specific sanitization rules in `src/allomorph/naming.py`:

1. **Path Separator Replacement (`/` and `\`):**
   - Slashes in tone names (such as `P/MM`) must never become filesystem subdirectories.
   - Slashes are automatically replaced with the **Unicode Division Slash (`∕`, `\u2215`)**:
     ```python
     name = name.replace("/", "\u2215").replace("\\", "\u2215")
     ```
   - Result: `P/MM Parallel` $\to$ `P∕MM Parallel` (renders as a clean visual slash without triggering directory nesting).
2. **Arrow Notation Replacement (`->` and `-->`):**
   - Directional arrows in multi-source model names are automatically sanitized to the **Unicode Rightwards Arrow (`→`, `\u2192`)**:
     ```python
     name = name.replace("-->", "\u2192").replace("->", "\u2192")
     ```
   - Result: `Fretless J [Bridge] -> Upright` $\to$ `Fretless J [Bridge] → Upright`.
3. **Quotation Marks:**
   - Avoid unescaped ASCII double quotes (`"`).
4. **Prohibited Characters:**
   - Never use `:`, `*`, `?`, `<`, `>`, or `|`, as these are illegal in Windows filenames.

---

## 7. Do's and Don'ts for Agents & Contributors

### DO:
- **DO follow the 3-token grammar:** `[Family] [Configuration] [Voicing Modifier]`.
- **DO use permitted vocabulary:** Draw tokens strictly from Section 3.
- **DO verify character lengths:** Ensure the unversioned display name is $\le 34$ characters for zero scroll on stage, and the versioned `.nam` basename never exceeds 64 chars.
- **DO use standard physical switch tags for multi-pickup source basses:** `[Neck]`, `[Bridge]`, `[Parallel]`, `[Series]`, `[Split]`.
- **DO verify with automated tests:** Run `uv run pytest tests/pipeline/test_pack.py tests/pipeline/test_versioning.py` after any voice modification.

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
- **DO NOT include the source instrument name in single-instrument `tone_name`:**
  - ❌ `34in Standard P Precision Vintage v7.1.1.nam` (43 chars, marquee scrolls)
  - ✅ `Precision Vintage [Split] v7.1.1.nam` (32 chars, zero-scroll)
- **DO NOT invent arbitrary abbreviations:**
  - ❌ `P-Bss Vntg [Splt] v7.1.1.nam` (cryptic and unprofessional)
  - ✅ `Precision Vintage [Split] v7.1.1.nam`
- **DO NOT mix and match word order:**
  - ❌ `Active Modern Jazz Pair`
  - ✅ `Jazz Pair Active`
