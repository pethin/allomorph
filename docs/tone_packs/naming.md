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
| `Mudbucker` | Gibson EB-0 / EB-2 Sidewinder neck humbucker | Iconic bassist term for the 18.0H coupled neck pickup |
| `Hofner` | Höfner 500/1 fully hollow Violin Bass Staple humbuckers | ASCII-safe spelling of Höfner for LCD & filesystem compatibility |
| `Dingwall` | Multiscale fanned-fret angled dual-coil humbuckers | Represents progressive multiscale clarity |
| `Upright` | Double bass acoustic bridge piezo force transducer | Replaces generic `Piezo` |

### 3.2 Configuration Tokens (`[Configuration]`)
Specifies the active coil topology or pickup selection on the modeled target instrument:

| Configuration Token | Meaning | When to Use |
| :--- | :--- | :--- |
| `Pair` | Both pickups engaged in parallel or series | Used for dual-pickup basses (`Jazz Pair`, `Hofner Pair`) |
| `Bridge` | Solo bridge pickup | Used when isolating the bridge coil (`Jazz Bridge`, `Dingwall Bridge`) |
| `Neck` | Solo neck pickup | Used when isolating the neck coil (`Jazz Neck`, `Rickenbacker Neck`) |
| `Violin` | Höfner 500/1 hollow violin archtop neck Staple | Used for `Hofner Violin Warm` |
| `Parallel` | Dual-coil or dual-pickup wired in parallel | Used for dual-coil models (`StingRay Parallel`, `P∕MM Parallel`, `Dingwall Parallel`) |
| `Series` | Dual-coil or dual-pickup wired in series | Used for high-output series models (`StingRay Series`, `P∕MM Series`, `Dingwall Series`, `Jazz Pair Series`) |
| *(Omitted)* | Inherent single-pickup architecture | Inherent for `Precision` (always split-coil), `Mudbucker` (always neck Sidewinder), and `Upright` (always bridge piezo) |

### 3.3 Voicing Modifier Tokens (`[Voicing Modifier]`)
Describes the musical tone contour or circuit state. Replaces all raw electronic component values (`22nF`, `47nF`, `100nF`) and artist nicknames with musician terms:

| Modifier Token | Musical Sonic Function | Replaced Legacy Terms | Analog / Physical Basis |
| :--- | :--- | :--- | :--- |
| `Open` | Wide-open tone control with full harmonic air | `Pair`, `60s` | CTS 250k/500k pot wide open |
| `Series` | Series coil summation for thick humbucker roar | `S-1`, `4-Way` | Series-wired dual coils (`Jazz Pair Series`, `Dingwall Series`) |
| `51 Slab` | 1951–1956 single-coil Precision pickup | `Tele P`, `51 P` | 2.8H Alnico V single-coil on roundwound strings |
| `Warm` | Classic rolled-off flatwound / Motown or Beatle thump | `47nF`, `ToneStyler 47nF` | 47nF shunt cap + flatwounds (`Precision Warm`) or hollow Staple + flatwounds (`Hofner Violin Warm`) |
| `Dub` | Ultra-deep sub-bass roll-off; reggae/dub | `100nF`, `ToneStyler 100nF` | 100nF shunt capacitor removes all top-end for massive sub |
| `Vintage` | Alnico V vintage winding or 4001 cap-in pair | `Vintage 62`, `4001 Pair` | Alnico V split-coil or 4001 dual-pickup with 4.7nF bridge HPF |
| `Modern` | Ceramic magnet / high output; crisp bite | `Ceramic`, `Modern P` | High-inductance ceramic coil with 500k harness |
| `Active` | Onboard active 2-band/3-band preamp EQ | `Active Buffer`, `Modern Jazz` | Active preamp with boosted lows/highs & isolated coils |
| `Passive` | Passive high-impedance RLC loading feel | `Vintage PJ` | Authentic passive RLC network with 250k pots & 750pF cable |
| `Growl` | Bridge-biased blend for singing fretless tone | `Jaco Bridge Growl` | Decoupled volume pot blend (100% bridge, 99% neck) |
| `Clank` | Aggressive pick bite and high-pass clank | `4.7nF HPF`, `4003` | In-line series 4.7nF capacitor rolling off sub-bass |
| `Deep` | Maximum low-end solid-body sidewinder sub resonance | `Ultra Series` | 18.0H coupled sidewinder coil on solid-body EB-0 |
| `Hollow` | Semi-hollow maple archtop Sidewinder resonance | `EB-2 Deep` | 18.0H Sidewinder coupled with 165 Hz semi-hollow body bloom & sustain sink |
| `Baritone` | Gibson choke + series cap high-pass mid snarl | `EB-2 Baritone`, `Choke` | 15.0H `GA-90C-1` shunt choke + 20nF series cap on semi-hollow EB-2 |
| `Piezo` | Bridge force response & rocking compliance | `Transducer`, `Wing` | Bridge force sensor simulation on 41.25" hollow upright scale |
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

Below is the definitive table of the 26 curated utilitarian & mix-ready storefront target voicings across the 18 catalog instruments, showing character lengths and screen viewport status:

| Target (`<inst_id>:<voicing_id>`) | Declarative `tone_name` | Source Pos Tag | Full Model Basename (`.nam`) | Unversioned / Versioned Chars | Screen Status | Musical Personality / Bassist Context |
| :--- | :--- | :---: | :--- | :---: | :--- | :--- |
| `34in_standard_jazz:pair_active` | `Jazz Pair Active` | `[Parallel]` | `Jazz Pair Active [Parallel] v7.1.1` | 27 / 34 | **Zero-Scroll** | Active 2-band EQ '70s-spacing slap scoop; hi-fi lows & highs |
| `34in_standard_jazz:pair_open` | `Jazz Pair Open` | `[Parallel]` | `Jazz Pair Open [Parallel] v7.1.1` | 25 / 32 | **Zero-Scroll** | Classic '60s dual single-coils ($63.5\text{ mm}$ bridge), tone open |
| `34in_standard_jazz:pair_series` | `Jazz Pair Series` | `[Parallel]` | `Jazz Pair Series [Parallel] v7.1.1` | 27 / 34 | **Zero-Scroll** | Fender S-1 series-wired Jazz coils ($7.0\text{ H}$); thick humbucker roar |
| `34in_standard_jazz:bridge_growl` | `Jazz Bridge Growl` | `[Bridge]` | `Jazz Bridge Growl [Bridge] v7.1.1` | 26 / 33 | **Zero-Scroll** | Singing Jaco bridge burp & vocal growl with neck-pot decoupling |
| `34in_standard_jazz:neck_warm` | `Jazz Neck Warm` | `[Neck]` | `Jazz Neck Warm [Neck] v7.1.1` | 21 / 28 | **Zero-Scroll** | Warm, woody '60s jazz neck single-coil bloom |
| `34in_standard_p:vintage_open` | `Precision Vintage` | `[Neck]` | `Precision Vintage [Neck] v7.1.1` | 24 / 31 | **Zero-Scroll** | Classic '62 Alnico V split-P, CTS 250k open |
| `34in_standard_pj:neck_active` | `Precision Active` | `[Neck]` | `Precision Active [Neck] v7.1.1` | 23 / 30 | **Zero-Scroll** | Fender Ultra II active split-P with 18V 3-band preamp |
| `34in_dingwall_sp1:super_p_solo` | `Dingwall Super P` | `[Neck]` | `Dingwall Super P [Neck] v7.1.1` | 23 / 30 | **Zero-Scroll** | Dingwall SP1 $32''\text{--}35''$ passive Neodymium Dual-P ($3.0\text{ kHz}$ peak) |
| `34in_standard_p:vintage_warm` | `Precision Warm` | `[Neck]` | `Precision Warm [Neck] v7.1.1` | 21 / 28 | **Zero-Scroll** | Classic Motown / Jamerson heavy flatwound woody thump |
| `34in_standard_pj:pair_active` | `PJ Active` | `[Parallel]` | `PJ Active [Parallel] v7.1.1` | 20 / 27 | **Zero-Scroll** | Ultra II Split-P + J-bridge with 18V 3-band preamp |
| `34in_standard_pj:pair_parallel` | `PJ Passive` | `[Parallel]` | `PJ Passive [Parallel] v7.1.1` | 21 / 28 | **Zero-Scroll** | Organic P-thump with J-bridge articulation |
| `34in_active_stingray:parallel_classic` | `StingRay Parallel` | `[Bridge]` | `StingRay Parallel [Bridge] v7.1.1` | 26 / 33 | **Zero-Scroll** | Classic 2-band active Music Man authority |
| `34in_active_stingray:series_punch` | `StingRay Series` | `[Bridge]` | `StingRay Series [Bridge] v7.1.1` | 24 / 31 | **Zero-Scroll** | Muscular, mid-forward high-output MM series bark |
| `33in_rickenbacker_4003:bridge_clank` | `Rickenbacker Clank` | `[Bridge]` | `Rickenbacker Clank [Bridge] v7.1.1` | 27 / 34 | **Zero-Scroll** | Classic 4003 bridge coil with 4.7nF high-pass clank |
| `33in_rickenbacker_4003:pair_vintage` | `Rickenbacker Vintage` | `[Parallel]` | `Rickenbacker Vintage [Parallel] v7.1.1` | 31 / 38 | **Zero-Scroll** | Vintage 4001 dual pair with 4.7nF bridge HPF |
| `33in_rickenbacker_4003:neck_warm` | `Rickenbacker Neck` | `[Neck]` | `Rickenbacker Neck [Neck] v7.1.1` | 24 / 31 | **Zero-Scroll** | Solo 4003 Toaster/Hi-Gain neck at 24th-fret node ($210\text{ mm}$, $2.34\text{ kHz}$ peak) |
| `34in_active_pmm:pmm_parallel` | `P∕MM Parallel` | `[Parallel]` | `P∕MM Parallel [Parallel] v7.1.1` | 24 / 31 | **Zero-Scroll** | Split-P + MM bridge (parallel coils) with LH-3 preamp |
| `34in_active_pmm:pmm_series` | `P∕MM Series` | `[Parallel]` | `P∕MM Series [Parallel] v7.1.1` | 22 / 29 | **Zero-Scroll** | Split-P + MM bridge (series coils) with LH-3 preamp |
| `30in_gibson_eb0:neck_deep` | `Mudbucker Deep` | `[Neck]` | `Mudbucker Deep [Neck] v7.1.1` | 21 / 28 | **Zero-Scroll** | Solid-body EB-0 18.0H coupled Sidewinder sub-bass rumble |
| `30in_gibson_eb2:neck_deep` | `Mudbucker Hollow` | `[Neck]` | `Mudbucker Hollow [Neck] v7.1.1` | 23 / 30 | **Zero-Scroll** | Semi-hollow EB-2 Sidewinder with 165 Hz cavity bloom & woody decay |
| `30in_gibson_eb2:neck_baritone` | `Mudbucker Baritone` | `[Neck]` | `Mudbucker Baritone [Neck] v7.1.1` | 25 / 32 | **Zero-Scroll** | Semi-hollow EB-2 Baritone choke + cap midrange rock bark |
| `30in_hofner_500_1:neck_warm` | `Hofner Violin Warm` | `[Neck]` | `Hofner Violin Warm [Neck] v7.1.1` | 25 / 32 | **Zero-Scroll** | 1962 Höfner 500/1 hollow Violin Bass neck Staple Beatle thump |
| `37in_multiscale_dingwall:bridge_clank` | `Dingwall Bridge` | `[Bridge]` | `Dingwall Bridge [Bridge] v7.1.1` | 24 / 31 | **Zero-Scroll** | High-tension fanned-fret progressive clarity |
| `37in_multiscale_dingwall:pair_parallel` | `Dingwall Parallel` | `[Parallel]` | `Dingwall Parallel [Parallel] v7.1.1` | 28 / 35 | **Zero-Scroll** | Modern scooped aggression, ultra-fast transients |
| `37in_multiscale_dingwall:pair_series` | `Dingwall Series` | `[Parallel]` | `Dingwall Series [Parallel] v7.1.1` | 26 / 33 | **Zero-Scroll** | Rotary Pos 3 Bridge + Middle in series; massive fanned-fret wall of sound |
| `41in_upright_bass:bridge_piezo` | `Upright Piezo` | `[Neck]` | `Upright Piezo [Neck] v7.1.1` | 20 / 27 | **Zero-Scroll** | Woody double-bass piezo bridge force sensor with hollow sustain decay |

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
