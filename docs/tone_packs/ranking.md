# Allomorph Storefront Voicing Ranking Architecture

This document defines the ranking and ordering system governing the digital twin voicings in **Allomorph Tone Packs** on [Tone3000](https://tone3000.com).

For tone naming rules and character budget constraints, see [`naming.md`](naming.md). For the complete pack catalog, see [`catalog.md`](catalog.md).

---

## 1. Core Principles

The Allomorph ranking architecture is built on four fundamental tenets designed to maximize user engagement, storefront clarity, and musical value:

### 1.1 Inverse-Availability Principle (Native Identity Exclusion)
When a player browses or loads a tone pack for their physical bass, the primary value proposition is **transformational capacity**—giving the player sounds they **cannot** physically achieve with their stock instrument.
- **Rule:** 1:1 base-instrument identity duplicates (`is_identity_voicing`) where the target pickup geometry, body construction (`body_type`), and circuit match the player's stock instrument voicing are **excluded** from the pack. Players do not need a neural network to emulate the exact physical pickup and circuit already installed in their guitar.
- **Example (Precision Bass):** Excludes `Precision Vintage` (`34in_standard_p:vintage_open`, since the player's bass already possesses this exact Alnico V split-coil CTS 250k circuit). Retains `Precision Active` (Ultra II 18V 3-band), `Dingwall Super P` (passive Neodymium Dual-P on $32''\text{--}35''$ multiscale), and `Precision Warm` (Jamerson heavy flatwounds + damped roll-off).
- **Example (Jazz Bass):** Excludes `Jazz Pair Open` and `Jazz Neck Warm` (matches stock 60s dual single-coils and neck single-coil), while retaining `Jazz Pair Active` ('70s $53.3\text{ mm}$ spacing + Sadowsky 2-band), `Jazz Pair Series` (S-1 series humbucker mode), and `Jazz Bridge Growl` (Jaco neck-pot decoupling).
- **Example (P/J Bass):** Excludes `PJ Passive` and `Precision Vintage` (matches the player's native passive P/J and solo P configurations).
- **Example (Active StingRay):** Excludes `StingRay Parallel` (matches the stock active bridge humbucker in parallel).

### 1.2 Popularity & Transformative Demand Weighting
Among the non-native/transformative voicings, tones are ranked according to **real-world musical demand and recording ubiquity**:
- **Tier 1 (Universal Workhorses):** Jazz Bass dual-single scooped slap and series punch (`Jazz Pair Active`, `Jazz Pair Open`, `Jazz Pair Series`, `Jazz Bridge Growl`, `Jazz Neck Warm`); Precision & Super P split-coils (`Precision Vintage`, `Precision Active`, `Dingwall Super P`, `Precision Warm`); Music Man StingRay active parallel bite and series punch.
- **Tier 2 (Modern Boutique & Rock Legends):** Lakland/Sandberg P∕MM active hybrids, Rickenbacker 4001/4003 prog-rock clank & 24th-fret toaster neck (`Rickenbacker Clank`, `Rickenbacker Vintage`, `Rickenbacker Neck`), Dingwall multi-scale modern metal clank (`Dingwall Bridge`, `Dingwall Parallel`, `Dingwall Series`).
- **Tier 3 (Specialty, Semi-Hollow & Hollow-Body Archetypes):** Solid-body Gibson EB-0 (`Mudbucker Deep`), semi-hollow 1958 Gibson EB-2 (`Mudbucker Hollow`, `Mudbucker Baritone`), and fully hollow 1962 Höfner 500/1 Violin Bass (`Hofner Violin Warm`).
- **Tier 4 (Acoustic Transducers):** `Upright Piezo` double bass bridge saddle force sensor simulation with hollow-archtop sustain absorption.

### 1.3 Strict Modular Topology Separation
Storefront categories must accurately reflect analog electrical topologies and physical coil arrangements. Arbitrary "catch-all" groupings are strictly forbidden:
- **Music Man StingRay Family:** Pure 2-band active dual-coil humbuckers placed at the bridge sweet spot are strictly isolated from hybrid models.
- **P∕MM Modern Hybrids:** Compound dual-pickup configurations (Split-P neck + Music Man bridge) form their own dedicated hybrid category.
- **Rickenbacker 4001/4003 Family:** 33.25" scale single-coil bridge HPF, dual-pair, and 24th-fret harmonic node neck models (`Rickenbacker Clank`, `Rickenbacker Vintage`, `Rickenbacker Neck`) form their own dedicated category.
- **Dingwall Multi-Scale Family:** 34"–37" fanned-fret active Neodymium FD3n bridge, series, and parallel models (`Dingwall Bridge`, `Dingwall Series`, `Dingwall Parallel`) form their own dedicated multi-scale category.
- **Gibson Mudbucker & Hollow-Body Family:** Solid-body and semi-hollow/hollow-archtop short-scale vintage legends (`Mudbucker Deep`, `Mudbucker Hollow`, `Mudbucker Baritone`, `Hofner Violin Warm`) form their own dedicated category.
- **Acoustic Transducers:** Piezo force sensor simulations are cleanly isolated from magnetic pickup guitars.

### 1.4 Utilitarian & Mix-Ready Storefront Selection Principle
Because Darkglass Anagram preset slots and Tone3000 pack listings must maximize practical stage and studio utility, every storefront target voicing must satisfy **both** the **Utilitarian Physics Test** and the **Mix-Ready Readiness Test**:

1. **Non-Invertible Physics or SNR-Preserving Bandwidth Extension (Utilitarian Physics Test):**
   - **Exclude** any voicing whose difference from another voicing in the same pack is merely a linear passive RC low-pass roll-off or a narrow spatial/electrical interpolation between neighboring voicings that the bassist can dial in with their onboard tone knob or a downstream EQ block (e.g., `Precision Dub` is omitted because `Precision Warm` already provides the non-linear heavy flatwound string physics, and `Precision 51 Slab` is omitted because its $\eta = 16.18\%$ / $2.48\text{ kHz}$ response sits directly between `Precision Vintage`, `Dingwall Super P`, and `Jazz Neck Warm`).
   - **Retain** voicings that provide **non-invertible electroacoustic physics**—multi-aperture comb filtering (`Jazz Pair Open`, `Jazz Pair Active`), $L/4$ 24th-fret harmonic node sensing (`Rickenbacker Neck` at $\eta = 24.87\%$), series vs. parallel coil inductance & core saturation (`Jazz Pair Series`, `StingRay Series`, `P∕MM Series`, `Dingwall Series`), passive Neodymium multi-scale split-P (`Dingwall Super P`), bridged-T LC choke networks (`Mudbucker Baritone`), string metallurgy transitions (`Precision Warm`), or semi-hollow/hollow-body mechanical admittance & dynamic sustain decay (`Mudbucker Hollow`, `Hofner Violin Warm`, `Upright Piezo`).
   - **Retain** voicings that **extend high-frequency bandwidth and transient reach without degrading SNR**—active internal buffers, 18V preamps (`Precision Active`, `Jazz Pair Active`, `PJ Active`, `P∕MM Parallel`), passive Neodymium low-inductance coils (`Dingwall Super P` at $3.05\text{ kHz}$), and $37''$ multiscale stainless wave-speed transformations (`Dingwall Bridge`, `Dingwall Parallel`, `Dingwall Series`). Because the NAM A2 neural network synthesizes high-frequency harmonic extension from non-linear string kinematics rather than applying linear high-shelf gain to analog noise, these models extend instrument frequency range with **zero noise-floor penalty**.

2. **Mix-Ready or Easily EQed to Mix-Ready (Mix-Ready Readiness Test):**
   - Every storefront target voicing must sit immediately in a live FOH or studio mix—or reach mix-ready perfection with a single broad EQ touch—without losing low-end fundamental authority or requiring complex corrective multi-band surgery.
   - Thin, un-blended solo bridge single-coils (`Jazz Bridge Open`) lack low-end fundamental support on stage and are replaced in storefront packs by their full-bodied, hum-canceling mix-ready counterparts: **`Jazz Bridge Growl`** (Jaco $81.3\% / 18.7\%$ bridge-biased blend at `neck_vol = 0.97` that fills in the midrange growl while retaining neck low-end weight), **`Jazz Pair Active`** (active '70s dual-coil pair with $53.3\text{ mm}$ bridge snap and deep Sadowsky 2-band bottom), and **`Jazz Pair Series`** (series-wired humbucker midrange punch).

---

## 2. Edition-Specific Ranking Hierarchies

### 2.1 Standard P/J Bass Edition (24 Voicings)
*Source Instrument:* Physical Split-P neck + Single-coil J bridge ($34''$ scale).  
*Native Exclusions:* `PJ Passive` and `Precision Vintage` (`is_identity_voicing`).

| Rank Range | Sonic Family | Rationale / Musical Character |
| :--- | :--- | :--- |
| **01–05** | **Jazz Bass Family** | #1 most requested non-native sound: `Jazz Pair Active`, `Jazz Pair Open`, `Jazz Pair Series`, `Jazz Bridge Growl`, and `Jazz Neck Warm`. |
| **06–07** | **Music Man StingRay Family** | Active humbucker punch, hollow mid-scoop (`StingRay Parallel`), and series bark (`StingRay Series`). |
| **08–09** | **P∕MM Modern Hybrids** | Modern boutique session tone (Sandberg California / Lakland 44-02) combining Split-P weight with Music Man bridge authority (`P∕MM Parallel`, `P∕MM Series`). |
| **10–12** | **Rickenbacker 4001/4003 Family** | `Rickenbacker Clank`, `Rickenbacker Vintage`, and `Rickenbacker Neck`. |
| **13–15** | **Dingwall Multi-Scale Family** | `Dingwall Bridge`, `Dingwall Series`, and `Dingwall Parallel`. |
| **16–19** | **Gibson Mudbucker & Hollow-Body Family** | `Mudbucker Deep`, `Mudbucker Hollow`, `Mudbucker Baritone`, and `Hofner Violin Warm`. |
| **20** | **Acoustic Transducers** | `Upright Piezo` bridge force transducer on $41.25''$ hollow upright scale. |
| **21–23** | **Precision Bass & Super P Family** | `Precision Active`, `Dingwall Super P`, and `Precision Warm`. |
| **24** | **P/J Bass Family** | `PJ Active` (Fender American Ultra II 18V 3-band). |

---

### 2.2 Mustang P/J Bass Edition (26 Voicings)
*Source Instrument:* Short-scale ($30''$) physical Split-P neck + Single-coil J bridge.  
*Native Exclusions:* None (all $34''$, $30.5''$, and hollow-body targets transform the solid-body $30''$ Mustang P/J into distinct instruments).

| Rank Range | Sonic Family | Rationale / Musical Character |
| :--- | :--- | :--- |
| **01–05** | **Jazz Bass Family** | Reconstructs full $34''$ wide-aperture dual-single comb scoop and series punch (`Jazz Pair Active`, `Jazz Pair Open`, `Jazz Pair Series`, `Jazz Bridge Growl`, `Jazz Neck Warm`) from short scale. |
| **06–07** | **Music Man StingRay Family** | High-demand active humbucker punch (`StingRay Parallel`) and series bark (`StingRay Series`). |
| **08–09** | **P∕MM Modern Hybrids** | Split-P + Music Man bridge dual-coil in parallel (`P∕MM Parallel`) and series (`P∕MM Series`). |
| **10–12** | **Rickenbacker 4001/4003 Family** | `Rickenbacker Clank`, `Rickenbacker Vintage`, and `Rickenbacker Neck`. |
| **13–15** | **Dingwall Multi-Scale Family** | `Dingwall Bridge`, `Dingwall Series`, and `Dingwall Parallel`. |
| **16–19** | **Gibson Mudbucker & Hollow-Body Family** | `Mudbucker Deep`, `Mudbucker Hollow`, `Mudbucker Baritone`, and `Hofner Violin Warm`. |
| **20** | **Acoustic Transducers** | `Upright Piezo` bridge force transducer on $41.25''$ hollow upright scale. |
| **21–24** | **Precision Bass & Super P Family** | Full $34''$/multiscale `Precision Vintage`, `Precision Active`, `Dingwall Super P`, and `Precision Warm`. |
| **25–26** | **P/J Bass Family** | `PJ Active` and `PJ Passive` in standard $34''$ scale tension. |

---

### 2.3 Standard Jazz Bass Edition (24 Voicings)
*Source Instrument:* Dual single-coils in 60s spacing ($155.6\text{ mm}$ neck, $63.5\text{ mm}$ bridge, $34''$ scale).  
*Native Exclusions:* `Jazz Pair Open` and `Jazz Neck Warm` (`is_identity_voicing`).

| Rank Range | Sonic Family | Rationale / Musical Character |
| :--- | :--- | :--- |
| **01–04** | **Precision Bass & Super P Family** | #1 most requested transformation for Jazz Bass players: `Precision Vintage`, `Precision Active`, `Dingwall Super P`, and `Precision Warm`. |
| **05–06** | **Music Man StingRay Family** | `StingRay Parallel` and `StingRay Series` bridge humbucker punch and slap clank. |
| **07–08** | **P/J Bass Family** | `PJ Active` and `PJ Passive` combining P-bass body with bridge single-coil cut. |
| **09–10** | **P∕MM Modern Hybrids** | `P∕MM Parallel` and `P∕MM Series` for modern boutique versatility. |
| **11–13** | **Rickenbacker 4001/4003 Family** | `Rickenbacker Clank`, `Rickenbacker Vintage`, and `Rickenbacker Neck`. |
| **14–16** | **Dingwall Multi-Scale Family** | `Dingwall Bridge`, `Dingwall Series`, and `Dingwall Parallel`. |
| **17–20** | **Gibson Mudbucker & Hollow-Body Family** | `Mudbucker Deep`, `Mudbucker Hollow`, `Mudbucker Baritone`, and `Hofner Violin Warm`. |
| **21** | **Acoustic Transducers** | `Upright Piezo` bridge force sensor on $41.25''$ hollow upright scale. |
| **22–24** | **Jazz Bass Family** | Non-native Jazz upgrades: NYC active '70s-spacing 2-band (`Jazz Pair Active`), S-1 series humbucker mod (`Jazz Pair Series`), and Jaco decoupled growl (`Jazz Bridge Growl`). |

---

### 2.4 Standard Precision Bass Edition (25 Voicings)
*Source Instrument:* Single split-coil P-Bass centered at $125.0\text{ mm}$ ($34''$ scale).  
*Native Exclusions:* `Precision Vintage` (`is_identity_voicing`).

| Rank Range | Sonic Family | Rationale / Musical Character |
| :--- | :--- | :--- |
| **01–05** | **Jazz Bass Family** | Top transformative capability: converts a single split-coil into 60s/70s dual-aperture and series Jazz voicings (`Jazz Pair Active`, `Jazz Pair Open`, `Jazz Pair Series`, `Jazz Bridge Growl`, `Jazz Neck Warm`). |
| **06–07** | **Music Man StingRay Family** | `StingRay Parallel` and `StingRay Series` bridge humbucker clank and punch. |
| **08–09** | **P/J Bass Family** | `PJ Active` and `PJ Passive` providing dual-pickup growl and articulation. |
| **10–11** | **P∕MM Modern Hybrids** | `P∕MM Parallel` and `P∕MM Series` for modern boutique versatility and punch. |
| **12–14** | **Rickenbacker 4001/4003 Family** | `Rickenbacker Clank`, `Rickenbacker Vintage`, and `Rickenbacker Neck`. |
| **15–17** | **Dingwall Multi-Scale Family** | `Dingwall Bridge`, `Dingwall Series`, and `Dingwall Parallel`. |
| **18–21** | **Gibson Mudbucker & Hollow-Body Family** | `Mudbucker Deep`, `Mudbucker Hollow`, `Mudbucker Baritone`, and `Hofner Violin Warm`. |
| **22** | **Acoustic Transducers** | `Upright Piezo` bridge force sensor on $41.25''$ hollow upright scale. |
| **23–25** | **Precision Bass & Super P Family** | `Precision Active`, `Dingwall Super P`, and `Precision Warm`. |

---

### 2.5 Preamp Soapbar Bass Edition (26 Voicings)
*Source Instrument:* Modern 34" preamp bass with dual passive Alnico dual-coil soapbars ($135.0\text{ mm}$ neck, $55.0\text{ mm}$ bridge, exposed round pole pieces) and 3-band active EQ buffer. Universally calibrated for 4, 5, and 6-string instruments.  
*Native Exclusions:* None (all 26 curated storefront targets transform the passive-soapbar + active-preamp topology into distinct iconic instruments).

| Rank Range | Sonic Family | Rationale / Musical Character |
| :--- | :--- | :--- |
| **01–04** | **Precision Bass & Super P Family** | #1 most requested transformation for preamp soapbar players: `Precision Vintage`, `Precision Active`, `Dingwall Super P`, and `Precision Warm`. |
| **05–09** | **Jazz Bass Family** | Vintage 1960s, series, and modern active '70s Jazz voicings (`Jazz Pair Open`, `Jazz Pair Active`, `Jazz Pair Series`, `Jazz Bridge Growl`, `Jazz Neck Warm`). |
| **10–11** | **Music Man StingRay Family** | `StingRay Parallel` and `StingRay Series`. |
| **12–13** | **P∕MM Modern Hybrids** | `P∕MM Parallel` and `P∕MM Series`. |
| **14–16** | **Rickenbacker 4001/4003 Family** | `Rickenbacker Clank`, `Rickenbacker Vintage`, and `Rickenbacker Neck`. |
| **17–19** | **Dingwall Multi-Scale Family** | `Dingwall Bridge`, `Dingwall Parallel`, and `Dingwall Series`. |
| **20–23** | **Gibson Mudbucker & Hollow-Body Family** | `Mudbucker Deep`, `Mudbucker Hollow`, `Mudbucker Baritone`, and `Hofner Violin Warm`. |
| **24** | **Acoustic Transducers** | `Upright Piezo` bridge force sensor on $41.25''$ hollow upright scale. |
| **25–26** | **P/J Bass Family** | `PJ Active` and `PJ Passive`. |

---

### 2.6 Active StingRay Bass Edition (25 Voicings)
*Source Instrument:* 34" active Music Man StingRay / Sterling Ray34 / Ray35 (4 and 5-string) with single bridge sweet-spot dual-coil humbucker ($66.0\text{ mm}$) and active 2-band or 3-band preamp. Universally calibrated down to low B ($30.87\text{ Hz}$).  
*Native Exclusions:* `StingRay Parallel` (`is_identity_voicing`).

| Rank Range | Sonic Family | Rationale / Musical Character |
| :--- | :--- | :--- |
| **01–04** | **Precision Bass & Super P Family** | #1 missing sound for StingRay players: warm low-mid thump of a neck pickup (`Precision Vintage`, `Precision Active`, `Dingwall Super P`, `Precision Warm`). |
| **05–09** | **Jazz Bass Family** | Dual single-coil wide-aperture comb scoop, NYC '70s slap sparkle, S-1 series punch, and Jaco growl (`Jazz Pair Active`, `Jazz Pair Open`, `Jazz Pair Series`, `Jazz Bridge Growl`, `Jazz Neck Warm`). |
| **10–11** | **P∕MM Modern Hybrids** | `P∕MM Parallel` and `P∕MM Series`. |
| **12–13** | **P/J Bass Family** | `PJ Active` and `PJ Passive`. |
| **14–16** | **Rickenbacker 4001/4003 Family** | `Rickenbacker Clank`, `Rickenbacker Vintage`, and `Rickenbacker Neck`. |
| **17–19** | **Dingwall Multi-Scale Family** | `Dingwall Bridge`, `Dingwall Parallel`, and `Dingwall Series`. |
| **20–23** | **Gibson Mudbucker & Hollow-Body Family** | `Mudbucker Deep`, `Mudbucker Hollow`, `Mudbucker Baritone`, and `Hofner Violin Warm`. |
| **24** | **Acoustic Transducers** | `Upright Piezo` bridge force sensor on $41.25''$ hollow upright scale. |
| **25** | **Music Man StingRay Family** | `StingRay Series` bark (+6.0 dB EMF, forward 2.19 kHz mids). |

---

### 2.7 Active EMG Bass Edition (26 Voicings)
*Source Instrument:* 34" active EMG soapbar bass (Spector / ESP LTD / Schecter) with dual EMG ceramic dual-blade soapbars ($135.0\text{ mm}$ neck, $55.0\text{ mm}$ bridge) and internal low-noise buffers calibrated under Option C ($f_r = 4.15\text{ kHz}, Q = 1.40$). Universally calibrated for 4, 5, and 6-string instruments.  
*Native Exclusions:* None (all 26 curated storefront targets transform active EMG dual-blade aperture and buffer response into iconic instruments).

| Rank Range | Sonic Family | Rationale / Musical Character |
| :--- | :--- | :--- |
| **01–04** | **Precision Bass & Super P Family** | #1 most requested transformation for active EMG players: `Precision Vintage`, `Precision Active`, `Dingwall Super P`, and `Precision Warm`. |
| **05–09** | **Jazz Bass Family** | Vintage 1960s, series, and modern active '70s Jazz voicings (`Jazz Pair Open`, `Jazz Pair Active`, `Jazz Pair Series`, `Jazz Bridge Growl`, `Jazz Neck Warm`). |
| **10–11** | **Music Man StingRay Family** | `StingRay Parallel` and `StingRay Series`. |
| **12–13** | **P∕MM Modern Hybrids** | `P∕MM Parallel` and `P∕MM Series`. |
| **14–16** | **Rickenbacker 4001/4003 Family** | `Rickenbacker Clank`, `Rickenbacker Vintage`, and `Rickenbacker Neck`. |
| **17–19** | **Dingwall Multi-Scale Family** | `Dingwall Bridge`, `Dingwall Parallel`, and `Dingwall Series`. |
| **20–23** | **Gibson Mudbucker & Hollow-Body Family** | `Mudbucker Deep`, `Mudbucker Hollow`, `Mudbucker Baritone`, and `Hofner Violin Warm`. |
| **24** | **Acoustic Transducers** | `Upright Piezo` bridge force sensor on $41.25''$ hollow upright scale. |
| **25–26** | **P/J Bass Family** | `PJ Active` and `PJ Passive`. |

---

## 3. Multi-Source Collection Architecture

For curated collections combining multiple physical instruments (e.g., Fretless-to-Upright packs where fretless Jazz, fretless active Jazz, and fretless Precision basses all target the Upright double bass sound), the ranking architecture shifts from target family grouping to **source instrument and configuration progression**:

1. **Source Grouping:** Bundles are grouped by source instrument archetype (e.g. Fretless Jazz Bass, Fretless Precision Bass).
2. **Control Position Progression:** Within each source instrument, voicings progress from bridge-biased fingerstyle (`[Bridge]`), through dual-pickup woody acoustic bloom (`[Parallel]`), to warm neck thump (`[Neck]`).
3. **Bespoke Product Copy:** Since each target transformation maps a distinct source guitar into the common target acoustic voice, the storefront narrative describes how each source instrument's pickup placement and fretless mwah translate into the target acoustic body resonance.

---

## 4. Governance & Quality Checklist

When modifying or generating new Tone3000 pack listings:
1. **Verify Native Identity Exclusion:** Confirm that $1:1$ identical base-instrument configurations (`is_identity_voicing`) are omitted.
2. **Verify Utilitarian & Mix-Ready Compliance:** Confirm that every target voicing provides non-invertible electroacoustic physics or SNR-preserving bandwidth extension, and sits mix-ready (or is easily EQed to mix-ready) with full low-end authority.
3. **Preserve Distinct Families:** Confirm StingRay is never merged with P/J, and P∕MM is never merged with Acoustic Transducers.
4. **Check Character Limits:** Ensure the total listing length strictly remains under 10,000 characters.
5. **Preserve Bracketed Tags:** Every multi-pickup listing must contain `[Parallel]`, `[Neck]`, or `[Bridge]` tags.
6. **Active Calibration Integrity:** Active editions must direct players to set onboard active EQs flat at center detents.
