# Allomorph Storefront Voicing Ranking Architecture

This document defines the ranking and ordering system governing the digital twin voicings in **Allomorph Tone Packs** on [Tone3000](https://tone3000.com).

For tone naming rules and character budget constraints, see [`naming.md`](naming.md). For the complete pack catalog, see [`catalog.md`](catalog.md).

---

## 1. Core Principles

The Allomorph ranking architecture is built on four fundamental tenets designed to maximize user engagement, storefront clarity, and musical value:

### 1.1 Inverse-Availability Principle (Native Identity Exclusion)
When a player browses or loads a tone pack for their physical bass, the primary value proposition is **transformational capacity**—giving the player sounds they **cannot** physically achieve with their stock instrument.
- **Rule:** 1:1 base-instrument identity duplicates (`is_identity_voicing`) where the target pickup geometry and circuit match the player's stock instrument voicing are **excluded** from the pack. Players do not need a neural network to emulate the exact physical pickup and circuit already installed in their guitar.
- **Example (Precision Bass):** Excludes `Precision Vintage` (`34in_standard_p:vintage_open`, since the player's bass already possesses this exact Alnico V split-coil CTS 250k circuit). Retains modified ToneStyler capacitive shunts (`Precision Mids`, `Precision Warm`, `Precision Dub`) and modern ceramic 500k configurations which deliver distinct electrical curves.
- **Example (Jazz Bass):** Excludes `Jazz Pair Open`, `Jazz Bridge Open`, and `Jazz Neck Warm` (matches stock dual single-coils, bridge single-coil, and neck single-coil).
- **Example (P/J Bass):** Excludes `PJ Passive`, `Precision Vintage`, and `Jazz Bridge Open` (matches the player's native 3-way switch positions).
- **Example (Active StingRay):** Excludes `StingRay Parallel` (matches the stock active bridge humbucker in parallel).

### 1.2 Popularity & Transformative Demand Weighting
Among the non-native/transformative voicings, tones are ranked according to **real-world musical demand and recording ubiquity**:
- **Tier 1 (Universal Workhorses):** Jazz Bass dual-single scooped slap and active Sadowsky NYC 2-band; Music Man StingRay active parallel bite and series punch.
- **Tier 2 (Modern Boutique & Rock Legends):** Lakland/Sandberg P∕MM active hybrids, Rickenbacker 4003 prog-rock clank with HPF and modern open bypass, Dingwall multi-scale modern metal clank (`Bridge`, `Middle`, `Parallel`).
- **Tier 3 (Specialty & Historical Archetypes):** Gibson EB-0 Mudbucker ultra-low series humbucker.
- **Tier 4 (Acoustic Transducers):** Upright Piezo double bass bridge saddle force sensor simulation.

### 1.3 Strict Modular Topology Separation
Storefront categories must accurately reflect analog electrical topologies and physical coil arrangements. Arbitrary "catch-all" groupings are strictly forbidden:
- **Music Man StingRay Family:** Pure 2-band active dual-coil humbuckers placed at the bridge sweet spot are strictly isolated from hybrid models.
- **P∕MM Modern Hybrids:** Compound dual-pickup configurations (Split-P neck + Music Man bridge) form their own dedicated hybrid category.
- **Progressive & Classic Rock Legends:** Historic and modern rock clank archetypes (Rickenbacker 4003, Dingwall, Gibson Mudbucker) are unified under a dedicated rock category.
- **Acoustic Transducers:** Piezo force sensor simulations are cleanly isolated from magnetic pickup guitars.

---

## 2. Edition-Specific Ranking Hierarchies

### 2.1 Standard P/J Bass Edition (21 Voicings)
*Source Instrument:* Physical Split-P neck + Single-coil J bridge ($34''$ scale).  
*Native Exclusions:* `PJ Passive`, `Precision Vintage`, and `Jazz Bridge Open` (`is_identity_voicing`).

| Rank Range | Sonic Family | Rationale / Musical Character |
| :--- | :--- | :--- |
| **01–04** | **Jazz Bass Family** | #1 most requested non-native sound: `Jazz Pair Active`, `Jazz Pair Open`, `Jazz Pair Mids`, `Jazz Bridge Growl`, plus `Jazz Neck Warm`. |
| **05–06** | **Music Man StingRay Family** | Active humbucker punch, hollow $2.5\text{ kHz}$ mid-scoop (`StingRay Parallel`), and series bark (`StingRay Series`). |
| **07–08** | **P∕MM Modern Hybrids** | Modern boutique session tone (Sandberg California / Lakland 44-02) combining Split-P weight with Music Man bridge authority (`P∕MM Parallel`, `P∕MM Series`). |
| **09–14** | **Progressive & Classic Rock Legends** | Aggressive pick clank and sub-bass weight: `Rickenbacker Clank`, `Rickenbacker Open`, `Dingwall Bridge`, `Dingwall Middle`, `Dingwall Parallel`, and `Mudbucker Deep`. |
| **15** | **Acoustic Transducers** | `Upright Piezo` bridge force transducer on $41.25''$ upright scale with velocity-to-force leaky integration. |
| **16–19** | **Precision Bass & Tone Shaper Family** | `Precision Active` and ToneStyler shunts (`Precision Mids`, `Precision Warm`, `Precision Dub`). |
| **20–21** | **P/J & Neck Foundations** | `PJ Active` (+4 dB @ 40 Hz & 4 kHz) and `Jazz Neck Warm`. |

---

### 2.2 Mustang P/J Bass Edition (24 Voicings)
*Source Instrument:* Short-scale ($30''$) physical Split-P neck + Single-coil J bridge.  
*Native Exclusions:* None (all $34''$ targets transform short-scale $30''$ string wave-speed continuum and pickup aperture geometry into full-scale targets).

| Rank Range | Sonic Family | Rationale / Musical Character |
| :--- | :--- | :--- |
| **01–06** | **Jazz Bass Family** | Reconstructs full $34''$ wide-aperture dual-single comb scoop (`Jazz Pair Active`, `Jazz Pair Open`, `Jazz Pair Mids`, `Jazz Bridge Growl`, `Jazz Bridge Open`, `Jazz Neck Warm`) from short scale. |
| **07–08** | **Music Man StingRay Family** | High-demand active humbucker punch (`StingRay Parallel`) and series bark (`StingRay Series`). |
| **09–10** | **P∕MM Modern Hybrids** | Split-P + Music Man bridge dual-coil in parallel (`P∕MM Parallel`) and series (`P∕MM Series`). |
| **11–16** | **Progressive & Classic Rock Legends** | `Rickenbacker Clank`, `Rickenbacker Open`, `Dingwall Bridge`, `Dingwall Middle`, `Dingwall Parallel`, and `Mudbucker Deep`. |
| **17** | **Acoustic Transducers** | `Upright Piezo` bridge force transducer on $41.25''$ upright scale. |
| **18–22** | **Precision Bass & Tone Shaper Family** | Full $34''$ scale `Precision Vintage`, `Precision Active`, `Precision Mids`, `Precision Warm`, and `Precision Dub`. |
| **23–24** | **P/J Bass Family** | `PJ Active` and `PJ Passive` in standard $34''$ scale tension. |

---

### 2.3 Standard Jazz Bass Edition (21 Voicings)
*Source Instrument:* Dual single-coils in 60s spacing ($155.6\text{ mm}$ neck, $63.5\text{ mm}$ bridge, $34''$ scale).  
*Native Exclusions:* `Jazz Pair Open`, `Jazz Bridge Open`, and `Jazz Neck Warm` (`is_identity_voicing`).

| Rank Range | Sonic Family | Rationale / Musical Character |
| :--- | :--- | :--- |
| **01–05** | **Precision Bass & Tone Shaper Family** | #1 most requested transformation for Jazz Bass players: `Precision Vintage`, `Precision Active`, `Precision Mids`, `Precision Warm`, and `Precision Dub`. |
| **06–07** | **Music Man StingRay Family** | `StingRay Parallel` and `StingRay Series` bridge humbucker punch and slap clank. |
| **08–09** | **P/J Bass Family** | `PJ Active` and `PJ Passive` combining P-bass body with bridge single-coil cut. |
| **10–11** | **P∕MM Modern Hybrids** | `P∕MM Parallel` and `P∕MM Series` for modern boutique versatility. |
| **12–17** | **Progressive & Classic Rock Legends** | `Rickenbacker Clank`, `Rickenbacker Open`, `Dingwall Bridge`, `Dingwall Middle`, `Dingwall Parallel`, and `Mudbucker Deep`. |
| **18** | **Acoustic Transducers** | `Upright Piezo` bridge force sensor on $41.25''$ upright scale. |
| **19–21** | **Jazz Bass Family** | Re-EQed NYC active 2-band (`Jazz Pair Active`), ToneStyler vocal mid bump (`Jazz Pair Mids`), and Jaco decoupled growl (`Jazz Bridge Growl`). |

---

### 2.4 Standard Precision Bass Edition (23 Voicings)
*Source Instrument:* Single split-coil P-Bass centered at $125.0\text{ mm}$ ($34''$ scale).  
*Native Exclusions:* `Precision Vintage` (`is_identity_voicing`).

| Rank Range | Sonic Family | Rationale / Musical Character |
| :--- | :--- | :--- |
| **01–06** | **Jazz Bass Family** | Top transformative capability: converts a single split-coil into dual-aperture 60s single-coils (`Jazz Pair Active`, `Jazz Pair Open`, `Jazz Pair Mids`, `Jazz Bridge Growl`, `Jazz Bridge Open`, `Jazz Neck Warm`). |
| **07–08** | **Music Man StingRay Family** | `StingRay Parallel` and `StingRay Series` bridge humbucker clank and punch. |
| **09–10** | **P/J Bass Family** | `PJ Active` and `PJ Passive` providing dual-pickup growl and articulation. |
| **11–12** | **P∕MM Modern Hybrids** | `P∕MM Parallel` and `P∕MM Series` for modern boutique versatility and punch. |
| **13–18** | **Progressive & Classic Rock Legends** | `Rickenbacker Clank`, `Rickenbacker Open`, `Dingwall Bridge`, `Dingwall Middle`, `Dingwall Parallel`, and `Mudbucker Deep`. |
| **19** | **Acoustic Transducers** | `Upright Piezo` bridge force sensor on $41.25''$ upright scale. |
| **20–23** | **Precision Bass & Tone Shaper Family** | `Precision Active` and ToneStyler shunts (`Precision Mids`, `Precision Warm`, `Precision Dub`). |

---

### 2.5 Preamp Soapbar Bass Edition (24 Voicings)
*Source Instrument:* Modern 34" preamp bass with dual passive Alnico dual-coil soapbars ($135.0\text{ mm}$ neck, $55.0\text{ mm}$ bridge, exposed round pole pieces) and 3-band active EQ buffer. Universally calibrated for 4, 5, and 6-string instruments.  
*Native Exclusions:* None (all 24 curated storefront targets transform the passive-soapbar + active-preamp topology into distinct iconic instruments).

| Rank Range | Sonic Family | Rationale / Musical Character |
| :--- | :--- | :--- |
| **01–05** | **Precision Bass & Tone Shaper Family** | #1 most requested transformation for preamp soapbar players: `Precision Vintage`, `Precision Active`, `Precision Mids`, `Precision Warm`, and `Precision Dub`. |
| **06–11** | **Jazz Bass Family** | Vintage 1960s and modern active Jazz voicings (`Jazz Pair Active`, `Jazz Pair Open`, `Jazz Pair Mids`, `Jazz Bridge Growl`, `Jazz Bridge Open`, `Jazz Neck Warm`). |
| **12–13** | **Music Man StingRay Family** | `StingRay Parallel` and `StingRay Series`. |
| **14–15** | **P∕MM Modern Hybrids** | `P∕MM Parallel` and `P∕MM Series`. |
| **16–21** | **Progressive & Classic Rock Legends** | `Rickenbacker Clank`, `Rickenbacker Open`, `Dingwall Bridge`, `Dingwall Middle`, `Dingwall Parallel`, and `Mudbucker Deep`. |
| **22** | **Acoustic Transducers** | `Upright Piezo` bridge force sensor on $41.25''$ upright scale. |
| **23–24** | **P/J Bass Family** | `PJ Active` and `PJ Passive`. |

---

### 2.6 Active StingRay Bass Edition (23 Voicings)
*Source Instrument:* 34" active Music Man StingRay / Sterling Ray34 / Ray35 (4 and 5-string) with single bridge sweet-spot dual-coil humbucker ($66.0\text{ mm}$) and active 2-band or 3-band preamp. Universally calibrated down to low B ($30.87\text{ Hz}$).  
*Native Exclusions:* `StingRay Parallel` (`is_identity_voicing`).

| Rank Range | Sonic Family | Rationale / Musical Character |
| :--- | :--- | :--- |
| **01–05** | **Precision Bass & Tone Shaper Family** | #1 missing sound for StingRay players: warm low-mid thump of a neck split-coil (`Precision Vintage`, `Precision Active`, `Precision Mids`, `Precision Warm`, `Precision Dub`). |
| **06–11** | **Jazz Bass Family** | Dual single-coil wide-aperture $1\text{ kHz}$ comb scoop, Marcus Miller slap sparkle, and Jaco bridge bite (`Jazz Pair Active`, `Jazz Pair Open`, `Jazz Pair Mids`, `Jazz Bridge Growl`, `Jazz Bridge Open`, `Jazz Neck Warm`). |
| **12–13** | **P∕MM Modern Hybrids** | `P∕MM Parallel` and `P∕MM Series`. |
| **14–15** | **P/J Bass Family** | `PJ Active` and `PJ Passive`. |
| **16–21** | **Progressive & Classic Rock Legends** | `Rickenbacker Clank`, `Rickenbacker Open`, `Dingwall Bridge`, `Dingwall Middle`, `Dingwall Parallel`, and `Mudbucker Deep`. |
| **22** | **Acoustic Transducers** | `Upright Piezo` bridge force sensor on $41.25''$ upright scale. |
| **23** | **Music Man StingRay Family** | `StingRay Series` bark (+2.5 dB gain, forward 2.1 kHz mids). |

---

### 2.7 Active EMG Bass Edition (24 Voicings)
*Source Instrument:* 34" active EMG soapbar bass (Spector / ESP LTD / Schecter) with dual EMG ceramic dual-blade soapbars ($135.0\text{ mm}$ neck, $55.0\text{ mm}$ bridge) and internal low-noise buffers calibrated under Option C ($f_r = 4.15\text{ kHz}, Q = 1.40$). Universally calibrated for 4, 5, and 6-string instruments.  
*Native Exclusions:* None (all 24 curated storefront targets transform active EMG dual-blade aperture and buffer response into iconic instruments).

| Rank Range | Sonic Family | Rationale / Musical Character |
| :--- | :--- | :--- |
| **01–05** | **Precision Bass & Tone Shaper Family** | #1 most requested transformation for active EMG players: `Precision Vintage`, `Precision Active`, `Precision Mids`, `Precision Warm`, and `Precision Dub`. |
| **06–11** | **Jazz Bass Family** | Vintage 1960s and modern active Jazz voicings (`Jazz Pair Active`, `Jazz Pair Open`, `Jazz Pair Mids`, `Jazz Bridge Growl`, `Jazz Bridge Open`, `Jazz Neck Warm`). |
| **12–13** | **Music Man StingRay Family** | `StingRay Parallel` and `StingRay Series`. |
| **14–15** | **P∕MM Modern Hybrids** | `P∕MM Parallel` and `P∕MM Series`. |
| **16–21** | **Progressive & Classic Rock Legends** | `Rickenbacker Clank`, `Rickenbacker Open`, `Dingwall Bridge`, `Dingwall Middle`, `Dingwall Parallel`, and `Mudbucker Deep`. |
| **22** | **Acoustic Transducers** | `Upright Piezo` bridge force sensor on $41.25''$ upright scale. |
| **23–24** | **P/J Bass Family** | `PJ Active` and `PJ Passive`. |

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
2. **Preserve Distinct Families:** Confirm StingRay is never merged with P/J, and P∕MM is never merged with Acoustic Transducers.
3. **Check Character Limits:** Ensure the total listing length strictly remains under 10,000 characters.
4. **Preserve Bracketed Tags:** Every multi-pickup listing must contain `[Parallel]`, `[Neck]`, or `[Bridge]` tags.
5. **Active Calibration Integrity:** Active editions must direct players to set onboard active EQs flat at center detents.
