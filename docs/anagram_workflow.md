# Darkglass Anagram Integration & Routing Architecture

This document provides a reference for deploying **Allomorph** IRs and NAM models onto the **Darkglass Anagram** pedalboard.

---

## 1. Optimal Block Layout (Architecture D Direct Single-Block Digital Twin)

The Darkglass Anagram allows up to 24 simultaneous blocks in series or parallel with 9 neural accelerator slots. Under Architecture D, Allomorph models pickup transformations into a single, high-fidelity neural front-end block:

```
[Hardware 1/4" Input]
          │  Peak calibrated to -6.0 to -3.0 dBFS on Anagram hardware meter
          ▼
┌────────────────────────────────────────────────────────┐
│ Block 1: Allomorph Digital Twin (NAM Preamp)           │  ◄── NAM Preamp Block (1 of 9 slots used)
│   └── "[Tone Name] [Position] v2.1.1.nam"              │      Captures target voicing, RLC resonance,
│   • End-to-end differential acoustic/circuit transform │      eddy currents, and non-linear magnetic feel
│   • Dynamic Lenz sag, back-EMF, & core hysteresis      │
│   • Calibrated RMS loudness matching                   │
└────────────────────────────────────────────────────────┘
          │
          ▼
┌────────────────────────────────────────────────────────┐
│ Block 2: Darkglass Drive / Preamp Engine               │  ◄── 8 Neural Slots Free!
│   ├── Microtubes B7K Ultra / Vintage Microtubes        │
│   └── Alpha·Omega / Microtubes Infinity                │
└────────────────────────────────────────────────────────┘
          │
          ▼
┌────────────────────────────────────────────────────────┐
│ Block 3: Speaker Cabinet Impulse Response (Cab IR)     │
│   └── Ampeg 8x10, Darkglass 4x10, or custom cab IR     │
└────────────────────────────────────────────────────────┘
          │
          ▼
┌────────────────────────────────────────────────────────┐
│ Block 4+: Time-Based Effects & Output Processing       │
│   └── Compression, Reverb, Chorus, Global EQ / Limiter │
└────────────────────────────────────────────────────────┘
          │
          ▼
[Stereo XLR Outputs to FOH / USB-C Audio]
```

> [!IMPORTANT]
> **The Golden Rule: Physical Knobs at 100% Wide Open**
> For Block 1 to perform an exact physical deconvolution ($0.00\text{ dB}$ flat identity baseline), keep your physical bass's volume and tone knobs completely wide open ($100\%$). Tone shaping (e.g. 22nF, 47nF Motown, 100nF Dub, Active preamp curves) is modeled authentically within the target digital twin.

> [!NOTE]
> **Single Neural Slot Economy:**
> By combining acoustic de-combing, RLC network transfer, and non-linear magnetic feel into a single slimmable A2 or nano model in Block 1, Allomorph uses only 1 of the Anagram's 9 neural slots. This leaves 8 neural slots completely free for drive engines, amplifier captures, and synths.
> For the complete table of training audio pairings, mathematical proofs, and CLI workflows, see [`docs/training.md`](training.md).

---

## 2. Hardware Input Gain Staging & Headroom Calibration

Active 18V EMG pickups provide immense dynamic headroom, delivering up to $+14\text{ dBu}$ of peak voltage on aggressive slap or heavy finger plucks. To preserve 100% linear conversion before neural processing in **Block 1 (Allomorph NAM)**:

1. **Adjust the Anagram Global Input Level / Pad:**
   * Navigate to the Anagram global I/O settings.
   * Play your most aggressive thumb slap and heavy finger strokes on the lowest string (Low B or E).
   * Adjust the analog input gain so that maximum peaks register cleanly between **$-6.0\text{ and } -3.0\text{ dBFS}$** on the hardware input meter.
2. **Prevent Hardware A/D Converter Clipping:**
   * Setting the input gain too hot will hard-clip the pedalboard's physical analog-to-digital converters before Block 1, introducing harsh inter-sample distortion that the NAM model cannot undo.
   * Setting the input gain too low will lower signal-to-noise ratio and prevent the model from engaging dynamic Alnico magnetic saturation.
3. **Downstream Processing in Block 2 (Preamp/Drive):**
   * Keeping input peaks at $-6.0\text{ to } -3.0\text{ dBFS}$ ensures the signal entering Block 1 mirrors the calibrated float window used during model training, allowing the downstream Darkglass drive in Block 2 (Microtubes B7K, Vintage Ultra) to distort organically.

---

## 3. Block 1 Level Trim & Perceived Loudness Matching

Passive and active multi-coil instruments often present notable level disparities:
* Splitting an MM humbucker to a single coil drops output level by $\sim 3\text{--}4\text{ dB}$.
* Switching pickups into Series configuration produces an inductive voltage surge of $+4\text{ to }+6\text{ dB}$.
* Active preamps provide $+3\text{ to }+5\text{ dB}$ of low/high shelving boost.

Allomorph models are calibrated with RMS loudness matching, bounded by a $-0.09\text{ dBFS}$ true-peak safety ceiling. On the pedalboard, you can fine-tune output level trim in Block 1:

| Voice Slug | Tone Name | Raw Offset | Recommended Block 1 Trim |
| :--- | :--- | :--- | :--- |
| **`jazz_pair_active`** | Jazz Pair Active | $+2.0\text{ dB}$ | $-2.0\text{ dB}$ (Controls active boost) |
| **`jazz_pair_open`** | Jazz Pair Open | $-0.5\text{ dB}$ | $+0.5\text{ dB}$ |
| **`jazz_pair_mids`** | Jazz Pair Mids | $-0.5\text{ dB}$ | $+0.5\text{ dB}$ |
| **`jazz_bridge_growl`**| Jazz Bridge Growl | $-1.2\text{ dB}$ | $+1.2\text{ dB}$ (Compensates pot decoupling) |
| **`jazz_bridge_open`** | Jazz Bridge Open | $-2.5\text{ dB}$ | $+2.5\text{ dB}$ (Compensates single-coil drop) |
| **`precision_active`** | Precision Active | $+1.5\text{ dB}$ | $-1.5\text{ dB}$ (Controls active preamp boost) |
| **`precision_vintage`**| Precision Vintage | $+0.5\text{ dB}$ | $+1.0\text{ dB}$ |
| **`precision_mids`** | Precision Mids | $+0.0\text{ dB}$ | $+1.0\text{ dB}$ |
| **`precision_warm`** | Precision Warm | $-1.0\text{ dB}$ | $+1.0\text{ dB}$ |
| **`precision_dub`** | Precision Dub | $-1.5\text{ dB}$ | $+1.5\text{ dB}$ |
| **`pj_active`** | PJ Active | $+2.0\text{ dB}$ | $-2.0\text{ dB}$ (Controls active boost) |
| **`pj_passive`** | PJ Passive | $+0.5\text{ dB}$ | $+0.5\text{ dB}$ |
| **`stingray_parallel`**| StingRay Parallel | $+2.5\text{ dB}$ | $-1.5\text{ dB}$ (Controls active boost) |
| **`stingray_series`** | StingRay Series | $+4.8\text{ dB}$ | $-3.5\text{ dB}$ (Controls series boost surge) |
| **`rickenbacker_clank`**| Rickenbacker Clank | $-1.5\text{ dB}$ | $+2.0\text{ dB}$ (Compensates series HPF cut) |
| **`p_mm_series`** | P∕MM Series | $+5.8\text{ dB}$ | $-4.0\text{ dB}$ (Prevents clipping downstream drives)|
| **`mudbucker_deep`** | Mudbucker Deep | $+6.2\text{ dB}$ | $-4.5\text{ dB}$ (Controls high-inductance surge) |
| **`dingwall_bridge`** | Dingwall Bridge | $+1.0\text{ dB}$ | $+0.5\text{ dB}$ |
| **`upright_acoustic`** | Upright Acoustic | $0.0\text{ dB}$ | $0.0\text{ dB}$ (Unity acoustic baseline; pair with 3 Sigma AST IRs) |
| **`studio_direct`** | Studio Direct | $0.0\text{ dB}$ | $0.0\text{ dB}$ (Transparent unity gain / dynamic feel) |
| **`studio_active`** | Studio Active | $+0.5\text{ dB}$ | $-0.5\text{ dB}$ (Unity gain active buffer) |
| **`studio_passive`** | Studio Passive | $+0.5\text{ dB}$ | $-0.5\text{ dB}$ (High-Z passive loading) |

---

## 4. Footswitching & Bank Organization

Group the pickup profiles into dedicated 3-button banks on the Anagram hardware:

### Bank 1: Modern & Vintage Jazz Foundations
* **Footswitch A:** `Jazz Pair Active [Parallel] v2.1.1.nam` (Modern Active Jazz - 2-Band Slap Scoop)
* **Footswitch B:** `Jazz Pair Open [Parallel] v2.1.1.nam` (Vintage 60s Jazz Bass Pair)
* **Footswitch C:** `Jazz Bridge Open [Bridge] v2.1.1.nam` (60s Jazz Bridge Single-Coil)

### Bank 2: Precision Bass Foundations
* **Footswitch A:** `Precision Active [Split] v2.1.1.nam` (Modern Ceramic Split-P - Active 2-Band Preamp)
* **Footswitch B:** `Precision Vintage [Split] v2.1.1.nam` (Vintage '62 Alnico V Split-P - Tone Open)
* **Footswitch C:** `Precision Warm [Split] v2.1.1.nam` (P-Bass 47nF Motown Flatwound Warmth)

### Bank 3: P/J Hybrid & Music Man Active
* **Footswitch A:** `PJ Active [Parallel] v2.1.1.nam` (Modern Active P/J)
* **Footswitch B:** `PJ Passive [Parallel] v2.1.1.nam` (Vintage Passive P/J)
* **Footswitch C:** `StingRay Parallel [Bridge] v2.1.1.nam` (Music Man Active 2-Band Humbucker Parallel)

### Bank 4: Series Punch & Maximum Inductance
* **Footswitch A:** `StingRay Series [Bridge] v2.1.1.nam` (Music Man Series Humbucker Mid-Punch)
* **Footswitch B:** `P∕MM Series [Series] v2.1.1.nam` (P/MM Series Sum)
* **Footswitch C:** `Mudbucker Deep [Neck] v2.1.1.nam` (Gibson Mudbucker Series)

### Bank 5: Multi-Scale, Acoustic & Vintage Filtered
* **Footswitch A:** `Rickenbacker Clank [Bridge] v2.1.1.nam` (Rickenbacker 4003 Bridge with HPF Clank)
* **Footswitch B:** `Dingwall Bridge [Bridge] v2.1.1.nam` (Dingwall Multi-Scale Bridge)
* **Footswitch C:** `Upright Acoustic v2.1.1.nam` (Upright Acoustic Bridge Transducer)

### Bank 6: Studio Voicings & Pure Dynamics
* **Footswitch A:** `Studio Direct v2.1.1.nam` (Pure Acoustic Aperture / Studio DI)
* **Footswitch B:** `Studio Active v2.1.1.nam` (Modern Active Buffer / Air Lift)
* **Footswitch C:** `Studio Passive v2.1.1.nam` (High-Z RLC & Cable Loading)

> [!TIP]
> **Studio Voicings Routing:**
> In Bank 6, `Studio Direct` preserves the instrument's exact physical aperture and applies only non-linear dynamic feel. `Studio Active` and `Studio Passive` apply true electrical circuit twins (active wideband buffer vs. 250k passive RLC loading) directly to the dry input.

---

## 4. Importing Files via Darkglass Suite

1. Connect your Darkglass Anagram to your computer via USB-C.
2. Open the **Darkglass Suite** application.
3. Navigate to the **NAM / Neural Capture** library.
4. Drag and drop trained `models/*.nam` captures into your user model library.
5. In your preset chain on the pedalboard, assign **Block 1** to your imported NAM capture.

---

## 5. Acoustic Upright Dual-Stage Architecture (Block 1 NAM + Block 3 3 Sigma AST IR)

When targeting an authentic upright double bass tone from a fretless or fretted electric bass (such as the 32" Fretless strung with **La Bella Low Tension Flats**), Allomorph splits the acoustic transformation into two specialized stages:

```
[32" Fretless Bass w/ La Bella LTF]
                 │
                 ▼
┌────────────────────────────────────────────────────────┐
│ Block 1: Allomorph NAM (`upright_acoustic`)            │
│   • Mathematical de-combing of EMG spatial aperture   │
│   • Leaky velocity-to-force integration (+6 dB/oct tilt)│
│   • Non-linear soft-knee bridge compliance (tanh)      │
│   • Subsonic stage rumble filter (32 Hz)               │
│   • Anti-double-damping deconvolution for flatwounds   │
└────────────────────────────────────────────────────────┘
                 │ (Pure simulated Realist/Underwood bridge force signal)
                 ▼
┌────────────────────────────────────────────────────────┐
│ Block 2: Transparent Acoustic Preamp / Optical Comp    │
│   • Subtle optical leveling (2:1 ratio)                │
└────────────────────────────────────────────────────────┘
                 │
                 ▼
┌────────────────────────────────────────────────────────┐
│ Block 3: Cab IR Loader (3 Sigma Upright Bass AST IR)   │
│   • Acoustic Sound Technology (AST) spruce top physics │
│   • Full 3/4 acoustic body & spruce soundboard cavity  │
│   • Helmholtz air resonance (~60 Hz)                   │
└────────────────────────────────────────────────────────┘
                 │
                 ▼
[Stereo XLR Outputs to FOH / Audio Interface]
```

### Why Both Blocks Are Necessary:
1. **An IR is Linear Time-Invariant (LTI):** It cannot deconvolve magnetic pickup comb notches, nor can it replicate the non-linear mechanical rocking of a double bass bridge under pizzicato attack. Feeding electric magnetic pickups straight into an acoustic IR sounds like an electric bass inside a hollow box.
2. **Block 1 Converts Pickup Physics:** Allomorph's NAM model transforms the magnetic velocity-sensing signal into a mechanical bridge force sensor, complete with dynamic compliance compression.
3. **Block 3 Radiates the Soundboard (3 Sigma AST):** Upright Piezo is engineered and recommended to be paired with **3 Sigma Audio Upright Bass AST (Acoustic Sound Technology) IRs**. The 3 Sigma AST impulse response receives the exact bridge force excitation it was designed for, radiating it through a resonant spruce soundboard and 3/4 double-bass body cavity.
4. **Anti-Double-Damping:** Because the 32" fretless is strung with La Bella Low Tension Flats, Allomorph's differential string engine automatically adjusts its acoustic damping curve, preventing the dull, muffled tone that occurs when a static acoustic low-pass filter is applied to already-dark flatwound strings.

> [!TIP]
> **Recommended 3 Sigma Audio IR Folder & File Tag:**
> In the official 3 Sigma Audio Upright Bass pack, select impulses from the **`Acoustic Upright Standard`** folder (identified by the **`AST`** file tag, e.g. `Upright Standard AST 1.wav`). 3 Sigma Audio notes:
> > *"Acoustic Upright Standard – IRs for Acoustic Upright Basses with a Standard Piezo Pickup. They are identified by the AST file tag. If your pickup is manufactured by a company not represented in any other folder, this is a great place to start."*
>
> Because Allomorph's Block 1 model synthesizes a canonical piezo bridge force transducer with direct buffer and leaky velocity integration, the `Acoustic Upright Standard` AST IR is the exact matching acoustic counterpart.

---

## 6. Future Milestone: Native Anagram Marketplace Block

While Allomorph currently deploys via the Anagram's stock Neural Amp Loader block, an upcoming roadmap goal is releasing an official **Anagram Marketplace Custom Block** (`marketplace.anagram.shop`):
* **All-in-One Voice Selector:** Instant rotary switching across all 21 pickup topologies and acoustic transducers directly within a single block.
* **Integrated Gain Normalization:** Automatically balances active boost, series boost, and single-coil level drop under the hood to ensure unity gain into Block 2.
* **Dynamic Control Emulation:** Real-time on-screen controls for volume pot loading ($500\text{ k}\Omega$ vs. $250\text{ k}\Omega$), active preamp boost, treble bleed networks, and cable capacitance ($750\text{ pF}$) loading.
