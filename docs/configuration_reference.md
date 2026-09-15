# Allomorph Configuration Reference

This guide provides a comprehensive specification of all configuration files in **Allomorph**, their schemas, data types, physical units, mathematical implications, and constraints.

---

## 1. Overview of Configuration Files

Allomorph organizes instrument models, target voices, and physical scale wave speeds under the `config/` directory:

```
config/
├── instruments/              # Unified instrument catalog & native target voicings
│   ├── 30in_emg_mmtw.toml    # 30" Short scale with EMG MMTW dual-mode pickup
│   ├── 30in_mustang_pj.toml  # 30" Short scale Fender Mustang Bass PJ (passive split-P + single J)
│   ├── 32in_custom_pmm.toml  # 32" Medium scale with Reverse PX + MMTWX + ABCX active blend
│   ├── 32in_fretless_pmm.toml # 32" Fretless Medium scale with PCSX + MMTWX
│   ├── 33in_rickenbacker_4003.toml # 33.25" Rickenbacker 4003 stereo Rick-O-Sound
│   ├── 34in_standard_p.toml  # 34" Standard Fender Precision Bass (passive datum)
│   ├── 34in_standard_jazz.toml # 34" Standard Fender Jazz Bass (passive datum)
│   ├── 34in_standard_pj.toml # 34" Standard P/J Bass (Fender PJ / Yamaha BB style)
│   ├── 34in_active_stingray.toml # 34" Standard Active StingRay (Music Man MM)
│   ├── 34in_preamp_soapbar.toml  # 34" Standard Preamp Dual-Soapbar (Ibanez SR / Yamaha TRBX / Sire F10)
│   ├── 34in_active_emg.toml      # 34" Standard Active EMG Bass (Option C Baseline)
│   ├── 34in_dingwall_sp1.toml    # 32"-35" Dingwall SP1 5-String (Dual-P + FD3n)
│   ├── 37in_multiscale_dingwall.toml # 34"-37" Multi-Scale Dingwall 5-String Combustion / NG (FD3n)
│   ├── 41in_upright_bass.toml    # 41.5" Orchestral 3/4 Double Bass (Piezo Bridge Transducer)
│   └── studio_direct.toml        # Studio Direct / Active Buffer / Passive RLC baseline
├── preamps.toml              # Reusable active preamp catalog (Sadowsky, StingRay, Aguilar, Dingwall)
├── scales.toml               # Physical scale lengths, wave speeds, and string dispersion
└── strings.toml              # Physical string core/wrap presets
```

---

## 2. Instrument Configuration Schema (`config/instruments/*.toml`)

An instrument configuration represents a **physical source bass** whose active signal is fed into Passivizer. It defines the instrument's vibrating scale length, string wave speeds, onboard pickups, physical coil locations, and the mapping from target voices to physical switch positions.

### Root Table Parameters

| Field | Type | Units | Required | Description |
| :--- | :--- | :--- | :---: | :--- |
| `id` | `string` | — | **Yes** | Unique identifier (e.g. `"30in_emg_mmtw"`). Used by CLI `--instrument <id>`. |
| `name` | `string` | — | **Yes** | Human-readable label (e.g. `"30\" Short Scale MM (EMG MMTW)"`). Embedded in NAM metadata. |
| `scale_length_in` | `float` | Inches | **Yes** | Vibrating string scale length in inches (e.g. `30.0`, `32.0`, `34.0`). |
| `scale_length_m` | `float` | Meters | **Yes** | Scale length in meters ($L_{\text{m}} = L_{\text{in}} \times 0.0254$). |
| `string_wave_speeds`| `array[float]`| m/s | **Yes** | Array of 4 (or 5) wave speeds from low to high string ($v = 2 \cdot L \cdot f_0$). |
| `default_pickup` | `string` | — | **Yes** | Pickup key within `[pickups]` used when no voice mapping or override is provided. |

#### Example:
```toml
id = "30in_emg_mmtw"
name = "30\" Short Scale MM (EMG MMTW)"
scale_length_in = 30.0
scale_length_m = 0.762
string_wave_speeds = [62.79, 83.82, 111.89, 149.35]
default_pickup = "mmtw_dual"
```

---

### Pickup Definitions (`[pickups.<pickup_id>]`)

Each entry defines a **physically selectable state** on the instrument (e.g., solo neck pickup, solo bridge dual-coil, single-coil split mode, or center-detent active blend).

> [!IMPORTANT]
> **Active Electronics Constraint:**
> All defined pickups must reflect **physically real switch/potentiometer states** on the actual instrument. For example:
> * Active pickups (EMG) cannot be wired in series. Active blends (e.g. EMG ABCX) operate strictly in **parallel**.
> * An EMG MMTW push/pull switch activates coils **L1 + L2** (dual-coil) or **L2 + L3** (bridge single-coil). There is no neck-coil-only mode.

| Field | Type | Units | Default | Description |
| :--- | :--- | :--- | :---: | :--- |
| `name` | `string` | — | Required | Descriptive label of the pickup / switch mode. |
| `position_from_bridge_m` | `float` | Meters | Required | Centerline distance from bridge saddle ($x = \text{datum}_{\text{mm}} / 1000$). |
| `aperture_width_in` | `float` | Inches | Required | Total magnetic sensing aperture width ($w$). |
| `coil_spacing_in` | `float` | Inches | `0.0` | Center-to-center distance ($d$) between dual coils. `0.0` for single-coils. |
| `type` | `string` | — | Required | Pickup architecture: `"single_coil"`, `"dual_coil_parallel"`, `"split_coil"`, or `"composite"`. |
| `pole_type` | `string` | — | `"rod"` | Spatial pole geometry: `"rod"` (2D cylindrical pole disc) or `"blade"` (1D bar slit). |
| `magnet_type` | `string` | — | `"alnico_v"` | Core magnet alloy: `"alnico_v"`, `"alnico_ii"`, `"alnico_iii"`, `"ceramic"`, `"hybrid"`, `"neodymium"`, `"piezo"`, `"active"`, or `"ideal"` (pure linear reference). |
| `circuit` | `table` | — | Optional | Embedded declarative SPICE netlist table (`[pickups.<id>.circuit]`) defining RLC parameters. |
| `resonant_frequency_hz` | `float` | Hz | Optional | Internal electrical resonant peak frequency ($f_r$) of the active preamp. |
| `q_factor` | `float` | — | `1.35` | Quality factor ($Q$) of the internal active resonant bump. |
| `coils` | `array[table]`| — | Optional | Array of individual physical coils for precise multi-coil/staggered acoustic modeling. |
| `components` | `array[table]`| — | Optional | Array of sub-pickups for active parallel blends (used when `type = "composite"`). |

---

### Sub-Coil Modeling (`coils = [...]`)

When a pickup consists of multiple or staggered coils (such as a split-coil Precision Bass or a dual-coil Music Man), the `coils` array models the spatial standing-wave envelope for each coil individually.

| Field | Type | Units | Default | Description |
| :--- | :--- | :--- | :---: | :--- |
| `strings` | `array[string\|int]` | — | `["all"]` | String bindings for this coil half: `["all"]`, `["E", "A"]`, `["D", "G"]`, or register halves `[1, 2]` (treble) and `[3, 4]` (bass). |
| `position_from_bridge_m` | `float` | Meters | Required | Physical distance from bridge saddle to this individual coil center. |
| `aperture_width_in` | `float` | Inches | Required | Magnetic aperture width of this specific coil. |
| `weight` | `float` | — | `1.0` | Amplitude contribution (e.g. `0.5` for two coils in parallel). |
| `polarity` | `float` | — | `1.0` | Phase polarity (`1.0` for in-phase, `-1.0` for reverse phase). |
| `pole_type` | `string` | — | `"rod"` | Coil-specific spatial geometry override: `"rod"` or `"blade"`. |

#### Example: EMG MMTW Dual-Coil vs. Single-Coil
```toml
[pickups.mmtw_dual]
name = "EMG MMTW Dual-Coil (Centerline)"
position_from_bridge_m = 0.0775
aperture_width_in = 1.50
coil_spacing_in = 0.90
type = "dual_coil_parallel"
resonant_frequency_hz = 2500.0
q_factor = 1.35
coils = [
    { strings = ["all"], position_from_bridge_m = 0.08893, aperture_width_in = 0.75, weight = 0.5 },
    { strings = ["all"], position_from_bridge_m = 0.06607, aperture_width_in = 0.75, weight = 0.5 }
]

[pickups.mmtw_single]
name = "EMG MMTW Single-Coil (Bridge Coil)"
position_from_bridge_m = 0.06607
aperture_width_in = 0.75
coil_spacing_in = 0.0
type = "single_coil"
resonant_frequency_hz = 3500.0
q_factor = 1.40
coils = [
    { strings = ["all"], position_from_bridge_m = 0.06607, aperture_width_in = 0.75, weight = 1.0 }
]
```

#### Example: Reverse Split-Coil Precision Pickup
```toml
[pickups.px]
name = "Reverse EMG PX Split-Coil (Neck)"
position_from_bridge_m = 0.1228
aperture_width_in = 1.10
coil_spacing_in = 0.0
type = "split_coil"
resonant_frequency_hz = 3200.0
q_factor = 1.40
coils = [
    # D/G coil is staggered further towards the neck (136.8mm)
    { strings = ["D", "G"], position_from_bridge_m = 0.1368, aperture_width_in = 1.10, weight = 1.0 },
    # E/A coil is staggered closer towards the bridge (108.8mm)
    { strings = ["E", "A"], position_from_bridge_m = 0.1088, aperture_width_in = 1.10, weight = 1.0 }
]
```

---

### Composite Active Blends (`type = "composite"`)

For instruments with an active blend control (like the **EMG ABCX**), the blend is modeled as an active parallel mix of individual pickups defined within the same file.

| Field | Type | Description |
| :--- | :--- | :--- |
| `components` | `array[table]` | List of sub-pickups and their blend proportions. |
| `components[i].pickup` | `string` | ID of the sub-pickup defined in `[pickups]`. |
| `components[i].weight` | `float` | Weighting factor (e.g. `0.5` for center detent). |

#### Example:
```toml
[pickups.blend_parallel]
name = "EMG PX + MMTWX Parallel (Center Detent)"
position_from_bridge_m = 0.0868
aperture_width_in = 0.88
coil_spacing_in = 0.0
type = "composite"
components = [
    { pickup = "px", weight = 0.5 },
    { pickup = "mmtwx_dual", weight = 0.5 }
]
```

---

### Pickup Affinity Routing (`[pickup_mapping]`)

The `[pickup_mapping]` table routes the 4 canonical position affinities (`neck`, `bridge`, `parallel`, `direct`) to the optimal physical pickup switch position on the player's instrument:

```toml
[pickup_mapping]
neck = "split_p"
bridge = "split_p"
parallel = "split_p"
direct = "split_p"
```

For multi-pickup instruments (such as a Jazz Bass or P/MM), each affinity maps to the corresponding physical coil configuration:
```toml
[pickup_mapping]
neck = "neck"
bridge = "bridge"
parallel = "pair_parallel"
direct = "neck"
```

---

## 3. Scale Lengths & Wave Speeds (`config/scales.toml`)

`config/scales.toml` defines the standard physical scales used for target acoustic scaling and wave speed calculations:

$$v_s = 2 \cdot L \cdot f_{0,s}$$

| Key | Type | Units | Description |
| :--- | :--- | :--- | :--- |
| `name` | `string` | — | Display label of the scale standard. |
| `scale_length_in` | `float` | Inches | Vibrating string length. |
| `scale_length_m` | `float` | Meters | Vibrating string length in meters. |
| `string_wave_speeds` | `array[float]` | m/s | Array of wave speeds for standard bass tuning ($E_1=41.2\text{ Hz}$, $A_1=55.0\text{ Hz}$, $D_2=73.4\text{ Hz}$, $G_2=98.0\text{ Hz}$). |

#### Supported Scales:
* **`30in`:** Short scale ($L = 0.762\text{ m}$, $v = [62.79, 83.82, 111.89, 149.35]\text{ m/s}$)
* **`32in`:** Medium scale ($L = 0.8128\text{ m}$, $v = [66.98, 89.41, 119.35, 159.31]\text{ m/s}$)
* **`34in`:** Standard long scale ($L = 0.8636\text{ m}$, $v = [71.16, 95.00, 126.81, 169.27]\text{ m/s}$)
* **`multiscale`:** Fanned-fret Dingwall scale ($34''\text{--}37''$, $L = 0.9398\text{ m}$, $v = [77.44, 98.50, 131.00, 169.27]\text{ m/s}$)
* **`multiscale_super`:** Compact fanned-fret Dingwall SP1 5-string ($32''\text{--}35''$, $L = 0.889\text{ m}$, $v = [54.88, 71.69, 93.60, 122.14, 159.31]\text{ m/s}$)
* **`upright`:** Standard 3/4 acoustic double bass ($41.5'' = 1.0541\text{ m}$, $v = [86.86, 115.95, 154.76, 206.59]\text{ m/s}$)

---

## 4. Target Voice Definitions (Native Instrument Voicings & Global VOICES Registry)

In modern Allomorph (v0.3.0 / DSP Gen 3), target voices are defined directly within the Unified Instrument Catalog (`config/instruments/*.toml`) under `[voicings.<id>]`. The `allomorph.config.voices` registry dynamically resolves these into validated `VoiceConfig` models. This ensures every target voice is rooted in an authentic, physically verified instrument with declarative SPICE circuits, precise coil apertures, and string setups:

| Field | Type | Units | Description |
| :--- | :--- | :--- | :--- |
| `name` | `string` | — | Descriptive title (e.g. `"Vintage 1962 Open"`). |
| `tone_name` | `string` | — | Standardized 2-to-3 token musician label (e.g. `"Precision Vintage"`). |
| `pickup` | `string` | Key | Physical pickup on this instrument generating the voice (`"split_p"`, `"bridge"`, etc.). |
| `affinity` | `string` | — | Position affinity for bundle partitioning: `"neck"`, `"bridge"`, `"parallel"`, or `"direct"`. |
| `circuit` | `table` | — | *(Optional)* Embedded declarative SPICE netlist override (e.g. pots, tone cap, active buffer). |
| `vol_pos` | `float` | $[0.0, 1.0]$ | Volume potentiometer wiper position (default: `1.0`). |
| `tone_pos` | `float` | $[0.0, 1.0]$ | Tone potentiometer wiper position (default: `1.0`). |
| `tone_cap_f` | `float` | Farads | Tone capacitor value in Farads (e.g. `4.7e-8` for 47nF). |
| `string_preset_override` | `string` | Key | Goal string preset from `config/strings.toml` (e.g. `"flatwound_vintage_heavy"`). |
| `gain_db` | `float` | dB | Saturation drive boost/cut in dB into non-linear magnetic saturation/compression, and informational visualizer vertical offset. Does not alter training level normalization. |
| `sensor_type` | `string` | `"magnetic"` | Physical sensor taxonomy: `"magnetic"`, `"bridge_force"`, or `"direct"`. |
| `preserve_aperture` | `bool` | `false` | Set `true` to preserve source instrument physical aperture (e.g. Studio Voicings). |

### Native Voicing Examples

#### 1. Standard P-Bass Vintage Open (`config/instruments/34in_standard_p.toml`) & Active P/J (`config/instruments/34in_standard_pj.toml`)
```toml
[voicings.vintage_open]
name = "Vintage 1962 Open"
tone_name = "Precision Vintage"
pickup = "split_p"
affinity = "neck"
vol_pos = 1.0
tone_pos = 1.0
tone_cap_f = 4.7e-08
string_preset_override = "roundwound_nickel_standard"

# In config/instruments/34in_standard_pj.toml:
[voicings.neck_active]
name = "Solo Modern Active Split P"
tone_name = "Precision Active"
pickup = "p_active"
affinity = "neck"
vol_pos = 1.0
tone_pos = 1.0
gain_db = 1.5
preamp_preset = "sadowsky_2band"
```

#### 2. Jazz Bass Pair Active & Open (`config/instruments/34in_standard_jazz.toml`)
```toml
[voicings.pair_open]
name = "Vintage 1960s Pair Open"
tone_name = "Jazz Pair Open"
pickup = "pair_parallel"
affinity = "parallel"
vol_pos = 1.0
tone_pos = 1.0
tone_cap_f = 4.7e-08
string_preset_override = "roundwound_nickel_standard"

[voicings.pair_active]
name = "Modern Active Jazz Pair"
tone_name = "Jazz Pair Active"
pickup = "pair_parallel"
affinity = "parallel"
magnet_type = "alnico_v"
gain_db = 1.0

[voicings.pair_active.circuit]
topology = "parallel"
active = true
preamp = "sadowsky_2band"
Rvol = 500000.0
```

---

## 5. Physical String Catalog Reference (`config/strings.toml`)

`config/strings.toml` defines the mechanical, viscoelastic, and acoustic properties of physical string sets. These parameters govern high-frequency mechanical damping, inharmonicity/tension class, dynamic bridge compliance, and low-frequency body bloom:

| Field | Type | Units | Description |
| :--- | :--- | :--- | :--- |
| `name` | `string` | — | Full human-readable display label (e.g. `"La Bella Low Tension Flats LTF-4A"`). |
| `type` | `string` | — | Construction classification: `"roundwound"`, `"flatwound"`, `"double_bass"`. |
| `wrap` | `string` | — | Outer wrap alloy: `"nickel"`, `"stainless"`, `"stainless_flat"`, `"chrome_steel"`. |
| `core` | `string` | — | Core wire geometry: `"hex"`, `"round"`, `"spiral_rope"`. |
| `tension_lbs` | `float` | lbs | Total 4-string set tension at pitch. |
| `damping_cutoff_hz` | `float` | Hz | Viscoelastic high-frequency mechanical roll-off corner frequency ($f_d$). |
| `damping_order` | `float` | — | High-frequency damping filter order ($n$). |
| `k_long` | `float` | — | Longitudinal core wire percussive clank coupling factor ($0.00\text{ to }0.35$). |

### Built-In String Presets:

1. **`roundwound_nickel_standard` (Global Default Baseline):**
   * Standard D'Addario EXL / Ernie Ball Slinky $.045\text{--}.105$.
   * $155.0\text{ lbs}$ tension, $f_d = 8500\text{ Hz}, n = 1.0, k_{\text{long}} = 0.20$.
2. **`roundwound_nickel_6string` (Extended 6-String Baseline):**
   * Universal 6-string D'Addario EXL170-6 / Ernie Ball Slinky $.032\text{--}.130$.
   * $230.0\text{ lbs}$ tension, $f_d = 8500\text{ Hz}, n = 1.0, k_{\text{long}} = 0.20$.
3. **`roundwound_stainless_clank` (Multi-Scale / Dingwall):**
   * Dingwall Custom $.045\text{--}.130$ high-tension stainless steel.
   * $180.0\text{ lbs}$ tension, $f_d = 12000\text{ Hz}, n = 1.0, k_{\text{long}} = 0.35$ (massive metallic clank).
4. **`flatwound_low_tension` (Smooth Fretless Thump):**
   * La Bella Low Tension Flats LTF-4A $.043\text{--}.100$ round core.
   * $132.0\text{ lbs}$ low tension, $f_d = 2800\text{ Hz}, n = 1.8$ (lower tension causes $+1.39\text{ dB}$ higher plucking compliance).
5. **`flatwound_vintage_heavy` (Motown / Jamerson 1954 Spec):**
   * La Bella 760M $.052\text{--}.110$ heavy hex core.
   * $195.0\text{ lbs}$ heavy tension, $f_d = 1800\text{ Hz}, n = 2.0, k_{\text{long}} = 0.05$ (massive ribbon shear damping rolls off highs above $1.8\text{ kHz}$).
6. **`double_bass_spirocore` (3/4 Upright Orchestral/Pizz):**
   * Thomastik-Infeld Spirocore / D'Addario Helicore Pizzicato $41.5''$ spiral rope core.
   * $265.0\text{ lbs}$ massive tension, $f_d = 3800\text{ Hz}, n = 2.0, k_{\text{long}} = 0.02$.

---

## 6. How to Add a Custom Instrument

To model your own bass in Passivizer:

1. Create a new file: `config/instruments/my_bass.toml`.
2. Measure:
   * Scale length ($L_{\text{in}}$).
   * Pickup centerline distances from the bridge saddle in millimeters ($x_{\text{mm}}$).
   * Active pickup resonant frequency ($f_r$) from manufacturer spec sheets (if active).
3. Fill out the schema:
   ```toml
   id = "my_custom_5str"
   name = "Custom 35\" 5-String Soapbar"
   scale_length_in = 35.0
   scale_length_m = 0.889
   string_wave_speeds = [53.28, 71.16, 95.00, 126.81, 169.27]
   default_pickup = "bridge_dual"

   [pickups.bridge_dual]
   name = "Dual-Coil Soapbar (Bridge)"
   position_from_bridge_m = 0.055
   aperture_width_in = 1.25
   coil_spacing_in = 0.75
   type = "dual_coil_parallel"
   pole_type = "blade"
   magnet_type = "ceramic"
   resonant_frequency_hz = 2800.0
   q_factor = 1.35
   coils = [
       { strings = ["all"], position_from_bridge_m = 0.0645, aperture_width_in = 0.60, weight = 0.5, pole_type = "blade" },
       { strings = ["all"], position_from_bridge_m = 0.0455, aperture_width_in = 0.60, weight = 0.5, pole_type = "blade" }
   ]
   ```
4. Run validation and preview:
   ```bash
   uv run allomorph --stage viz --instrument my_custom_5str
   ```
