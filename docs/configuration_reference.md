# Allomorph Configuration Reference

This guide provides a comprehensive specification of all configuration files in **Allomorph**, their schemas, data types, physical units, mathematical implications, and fail-fast validation constraints.

---

## 1. Overview of Configuration Files

Allomorph organizes instrument models, target voices, active preamps, physical scale wave speeds, string metallurgy presets, and visualizer settings under the `config/` directory:

```
config/
├── instruments/                        # Unified instrument catalog & native voicings (16 instruments)
│   ├── 30in_emg_mmtw.toml              # 30.0" Short scale with EMG MMTWX dual-mode pickup
│   ├── 30in_gibson_eb0.toml            # 30.5" Short scale 1960s Gibson EB-0 Sidewinder Mudbucker
│   ├── 30in_mustang_pj.toml            # 30.0" Short scale Fender Mustang Bass PJ
│   ├── 32in_custom_pmm.toml            # 32.0" Medium scale with Reverse EMG PX + MMTWX + ABCX blend
│   ├── 32in_fretless_pmm.toml          # 32.0" Fretless Medium scale with EMG PCSX + MMTWX + ABCX blend
│   ├── 33in_rickenbacker_4003.toml     # 33.25" Rickenbacker 4003 with Vintage Tone Push-Pull
│   ├── 34in_active_emg.toml            # 34.0" 5-String Active EMG 40DC Dual-Soapbar
│   ├── 34in_active_pmm.toml            # 34.0" Modern Active P/MM (Lakland 44-02 / Sandberg VM4)
│   ├── 34in_active_stingray.toml       # 34.0" Music Man StingRay (2-Band Preamp)
│   ├── 34in_dingwall_sp1.toml          # 32.0"-35.0" Multi-scale Dingwall SP1 5-String
│   ├── 34in_preamp_soapbar.toml        # 34.0" 5-String Passive Soapbar + Aguilar OBP-3
│   ├── 34in_standard_jazz.toml         # 34.0" Standard 1960s Fender Jazz Bass
│   ├── 34in_standard_p.toml            # 34.0" Standard 1962 Fender Precision Bass
│   ├── 34in_standard_pj.toml           # 34.0" Standard P/J & Fender American Ultra II Active P/J
│   ├── 37in_multiscale_dingwall.toml   # 34.0"-37.0" Multi-scale Dingwall Combustion / NG3 5-String
│   └── 41in_upright_bass.toml          # 41.25" Orchestral 3/4 Double Bass (Piezo Bridge Transducer)
├── packs/                              # Storefront pack manifests & affinity bundle definitions
├── naming.toml                         # Tone naming vocabulary & pedalboard character budgets
├── preamps.toml                        # Reusable active preamp catalog (8 preamps)
├── scales.toml                         # Physical scale lengths, wave speeds, and multiscale spans (9 scales)
├── strings.toml                        # Physical string core/wrap metallurgy presets (7 presets)
└── visualizer.toml                     # Visualizer storefront catalog & source comparison baselines
```

---

## 2. Instrument Configuration Schema (`config/instruments/*.toml`)

Each file in `config/instruments/*.toml` defines a complete **Physical & Digital Twin** (`InstrumentConfig`): its vibrating scale length, string wave speeds, onboard pickups (`[pickups.<id>]`), physical sub-coil apertures (`coils = [...]`), electrical MNA harnesses (`[harnesses.<id>]`), and playable switch/potentiometer voicings (`[voicings.<id>]`).

### 2.1 Root Table Parameters (`InstrumentConfig`)

| Field | Type | Units | Default | Description |
| :--- | :--- | :--- | :---: | :--- |
| `id` | `string` | — | **Required** | Unique identifier matching filename stem (e.g. `"30in_emg_mmtw"`). |
| `name` | `string` | — | **Required** | Human-readable instrument label embedded in NAM metadata and charts. |
| `scale_length_in` | `float` | Inches | **Required** | Vibrating string scale length (longest string on multiscale basses). |
| `scale_length_m` | `float` | Meters | **Required** | Vibrating scale length in meters ($L_{\text{m}} = L_{\text{in}} \times 0.0254$). |
| `multiscale` | `bool` | — | `false` | Set `true` for fanned-fret multiscale instruments. |
| `scale_length_short_in` | `float` | Inches | `None` | Treble-side (G-string) scale length in inches when `multiscale = true`. |
| `scale_length_long_in` | `float` | Inches | `None` | Bass-side (B/E-string) scale length in inches when `multiscale = true`. |
| `string_wave_speeds` | `array[float]` | m/s | **Required** | Array of 4 or 5 string wave speeds from lowest to highest string ($v_s = 2 L_s f_{0,s}$). |
| `string_preset` | `string` | Key | `"roundwound_nickel_standard"` | Default string preset key in `config/strings.toml`. |
| `string` | `table` | — | `None` | Optional inline custom string physics block (`CustomStringConfig`). |
| `preamp` | `string` | Key | `None` | Optional instrument-level active preamp preset key from `config/preamps.toml`. |

---

### 2.2 Pickup Definitions (`[pickups.<pickup_id>]` — `PickupConfig`)

Each entry in `[pickups.<id>]` represents a physical transducer mounted on the instrument body.

| Field | Type | Units | Default | Description |
| :--- | :--- | :--- | :---: | :--- |
| `name` | `string` | — | **Required** | Descriptive label of the physical pickup. |
| `type` | `string` | — | **Required** | Pickup architecture: `"single_coil"`, `"split_coil"`, `"dual_coil_parallel"`, `"dual_coil_series"`, `"humbucker"`, or `"piezo"`. |
| `position_from_bridge_m` | `float` | Meters | `None` | Centerline distance from bridge saddle ($x = \text{datum}_{\text{mm}} / 1000$). |
| `aperture_width_in` | `float` | Inches | `1.0` | Total magnetic sensing aperture width ($w$). |
| `pole_type` | `string` | — | `"rod"` | Spatial pole geometry: `"rod"` (2D cylindrical pole disc) or `"blade"` (1D bar slit). |
| `magnet_type` | `string` | — | `None` | Core metallurgy key in `MAGNET_PROPERTIES`: `"alnico_v"`, `"alnico_ii"`, `"alnico_iii"`, `"ceramic"`, `"ceramic_steel"`, `"neodymium"`, `"piezo"`, `"active"`, or `"ideal"`. Validated fail-fast at load time. |
| `alpha` | `float` | — | `None` | Optional non-linear asymmetric flux quadratic coefficient override. |
| `has_internal_buffer` | `bool` | — | `false` | Set `true` for active pickups with an internal differential op-amp buffer (e.g., EMG X-series / DC series). Isolates internal coil RLC resonance from external cable/pot loading and divides coil EMF by $N_{\text{coils}}$ before summing. |
| `buffer_topology` | `string` | — | `"x_series"` | Internal active op-amp headroom architecture: `"x_series"` (18V high-headroom) or `"classic"` (9V standard). |
| `buffer_rout` | `float` | $\Omega$ | `10000.0` | Output impedance of the internal active pickup buffer ($10\text{ k}\Omega$ for EMG). |
| `V_sat` | `float` | Volts | `None` | Optional pickup saturation threshold override. |
| `tau_th` | `float` | Seconds | `None` | Optional thermal recovery time constant override. |
| `c_th` | `float` | — | `None` | Optional thermal compression depth override. |
| `k_var` | `float` | — | `None` | Optional nonlinear varactor capacitance modulation override. |
| `coils` | `array[table]` | — | `[]` | Array of physical sub-coils (`CoilConfig`) with individual positions, apertures, and RLC parameters. |

---

### 2.3 Sub-Coil Modeling (`coils = [...]` — `CoilConfig`)

Every pickup defines one or more physical sub-coils with exact mechanical coordinates and electrical RLC parameters:

| Field | Type | Units | Default | Description |
| :--- | :--- | :--- | :---: | :--- |
| `id` | `string` | — | `""` | Coil terminal identifier (e.g. `"forward"`, `"rearward"`, `"neck"`, `"bridge"`), addressed in harness netlists as `pickups.<pickup_id>.<coil_id>.hot` / `.cold`. |
| `strings` | `array[string\|int]` | — | `["all"]` | String bindings for split/staggered coils: `["all"]`, `["E", "A"]`, `["D", "G"]`, or register indices `[1, 2]` / `[3, 4]` / `[3, 4, 5]`. |
| `position_from_bridge_m` | `float` | Meters | **Required** | Physical distance from bridge saddle to this coil's sensing centerline. |
| `aperture_width_in` | `float` | Inches | **Required** | Magnetic sensing window width of this specific coil. |
| `weight` | `float` | — | `1.0` | Spatial amplitude contribution weight. Automatically re-normalized to sum to `1.0` when a full-string-bed multi-coil pickup is coil-split to a single active coil. |
| `polarity` | `float` | — | `1.0` | Magnetic/electrical polarity (`1.0` in-phase, `-1.0` reverse phase). |
| `pole_type` | `string` | — | `"rod"` | Spatial sensing geometry: `"rod"` or `"blade"`. |
| `L` | `float` | Henries | `None` | Coil inductance ($L$). |
| `Rdc` | `float` | $\Omega$ | `None` | Coil DC winding resistance ($R_{\text{dc}}$). |
| `Reddy` | `float` | $\Omega$ | `None` | Parallel core/pole eddy-current loss resistance ($R_{\text{eddy}}$). |
| `Ccoil` | `float` | Farads | `None` | Parasitic inter-winding turn-to-turn capacitance ($C_{\text{coil}}$). |

---

### 2.4 Electrical Harnesses (`[harnesses.<harness_id>]` — `HarnessConfig`)

Harnesses define the complete Modified Nodal Analysis (MNA) netlist connecting pickup coils, potentiometers, switches, tone capacitors, cable capacitance, and active preamps:

| Field | Type | Units | Default | Description |
| :--- | :--- | :--- | :---: | :--- |
| `type` | `string` | — | `"passive"` | Circuit architecture: `"passive"`, `"active_preamp"`, or `"direct"`. |
| `preamp` | `string` | Key | `None` | Preamp preset key in `config/preamps.toml` when `type = "active_preamp"`. |
| `cable_pf` | `float` | pF | `750.0` | Instrument cable shunt capacitance in picofarads ($C_{\text{cable}}$). |
| `load_resistance` | `float` | $\Omega$ | `1000000.0` | Downstream amplifier / Darkglass Anagram input impedance ($R_{\text{load}} = 1\text{ M}\Omega$). |
| `k_mutual` | `float` | — | `0.0` | Default mutual inductive coupling coefficient ($k \in [0, 1)$) between coils of the same pickup. |
| `wiring` | `array[array[string]]` | — | `[]` | Always-connected node pairs `[["node_a", "node_b"], ...]`. All terminal paths are validated fail-fast against declared pickups and controls. |
| `controls` | `table[ControlElementConfig]` | — | `{}` | Potentiometers and active preamp EQ band controls (`[harnesses.<h>.controls.<c>]`). |
| `switches` | `table[SwitchConfig]` | — | `{}` | Multi-position switches (`[harnesses.<h>.switches.<s>]`) with position-specific `connect`, `components`, and `k_mutual`. |

#### Control Elements (`ControlElementConfig`)

| Field | Type | Units | Default | Description |
| :--- | :--- | :--- | :---: | :--- |
| `type` | `string` | — | `"pot"` | `"pot"` (passive volume, tone, or MN blend potentiometer) or `"preamp_band"` (active EQ shelf/bell knob). |
| `resistance` | `float` | $\Omega$ | `250000.0` | Total potentiometer track resistance ($R_{\text{pot}}$). |
| `taper` | `string` | — | `"audio_15"` | Resistance taper law: `"linear"`, `"audio_15"`, `"audio_10"`, `"audio_30"`, `"reverse_audio"`, or `"mn_blend"`. |
| `cap` | `float` | Farads | `None` | Series tone capacitor ($C_{\text{tone}}$) when the pot acts as a variable RC tone shunt. |
| `buffered` | `bool` | — | `false` | When `true` on an `mn_blend` pot in an `active_preamp` harness (e.g., Lakland LH-3 active pan pot), each pickup feeds an isolated high-$Z$ FET buffer input prior to panning, eliminating passive coil-to-coil loading. |
| `band` | `string` | — | `None` | Target band name in `PreampConfig.bands` when `type = "preamp_band"` (validated fail-fast). |
| `max_boost_db` | `float` | dB | `12.0` | Full-throw (`pos = 1.0`) boost (or $\pm$ boost/cut range for center-detent bands) when `type = "preamp_band"`. |
| `default` | `float` | $[0.0, 1.0]$ | `1.0` | Default knob wiper position (`0.5` = center detent). For `preamp_band` controls: if `band.gain_db == 0.0` in `preamps.toml` (center-detent boost/cut preamp), `pos = 0.5` yields $0.0\text{ dB}$; if `band.gain_db != 0.0` (boost-only or sweet-spot preset), `pos = 0.5` yields `band.gain_db`. |

---

### 2.5 Native Voicing Definitions (`[voicings.<voicing_id>]` — `VoicingConfig`)

Every playable tone on an instrument is declared under `[voicings.<id>]`:

| Field | Type | Units | Default | Description |
| :--- | :--- | :--- | :---: | :--- |
| `name` | `string` | — | **Required** | Descriptive title (e.g. `"Vintage 1962 Open"`). |
| `tone_name` | `string` | — | **Required** | Standardized 2-to-3 token musician label (e.g. `"Precision Vintage"`). |
| `affinity` | `string` | — | `"neck"` | Position affinity for Tone3000 bundle partitioning: `"neck"`, `"bridge"`, or `"parallel"`. |
| `harness` | `string` | Key | `""` | Active harness key in `[harnesses]`. |
| `controls` | `table[float]` | $[0.0, 1.0]$ | `{}` | Map of control IDs to knob positions (e.g. `{ neck_vol = 0.99, bridge_vol = 1.0, tone = 1.0 }`). |
| `switch` | `string` | Key | `None` | Shorthand switch position name when the harness has a single switch. |
| `switches` | `table[string]` | Key | `{}` | Map of switch IDs to position names (e.g. `{ selector = "p_mm", coil_split = "parallel" }`). |
| `components` | `table[float]` | SI | `{}` | Validated component value overrides (`"cable_pf"`, `"load_resistance"`, `"k_mutual"`, `"Rtone"`, `"controls.<id>.resistance\|cap\|Rtone"`, `"pickups.<id>[.<coil>].L\|Rdc\|Reddy\|Ccoil"`, `"preamp.<band>.freq_hz"`). Unknown keys raise `KeyError`. |
| `preamp` | `string` | Key | `None` | Optional voicing-level preamp preset override from `config/preamps.toml`. |
| `gain_db` | `float` | dB | `0.0` | Drive offset in dB into non-linear magnetic saturation and visualizer vertical display offset. |
| `string_preset_override` | `string` | Key | `None` | Optional string preset override from `config/strings.toml` (e.g. `"flatwound_vintage_heavy"`). Unknown keys raise `KeyError`. |
| `sensor_type` | `string` | — | `"magnetic"` | Physical transducer taxonomy: `"magnetic"` or `"bridge_force"`. |

---

## 3. Scale Lengths & Wave Speeds (`config/scales.toml`)

`config/scales.toml` defines the 9 physical scale standards used across all 16 instruments ($v_s = 2 L_s f_{0,s}$):

| Scale Key | Scale Length ($L_{\text{in}}$) | Metric ($L_{\text{m}}$) | Multi-Scale Span | String Wave Speeds ($v_s$, m/s) | Representative Instruments |
| :--- | :--- | :--- | :---: | :--- | :--- |
| **`30in`** | $30.0''$ | $0.76200\text{ m}$ | Fixed | $[62.79, 83.82, 111.89, 149.35]$ | `30in_emg_mmtw`, `30in_mustang_pj` |
| **`30.5in`** | $30.5''$ | $0.77470\text{ m}$ | Fixed | $[63.83, 85.22, 113.75, 151.84]$ | `30in_gibson_eb0` |
| **`32in`** | $32.0''$ | $0.81280\text{ m}$ | Fixed | $[66.98, 89.41, 119.35, 159.31]$ | `32in_custom_pmm`, `32in_fretless_pmm` |
| **`33.25in`** | $33.25''$ | $0.84455\text{ m}$ | Fixed | $[69.60, 92.90, 124.01, 165.53]$ | `33in_rickenbacker_4003` |
| **`34in`** | $34.0''$ | $0.86360\text{ m}$ | Fixed | $[71.16, 95.00, 126.81, 169.27]$ | `34in_standard_p`, `34in_standard_jazz`, `34in_standard_pj`, `34in_active_stingray`, `34in_active_pmm` |
| **`34in_5string`** | $34.0''$ | $0.86360\text{ m}$ | Fixed (5-str) | $[53.31, 71.16, 95.00, 126.81, 169.27]$ | `34in_active_emg`, `34in_preamp_soapbar` |
| **`multiscale_super`** | $35.0''$ | $0.88900\text{ m}$ | $32.0''\text{--}35.0''$ (5-str) | $[54.88, 71.69, 93.60, 122.14, 159.31]$ | `34in_dingwall_sp1` |
| **`multiscale`** | $37.0''$ | $0.93980\text{ m}$ | $34.0''\text{--}37.0''$ (5-str) | $[58.02, 75.88, 99.19, 129.60, 169.27]$ | `37in_multiscale_dingwall` |
| **`upright`** | $41.25''$ | $1.04775\text{ m}$ | Fixed | $[86.33, 115.25, 153.86, 205.36]$ | `41in_upright_bass` |

---

## 4. Active Preamp Catalog (`config/preamps.toml`)

`config/preamps.toml` defines reusable active onboard preamps (`PreampConfig`) with FET/op-amp input impedance ($R_{\text{in}} \parallel C_{\text{in}}$), output impedance ($R_{\text{out}}$), and analog EQ bands (`PreampBandConfig`):

| Preamp Key | Archetype | $R_{\text{in}}$ | $C_{\text{in}}$ | $R_{\text{out}}$ | Default EQ Contour |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **`sadowsky_2band`** | Sadowsky NYC 2-Band Boost-Only | $1.0\text{ M}\Omega$ | $150\text{ pF}$ | $100\,\Omega$ | $+3.5\text{ dB}$ @ $60\text{ Hz}$ low shelf, $+3.5\text{ dB}$ @ $3.5\text{ kHz}$ high shelf |
| **`stingray_2band`** | Music Man StingRay 2-Band | $1.0\text{ M}\Omega$ | $850\text{ pF}$ | $100\,\Omega$ | $+1.8\text{ dB}$ @ $80\text{ Hz}$ low shelf, $+2.2\text{ dB}$ @ $4.0\text{ kHz}$ high shelf |
| **`aguilar_3band`** | Aguilar OBP-3 18V 3-Band | $1.0\text{ M}\Omega$ | $100\text{ pF}$ | $100\,\Omega$ | $+2.0\text{ dB}$ @ $40\text{ Hz}$ low shelf, $-2.0\text{ dB}$ @ $800\text{ Hz}$ bell, $+2.0\text{ dB}$ @ $6.5\text{ kHz}$ high shelf |
| **`glockenklang_2band`** | Glockenklang 2-Band | $1.0\text{ M}\Omega$ | $470\text{ pF}$ | $100\,\Omega$ | $+2.0\text{ dB}$ @ $40\text{ Hz}$ low shelf, $+2.0\text{ dB}$ @ $4.0\text{ kHz}$ high shelf |
| **`lakland_lh3`** | Lakland LH-3 3-Band Hybrid | $1.0\text{ M}\Omega$ | $470\text{ pF}$ | $100\,\Omega$ | $+1.5\text{ dB}$ @ $50\text{ Hz}$ low shelf, $+1.0\text{ dB}$ @ $600\text{ Hz}$ bell, $+2.0\text{ dB}$ @ $7.0\text{ kHz}$ high shelf |
| **`fender_ultra2_3band`** | Fender American Ultra II 18V 3-Band | $1.0\text{ M}\Omega$ | $100\text{ pF}$ | $100\,\Omega$ | Flat center detent ($0.0\text{ dB}$ @ $40\text{ Hz}$, $325/750\text{ Hz}$, $8.0\text{ kHz}$) |
| **`dingwall_active`** | Dingwall / Glockenklang 3-Band | $0.5\text{ M}\Omega$ | $100\text{ pF}$ | $1,000\,\Omega$ | Flat center detent ($0.0\text{ dB}$ @ $70\text{ Hz}$ bell, $500\text{ Hz}$ bell, $2.8\text{ kHz}$ bell) |
| **`flat_buffer`** | Transparent Unity FET Buffer | $1.0\text{ M}\Omega$ | $30\text{ pF}$ | $100\,\Omega$ | Unity passband ($0.0\text{ dB}$) |

---

## 5. Physical String Catalog Reference (`config/strings.toml`)

`config/strings.toml` defines the viscoelastic wrap damping ($f_d, n$), tension compliance, core geometry, and longitudinal clank coupling ($k_{\text{long}}$) for 7 physical string sets:

| Preset Key | Archetype | Type / Wrap / Core | Tension | Cutoff $f_d$ | Order $n$ | $k_{\text{long}}$ |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| **`roundwound_nickel_standard`** | D'Addario EXL165 / Ernie Ball Slinky ($.045\text{--}.105$) | Roundwound / Nickel / Hex | $155.0\text{ lbs}$ | $8,500\text{ Hz}$ | $1.5$ | $0.20$ |
| **`roundwound_short_scale`** | D'Addario EXL160S Short Scale ($.050\text{--}.105$) | Roundwound / Nickel / Hex | $138.0\text{ lbs}$ | $8,500\text{ Hz}$ | $1.5$ | $0.18$ |
| **`roundwound_nickel_6string`** | D'Addario EXL170-6 / Slinky 6-String ($.032\text{--}.130$) | Roundwound / Nickel / Hex | $230.0\text{ lbs}$ | $8,500\text{ Hz}$ | $1.5$ | $0.20$ |
| **`roundwound_stainless_clank`** | Dingwall Custom Stainless Steel ($.045\text{--}.130$) | Roundwound / Stainless / Hex | $180.0\text{ lbs}$ | $12,000\text{ Hz}$ | $1.5$ | $0.35$ |
| **`flatwound_low_tension`** | La Bella Low Tension Flats LTF-4A ($.042\text{--}.100$) | Flatwound / Stainless Flat / Round | $132.0\text{ lbs}$ | $2,800\text{ Hz}$ | $1.8$ | $0.08$ |
| **`flatwound_vintage_heavy`** | La Bella 760M 1954 Jamerson Heavy Flats ($.052\text{--}.110$) | Flatwound / Stainless Flat / Hex | $195.0\text{ lbs}$ | $1,800\text{ Hz}$ | $2.0$ | $0.05$ |
| **`double_bass_spirocore`** | Thomastik-Infeld Spirocore 3/4 Orchestral | Double Bass / Chrome Steel / Spiral Rope | $265.0\text{ lbs}$ | $3,200\text{ Hz}$ | $2.0$ | $0.02$ |

---

## 6. How to Add a Custom Instrument

1. Create `config/instruments/<id>.toml` with `scale_length_in`, `scale_length_m`, `string_wave_speeds`, `[pickups.<id>]`, `[harnesses.<id>]`, and `[voicings.<id>]`.
2. Ensure all pickup `magnet_type` keys exist in `MAGNET_PROPERTIES`, all harness `wiring`/`connect` terminals match declared pickups and controls, and all `preamp_band` controls reference valid band names in `config/preamps.toml`.
3. Run the test suite and visualizer to verify:
   ```bash
   uv run pytest
   uv run allomorph-viz
   ```
