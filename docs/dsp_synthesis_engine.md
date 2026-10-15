# Allomorph - Acoustic & String Vibration DSP Synthesis Engine

**Document Version:** 1.0.0  
**Package Reference:** [`src/allomorph/dsp/`](../src/allomorph/dsp/)  
**Primary Modules:** [`dry.py`](../src/allomorph/dsp/dry.py), [`plucks.py`](../src/allomorph/dsp/plucks.py), [`sweeps.py`](../src/allomorph/dsp/sweeps.py), [`loudness.py`](../src/allomorph/dsp/loudness.py), [`fir.py`](../src/allomorph/dsp/fir.py), [`convolution.py`](../src/allomorph/dsp/convolution.py), [`io.py`](../src/allomorph/dsp/io.py)  

---

## 1. Executive Summary & Purpose

The Allomorph DSP Synthesis Engine (`allomorph.dsp`) is an analytical, first-principles digital signal processing suite responsible for generating the master excitation audio (`audio/input.wav`), synthesizing physically accurate modal string plucks, generating logarithmic sweeps with string damping, calculating ITU-R BS.1770-4 gated loudness matching, and handling 24-bit 48 kHz bit-exact audio I/O.

Unlike generic pink-noise or impulse test signals, the Allomorph DSP excitation engine generates a **scientifically calibrated multi-technique string excitation suite** that provides comprehensive spectral and dynamic coverage while strictly observing physical string harmonic damping laws to prevent non-linear artifacts in downstream neural models.

---

## 2. DSP Subsystem Architecture

```
src/allomorph/dsp/
├── dry.py           # Master excitation assembler, multi-technique suite, chirp tiers
├── plucks.py        # 2D modal string vibration, inharmonicity, pitch sag, techniques
├── sweeps.py        # Logarithmic frequency chirps with string tilt damping
├── loudness.py      # ITU-R BS.1770-4 K-weighted gated LUFS, true-peak, and crest factor
├── conditioning.py  # 8 Hz DC blocking, RLC-colored Johnson-Nyquist dither (-108 dBFS), and LUFS normalization
├── fir.py           # Pure real-FFT homomorphic cepstrum minimum-phase FIR synthesis
├── convolution.py   # Overlap-save FFT convolution with SIMD optimizations
├── io.py            # 24-bit 48 kHz PCM little-endian byte packing and WAV I/O
└── constants.py     # Master DSP constants (FS=48000, TARGET_LUFS=-20.50, CALIBRATION_PEAK_CEILING=0.9900)
```

---

## 3. Physical String Harmonic Damping & Chirp Synthesis (`sweeps.py`, `dry.py`)

### 3.1 The Physical String Tilt Law ($f_{\text{corner}} = 1200\text{ Hz}$)

Real vibrating electric bass strings exhibit steep high-frequency energy decay due to internal viscoelastic damping within the steel core and nickel/steel windings:
- Fundamental register ($40\text{--}120\text{ Hz}$): Full displacement excursion ($0\text{ dB}$).
- Clank & bite register ($2\text{--}4\text{ kHz}$): Vibrational energy down by $-25\text{ to } -35\text{ dB}$.
- Treble air register ($8\text{--}15\text{ kHz}$): Vibrational energy down by $-43.5\text{ to } -60\text{ dB}$.

To prevent unphysical high-frequency drive from destabilizing downstream neural network training, all synthetic logarithmic frequency chirps apply continuous physical string tilt filtering (`-12 dB/octave` second-order roll-off):

$$H_{\text{tilt}}(f) = \frac{1}{1 + \left(\frac{f}{f_{\text{corner}}}\right)^2}, \quad f_{\text{corner}} = 1200.0\text{ Hz}$$

```python
# src/allomorph/dsp/sweeps.py
if string_tilt:
    # Continuous instantaneous frequency vector:
    f_t = f_start * (f_end / f_start) ** (t / dur)
    tilt_gain = 1.0 / (1.0 + (f_t / f_corner) ** 2)
    sig = sig * tilt_gain
```

### 3.2 Elimination of Untilted $22\text{ kHz}$ Flat Air Sweeps
In earlier versions, flat $22\text{ kHz}$ chirps without string tilt were tested to force high-frequency differentiator learning. This resulted in catastrophic zero-crossing rectification buzzing (+19 dB louder than reference) in WaveNet models because the neural network received 150x more high-frequency energy than physically possible from a vibrating bass string.

**Invariant:** All chirps in `synth_log_chirps` and `generate_optimal_bass_dry` strictly default to `string_tilt=True` with $f_{\text{corner}} = 1200\text{ Hz}$.

---

## 4. 2D Modal String Vibration & Articulation Suite (`plucks.py`)

### 4.1 Biaxial Modal String Vibration with Inharmonicity

String vibration is evaluated using a 2D modal model combining orthogonal vertical (perpendicular to pickup face) and horizontal (parallel to fretboard) vibration planes:

$$y(t) = \sum_{n=1}^{N_{\text{partials}}} A_n \cdot \left[ 0.65\sin(2\pi f_n t + \phi_{\text{sag}, n}) \cdot e^{-d_{v, n} t} + 0.35\sin(2\pi (f_n + \Delta f_{\text{split}}) t + \phi_{\text{sag}, n}) \cdot e^{-d_{h, n} t} \right]$$

1. **Inharmonicity Stiffness Dispersion ($B_s$) with Shear-Wave Saturation ($f_{\text{disp,max}} = 3500\text{ Hz}$):**
   $$f_n = n \cdot f_0 \cdot \sqrt{1 + \frac{B_s \cdot n^2}{1 + (n f_0 / f_{\text{disp,max}})^2}}$$
   where $B_s$ is interpolated from empirical string stiffness calibration tables via the exact $C^\infty$ Gaussian RBF solver (`get_inharmonicity_for_f0` in `allomorph.physics.strings`), matching the pickup spatial dispersion tensor (`compute_string_dispersion_tensor` in `allomorph.physics.aperture`).
2. **Attack Pitch Sag (Non-Linear Elongation Transient):**
   During the initial pluck attack, high initial displacement increases string tension $\Delta T \propto \int (\partial y/\partial x)^2 dx$, momentarily sharpening the pitch:
   $$\Delta f(t) = \Delta_{\text{sag}} \cdot e^{-t / \tau_{\text{sag}}}, \quad \Delta_{\text{sag}} = 2.5\text{ Hz}, \quad \tau_{\text{sag}} = 80\text{ ms}\text{ }(0.08\text{ s})$$
3. **Biaxial Damping Decay Rates & Spectral Roll-off:**
   Vertical polarization couples directly to the magnetic field and body bridge witness point, decaying faster than horizontal polarization:
   $$d_{v, n} = (0.70 + 0.10 n + 0.00025 n^2) \cdot m_{\text{decay}}, \quad d_{h, n} = (0.35 + 0.05 n + 0.00012 n^2) \cdot m_{\text{decay}}$$
   while partial amplitudes follow $A_n \propto n^{-\gamma_{\text{tech}}}$ with $C^\infty$ quadrature pluck-position and fingertip floors.

### 4.2 Luthier Articulation Techniques

The engine synthesizes five distinct playing techniques:
- **Fingerstyle:** Smooth fleshy pad contact with warm low-mid bloom ($\gamma_{\text{tech}} = 1.45$, $m_{\text{decay}} = 1.0$).
- **Pick Strike:** Sharp plectrum attack with instantaneous transient burst and enhanced upper partials ($\gamma_{\text{tech}} = 1.10$, $m_{\text{decay}} = 1.0$).
- **Slap Thumb & Pop:** Aggressive percussive impact with boundary fret slap click and extended dynamic crest factor ($\gamma_{\text{tech}} = 0.95$, $m_{\text{decay}} = 1.3$).
- **Palm-Muted Staccato:** Viscoelastic hand damping filtering partials above $600\text{ Hz}$ via $C^\infty$ softplus exponential roll-off ($\gamma_{\text{tech}} = 1.80$, $m_{\text{decay}} = 6.0$).
- **Natural Harmonics:** Pure crystalline sinusoidal overtones generated at nodes ($L/2, L/3, L/4$) with 5 partials, strictly omitting non-harmonic string modes.

---

## 5. Broadband RMS Level Calibration, ITU-R BS.1770-4 Telemetry & Peak Safety (`conditioning.py`, `loudness.py`)

### 5.1 The NAM Training Broadband RMS Reference Standard ($-20.50\text{ dBFS}$ RMS)

All dry excitation tracks (`audio/input.wav`) and NAM training stems (normalized at the start of training via `normalize_training_stem`) are leveled to the broadband RMS reference standard (`TARGET_RMS_DBFS`) so Block 1 Digital Twins maintain exact $0.00\text{ dB}$ broadband voltage gain staging into downstream pedalboard blocks:
$$\text{Target Broadband RMS Level} = -20.50\text{ dBFS RMS}$$
$$\text{True-Peak Safety Ceiling} = 0.9900\text{ (}-0.09\text{ dBFS)}$$

### 5.2 K-Weighted Integrated Loudness Algorithm

1. **Stage 1: High-Shelf Pre-Filter ($+4.0\text{ dB}$ at $1.5\text{ kHz}$):**
   $$H_{\text{shelf}}(z) = \frac{b_0 + b_1 z^{-1} + b_2 z^{-2}}{1 + a_1 z^{-1} + a_2 z^{-2}}$$
   Simulates acoustic head diffraction and pinna filtering.
2. **Stage 2: RLB High-Pass Filter ($f_c \approx 38\text{ Hz}$):**
   $$H_{\text{RLB}}(z) = \frac{1.0 - 2.0 z^{-1} + 1.0 z^{-2}}{1 - 1.99004745 z^{-1} + 0.99007225 z^{-2}}$$
   Attenuates sub-audible mechanical rumblings.
3. **Stage 3: Gated Power Integration:**
   - Absolute threshold: $-70.0\text{ LKFS}$.
   - Relative threshold: $-10.0\text{ dB}$ below un-gated mean power.
   - Gated blocks ($400\text{ ms}$, $75\%$ overlap) calculate true perceived stage loudness.

### 5.3 Transparent Linear Peak-Based Scaling

To prevent artificial dynamic compression or distortion on transient peaks:
- If a target voicing features extreme resonant peaks or scale snap, it is scaled **strictly with linear gain**:
  $$g_{\text{peak}} = \frac{0.9900}{\max(|y_{\text{true\_peak}}|)}$$
- Digital dynamic range compressors or soft-clippers are **never applied** to target stems during leveling, providing $>20.4\text{ dB}$ of clean digital headroom.

---

## 6. Post-Conditioning & Minimum-Phase FIR Synthesis (`conditioning.py`, `fir.py`)

### 6.1 Sub-Audible DC Blocking & RLC-Colored Thermal Dither (`conditioning.py`)
1. **Sub-Audible DC Blocking (`apply_dc_block`):** Applies an $8.0\text{ Hz}$ high-pass filter followed by exact mean subtraction so every rendered stem has zero DC offset.
2. **RLC-Colored Johnson-Nyquist Thermal Dither (`apply_johnson_dither`):** Shapes deterministic white noise through the minimum-phase FIR of the composite passive RLC transfer function and injects it at $-108.0\text{ dBFS}$ RMS to prevent 24-bit PCM quantization limit cycles during silence and ring-out tails.

### 6.2 Minimum-Phase Synthesis via Pure Real-FFT Homomorphic Cepstrum (`fir.py`)
For zero-latency pedalboard execution, linear magnitude responses $|H[k]|$ are converted to causal minimum-phase FIR kernels ($N = 4096$ taps, $N_{\text{fft}} = 16384$) via a pure real-FFT pipeline (`irfft` $\to$ causal folding $\to$ `rfft` $\to$ `irfft`) with cepstral power regularization floor $\epsilon = 10^{-12}$ ($-120\text{ dB}$):
1. **Real Cepstrum Computation:** Given real, conjugate-symmetric log-magnitude spectrum $L[k] = \frac{1}{2}\ln(|H[k]|^2 + 10^{-12})$ for $k \in [0, N]$:
   $$\hat{c}[n] = \text{irfft}(L)[n]$$
2. **Causal Liftering:** Zero non-causal negative cepstral time indices and double positive time indices:
   $$c_{\text{min}}[n] = \begin{cases} \hat{c}[0], & n = 0 \\ 2\hat{c}[n], & 1 \le n < N \\ \hat{c}[N], & n = N \\ 0, & N < n < 2N \end{cases}$$
3. **Analytic Spectrum & Direct Inverse Real FFT:**
   $$S[k] = \text{rfft}(c_{\text{min}})[k] = \ln|H[k]| + j \theta_{\text{min}}[k], \quad h[n] = \text{irfft}(\exp(S))[:N]$$
4. **Real-Analytic Tail Windowing:** The trailing 15% of taps are tapered with $w(t) = 1.0 - S_\infty(t)$ (`cinf_smoothstep`), vanishing all boundary derivatives at buffer termination and guaranteeing zero latency with peak energy concentrated at tap 0.

---

## 7. Fast Overlap-Save Convolution & 24-bit PCM Audio I/O (`io.py`)

1. **Overlap-Save FFT Convolution (`convolution.py`):**
   - Audio tracks are processed in overlapping power-of-two blocks ($N=16,384$), broadcasting impulse responses across channels using vector SIMD FFTs.
2. **24-bit Little-Endian Byte Packing (`io.py`):**
   - Samples are clipped to $[-1.0, 1.0 - 2^{-23}]$, scaled to integer range $[-8388608, 8388607]$, and packed via fast C memory view slicing:
     ```python
     i32 = (samples * 8388607.0).astype("<i4")
     b24 = i32.view(np.uint8).reshape(-1, 4)[:, :3].flatten().tobytes()
     ```
   - Avoids Python byte concatenation loops, writing $10\text{ MB}$ WAV files in $<12\text{ ms}$.
