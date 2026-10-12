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
├── conditioning.py  # Pre/de-emphasis filters, phase unwrapping, Hilbert transform
├── fir.py           # Zero-phase Blackman/Hamming FIR filter design
├── convolution.py   # Overlap-save FFT convolution with SIMD optimizations
├── io.py            # 24-bit 48 kHz PCM little-endian byte packing and WAV I/O
└── constants.py     # Master DSP constants (FS=48000, target LUFS=-20.50, peak=0.9900)
```

---

## 3. Physical String Harmonic Damping & Chirp Synthesis (`sweeps.py`, `dry.py`)

### 3.1 The Physical String Tilt Law ($f_{\text{corner}} = 1200\text{ Hz}$)

Real vibrating electric bass strings exhibit steep high-frequency energy decay due to internal viscoelastic damping within the steel core and nickel/steel windings:
- Fundamental register ($40\text{--}120\text{ Hz}$): Full displacement excursion ($0\text{ dB}$).
- Clank & bite register ($2\text{--}4\text{ kHz}$): Vibrational energy down by $-25\text{ to } -35\text{ dB}$.
- Treble air register ($8\text{--}15\text{ kHz}$): Vibrational energy down by $-43.5\text{ to } -60\text{ dB}$.

To prevent unphysical high-frequency drive from destabilizing downstream neural network training, all synthetic logarithmic frequency chirps apply continuous physical string tilt filtering:

$$H_{\text{tilt}}(f) = \frac{1}{\sqrt{1 + \left(\frac{f}{f_{\text{corner}}}\right)^2}}, \quad f_{\text{corner}} = 1200.0\text{ Hz}$$

```python
# src/allomorph/dsp/sweeps.py
if string_tilt:
    # Continuous instantaneous frequency vector:
    f_t = f_start * (f_end / f_start) ** (t / dur)
    tilt_gain = 1.0 / np.sqrt(1.0 + (f_t / f_corner) ** 2)
    sig = sig * tilt_gain
```

### 3.2 Elimination of Untilted $22\text{ kHz}$ Flat Air Sweeps
In earlier versions, flat $22\text{ kHz}$ chirps without string tilt were tested to force high-frequency differentiator learning. This resulted in catastrophic zero-crossing rectification buzzing (+19 dB louder than reference) in WaveNet models because the neural network received 150x more high-frequency energy than physically possible from a vibrating bass string.

**Invariant:** All chirps in `synth_log_chirps` and `generate_optimal_bass_dry` strictly default to `string_tilt=True` with $f_{\text{corner}} = 1200\text{ Hz}$.

---

## 4. 2D Modal String Vibration & Articulation Suite (`plucks.py`)

### 4.1 Biaxial Modal String Vibration with Inharmonicity

String vibration is evaluated using a 2D modal model combining orthogonal vertical (perpendicular to pickup face) and horizontal (parallel to fretboard) vibration planes:

$$y(t) = \sum_{n=1}^{N_{\text{partials}}} A_n \cdot \left[ \sin(2\pi f_n t + \phi_{v, n}) \cdot e^{-d_{v, n} t} + \alpha_h \sin(2\pi f_n t + \phi_{h, n}) \cdot e^{-d_{h, n} t} \right]$$

1. **Inharmonicity Stiffness Dispersion ($B_s$):**
   $$f_n = n \cdot f_0 \cdot \sqrt{1 + B_s \cdot n^2}$$
   where $B_s$ is interpolated from empirical string stiffness calibration tables via exact Gaussian RBF solvers.
2. **Attack Pitch Sag (Non-Linear Elongation Transient):**
   During the initial pluck attack ($<40\text{ ms}$), high initial displacement increases string tension $\Delta T \propto \int (\partial y/\partial x)^2 dx$, momentarily sharpening the pitch:
   $$f(t) = f_0 \cdot \left(1.0 + \Delta_{\text{sag}} \cdot e^{-t / \tau_{\text{sag}}}\right), \quad \tau_{\text{sag}} \approx 30\text{--}45\text{ ms}$$
3. **Biaxial Damping Decay Rates:**
   Vertical polarization couples directly to the magnetic field and body bridge witness point, decaying faster than horizontal polarization:
   $$d_{v, n} = (0.70 + 0.10 n + 0.00025 n^2) \cdot \gamma_{\text{tech}}$$
   $$d_{h, n} = (0.35 + 0.05 n + 0.00012 n^2) \cdot \gamma_{\text{tech}}$$

### 4.2 Luthier Articulation Techniques

The engine synthesizes five distinct playing techniques:
- **Fingerstyle:** Smooth fleshy pad contact with warm low-mid bloom ($\gamma_{\text{tech}} = 1.45$).
- **Pick Strike:** Sharp plectrum attack with instantaneous transient burst and enhanced upper partials ($\gamma_{\text{tech}} = 1.10$).
- **Slap Thumb & Pop:** Aggressive percussive impact with boundary fret slap click and extended dynamic crest factor ($\gamma_{\text{tech}} = 0.95$).
- **Palm-Muted Staccato:** Viscoelastic hand damping filtering partials above $600\text{ Hz}$ via exponential rolloff ($\gamma_{\text{tech}} = 1.80$).
- **Natural Harmonics:** Pure crystalline sinusoidal overtones generated at nodes ($L/2, L/3, L/4$) with 5 partials, strictly omitting non-harmonic string modes.

---

## 5. ITU-R BS.1770-4 Loudness Matching & Peak Safety (`loudness.py`)

### 5.1 The NAM Training Loudness Reference Standard ($-20.50\text{ LUFS}$)

All dry excitation tracks and target wet simulation stems are leveled to the official NAM reference standard:
$$\text{Target Integrated Loudness} = -20.50\text{ LUFS}$$
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

## 6. Zero-Phase FIR Design & Minimum-Phase Synthesis (`fir.py`, `conditioning.py`)

### 6.1 Type-I Linear-Phase FIR Filter Design
Implemented directly in NumPy without heavy external dependencies:
$$h_{\text{win}}[n] = h_{\text{ideal}}[n] \cdot w_{\text{Blackman}}[n]$$
$$w_{\text{Blackman}}[n] = 0.42 - 0.50 \cos\left(\frac{2\pi n}{M-1}\right) + 0.08 \cos\left(\frac{4\pi n}{M-1}\right)$$
- Normalized to exact $1.00$ ($0.00\text{ dB}$) passband gain.
- Evaluated via PyTorch GPU GEMM 1D convolution (`F.conv1d`) with `padding="valid"` to eliminate boundary step transients.

### 6.2 Minimum-Phase Synthesis via Real-Cepstrum Hilbert Transform
For zero-latency pedalboard execution, linear magnitude responses $|H(f)|$ are converted to causal minimum-phase FIR kernels:
1. Log-magnitude spectrum: $X_{\text{log}}[k] = \ln(|H[k]| + \epsilon)$.
2. Real cepstrum: $c[n] = \text{IFFT}(X_{\text{log}})$.
3. Causal windowing: $c_{\text{min}}[n] = c[n] \cdot \text{sgn\_window}[n]$.
4. Minimum-phase impulse response: $h_{\text{min}}[n] = \text{real}(\text{FFT}(\exp(\text{FFT}(c_{\text{min}}))))$.
Guarantees strictly zero latency with peak energy concentrated at tap 0.

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
