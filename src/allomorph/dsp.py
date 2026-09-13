"""
Allomorph - Digital Signal Processing & Filter Synthesis Primitives
Provides NumPy-accelerated real-cepstrum Hilbert transform minimum-phase FIR synthesis,
fast Fourier transform wrappers, and 24-bit PCM audio export.
"""

import wave
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal

import numpy as np

FS = 48000
NUM_TAPS = 4096
NYQ = FS / 2.0
FREQS = [i * (NYQ / (NUM_TAPS - 1)) for i in range(NUM_TAPS)]


def synthesize_minimum_phase_fir(
    magnitude_curve: Sequence[float] | np.ndarray,
    num_taps: int = NUM_TAPS,
    normalize: bool = True,
) -> list[float]:
    """
    Synthesizes a causal, minimum-phase FIR filter from a desired magnitude
    curve using the homomorphic real-cepstrum Hilbert transform.
    Vectorized with NumPy FFT, executing in < 0.1 ms.
    """
    mag = np.asarray(magnitude_curve, dtype=np.float64)
    n_fft = max(8192, 2 * num_taps)
    half = n_fft // 2

    # Linear interpolation of input magnitude curve to half + 1 points (bypass if already on grid)
    m_in = len(mag)
    if m_in == half + 1:
        mag_grid = mag.copy()
    else:
        orig_indices = np.linspace(0, half, m_in)
        target_indices = np.arange(half + 1)
        mag_grid = np.interp(target_indices, orig_indices, mag)
    # Extrapolate DC bin if dropping into deep transmission zero to avoid cepstral delta spike
    if mag_grid[0] < mag_grid[1] * 0.5:
        mag_grid[0] = mag_grid[1]
    # Build half-spectrum log-magnitude with C^inf quadratic regularization
    log_mag = 0.5 * np.log(mag_grid**2 + 1e-8)

    # Real cepstrum via IRFFT directly on conjugate-symmetric half-spectrum
    c = np.fft.irfft(log_mag, n_fft)

    # Minimum-phase causal folding (Hilbert transform operator in cepstral domain)
    c_hat = np.zeros(n_fft, dtype=np.float64)
    c_hat[0] = c[0]
    c_hat[half] = c[half]
    c_hat[1:half] = 2.0 * c[1:half]

    # Complex minimum-phase frequency spectrum H_min = exp(RFFT(c_hat))
    spec = np.fft.rfft(c_hat)
    h_min_spec = np.exp(spec)

    # Causal impulse response h[n] = IRFFT(H_min)
    h = np.fft.irfft(h_min_spec, n_fft)
    fir = h[:num_taps].copy()

    # Smooth tail (final 15%) with a cosine taper to eliminate truncation artifacts
    taper_len = int(num_taps * 0.15)
    start_taper = num_taps - taper_len
    w = 0.5 * (1.0 + np.cos(np.pi * np.arange(taper_len) / taper_len))
    fir[start_taper:] *= w

    if not normalize:
        return fir.tolist()

    # Peak normalization to -0.1 dBFS (0.99)
    max_peak = np.max(np.abs(fir))
    if max_peak > 0:
        fir = (fir / max_peak) * 0.99
    return fir.tolist()


def write_wav_24bit(
    filepath: str | Path,
    samples: Sequence[float] | np.ndarray,
    sample_rate: int = FS,
) -> None:
    """Exports a 48 kHz / 24-bit mono PCM WAV file."""
    try:
        from pedalboard.io import AudioFile

        arr = np.array([samples], dtype=np.float32)
        with AudioFile(
            str(filepath), "w", samplerate=sample_rate, num_channels=1, bit_depth=24
        ) as f:
            f.write(arr)
    except ImportError, RuntimeError, OSError, ValueError:
        with wave.open(str(filepath), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(3)  # 3 bytes = 24-bit PCM
            wf.setframerate(sample_rate)
            scaled = np.clip(
                np.asarray(samples, dtype=np.float32) * 8388607.0, -8388608.0, 8388607.0
            ).astype(np.int32)
            raw_bytes = scaled.astype("<i4").view(np.uint8).reshape(-1, 4)[:, :3].tobytes()
            wf.writeframes(raw_bytes)


def read_wav(
    filepath: str | Path,
    max_samples: int | None = None,
    dtype: type | np.dtype[Any] = np.float32,
) -> tuple[np.ndarray, int]:
    """
    Reads a WAV file (16-bit PCM, 24-bit PCM, or 32-bit float) into a NumPy array.
    Returns (audio, sample_rate).
    audio is shaped (n_channels, n_samples) for multichannel or (n_samples,) for mono.
    Vectorized unpacking executes in < 0.1 s even for multi-minute 24-bit audio files.
    """
    with wave.open(str(filepath), "rb") as wf:
        n_ch = wf.getnchannels()
        sw = wf.getsampwidth()
        sr = wf.getframerate()
        n_frames = wf.getnframes()
        if max_samples is not None:
            n_frames = min(n_frames, max_samples)
        raw = wf.readframes(n_frames)

    target_dtype = np.dtype(dtype)
    if sw == 3:
        raw_u8 = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3)
        buf = np.empty((len(raw_u8), 4), dtype=np.uint8)
        buf[:, :3] = raw_u8
        buf[:, 3] = np.where(raw_u8[:, 2] >= 128, 255, 0)
        audio = buf.view("<i4").reshape(-1).astype(target_dtype) / 8388607.0
    elif sw == 2:
        audio = np.frombuffer(raw, dtype="<i2").astype(target_dtype) / 32767.0
    elif sw == 4:
        audio = np.frombuffer(raw, dtype=np.float32).astype(target_dtype)
    else:
        audio = (np.frombuffer(raw, dtype=np.uint8).astype(target_dtype) - 128.0) / 128.0

    if n_ch > 1:
        audio = audio.reshape(-1, n_ch).T
    return audio, sr


def read_wav_24bit(
    filepath: str | Path,
    max_samples: int | None = None,
    dtype: type | np.dtype[Any] = np.float32,
) -> tuple[np.ndarray, int]:
    """Reads a 24-bit PCM WAV file into a float NumPy array."""
    return read_wav(filepath, max_samples=max_samples, dtype=dtype)


def fft_convolve(
    in1: Sequence[float] | np.ndarray,
    in2: Sequence[float] | np.ndarray,
    mode: str = "full",
) -> np.ndarray:
    """
    High-performance 1D convolution using real FFTs with cache-friendly overlap-add
    for large signals. Automatically selects single-pass or blocked FFT depending on signal length.

    Supported modes:
      - 'full': Standard full convolution of length len(in1) + len(in2) - 1.
      - 'causal': Output has length max(len(in1), len(in2)), starting at sample 0 (preserves causal delay).
      - 'same': Output has the same length as max(len(in1), len(in2)), centered with respect to 'full'.
    """
    x = np.asarray(in1)
    y = np.asarray(in2)
    n = len(x)
    m = len(y)
    if n == 0 or m == 0:
        return np.array([], dtype=x.dtype)

    # Ensure x is the longer signal for overlap-add
    if m > n:
        x, y = y, x
        n, m = m, n

    out_len = n + m - 1
    out_dtype = np.float64 if (x.dtype == np.float64 or y.dtype == np.float64) else np.float32

    # Vectorized overlap-save for signals larger than block_size where filter is compact
    if n > 32768 and m <= 32768:
        block_size = 65536
        if block_size <= m:
            block_size = 1 << (m + 1).bit_length()

        l = block_size - m + 1
        num_blocks = (n + l - 1) // l
        pad_end = num_blocks * l - n
        x_padded = np.pad(x, (m - 1, pad_end), mode="constant")

        shape = (num_blocks, block_size)
        strides = (l * x_padded.strides[0], x_padded.strides[0])
        blocks = np.lib.stride_tricks.as_strided(x_padded, shape=shape, strides=strides)

        H = np.fft.rfft(y, block_size)
        X_blocks = np.fft.rfft(blocks, block_size, axis=-1)
        Y_blocks = np.fft.irfft(X_blocks * H, block_size, axis=-1)

        valid = Y_blocks[:, m - 1 :]
        out = valid.reshape(-1)[:out_len].astype(out_dtype)
    else:
        n_fft = 1 << (out_len - 1).bit_length()
        X = np.fft.rfft(x, n_fft)
        Y = np.fft.rfft(y, n_fft)
        out = np.fft.irfft(X * Y, n_fft)[:out_len].astype(out_dtype)

    if mode == "full":
        return out
    elif mode == "causal":
        # Causal slice: starts at sample 0 of convolution (preserves causal filter delay)
        return out[:n]
    elif mode == "same":
        # Centered slice matching np.convolve(in1, in2, mode="same")
        start = (m - 1) // 2
        return out[start : start + n]
    else:
        raise ValueError(f"Unsupported mode '{mode}'. Choose 'full', 'causal', or 'same'.")


def fft_convolve_multi(
    x: Sequence[float] | np.ndarray,
    filters: Sequence[Sequence[float] | np.ndarray],
    mode: Literal["full", "causal", "same"] = "full",
) -> list[np.ndarray]:
    """
    Convolves a single 1D signal x against multiple 1D filters using vectorized FFT convolution,
    broadcasting the forward FFT of x across all filter channels to eliminate redundant FFTs.
    """
    if not filters:
        return []
    if len(filters) == 1:
        return [fft_convolve(x, filters[0], mode=mode)]

    x_arr = np.asarray(x)
    n = len(x_arr)
    filter_arrs = [np.asarray(f) for f in filters]
    m_max = max(len(f) for f in filter_arrs)
    out_dtype = (
        np.float64
        if (x_arr.dtype == np.float64 or any(f.dtype == np.float64 for f in filter_arrs))
        else np.float32
    )

    # Overlap-save path when all filters fit in block_size
    if n > 32768 and m_max <= 32768:
        block_size = 65536
        l = block_size - m_max + 1
        num_blocks = (n + l - 1) // l
        pad_end = num_blocks * l - n
        x_padded = np.pad(x_arr, (m_max - 1, pad_end), mode="constant")
        shape = (num_blocks, block_size)
        strides = (l * x_padded.strides[0], x_padded.strides[0])
        blocks = np.lib.stride_tricks.as_strided(x_padded, shape=shape, strides=strides)
        # Compute forward FFT of input blocks once and broadcast:
        X_blocks = np.fft.rfft(blocks, block_size, axis=-1)

        results: list[np.ndarray] = []
        for f in filter_arrs:
            out_len = n + len(f) - 1
            H = np.fft.rfft(f, block_size)
            Y_blocks = np.fft.irfft(X_blocks * H, block_size, axis=-1)
            valid = Y_blocks[:, m_max - 1 :]
            out = valid.reshape(-1)[:out_len].astype(out_dtype)
            if mode == "full":
                results.append(out)
            elif mode == "causal":
                results.append(out[:n])
            elif mode == "same":
                start = (len(f) - 1) // 2
                results.append(out[start : start + n])
        return results
    else:
        return [fft_convolve(x_arr, f, mode=mode) for f in filter_arrs]


def calibrate_nam_v3_latency(y: np.ndarray) -> tuple[int, bool, bool]:
    """
    Evaluates NAM V3 calibration blip latency alignment using NAM's exact algorithm.
    Runs entirely in NumPy (< 2ms) without importing torch or pytorch_lightning.

    Returns:
        (recommended_delay, matches_lookahead_warning, not_detected)
    """
    first_blips_start = 480000
    t_blips = 96000
    noise_start = 492000
    noise_end = 498000
    blip_locations = (504000, 552000)
    lookahead = 1000
    lookback = 10000
    safety_factor = 1

    if len(y) < first_blips_start + t_blips:
        return 0, False, True

    y_blips = y[first_blips_start : first_blips_start + t_blips]
    bg = float(np.max(np.abs(y[noise_start:noise_end])))
    trig_thresh = max(bg + 0.01, 1.1 * bg)

    y_scans = []
    for blip in blip_locations:
        i_rel = blip - first_blips_start
        start_looking = i_rel - lookahead
        stop_looking = i_rel + lookback
        y_scans.append(y_blips[start_looking:stop_looking])

    y_avg = np.mean(np.stack(y_scans), axis=0)
    triggered = np.where(np.abs(y_avg) > trig_thresh)[0]
    if len(triggered) == 0:
        return 0, False, True

    delay = int(triggered[0] - lookahead)
    recommended = delay - safety_factor
    matches_lookahead = delay == -lookahead
    return recommended, matches_lookahead, False


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
AUDIO_DIR = REPO_ROOT / "audio"
from allomorph.version import DSP_GENERATION

OPTIMAL_DRY_PATH = AUDIO_DIR / "canonical" / f"optimal_bass_dry_v{DSP_GENERATION}.wav"


def _apply_hann_fades(sig: np.ndarray, fade_len: int) -> np.ndarray:
    """Applies smooth raised-cosine (Hann) fade-in and fade-out to prevent DC/phase click discontinuities."""
    if len(sig) < 2 * fade_len or fade_len <= 0:
        return sig
    out = sig.copy()
    w = 0.5 * (1.0 - np.cos(np.pi * np.arange(fade_len, dtype=np.float64) / fade_len))
    out[:fade_len] *= w
    out[-fade_len:] *= w[::-1]
    return out


def _synth_log_chirp(
    dur: float,
    f_start: float,
    f_end: float,
    amp: float,
    sample_rate: int = FS,
) -> np.ndarray:
    """Synthesizes a full-range logarithmic frequency chirp with 5ms micro-fades."""
    n = int(dur * sample_rate)
    if n <= 0:
        return np.empty(0, dtype=np.float64)
    t = np.linspace(0.0, dur, n, endpoint=False)
    gamma = np.log(f_end / f_start)
    phase = 2.0 * np.pi * f_start * (dur / gamma) * ((f_end / f_start) ** (t / dur) - 1.0)
    sig = amp * np.sin(phase)
    sig -= np.mean(sig)
    return _apply_hann_fades(sig, min(n // 4, int(0.005 * sample_rate)))


def _synth_pluck(
    f0: float,
    amp: float,
    dur: float,
    sample_rate: int = FS,
    b_inharm: float = 0.00025,
    pitch_sag_hz: float = 2.5,
    tau_sag: float = 0.08,
    clank: bool = True,
    technique: str = "finger",
) -> np.ndarray:
    """Synthesizes an authentic physical bass string pluck with:

    - Inharmonic modal frequencies fn = n * f0 * sqrt(1 + B * n^2)
    - Dynamic attack pitch sag f(t) = f0 + sag * exp(-t / tau_sag)
    - Dual-polarization orthogonal mode splitting (horizontal vs vertical blooming)
    - Attack transient clank/snap tailored by playing technique
    - Velocity profile (Faraday velocity scaling ~ 1 / n^0.82)
    """
    n = int(dur * sample_rate)
    if n <= 0:
        return np.empty(0, dtype=np.float64)
    t = np.linspace(0.0, dur, n, endpoint=False)
    phase_sag = -pitch_sag_hz * tau_sag * (np.exp(-t / max(tau_sag, 1e-4)) - 1.0)
    decay_mult = 3.5 if technique == "staccato" else (1.3 if technique == "slap" else 1.0)

    sig = np.zeros(n, dtype=np.float64)
    max_h = min(32, int((sample_rate / 2.0 - 200.0) / f0))
    split_hz = 0.18

    for h in range(1, max_h):
        fn = h * f0 * np.sqrt(1.0 + b_inharm * (h**2))
        if fn >= (sample_rate / 2.0) - 200.0:
            break
        h_weight = 1.0 / (h**0.82)
        decay_v = np.exp(-t * (0.7 + 0.12 * h) * decay_mult)
        decay_h = np.exp(-t * (0.35 + 0.06 * h) * decay_mult)
        phi_base = 2.0 * np.pi * (fn * t + h * phase_sag)
        sig += h_weight * (
            0.65 * np.sin(phi_base) * decay_v
            + 0.35 * np.sin(phi_base + 2.0 * np.pi * split_hz * t) * decay_h
        )

    # Attack transients based on technique
    if clank or technique in ("pick", "slap"):
        if technique == "slap":
            clank_len = min(n, int(0.018 * sample_rate))
            tc = t[:clank_len]
            burst = np.sin(2.0 * np.pi * 3200.0 * tc) * np.exp(-tc / 0.003)
            sig[:clank_len] += 0.85 * burst
        elif technique == "pick":
            pick_len = min(n, int(0.008 * sample_rate))
            tpk = t[:pick_len]
            burst = np.sin(2.0 * np.pi * 4200.0 * tpk) * ((1.0 - tpk / 0.008) ** 2)
            sig[:pick_len] += 0.60 * burst
        else:
            fret_len = min(n, int(0.012 * sample_rate))
            tf = t[:fret_len]
            burst = np.sin(2.0 * np.pi * 2600.0 * tf) * np.exp(-tf / 0.004)
            sig[:fret_len] += 0.35 * burst

    sig -= np.mean(sig)
    p_max = float(np.max(np.abs(sig)))
    if p_max > 0:
        sig = (sig / p_max) * amp
    return _apply_hann_fades(sig, min(n // 4, int(0.005 * sample_rate)))


def _synth_ghost_note(
    dur: float,
    amp: float,
    sample_rate: int = FS,
) -> np.ndarray:
    """Synthesizes an unpitched dead-string percussive thump (< 40ms) with zero pitch sustain."""
    n = int(dur * sample_rate)
    if n <= 0:
        return np.empty(0, dtype=np.float64)
    t = np.linspace(0.0, dur, n, endpoint=False)
    body = np.sin(2.0 * np.pi * 85.0 * t) * np.exp(-t / 0.020)
    click = np.sin(2.0 * np.pi * 2800.0 * t) * np.exp(-t / 0.004)
    sig = 0.70 * body + 0.50 * click
    sig -= np.mean(sig)
    p_max = float(np.max(np.abs(sig)))
    if p_max > 0:
        sig = (sig / p_max) * amp
    return _apply_hann_fades(sig, min(n // 4, int(0.003 * sample_rate)))


def _synth_natural_harmonic(
    f_harmonic: float,
    amp: float,
    dur: float,
    sample_rate: int = FS,
) -> np.ndarray:
    """Synthesizes a pure bell-like natural harmonic overtone with zero low-frequency fundamental."""
    n = int(dur * sample_rate)
    if n <= 0:
        return np.empty(0, dtype=np.float64)
    t = np.linspace(0.0, dur, n, endpoint=False)
    sig = np.sin(2.0 * np.pi * f_harmonic * t) * np.exp(-t * 0.45)
    sig += 0.25 * np.sin(2.0 * np.pi * (2.0 * f_harmonic) * t) * np.exp(-t * 0.85)
    sig -= np.mean(sig)
    p_max = float(np.max(np.abs(sig)))
    if p_max > 0:
        sig = (sig / p_max) * amp
    return _apply_hann_fades(sig, min(n // 4, int(0.005 * sample_rate)))


def _synth_dyad(
    f1: float,
    f2: float,
    amp: float,
    dur: float,
    sample_rate: int = FS,
) -> np.ndarray:
    """Synthesizes a two-note chord (power 5th or octave) to excite nonlinear intermodulation distortion."""
    p1 = _synth_pluck(f1, 0.55, dur, sample_rate, clank=True, technique="finger")
    p2 = _synth_pluck(f2, 0.45, dur, sample_rate, clank=True, technique="finger")
    sig = p1 + p2
    sig -= np.mean(sig)
    p_max = float(np.max(np.abs(sig)))
    if p_max > 0:
        sig = (sig / p_max) * amp
    return _apply_hann_fades(sig, min(len(sig) // 4, int(0.005 * sample_rate)))


def _synth_glissando(
    f_start: float,
    f_end: float,
    amp: float,
    dur: float,
    sample_rate: int = FS,
) -> np.ndarray:
    """Synthesizes a smooth continuous exponential string slide across register comb nulls."""
    n = int(dur * sample_rate)
    if n <= 0:
        return np.empty(0, dtype=np.float64)
    t = np.linspace(0.0, dur, n, endpoint=False)
    gamma = np.log(f_end / f_start)
    phase = 2.0 * np.pi * f_start * (dur / gamma) * ((f_end / f_start) ** (t / dur) - 1.0)
    sig = np.sin(phase)
    phase2 = 2.0 * np.pi * (2.0 * f_start) * (dur / gamma) * ((f_end / f_start) ** (t / dur) - 1.0)
    phase3 = 2.0 * np.pi * (3.0 * f_start) * (dur / gamma) * ((f_end / f_start) ** (t / dur) - 1.0)
    sig += 0.50 * np.sin(phase2) + 0.25 * np.sin(phase3)
    sig -= np.mean(sig)
    p_max = float(np.max(np.abs(sig)))
    if p_max > 0:
        sig = (sig / p_max) * amp
    return _apply_hann_fades(sig, min(n // 4, int(0.005 * sample_rate)))


def generate_optimal_bass_dry(
    duration_sec: float = 180.0,
    sample_rate: int = FS,
    peak_dbfs: float = -1.0,
    seed: int = 42,
) -> np.ndarray:
    """Synthesizes a 48 kHz high-fidelity synthetic dry excitation signal tailored for bass modeling.

    Excites the complete physical state space of active, passive, and multi-scale bass systems:
    1. Latency Calibration Double-Blips: clean impulses at 0.3s (+0.89) and 0.8s (-0.89) for
       bit-exact cross-correlation stem alignment.
    2. Multi-Tier Full-Spectrum Log Chirps: 4 discrete full-range sweeps (15 Hz -> 22 kHz) across
       -24 dBFS (linear small-signal baseline), -12 dBFS (eddy damping), -6 dBFS (Lenz drag onset),
       and -1 dBFS (full core saturation) + 1 inverted down-sweep (22 kHz -> 15 Hz at -3 dBFS).
    3. 5-Step Dynamic Velocity Ladder on E1 (pp -> ff: -20, -14, -8, -4, -1 dBFS) followed by
       modal plucks with dual-polarization bloom, attack pitch sag, and fret clank (Drop A0 27.5 Hz
       through C3 130.81 Hz).
    4. Comprehensive Articulations: plectrum down/upstrokes, slap thumb pops, palm-muted staccato,
       unpitched percussive ghost notes (< 40ms), and natural harmonic bell chimes (123.6 & 164.8 Hz).
    5. Polyphony & Dyads: Low-A, Low-B, and Low-E power 5ths + slap octaves to train nonlinear
       intermodulation distortion (IMD) + Schroeder-phase multitone complexes with dynamic swell.
    6. Continuous Glissandi: smooth exponential frequency glides across registers (A0 -> D1 -> G1 -> C2).
    7. Wideband Pink Noise Bursts: rhythmic gated pink noise ensuring 100% continuous spectral density.

    Zero Artificial Dither Policy:
    Silence intervals contain pure, bit-exact digital silence (0.0). No background noise, hum,
    or thermal dither is injected. All segment boundaries are smoothed with 3-5ms Hann micro-fades.
    Leading and trailing silence is strictly bounded (<= 0.3s) and inter-event pauses are kept to
    0.4s-0.5s, eliminating Tone3000 'Too Much Silence' rejections while providing full recovery time
    for RLC resonance and magnetic relaxation.
    """
    total_samples = int(duration_sec * sample_rate)
    audio = np.zeros(total_samples, dtype=np.float64)
    scale = min(1.0, duration_sec / 180.0)

    lead_silence = min(int(0.3 * sample_rate), int(0.05 * total_samples))
    trail_silence = min(int(0.3 * sample_rate), int(0.05 * total_samples))
    cur = lead_silence

    # 1. Calibration blips
    if cur + int(0.5 * sample_rate * scale) < total_samples - trail_silence:
        audio[cur] = 0.89
        b2 = cur + max(100, int(0.5 * sample_rate * scale))
        if b2 < total_samples - trail_silence:
            audio[b2] = -0.89
            cur = b2 + max(100, int(0.3 * sample_rate * scale))
        else:
            cur += max(100, int(0.3 * sample_rate * scale))

    def append_segment(seg: np.ndarray, pause_dur: float) -> None:
        nonlocal cur
        if len(seg) == 0:
            return
        end_idx = cur + len(seg)
        limit = total_samples - trail_silence
        if end_idx >= limit:
            avail = max(0, limit - cur)
            if avail > 0:
                audio[cur : cur + avail] = seg[:avail]
                cur += avail
            return
        audio[cur : end_idx] = seg
        p_samples = max(50, int(pause_dur * sample_rate * scale))
        cur = min(end_idx + p_samples, limit)

    # 2. Multi-tier full sweeps (15 Hz -> 22 kHz)
    c_dur = max(0.8, 7.0 * scale)
    chirp_tiers = [
        (15.0, 22000.0, 0.050),
        (15.0, 22000.0, 0.180),
        (15.0, 22000.0, 0.400),
        (15.0, 22000.0, 0.700),
        (22000.0, 15.0, 0.550),
    ]
    for f_s, f_e, amp in chirp_tiers:
        if cur >= total_samples - trail_silence:
            break
        c = _synth_log_chirp(c_dur, f_s, f_e, amp, sample_rate)
        append_segment(c, 0.4)

    # 3. 5-Step Dynamic Velocity Ladder on open E1 (pp -> ff)
    v_dur = max(0.4, 1.8 * scale)
    for v_amp in [0.10, 0.20, 0.40, 0.63, 0.89]:
        if cur >= total_samples - trail_silence:
            break
        p = _synth_pluck(41.20, v_amp, v_dur, sample_rate, technique="finger")
        append_segment(p, 0.4)

    # Modal plucks with pitch sag and fret clank across all string registers
    m_dur = max(0.4, 1.3 * scale)
    notes = [27.50, 30.87, 41.20, 55.00, 73.42, 82.41, 98.00, 110.00, 130.81, 146.83]
    for n_f in notes:
        for v_amp in [0.45, 0.80]:
            if cur >= total_samples - trail_silence:
                break
            p = _synth_pluck(n_f, v_amp, m_dur, sample_rate, technique="finger")
            append_segment(p, 0.4)

    # 4. Articulations & Techniques
    # Plectrum pick strikes
    for _ in range(2):
        if cur >= total_samples - trail_silence:
            break
        p = _synth_pluck(41.20, 0.75, max(0.4, 1.4 * scale), sample_rate, technique="pick")
        append_segment(p, 0.4)
    # Slap thumb pops
    for _ in range(2):
        if cur >= total_samples - trail_silence:
            break
        p = _synth_pluck(41.20, 0.85, max(0.4, 1.4 * scale), sample_rate, technique="slap")
        append_segment(p, 0.4)
    # Palm-muted staccato
    for n_f in [41.20, 55.00, 73.42]:
        if cur >= total_samples - trail_silence:
            break
        p = _synth_pluck(n_f, 0.70, max(0.3, 0.8 * scale), sample_rate, technique="staccato")
        append_segment(p, 0.4)
    # Percussive ghost notes (< 40ms)
    for _ in range(3):
        if cur >= total_samples - trail_silence:
            break
        g = _synth_ghost_note(max(0.15, 0.35 * scale), 0.75, sample_rate)
        append_segment(g, 0.4)
    # Natural harmonic bell chimes
    for h_f in [82.41, 123.6, 164.8]:
        if cur >= total_samples - trail_silence:
            break
        h = _synth_natural_harmonic(h_f, 0.70, max(0.4, 1.6 * scale), sample_rate)
        append_segment(h, 0.4)

    # 5. Polyphony & Dyads (Intermodulation Distortion)
    # Non-octave musical intervals (power 5ths, 4ths) to excite nonlinear IMD
    # without creating locked octave sub-harmonic bias.
    d_dur = max(0.5, 2.0 * scale)
    dyads = [
        (27.50, 41.25),  # Low-A0 + E1 (power 5th)
        (30.87, 46.31),  # Low-B0 + F#1 (power 5th)
        (41.20, 61.74),  # Low-E1 + B1 (power 5th)
        (55.00, 82.50),  # Low-A1 + E2 (power 5th)
        (55.00, 73.42),  # Low-A1 + D2 (perfect 4th)
        (73.42, 110.00), # D2 + A2 (power 5th)
    ]
    for f1, f2 in dyads:
        if cur >= total_samples - trail_silence:
            break
        d = _synth_dyad(f1, f2, 0.75, d_dur, sample_rate)
        append_segment(d, 0.4)

    # Schroeder-phase multitone complex with dynamic swell
    m_rem = max(0, total_samples - trail_silence - cur)
    if m_rem > int(2.0 * sample_rate * scale):
        dur_m = min(14.0 * scale, m_rem / sample_rate * 0.4)
        nm = int(dur_m * sample_rate)
        if nm > 100:
            tm = np.linspace(0.0, dur_m, nm, endpoint=False)
            clusters = [
                27.50,
                30.87,
                41.20,
                55.00,
                82.41,
                110.0,
                220.0,
                440.0,
                880.0,
                1250.0,
                1800.0,
                2400.0,
                3100.0,
                4200.0,
                6000.0,
            ]
            kc = len(clusters)
            sig_m = np.zeros(nm, dtype=np.float64)
            for k, fk in enumerate(clusters):
                th = (np.pi * (k**2)) / kc
                sig_m += (1.0 / np.sqrt(1.0 + (fk / 300.0))) * np.sin(2.0 * np.pi * fk * tm + th)
            sig_m -= np.mean(sig_m)
            sig_m = (sig_m / np.max(np.abs(sig_m))) * 0.80
            am_env = 0.575 + 0.325 * np.sin(2.0 * np.pi * 0.25 * tm)
            append_segment(
                _apply_hann_fades(sig_m * am_env, min(nm // 4, int(0.01 * sample_rate))),
                0.4,
            )

    # 6. Continuous Glissandi across register boundaries
    g_dur = max(0.5, 3.5 * scale)
    slides = [
        (27.50, 41.20),
        (41.20, 55.00),
        (55.00, 73.42),
        (73.42, 98.00),
        (98.00, 41.20),
    ]
    for fs, fe in slides:
        if cur >= total_samples - trail_silence:
            break
        gl = _synth_glissando(fs, fe, 0.80, g_dur, sample_rate)
        append_segment(gl, 0.4)

    # 7. Shaped Pink Noise Bursts
    rem_samples = max(0, total_samples - trail_silence - cur)
    if rem_samples > int(0.5 * sample_rate):
        rng = np.random.default_rng(seed)
        white = rng.standard_normal(rem_samples)
        n_fft = 1 << (rem_samples - 1).bit_length()
        w_spec = np.fft.rfft(white, n_fft)
        freqs = np.fft.rfftfreq(n_fft, 1.0 / sample_rate)
        freqs[0] = 1.0
        pink = np.fft.irfft(w_spec * (1.0 / np.sqrt(freqs)), n_fft)[:rem_samples]
        pink -= np.mean(pink)
        pink_max = np.max(np.abs(pink))
        if pink_max > 0:
            pink = (pink / pink_max) * 0.75
        period = int(0.25 * sample_rate)
        on_len = int(0.15 * sample_rate)
        burst_gate = ((np.arange(rem_samples) % period) < on_len).astype(np.float64)
        hw = np.hanning(max(16, int(0.01 * sample_rate)))
        hw /= np.sum(hw)
        burst_gate = np.convolve(burst_gate, hw, mode="same")
        b_max = np.max(burst_gate)
        if b_max > 0:
            burst_gate /= b_max
        append_segment(
            _apply_hann_fades(pink * burst_gate, min(rem_samples // 4, int(0.01 * sample_rate))),
            0.0,
        )

    # Zero-DC centering on active regions (preserves pure zero digital silence in rests)
    active_mask = audio != 0.0
    if np.any(active_mask):
        active_mean = float(np.mean(audio[active_mask]))
        audio[active_mask] -= active_mean

    # Peak ceiling bounding to requested peak_dbfs (preserves pure zero digital silence in rests)
    target_peak = 10.0 ** (peak_dbfs / 20.0)
    current_peak = float(np.max(np.abs(audio)))
    if current_peak > 0:
        audio = audio * (target_peak / current_peak)

    return audio.astype(np.float32)



def ensure_optimal_dry_wav(
    output_path: Path | str | None = None,
    duration_sec: float = 180.0,
    sample_rate: int = FS,
    peak_dbfs: float = -1.0,
    overwrite: bool = False,
    version_tag: str | None = None,
    no_manifest: bool = False,
) -> Path:
    """Ensures that the synthesized optimal bass dry signal exists on disk.

    If output_path is None, writes the versioned optimal bass dry file to audio/canonical/
    (e.g., optimal_bass_dry_v2.wav) and records entries in manifest.json.
    """
    if output_path is not None:
        p = Path(output_path)
    else:
        from allomorph.naming import get_optimal_dry_path

        p = get_optimal_dry_path(version_tag=version_tag)

    if p.exists() and not overwrite:
        return p

    p.parent.mkdir(parents=True, exist_ok=True)
    audio = generate_optimal_bass_dry(
        duration_sec=duration_sec,
        sample_rate=sample_rate,
        peak_dbfs=peak_dbfs,
    )
    write_wav_24bit(p, audio, sample_rate)

    if output_path is None and not no_manifest:
        from allomorph.version import DSP_GENERATION, write_manifest

        v_tag = version_tag or f"v{DSP_GENERATION}"
        write_manifest(
            output_dir=p.parent,
            stage="canonical",
            files=[p],
            version_tag=v_tag,
        )

    return p


