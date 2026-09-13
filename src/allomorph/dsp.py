"""
Allomorph - Digital Signal Processing & Filter Synthesis Primitives
Provides NumPy-accelerated real-cepstrum Hilbert transform minimum-phase FIR synthesis,
fast Fourier transform wrappers, and 24-bit PCM audio export.
"""

import wave
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

if TYPE_CHECKING:
    prange = range

    def njit(*args: Any, **kwargs: Any) -> Callable[[Any], Any]:
        def decorator(func: Any) -> Any:
            return func

        return decorator

    _HAS_NUMBA = False
else:
    try:
        from numba import njit, prange

        _HAS_NUMBA = True
    except ImportError:

        def njit(*args: Any, **kwargs: Any) -> Callable[[Any], Any]:
            def decorator(func: Any) -> Any:
                return func

            return decorator

        def prange(*args: Any) -> Any:
            return range(*args)

        _HAS_NUMBA = False

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


@njit(fastmath=True)
def _cinf_smoothstep_kernel(t: np.ndarray, out: np.ndarray) -> None:
    n = len(t)
    for i in range(n):
        ti = t[i]
        if ti <= 0.0:
            out[i] = 0.0
        elif ti >= 1.0:
            out[i] = 1.0
        else:
            arg = (1.0 - 2.0 * ti) / (ti * (1.0 - ti))
            if arg > 80.0:
                arg = 80.0
            elif arg < -80.0:
                arg = -80.0
            out[i] = 1.0 / (1.0 + np.exp(arg))


def _cinf_smoothstep(t: np.ndarray) -> np.ndarray:
    """Computes the real-analytic C^infinity smoothstep mollifier transition function S(t).

    All derivatives of all orders are identically zero at t <= 0 and t >= 1, eliminating
    high-order spectral boundary leakage into digital silence.
    """
    out = np.zeros_like(t, dtype=np.float64)
    _cinf_smoothstep_kernel(t, out)
    return out


@njit(fastmath=True, parallel=True)
def _accumulate_pluck_modes_simd(
    sig: np.ndarray,
    t: np.ndarray,
    phase_sag: np.ndarray,
    fns: np.ndarray,
    hws: np.ndarray,
    drvs: np.ndarray,
    drhs: np.ndarray,
    hs: np.ndarray,
    split_hz: float,
) -> None:
    n = len(t)
    num_modes = len(fns)
    two_pi = 2.0 * np.pi
    two_pi_split = two_pi * split_hz
    for i in prange(n):
        ti = t[i]
        sag_i = phase_sag[i]
        acc = 0.0
        for m in range(num_modes):
            fn = fns[m]
            hw = hws[m]
            drv = drvs[m]
            drh = drhs[m]
            h = hs[m]
            phi_base = two_pi * (fn * ti + h * sag_i)
            decay_v = np.exp(-ti * drv)
            decay_h = np.exp(-ti * drh)
            acc += hw * (
                0.65 * np.sin(phi_base) * decay_v
                + 0.35 * np.sin(phi_base + two_pi_split * ti) * decay_h
            )
        sig[i] += acc


@njit(fastmath=True)
def _accumulate_ringout_phasors_simd(
    sig: np.ndarray,
    n: int,
    dt: float,
    fns: np.ndarray,
    hws: np.ndarray,
    drvs: np.ndarray,
    drhs: np.ndarray,
    split_hz: float,
) -> None:
    num_modes = len(fns)
    two_pi = 2.0 * np.pi
    for m in range(num_modes):
        fn = fns[m]
        hw = hws[m]
        drv = drvs[m]
        drh = drhs[m]

        w_v = two_pi * fn
        mult_v = np.exp(-drv * dt) * (np.cos(w_v * dt) + 1j * np.sin(w_v * dt))
        w_h = two_pi * (fn + split_hz)
        mult_h = np.exp(-drh * dt) * (np.cos(w_h * dt) + 1j * np.sin(w_h * dt))

        z_v = 1.0 + 0.0j
        z_h = 1.0 + 0.0j
        c_v = 0.65 * hw
        c_h = 0.35 * hw
        for i in range(n):
            sig[i] += c_v * z_v.imag + c_h * z_h.imag
            z_v *= mult_v
            z_h *= mult_h


@njit(fastmath=True, parallel=True)
def _accumulate_vibrato_modes_simd(
    sig: np.ndarray,
    t: np.ndarray,
    base_phase: np.ndarray,
    fns: np.ndarray,
    hws: np.ndarray,
    drs: np.ndarray,
    hs: np.ndarray,
) -> None:
    n = len(t)
    num_modes = len(fns)
    for i in prange(n):
        ti = t[i]
        bp_i = base_phase[i]
        acc = 0.0
        for m in range(num_modes):
            hw = hws[m]
            dr = drs[m]
            h = hs[m]
            decay = np.exp(-ti * dr)
            acc += hw * np.sin(h * bp_i) * decay
        sig[i] += acc


@njit(fastmath=True, parallel=True)
def _accumulate_glissando_modes_simd(
    sig: np.ndarray,
    base_phase: np.ndarray,
    ks: np.ndarray,
    k_weights: np.ndarray,
) -> None:
    n = len(base_phase)
    num_k = len(ks)
    for i in prange(n):
        bp_i = base_phase[i]
        acc = 0.0
        for m in range(num_k):
            k = ks[m]
            kw = k_weights[m]
            acc += kw * np.sin(k * bp_i)
        sig[i] += acc


def _apply_cinf_fades(sig: np.ndarray, fade_len: int) -> np.ndarray:
    """Applies C^infinity smooth mollifier fade-in and fade-out to guarantee zero high-order
    spectral leakage and infinitely differentiable boundary transitions into digital silence.
    """
    if len(sig) < 2 * fade_len or fade_len <= 0:
        return sig
    out = sig.copy()
    t_in = np.linspace(0.0, 1.0, fade_len, endpoint=False)
    w_in = _cinf_smoothstep(t_in)
    out[:fade_len] *= w_in
    out[-fade_len:] *= w_in[::-1]
    return out


# Backward-compatibility alias ensuring full C^infinity boundary smoothness across all synthesis stages
_apply_hann_fades = _apply_cinf_fades


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


def _calc_inharmonicity_b(f0: float) -> float:
    """Calculates physical string inharmonicity coefficient B as a function of fundamental frequency f0.

    Heavy low-B0 and low-E1 wound strings exhibit higher stiffness (B ~ 0.00020 - 0.00025),
    while thin upper register strings exhibit lower inharmonicity (B ~ 0.00008 - 0.00010).
    """
    return float(0.00008 + 0.00018 * np.exp(-f0 / 75.0))


def _synth_pluck(
    f0: float,
    amp: float,
    dur: float,
    sample_rate: int = FS,
    b_inharm: float | None = None,
    pitch_sag_hz: float = 2.5,
    tau_sag: float = 0.08,
    clank: bool = True,
    technique: str = "finger",
) -> np.ndarray:
    """Synthesizes an authentic physical bass string pluck with:

    - Register-dependent inharmonic modal frequencies fn = n * f0 * sqrt(1 + B(f0) * n^2)
    - Dynamic attack pitch sag f(t) = f0 + sag * exp(-t / tau_sag)
    - Dual-polarization orthogonal mode splitting (horizontal vs vertical blooming)
    - Multi-cycle kinematic fret collision clank/snap tailored by playing technique
    - Pluck position geometric scaling with quadrature floor g_pos(h) = max(sin(h * pi * xp/L), 0.25)
    - Technique-dependent spectral decay exponent (slap=0.65, pick=0.72, finger=0.92, palm_mute=1.60)
    - Quadratic saddle bending damping (0.7 + 0.10*h + 0.00025*h^2)
    """
    n = int(dur * sample_rate)
    if n <= 0:
        return np.empty(0, dtype=np.float64)
    t = np.linspace(0.0, dur, n, endpoint=False)
    phase_sag = -pitch_sag_hz * tau_sag * (np.exp(-t / max(tau_sag, 1e-4)) - 1.0)
    decay_mult = (
        6.0
        if technique == "palm_mute"
        else (3.5 if technique == "staccato" else (1.3 if technique == "slap" else 1.0))
    )

    b_coeff = _calc_inharmonicity_b(f0) if b_inharm is None else b_inharm
    sig = np.zeros(n, dtype=np.float64)
    max_h = min(128, int((sample_rate / 2.0 - 200.0) / f0))
    split_hz = 0.18

    gamma = (
        0.65
        if technique == "slap"
        else (0.72 if technique == "pick" else (1.60 if technique == "palm_mute" else 0.92))
    )

    fns: list[float] = []
    hws: list[float] = []
    drvs: list[float] = []
    drhs: list[float] = []
    hs: list[float] = []
    for h in range(1, max_h):
        fn = h * f0 * np.sqrt(1.0 + b_coeff * (h**2))
        if fn >= (sample_rate / 2.0) - 200.0:
            break
        geo_pos = max(float(np.sin(h * np.pi * 0.14)), 0.25)
        h_weight = (1.0 / (h**gamma)) * geo_pos
        if technique == "palm_mute" and fn > 600.0:
            h_weight *= float(np.exp(-(fn - 600.0) / 300.0))
        d_rate_v = (0.7 + 0.10 * h + 0.00025 * (h**2)) * decay_mult
        d_rate_h = (0.35 + 0.05 * h + 0.00012 * (h**2)) * decay_mult
        fns.append(fn)
        hws.append(h_weight)
        drvs.append(d_rate_v)
        drhs.append(d_rate_h)
        hs.append(float(h))

    if fns:
        _accumulate_pluck_modes_simd(
            sig,
            t,
            phase_sag,
            np.asarray(fns, dtype=np.float64),
            np.asarray(hws, dtype=np.float64),
            np.asarray(drvs, dtype=np.float64),
            np.asarray(drhs, dtype=np.float64),
            np.asarray(hs, dtype=np.float64),
            split_hz,
        )

    # Initial physical string displacement asymmetry (exercises even-order alpha_2 magnetic nonlinearity)
    if amp >= 0.70 and technique != "palm_mute":
        asym = 0.12 * amp * np.exp(-t / 0.012)
        sig += asym

    # Attack transients based on technique and C^infinity kinematic fret collision
    if clank or technique in ("pick", "slap"):
        if technique == "slap":
            burst = np.sin(2.0 * np.pi * 3200.0 * t) * np.exp(-t / 0.004)
            # C^infinity softplus kinematic contact force: string strikes fretwire on negative excursions
            contact_force = np.logaddexp(0.0, 12.0 * (-np.sin(2.0 * np.pi * f0 * t))) / 12.0
            collision = (contact_force**3) * np.exp(-t / 0.025)
            sig += 0.85 * burst + 0.45 * collision
        elif technique == "pick":
            burst = np.sin(2.0 * np.pi * 4200.0 * t) * np.exp(-t / 0.0035)
            sig += 0.60 * burst
        elif technique == "palm_mute":
            pass  # Zero clank on palm-muted thumps
        else:
            burst = np.sin(2.0 * np.pi * 2600.0 * t) * np.exp(-t / 0.004)
            if amp >= 0.70:
                # Multi-cycle fret buzz on hard finger plucks decaying over ~50ms
                contact_force = np.logaddexp(0.0, 12.0 * (-np.sin(2.0 * np.pi * f0 * t))) / 12.0
                collision = (contact_force**3) * np.exp(-t / 0.018)
                sig += 0.35 * burst + 0.30 * collision
            else:
                sig += 0.35 * burst

    sig -= np.mean(sig)
    p_max = float(np.max(np.abs(sig)))
    if p_max > 0:
        sig = (sig / p_max) * amp
    return _apply_hann_fades(sig, min(n // 4, int(0.005 * sample_rate)))


def _synth_ghost_note(
    dur: float,
    amp: float,
    sample_rate: int = FS,
    seed: int = 101,
) -> np.ndarray:
    """Synthesizes an authentic unpitched dead-string percussive thump (< 40ms)
    combining a damped body/bridge resonance cluster (115 Hz, 230 Hz, 380 Hz)
    with a broadband pick/flesh friction scrape (1.0 - 5.0 kHz).
    """
    n = int(dur * sample_rate)
    if n <= 0:
        return np.empty(0, dtype=np.float64)
    t = np.linspace(0.0, dur, n, endpoint=False)
    m1 = np.sin(2.0 * np.pi * 115.0 * t) * np.exp(-t / 0.016)
    m2 = np.sin(2.0 * np.pi * 230.0 * t) * np.exp(-t / 0.012)
    m3 = np.sin(2.0 * np.pi * 380.0 * t) * np.exp(-t / 0.008)
    body = 0.50 * m1 + 0.35 * m2 + 0.25 * m3

    # Deterministic broadband high-frequency flesh/fret contact transient
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal(n)
    scrape = np.diff(noise, prepend=noise[0])
    scrape_click = np.sin(2.0 * np.pi * 3200.0 * t) * np.exp(-t / 0.004)
    transient = (0.35 * scrape + 0.40 * scrape_click) * np.exp(-t / 0.006)

    sig = body + transient
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
    num_partials: int = 5,
) -> np.ndarray:
    """Synthesizes a pure crystalline natural harmonic overtone cascade with high mechanical Q
    and zero low-frequency fundamental masking, providing high-SNR RLC resonance excitation.
    """
    n = int(dur * sample_rate)
    if n <= 0:
        return np.empty(0, dtype=np.float64)
    t = np.linspace(0.0, dur, n, endpoint=False)
    sig = np.zeros(n, dtype=np.float64)
    for k in range(1, num_partials + 1):
        fk = k * f_harmonic
        if fk >= (sample_rate / 2.0) - 200.0:
            break
        decay_rate = 0.25 + 0.10 * k
        decay = np.exp(-t * decay_rate)
        weight = 1.0 / (k**0.85)
        sig += weight * np.sin(2.0 * np.pi * fk * t) * decay

    # C^infinity fingertip release chime transient
    chime = np.sin(2.0 * np.pi * 4800.0 * t) * np.exp(-t / 0.003)
    sig += 0.20 * chime

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
    stagger_ms: float = 12.0,
) -> np.ndarray:
    """Synthesizes a two-note chord (power 5th or root-tenth) with micro-staggered string pluck
    to excite nonlinear intermodulation distortion without artificial sample-0 transient collision.
    """
    stagger_samples = int((stagger_ms / 1000.0) * sample_rate)
    dur2 = max(0.2, dur - (stagger_ms / 1000.0))
    p1 = _synth_pluck(f1, 0.55, dur, sample_rate, clank=True, technique="finger")
    p2 = _synth_pluck(f2, 0.45, dur2, sample_rate, clank=True, technique="finger")
    total_len = max(len(p1), stagger_samples + len(p2))
    sig = np.zeros(total_len, dtype=np.float64)
    sig[: len(p1)] += p1
    sig[stagger_samples : stagger_samples + len(p2)] += p2
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
    num_harmonics: int = 8,
) -> np.ndarray:
    """Synthesizes an authentic continuous string glissando across register comb nulls
    and pickup RLC resonances with multi-harmonic extension and wound-string wrap friction.
    """
    n = int(dur * sample_rate)
    if n <= 0:
        return np.empty(0, dtype=np.float64)
    t = np.linspace(0.0, dur, n, endpoint=False)
    gamma = np.log(f_end / f_start)
    base_phase = 2.0 * np.pi * f_start * (dur / gamma) * ((f_end / f_start) ** (t / dur) - 1.0)
    sig = np.zeros(n, dtype=np.float64)
    ks: list[float] = []
    kws: list[float] = []
    for k in range(1, num_harmonics + 1):
        max_fk = k * max(f_start, f_end)
        if max_fk >= (sample_rate / 2.0) - 200.0:
            break
        ks.append(float(k))
        kws.append(1.0 / (k**0.88))

    if ks:
        _accumulate_glissando_modes_simd(
            sig,
            base_phase,
            np.asarray(ks, dtype=np.float64),
            np.asarray(kws, dtype=np.float64),
        )

    num_frets = abs(12.0 * np.log2(f_end / f_start))
    if num_frets > 1.0:
        fret_clicks = np.sin(2.0 * np.pi * num_frets * (t / dur)) ** 16
        sig += 0.08 * fret_clicks * np.sin(2.0 * np.pi * 3200.0 * t)

    sig -= np.mean(sig)
    p_max = float(np.max(np.abs(sig)))
    if p_max > 0:
        sig = (sig / p_max) * amp
    return _apply_hann_fades(sig, min(n // 4, int(0.005 * sample_rate)))


def _synth_ghost_rake(
    f0: float,
    amp: float,
    dur: float,
    sample_rate: int = FS,
) -> np.ndarray:
    """Synthesizes a percussive funk dead-note rake (3 muted clicks leading into an accented downbeat)."""
    c1 = _synth_ghost_note(0.040, amp * 0.70, sample_rate, seed=101)
    c2 = _synth_ghost_note(0.040, amp * 0.75, sample_rate, seed=102)
    c3 = _synth_ghost_note(0.045, amp * 0.85, sample_rate, seed=103)
    p_dur = max(0.2, dur - 0.125)
    p = _synth_pluck(f0, amp, p_dur, sample_rate, clank=True, technique="finger")
    return np.concatenate([c1, c2, c3, p])


def _synth_groove_burst(
    f0: float,
    amp: float,
    bpm: float,
    count: int,
    sample_rate: int = FS,
    technique: str = "finger",
    rest_ms: float = 25.0,
) -> np.ndarray:
    """Synthesizes rapid repeated plucks at a given tempo with short micro-rests,
    exercising steady-state non-zero initial envelope memory ('pumped' core saturation).
    """
    interval_sec = 60.0 / (bpm * 4.0) if bpm > 0 else 0.125
    rest_sec = rest_ms / 1000.0
    pluck_dur = max(0.04, interval_sec - rest_sec)
    p = _synth_pluck(f0, amp, pluck_dur, sample_rate, clank=True, technique=technique)
    rest_samples = max(10, int(rest_sec * sample_rate))
    rest = np.zeros(rest_samples, dtype=np.float64)
    pieces: list[np.ndarray] = []
    for i in range(count):
        stroke_amp = 1.0 if (i % 4 == 0) else (0.86 if i % 2 == 0 else 0.78)
        pieces.append(p * stroke_amp)
        if i < count - 1:
            pieces.append(rest)
    return np.concatenate(pieces)


def _synth_slap_pop_pair(
    f_slap: float,
    f_pop: float,
    amp: float,
    gap_ms: float = 70.0,
    dur: float = 1.4,
    sample_rate: int = FS,
) -> np.ndarray:
    """Synthesizes an authentic funk slap-and-pop pair: heavy thumb slap followed by octave pop."""
    slap_part = _synth_pluck(f_slap, amp, dur, sample_rate, clank=True, technique="slap")
    pop_dur = max(0.4, dur - gap_ms / 1000.0)
    pop_part = _synth_pluck(f_pop, amp * 0.90, pop_dur, sample_rate, clank=True, technique="pick")
    gap_samples = int((gap_ms / 1000.0) * sample_rate)
    total_len = max(len(slap_part), gap_samples + len(pop_part))
    composite = np.zeros(total_len, dtype=np.float64)
    composite[: len(slap_part)] += slap_part
    composite[gap_samples : gap_samples + len(pop_part)] += pop_part
    composite -= np.mean(composite)
    p_max = float(np.max(np.abs(composite)))
    if p_max > 0:
        composite = (composite / p_max) * amp
    return _apply_hann_fades(composite, min(len(composite) // 4, int(0.005 * sample_rate)))


def _synth_vibrato_pluck(
    f0: float,
    amp: float,
    dur: float,
    mod_rate: float = 5.0,
    mod_depth_cents: float = 25.0,
    sample_rate: int = FS,
) -> np.ndarray:
    """Synthesizes a sustained bass pluck with left-hand finger vibrato sweeping across RLC resonances."""
    n = int(dur * sample_rate)
    if n <= 0:
        return np.empty(0, dtype=np.float64)
    t = np.linspace(0.0, dur, n, endpoint=False)
    delta = 2.0 ** (mod_depth_cents / 1200.0) - 1.0
    phase_mod = -delta * (f0 / mod_rate) * np.cos(2.0 * np.pi * mod_rate * t)
    base_phase = 2.0 * np.pi * f0 * t + 2.0 * np.pi * phase_mod
    sig = np.zeros(n, dtype=np.float64)
    max_h = min(110, int((sample_rate / 2.0 - 200.0) / f0))
    fns: list[float] = []
    hws: list[float] = []
    drs: list[float] = []
    hs: list[float] = []
    for h in range(1, max_h):
        fn = h * f0
        if fn >= (sample_rate / 2.0) - 200.0:
            break
        geo_pos = max(float(np.sin(h * np.pi * 0.14)), 0.25)
        h_weight = (1.0 / (h**0.92)) * geo_pos
        decay_rate = 0.5 + 0.07 * h + 0.00018 * (h**2)
        fns.append(fn)
        hws.append(h_weight)
        drs.append(decay_rate)
        hs.append(float(h))

    if fns:
        _accumulate_vibrato_modes_simd(
            sig,
            t,
            base_phase,
            np.asarray(fns, dtype=np.float64),
            np.asarray(hws, dtype=np.float64),
            np.asarray(drs, dtype=np.float64),
            np.asarray(hs, dtype=np.float64),
        )
    sig -= np.mean(sig)
    p_max = float(np.max(np.abs(sig)))
    if p_max > 0:
        sig = (sig / p_max) * amp
    return _apply_hann_fades(sig, min(n // 4, int(0.005 * sample_rate)))


def _synth_long_ringout(
    f0: float,
    amp: float,
    dur: float,
    sample_rate: int = FS,
    b_inharm: float | None = None,
) -> np.ndarray:
    """Synthesizes an extended uninterrupted bass ring-out decaying across > 75 dB into pure digital silence.

    Trains the neural network's convolutional receptive field to transition smoothly without noise-gate chatter.
    """
    n = int(dur * sample_rate)
    if n <= 0:
        return np.empty(0, dtype=np.float64)
    b_coeff = _calc_inharmonicity_b(f0) if b_inharm is None else b_inharm
    sig = np.zeros(n, dtype=np.float64)
    max_h = min(128, int((sample_rate / 2.0 - 200.0) / f0))
    split_hz = 0.16
    fns: list[float] = []
    hws: list[float] = []
    drvs: list[float] = []
    drhs: list[float] = []
    for h in range(1, max_h):
        fn = h * f0 * np.sqrt(1.0 + b_coeff * (h**2))
        if fn >= (sample_rate / 2.0) - 200.0:
            break
        geo_pos = max(float(np.sin(h * np.pi * 0.14)), 0.25)
        h_weight = (1.0 / (h**0.92)) * geo_pos
        decay_v_rate = 0.35 + 0.05 * h + 0.00012 * (h**2)
        decay_h_rate = 0.20 + 0.03 * h + 0.00006 * (h**2)
        fns.append(fn)
        hws.append(h_weight)
        drvs.append(decay_v_rate)
        drhs.append(decay_h_rate)

    if fns:
        _accumulate_ringout_phasors_simd(
            sig,
            n,
            1.0 / sample_rate,
            np.asarray(fns, dtype=np.float64),
            np.asarray(hws, dtype=np.float64),
            np.asarray(drvs, dtype=np.float64),
            np.asarray(drhs, dtype=np.float64),
            split_hz,
        )
    sig -= np.mean(sig)
    p_max = float(np.max(np.abs(sig)))
    if p_max > 0:
        sig = (sig / p_max) * amp
    fade_samples = min(n // 8, int(0.05 * sample_rate))
    return _apply_hann_fades(sig, fade_samples)


def _synth_two_tone_probe(
    f1: float,
    f2: float,
    amp: float,
    dur: float,
    sample_rate: int = FS,
) -> np.ndarray:
    """Synthesizes a precision CCIF/DIN two-tone intermodulation probe burst to isolate Volterra kernels."""
    n = int(dur * sample_rate)
    if n <= 0:
        return np.empty(0, dtype=np.float64)
    t = np.linspace(0.0, dur, n, endpoint=False)
    sig = 0.50 * np.sin(2.0 * np.pi * f1 * t) + 0.50 * np.sin(2.0 * np.pi * f2 * t)
    sig -= np.mean(sig)
    p_max = float(np.max(np.abs(sig)))
    if p_max > 0:
        sig = (sig / p_max) * amp
    return _apply_hann_fades(sig, min(n // 4, int(0.005 * sample_rate)))


def generate_optimal_bass_dry(
    duration_sec: float = 240.0,
    sample_rate: int = FS,
    peak_dbfs: float = -1.0,
    seed: int = 42,
) -> np.ndarray:
    """Synthesizes a 48 kHz high-fidelity 4-minute synthetic dry excitation signal tailored for bass modeling.

    Implements a 15-vector physical excitation architecture covering the complete state space of bass pickups:
    1. Latency Calibration Double-Blips: clean impulses at 0.3s (+0.89) and 0.8s (-0.89) ensuring Tone3000
       detects non-V3 prelude format and defaults latency to 0.
    2. Slew-Rate Diverse Log Chirps: fast (1.8s) and slow (8.0s) full-range sweeps (15 Hz -> 22 kHz) across
       -24 dBFS to -1 dBFS + inverted sweep (22 kHz -> 15 Hz).
    3. 7-Step Dynamic Velocity Ladder on E1 (pp -> fff: -28 dBFS to -0.4 dBFS).
    4. Long-Decay Continuous Ring-Outs (5.0s - 5.5s) decaying across 75 dB to eliminate noise-gate cutoff artifacts.
    5. Modal Plucks across Registers: B0 through E3 with heavy-string attack pitch sag and unilateral fret buzz.
    6. Comprehensive Bass Articulations: rapid groove bursts (120 & 140 BPM), slap-and-pop pairs, funk ghost rakes,
       palm-muted Motown thuds, finger vibrato, and natural harmonics.
    7. Polyphony, Dyads, Tenths, CCIF Probes & Schroeder Multitone: root-fifths, upper-register tenths,
       CCIF resonant probes (3.0-4.25 kHz), and Schroeder multitone with 800 Hz roll-off corner.
    8. Continuous Glissandi: smooth exponential slides traversing the entire 24-fret fingerboard up to G4 (392 Hz).
    9. Shaped Pink Noise Bursts: rhythmic gated pink noise ensuring 100% continuous spectral density, adaptively
       filling to exact duration with bounded trailing silence.

    Zero Artificial Dither Policy:
    Silence intervals contain pure, bit-exact digital silence (0.0). No background noise, hum,
    or thermal dither is injected. All segment boundaries are smoothed with 3-5ms Hann micro-fades.
    Leading and trailing silence is strictly bounded (<= 0.3s) and inter-event pauses provide full
    recovery time for RLC resonance and magnetic relaxation.
    """
    total_samples = int(duration_sec * sample_rate)
    audio = np.zeros(total_samples, dtype=np.float64)
    scale = min(1.0, duration_sec / 240.0)

    lead_silence = min(int(0.3 * sample_rate), int(0.05 * total_samples))
    trail_silence = min(int(0.3 * sample_rate), int(0.05 * total_samples))
    cur = lead_silence

    # 1. Calibration blips (at 0.3s and 0.8s)
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

    # 2. Multi-tier full sweeps (15 Hz -> 22 kHz) with Slew-Rate Diversity
    chirp_tiers = [
        (15.0, 22000.0, 0.050, 12.0),  # Slow precision sweep (-24 dBFS linear baseline, spans 10-12s V3 window)
        (15.0, 22000.0, 0.220, 1.8),  # Fast dynamic sweep (-13 dBFS eddy onset)
        (15.0, 22000.0, 0.500, 7.0),  # Slow high-res sweep (-6 dBFS Lenz drag)
        (15.0, 22000.0, 0.890, 1.8),  # Fast extreme slew sweep (-1 dBFS 16 kHz limit)
        (22000.0, 15.0, 0.650, 5.0),  # Inverted down-sweep (-3.7 dBFS)
    ]
    for f_s, f_e, amp, dur_base in chirp_tiers:
        if cur >= total_samples - trail_silence:
            break
        dur_c = max(0.6, dur_base * scale)
        c = _synth_log_chirp(dur_c, f_s, f_e, amp, sample_rate)
        append_segment(c, 0.4)

    # 3. 7-Step Dynamic Velocity Ladder on open E1 (pp -> fff)
    v_dur = max(0.4, 1.8 * scale)
    velocity_tiers = [0.04, 0.10, 0.22, 0.40, 0.62, 0.82, 0.95]
    for v_amp in velocity_tiers:
        if cur >= total_samples - trail_silence:
            break
        p = _synth_pluck(41.20, v_amp, v_dur, sample_rate, technique="finger")
        append_segment(p, 0.4)

    # 4. Long-Decay Continuous Ring-Outs (Gate-Free Tail Linearity across > 75 dB)
    r_dur_e = max(1.0, 5.5 * scale)
    r_dur_a = max(1.0, 5.0 * scale)
    for f_ring, dur_ring in [(41.20, r_dur_e), (55.00, r_dur_a)]:
        if cur >= total_samples - trail_silence:
            break
        r = _synth_long_ringout(f_ring, 0.85, dur_ring, sample_rate)
        append_segment(r, 0.5)

    # 5. Modal plucks with pitch sag and heavy-string inharmonicity across registers
    m_dur = max(0.4, 1.3 * scale)
    notes = [27.50, 30.87, 41.20, 55.00, 73.42, 98.00, 130.81, 164.81]
    for n_f in notes:
        sag = 4.5 if n_f < 35.0 else (2.5 if n_f < 60.0 else 1.2)
        for v_amp in [0.45, 0.82]:
            if cur >= total_samples - trail_silence:
                break
            p = _synth_pluck(n_f, v_amp, m_dur, sample_rate, pitch_sag_hz=sag, technique="finger")
            append_segment(p, 0.35)

    # 6. Bass Articulations & Playing Techniques
    # Rapid groove bursts (pumped core envelope memory)
    if cur < total_samples - trail_silence:
        gb1 = _synth_groove_burst(41.20, 0.82, bpm=120.0, count=6, sample_rate=sample_rate, technique="finger")
        append_segment(gb1, 0.4)
    if cur < total_samples - trail_silence:
        gb2 = _synth_groove_burst(55.00, 0.80, bpm=140.0, count=8, sample_rate=sample_rate, technique="pick")
        append_segment(gb2, 0.4)
    # Slap-and-pop pairs (thumb slap -> octave pop)
    for f_slap, f_pop in [(41.20, 82.41), (55.00, 110.00)]:
        if cur >= total_samples - trail_silence:
            break
        sp = _synth_slap_pop_pair(f_slap, f_pop, 0.89, gap_ms=70.0, dur=max(0.4, 1.4 * scale), sample_rate=sample_rate)
        append_segment(sp, 0.4)
    # Funk ghost-note percussive rakes & isolated ghost clicks
    if cur < total_samples - trail_silence:
        rake = _synth_ghost_rake(41.20, 0.85, max(0.4, 1.0 * scale), sample_rate)
        append_segment(rake, 0.4)
    for _ in range(2):
        if cur >= total_samples - trail_silence:
            break
        g = _synth_ghost_note(max(0.15, 0.30 * scale), 0.75, sample_rate)
        append_segment(g, 0.35)
    # Palm-muted "Motown / Dub" thuds (heavy exponential damping)
    for n_f in [41.20, 55.00, 73.42]:
        if cur >= total_samples - trail_silence:
            break
        pm = _synth_pluck(n_f, 0.85, max(0.3, 0.7 * scale), sample_rate, technique="palm_mute")
        append_segment(pm, 0.4)
    # Sustained plucks with finger vibrato (RLC resonance sweeping)
    for v_f in [55.00, 73.42]:
        if cur >= total_samples - trail_silence:
            break
        vib = _synth_vibrato_pluck(v_f, 0.78, max(0.6, 2.5 * scale), mod_rate=5.0, mod_depth_cents=25.0, sample_rate=sample_rate)
        append_segment(vib, 0.4)
    # Plectrum downstroke/upstroke strikes
    for _ in range(2):
        if cur >= total_samples - trail_silence:
            break
        p = _synth_pluck(41.20, 0.78, max(0.4, 1.4 * scale), sample_rate, technique="pick")
        append_segment(p, 0.4)
    # Natural harmonic bell chimes
    for h_f in [82.41, 123.6, 164.8]:
        if cur >= total_samples - trail_silence:
            break
        h = _synth_natural_harmonic(h_f, 0.70, max(0.4, 1.6 * scale), sample_rate)
        append_segment(h, 0.4)

    # 7. Polyphony, Dyads, Tenths, CCIF Probes & Schroeder Multitone
    d_dur = max(0.5, 2.0 * scale)
    low_dyads = [
        (27.50, 41.25),  # Low-A0 + E1
        (30.87, 46.31),  # Low-B0 + F#1
        (41.20, 61.74),  # Low-E1 + B1
        (55.00, 82.50),  # Low-A1 + E2
    ]
    for f1, f2 in low_dyads:
        if cur >= total_samples - trail_silence:
            break
        d = _synth_dyad(f1, f2, 0.75, d_dur, sample_rate)
        append_segment(d, 0.4)
    # Upper-register root-tenths (melodic polyphony & mid-band IMD)
    tenths = [
        (82.41, 207.65),   # E2 + G#3 (tenth)
        (110.00, 277.18),  # A2 + C#4 (tenth)
        (146.83, 369.99),  # D3 + F#4 (tenth)
    ]
    for f1, f2 in tenths:
        if cur >= total_samples - trail_silence:
            break
        dt = _synth_dyad(f1, f2, 0.72, max(0.4, 1.8 * scale), sample_rate)
        append_segment(dt, 0.4)
    # High-frequency CCIF / DIN two-tone intermodulation probes
    ccif_probes = [
        (3000.0, 3200.0),  # Delta f = 200 Hz, RLC resonance band
        (4000.0, 4250.0),  # Delta f = 250 Hz, upper resonance band
        (2000.0, 2150.0),  # Delta f = 150 Hz, upper-mid presence
    ]
    probe_dur = max(0.4, 2.0 * scale)
    for f1, f2 in ccif_probes:
        if cur >= total_samples - trail_silence:
            break
        pr = _synth_two_tone_probe(f1, f2, 0.65, probe_dur, sample_rate)
        append_segment(pr, 0.4)
    # Schroeder-phase multitone complex with 800 Hz roll-off corner
    m_rem = max(0, total_samples - trail_silence - cur)
    if m_rem > int(2.0 * sample_rate * scale):
        dur_m = min(14.0 * scale, m_rem / sample_rate * 0.35)
        nm = int(dur_m * sample_rate)
        if nm > 100:
            tm = np.linspace(0.0, dur_m, nm, endpoint=False)
            clusters = [
                27.50, 30.87, 41.20, 55.00, 82.41, 110.0, 220.0,
                440.0, 880.0, 1250.0, 1800.0, 2400.0, 3100.0, 4200.0, 6000.0,
            ]
            kc = len(clusters)
            sig_m = np.zeros(nm, dtype=np.float64)
            for k, fk in enumerate(clusters):
                th = (np.pi * (k**2)) / kc
                weight = 1.0 / np.sqrt(1.0 + (fk / 800.0) ** 1.1)
                sig_m += weight * np.sin(2.0 * np.pi * fk * tm + th)
            sig_m -= np.mean(sig_m)
            sig_m = (sig_m / np.max(np.abs(sig_m))) * 0.80
            am_env = 0.575 + 0.325 * np.sin(2.0 * np.pi * 0.25 * tm)
            append_segment(
                _apply_hann_fades(sig_m * am_env, min(nm // 4, int(0.01 * sample_rate))),
                0.4,
            )

    # 8. Continuous Glissandi traversing full fretboard comb nulls up to G4 (392 Hz)
    g_dur = max(0.5, 3.2 * scale)
    slides = [
        (27.50, 41.20),
        (41.20, 73.42),
        (73.42, 146.83),
        (146.83, 293.66),
        (293.66, 392.00),  # High G4 on 24th fret
        (392.00, 41.20),   # Full-fingerboard downward slide
    ]
    for fs, fe in slides:
        if cur >= total_samples - trail_silence:
            break
        gl_dur = max(0.6, 4.0 * scale) if fe < fs else g_dur
        gl = _synth_glissando(fs, fe, 0.80, gl_dur, sample_rate)
        append_segment(gl, 0.4)

    # 9. Shaped Pink Noise Bursts adaptively filling to exact duration
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
    duration_sec: float = 240.0,
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


