"""
Logarithmic sine sweep (Farina chirp) synthesis, deconvolution, and harmonic distortion extraction.
"""

import math

import numpy as np

from allomorph.dsp.constants import FS, apply_cinf_fades, cinf_smoothstep


def _synth_log_chirp(
    dur: float,
    f_start: float,
    f_end: float,
    amp: float,
    sample_rate: int = FS,
    string_tilt: bool = True,
    f_corner: float = 1200.0,
) -> np.ndarray:
    """Synthesizes a full-range logarithmic frequency chirp with 5ms micro-fades and physical string harmonic tilt."""
    n = int(dur * sample_rate)
    if n <= 0:
        return np.empty(0, dtype=np.float64)
    t = np.linspace(0.0, dur, n, endpoint=False)
    gamma = np.log(f_end / f_start)
    phase = 2.0 * np.pi * f_start * (dur / gamma) * ((f_end / f_start) ** (t / dur) - 1.0)
    if string_tilt and f_corner > 0.0:
        inst_freq = f_start * ((f_end / f_start) ** (t / dur))
        tilt = 1.0 / (1.0 + (inst_freq / f_corner) ** 2)
        sig = amp * tilt * np.sin(phase)
    else:
        sig = amp * np.sin(phase)
    sig -= np.mean(sig)
    return apply_cinf_fades(sig, min(n // 4, int(0.005 * sample_rate)))


def synthesize_fast_log_sweep(
    n_samples: int = 16384,
    f_start: float = 10.0,
    f_end: float = 24000.0,
    sr: int = FS,
    target_dbfs: float = -20.5,
    fade_len: int = 144,
    tail_len: int = 2048,
) -> np.ndarray:
    """Synthesizes a full-range logarithmic sine sweep (Farina chirp) with smooth C^infinity boundary fades,

    trailing silence headroom to allow filter impulse responses to ring down without boundary truncation,
    and calibrated RMS target level.
    """
    if n_samples <= 0:
        return np.empty(0, dtype=np.float64)
    tail = max(0, min(tail_len, n_samples // 2))
    n_chirp = n_samples - tail
    dur = float(n_chirp) / float(sr)
    t = np.linspace(0.0, dur, n_chirp, endpoint=False)
    gamma = math.log(f_end / f_start)
    phase = 2.0 * math.pi * f_start * (dur / gamma) * (np.exp((t / dur) * gamma) - 1.0)
    sig = np.sin(phase)
    sig = sig - float(np.mean(sig))

    if fade_len > 0 and 2 * fade_len < n_chirp:
        sig = apply_cinf_fades(sig, fade_len)

    if tail > 0:
        sig = np.pad(sig, (0, tail), mode="constant")

    rms = float(np.sqrt(np.mean(sig**2)))
    target_rms = 10.0 ** (target_dbfs / 20.0)
    if rms > 1e-9:
        sig = sig * (target_rms / rms)
    return sig.astype(np.float64)


def deconvolve_log_sweep(
    y_wet: np.ndarray,
    x_sweep: np.ndarray,
    *,
    f_start: float = 10.0,
    f_end: float = 24000.0,
    sr: int = FS,
    gate_taps: int | None = 4096,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Performs calibrated Farina time-reversed deconvolution of a stem output against an input log sweep.

    Returns:
        (f_bins, H_complex, h_impulse_full)
    where:
        - f_bins: frequency vector for H_complex
        - H_complex: complex transfer function calibrated such that loopback (y = x) is bit-exact 0.00 dB
        - h_impulse_full: full unwindowed time-domain impulse response containing linear response
          aligned at sample 0, and harmonic distortion components (Farina harmonics) at negative time offsets.
    """
    n_in = max(len(y_wet), len(x_sweep))
    n_fft = 1 << (n_in * 2 - 1).bit_length()

    X = np.fft.rfft(x_sweep, n=n_fft)
    Y = np.fft.rfft(y_wet, n=n_fft)

    # Regularized spectral inversion: H_inv = X* / (|X|^2 + eps)
    eps = 1e-12 * float(np.max(np.abs(X)) ** 2)
    H_inv = np.conj(X) / (np.abs(X) ** 2 + eps)

    h_full = np.fft.irfft(Y * H_inv, n=n_fft)

    if gate_taps is not None and 0 < gate_taps < n_fft:
        h_gated = h_full[:gate_taps].copy()
        fade_len = min(64, gate_taps // 16)
        if fade_len > 0:
            t_out = np.linspace(0.0, 1.0, fade_len, endpoint=False)
            w_out = cinf_smoothstep(t_out)[::-1]
            h_gated[-fade_len:] *= w_out
        H_complex = np.fft.rfft(h_gated, n=gate_taps)
        f_bins = np.fft.rfftfreq(gate_taps, 1.0 / sr)
        return f_bins, H_complex, h_full
    else:
        H_complex = np.fft.rfft(h_full, n=n_fft)
        f_bins = np.fft.rfftfreq(n_fft, 1.0 / sr)
        return f_bins, H_complex, h_full


def extract_farina_harmonics(
    h_time: np.ndarray,
    n_samples: int = 16384,
    f_start: float = 10.0,
    f_end: float = 24000.0,
    sr: int = FS,
    win_len: int = 512,
    tail_len: int = 2048,
) -> dict[str, float]:
    """Extracts fundamental energy, 2nd harmonic, 3rd harmonic, and THD from a Farina deconvolved impulse response."""
    tail = max(0, min(tail_len, n_samples // 2))
    n_chirp = n_samples - tail
    dur = float(n_chirp) / float(sr)
    gamma = math.log(f_end / f_start)
    n_fft = len(h_time)

    # Fundamental window around sample 0
    w_fund = min(win_len, n_fft // 8)
    e1 = float(np.sum(h_time[:w_fund] ** 2))

    # 2nd harmonic position: arrival earlier by dt2
    dt2 = dur * math.log(2.0) / gamma
    idx2 = (n_fft - round(dt2 * sr)) % n_fft
    s2_start = (idx2 - win_len // 2) % n_fft
    s2_end = (idx2 + win_len // 2) % n_fft
    if s2_start < s2_end:
        e2 = float(np.sum(h_time[s2_start:s2_end] ** 2))
    else:
        e2 = float(np.sum(h_time[s2_start:] ** 2) + np.sum(h_time[:s2_end] ** 2))

    # 3rd harmonic position: arrival earlier by dt3
    dt3 = dur * math.log(3.0) / gamma
    idx3 = (n_fft - round(dt3 * sr)) % n_fft
    s3_start = (idx3 - win_len // 2) % n_fft
    s3_end = (idx3 + win_len // 2) % n_fft
    if s3_start < s3_end:
        e3 = float(np.sum(h_time[s3_start:s3_end] ** 2))
    else:
        e3 = float(np.sum(h_time[s3_start:] ** 2) + np.sum(h_time[:s3_end] ** 2))

    e1_safe = max(e1, 1e-12)
    thd2 = math.sqrt(e2 / e1_safe)
    thd3 = math.sqrt(e3 / e1_safe)
    thd_tot = math.sqrt((e2 + e3) / e1_safe)

    return {
        "thd_percent": round(thd_tot * 100.0, 3),
        "thd_db": round(20.0 * math.log10(max(thd_tot, 1e-6)), 2),
        "thd2_percent": round(thd2 * 100.0, 3),
        "thd2_db": round(20.0 * math.log10(max(thd2, 1e-6)), 2),
        "thd3_percent": round(thd3 * 100.0, 3),
        "thd3_db": round(20.0 * math.log10(max(thd3, 1e-6)), 2),
    }
