"""
Audio post-processing and output conditioning primitives:
1. Sub-audible 8 Hz DC blocking filter and mean offset nulling
2. Passive RLC-colored Johnson-Nyquist thermal noise dither (-108 dBFS)
3. Calibrated LUFS/RMS/Peak level matching with true-peak ceiling limiting
"""

import math

import numpy as np
import pedalboard

from allomorph.dsp.constants import FS
from allomorph.dsp.convolution import fft_convolve
from allomorph.dsp.fir import synthesize_minimum_phase_fir
from allomorph.dsp.io import compute_true_peak
from allomorph.dsp.loudness import compute_lufs

CALIBRATION_PEAK_CEILING = 0.9900  # -0.09 dBFS true-peak headroom ceiling


def get_white_noise_vector(n_samples: int, seed: int = 42) -> np.ndarray:
    """Generates a deterministic 1D Gaussian white noise vector."""
    rng = np.random.default_rng(seed=seed)
    return rng.standard_normal(n_samples).astype(np.float64)


def apply_dc_block(
    audio: np.ndarray,
    sample_rate: int = FS,
    cutoff_hz: float = 8.0,
) -> np.ndarray:
    """Applies a sub-audible high-pass filter (default 8 Hz) and subtracts mean DC offset."""
    if len(audio) == 0:
        return audio.copy()
    hp = pedalboard.HighpassFilter(cutoff_frequency_hz=cutoff_hz)
    in_arr = audio.astype(np.float32)
    if in_arr.ndim == 1:
        filtered = hp(in_arr[np.newaxis, :], sample_rate)[0].astype(np.float64)
    else:
        filtered = hp(in_arr, sample_rate).astype(np.float64)
    filtered = filtered - float(np.mean(filtered))
    return filtered


def apply_johnson_dither(
    audio: np.ndarray,
    h_total: np.ndarray,
    target_rms_dbfs: float = -108.0,
    num_taps: int = 512,
    seed: int = 42,
) -> np.ndarray:
    """Shapes Johnson-Nyquist thermal noise dither through the composite passive RLC transfer function

    and adds it at target_rms_dbfs (-108 dBFS standard).
    """
    n_samples = len(audio)
    if n_samples == 0:
        return audio.copy()
    white_noise = get_white_noise_vector(n_samples, seed=seed)
    dither_fir = synthesize_minimum_phase_fir(h_total, num_taps=num_taps, normalize=True)
    colored_noise = fft_convolve(
        white_noise, np.asarray(dither_fir, dtype=np.float64), mode="causal"
    )[:n_samples]
    colored_rms = max(float(np.sqrt(np.mean(colored_noise**2))), 1e-9)
    target_dither_rms = 10.0 ** (target_rms_dbfs / 20.0)
    dither = (colored_noise / colored_rms) * target_dither_rms
    return audio + dither


def apply_calibrated_normalization(
    audio: np.ndarray,
    reference_audio: np.ndarray | None = None,
    mode: str = "auto",
    target_dbfs: float | None = None,
    sample_rate: int = FS,
    peak_ceiling: float = CALIBRATION_PEAK_CEILING,
) -> np.ndarray:
    """Normalizes output audio levels according to ITU-R BS.1770-4 gated LUFS, RMS, or Peak,

    and enforces an absolute true-peak headroom ceiling (default -0.09 dBFS / 0.9900).
    """
    if len(audio) == 0:
        return audio.copy()

    ref = reference_audio if reference_audio is not None else audio
    ref_mono = ref[0] if ref.ndim > 1 else ref
    filtered = audio.copy()

    if mode in ("auto", "lufs"):
        in_lufs = compute_lufs(ref_mono, sample_rate=sample_rate)
        cur_lufs = compute_lufs(filtered, sample_rate=sample_rate)
        if not (
            math.isinf(cur_lufs)
            or math.isnan(cur_lufs)
            or math.isinf(in_lufs)
            or math.isnan(in_lufs)
        ):
            target_lufs = float(target_dbfs) if target_dbfs is not None else in_lufs
            gain_db = target_lufs - cur_lufs
            filtered = filtered * (10.0 ** (gain_db / 20.0))
        else:
            in_rms = float(np.sqrt(np.mean(ref_mono**2)))
            out_rms = float(np.sqrt(np.mean(filtered**2)))
            target_rms = 10.0 ** (float(target_dbfs) / 20.0) if target_dbfs is not None else in_rms
            if out_rms > 1e-9:
                filtered = filtered * (target_rms / out_rms)
    elif mode == "rms":
        target_rms = (
            10.0 ** (float(target_dbfs) / 20.0)
            if target_dbfs is not None
            else float(np.sqrt(np.mean(ref_mono**2)))
        )
        cur_rms = float(np.sqrt(np.mean(filtered**2)))
        if cur_rms > 1e-9:
            filtered = filtered * (target_rms / cur_rms)
    elif mode == "peak":
        target_peak = (
            10.0 ** (float(target_dbfs) / 20.0)
            if target_dbfs is not None
            else float(np.max(np.abs(ref_mono)))
        )
        cur_peak = float(np.max(np.abs(filtered)))
        if cur_peak > 1e-9:
            filtered = filtered * (target_peak / cur_peak)

    if mode not in ("none", "raw"):
        tp = compute_true_peak(filtered, sample_rate=sample_rate)
        if tp > peak_ceiling and tp > 1e-9:
            filtered = filtered * (peak_ceiling / tp)

    return filtered
