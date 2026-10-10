"""
ITU-R BS.1770-4 / EBU R128 Integrated Loudness (LUFS) Analysis.
Decomposed into pure, property-testable stages:
1. K-weighting pre-filtering (Stage 1 high-shelf + Stage 2 RLB high-pass)
2. Block-level mean-square energy computation (400 ms sliding window, 75% overlap)
3. Dual-threshold gating (absolute -70 LKFS and relative -10 LU)
"""

import math

import numpy as np

from allomorph.dsp.constants import FS, apply_biquad


def apply_k_weighting_filter(
    audio: np.ndarray,
    sample_rate: int = FS,
) -> np.ndarray:
    """Applies the two-stage ITU-R BS.1770-4 K-weighting pre-filter.

    Stage 1: High-shelf pre-filter (simulating human head acoustic diffraction, ~+4 dB boost).
    Stage 2: RLB high-pass filter (2nd-order Butterworth at ~38 Hz, modeling bass acoustic roll-off).
    """
    mono = audio[0] if audio.ndim > 1 else audio
    mono = np.asarray(mono, dtype=np.float64)
    if len(mono) == 0:
        return np.empty(0, dtype=np.float64)

    if sample_rate == 48000:
        b_pre = (1.53512485958697, -2.69169618940638, 1.19839281085285)
        a_pre = (1.0, -1.69065929318241, 0.73248077421585)
        b_rlb = (1.0, -2.0, 1.0)
        a_rlb = (1.0, -1.99004745483398, 0.99007225036621)
    else:
        # Exact ITU-R BS.1770-4 bilinear transform for arbitrary sample rate
        k_pre = math.tan(math.pi * 1681.9744509555319 / sample_rate)
        vh = 10.0 ** (3.99984385397 / 20.0)
        vb = vh**0.499666774155
        a0_pre = 1.0 + k_pre / 0.7071752369554193 + k_pre * k_pre
        b_pre = (
            (vh + vb * k_pre / 0.7071752369554193 + k_pre * k_pre) / a0_pre,
            2.0 * (k_pre * k_pre - vh) / a0_pre,
            (vh - vb * k_pre / 0.7071752369554193 + k_pre * k_pre) / a0_pre,
        )
        a_pre = (
            1.0,
            2.0 * (k_pre * k_pre - 1.0) / a0_pre,
            (1.0 - k_pre / 0.7071752369554193 + k_pre * k_pre) / a0_pre,
        )

        k_rlb = math.tan(math.pi * 38.13547087602444 / sample_rate)
        q_rlb = 0.5003270373253953
        a0_rlb = 1.0 + k_rlb / q_rlb + k_rlb * k_rlb
        b_rlb = (1.0 / a0_rlb, -2.0 / a0_rlb, 1.0 / a0_rlb)
        a_rlb = (
            1.0,
            2.0 * (k_rlb * k_rlb - 1.0) / a0_rlb,
            (1.0 - k_rlb / q_rlb + k_rlb * k_rlb) / a0_rlb,
        )

    y_pre = apply_biquad(mono, b_pre, a_pre)
    return apply_biquad(y_pre, b_rlb, a_rlb)


def compute_gated_loudness_blocks(
    k_filtered: np.ndarray,
    sample_rate: int = FS,
    block_size_ms: float = 400.0,
    hop_size_ms: float = 100.0,
) -> np.ndarray:
    """Computes mean-square energy across 75% overlapping rectangular blocks."""
    n = len(k_filtered)
    block_samples = int((block_size_ms / 1000.0) * sample_rate)
    step_samples = int((hop_size_ms / 1000.0) * sample_rate)

    if n < block_samples:
        z = float(np.mean(k_filtered**2)) if n > 0 else 0.0
        return np.array([z], dtype=np.float64)

    num_blocks = (n - block_samples) // step_samples + 1
    shape = (num_blocks, block_samples)
    strides = (step_samples * k_filtered.strides[0], k_filtered.strides[0])
    blocks = np.lib.stride_tricks.as_strided(k_filtered, shape=shape, strides=strides)
    return np.mean(blocks**2, axis=-1)


def apply_itu_gating(
    block_energies: np.ndarray,
    abs_thresh_lkfs: float = -70.0,
    rel_offset_lu: float = -10.0,
) -> float:
    """Applies dual-threshold gating (absolute threshold followed by relative threshold)

    and evaluates integrated loudness in LUFS: L = -0.691 + 10 * log10(mean(z_gated)).
    """
    if len(block_energies) == 0:
        return -100.0

    # Step 1: Absolute threshold gating (-70 LKFS)
    gamma_a = 10.0 ** ((abs_thresh_lkfs + 0.691) / 10.0)
    valid_a = block_energies > gamma_a
    if not np.any(valid_a):
        return -100.0

    z_valid_a = block_energies[valid_a]
    z_avg_a = float(np.mean(z_valid_a))

    # Step 2: Relative threshold gating (e.g. -10 LU relative to absolute-gated average)
    gamma_r = z_avg_a * (10.0 ** (rel_offset_lu / 10.0))
    valid_r = z_valid_a > gamma_r
    if not np.any(valid_r):
        return -100.0

    z_final = float(np.mean(z_valid_a[valid_r]))
    return float(-0.691 + 10.0 * math.log10(max(z_final, 1e-12)))


def compute_lufs(
    audio: np.ndarray,
    sample_rate: int = FS,
) -> float:
    """Computes integrated loudness in LUFS according to ITU-R BS.1770-4 / EBU R128.

    Implements:
      1. Stage 1 High Shelf Pre-filter (f0 ~ 1.5 kHz, +4 dB boost)
      2. Stage 2 RLB High-Pass Filter (2nd-order Butterworth f0 ~ 38 Hz)
      3. Gated mean-square integration (400 ms blocks, 75% overlap, -70 LKFS / -10 LU relative gating)
    """
    mono = audio[0] if audio.ndim > 1 else audio
    mono = np.asarray(mono, dtype=np.float64)
    if len(mono) == 0:
        return -100.0

    y_k = apply_k_weighting_filter(mono, sample_rate=sample_rate)
    block_samples = int(0.400 * sample_rate)
    if len(mono) < block_samples:
        z = float(np.mean(y_k**2))
        return float(-0.691 + 10.0 * math.log10(max(z, 1e-12)))

    z_blocks = compute_gated_loudness_blocks(y_k, sample_rate=sample_rate)
    return apply_itu_gating(z_blocks, abs_thresh_lkfs=-70.0, rel_offset_lu=-10.0)
