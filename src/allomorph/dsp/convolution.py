"""
High-performance FFT convolution and latency calibration routines.
"""

from collections.abc import Sequence
from typing import Literal

import numpy as np


def fft_convolve(
    in1: Sequence[float] | np.ndarray,
    in2: Sequence[float] | np.ndarray,
    mode: str = "full",
) -> np.ndarray:
    """High-performance 1D convolution using real FFTs with cache-friendly overlap-add

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
        num_blocks = (out_len + l - 1) // l
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
        return out[:n]
    elif mode == "same":
        start = (m - 1) // 2
        return out[start : start + n]
    else:
        raise ValueError(f"Unsupported mode '{mode}'. Choose 'full', 'causal', or 'same'.")


def fft_convolve_multi(
    x: Sequence[float] | np.ndarray,
    filters: Sequence[Sequence[float] | np.ndarray],
    mode: Literal["full", "causal", "same"] = "full",
) -> list[np.ndarray]:
    """Convolves a single 1D signal x against multiple 1D filters using vectorized FFT convolution,

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

    if n > 32768 and m_max <= 32768:
        block_size = 65536
        l = block_size - m_max + 1
        out_len_max = n + m_max - 1
        num_blocks = (out_len_max + l - 1) // l
        pad_end = num_blocks * l - n
        x_padded = np.pad(x_arr, (m_max - 1, pad_end), mode="constant")
        shape = (num_blocks, block_size)
        strides = (l * x_padded.strides[0], x_padded.strides[0])
        blocks = np.lib.stride_tricks.as_strided(x_padded, shape=shape, strides=strides)
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
    """Evaluates NAM V3 calibration blip latency alignment using NAM's exact algorithm.

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
