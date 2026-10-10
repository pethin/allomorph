"""
Audio file I/O and true-peak intersample analysis primitives.
"""

import math
import wave
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from allomorph.dsp.constants import FS


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

    audio = np.clip(audio, -1.0, 1.0)
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


def compute_true_peak(
    audio: np.ndarray,
    sample_rate: int = FS,
    oversample: int = 4,
) -> float:
    """Computes inter-sample true-peak amplitude using 4x sinc/FFT interpolation (ITU-R BS.1770-4)."""
    mono = audio[0] if audio.ndim > 1 else audio
    mono = np.asarray(mono, dtype=np.float64)
    n = len(mono)
    if n == 0:
        return 0.0

    chunk_size = 65536
    overlap = 1024
    step = chunk_size - overlap
    max_peak = 0.0

    for start in range(0, n, step):
        chunk = mono[start : start + chunk_size]
        c_len = len(chunk)
        if c_len < 32:
            max_peak = max(max_peak, float(np.max(np.abs(chunk))))
            continue

        n_up = c_len * oversample
        X = np.fft.rfft(chunk)
        n_bins = len(X)
        X_padded = np.zeros(n_up // 2 + 1, dtype=np.complex128)
        X_padded[:n_bins] = X
        X_padded[n_bins - 1] *= 0.5
        interpolated = np.fft.irfft(X_padded, n=n_up) * oversample

        valid_start = overlap * oversample // 2 if start > 0 else 0
        valid_end = n_up - (overlap * oversample // 2) if (start + chunk_size < n) else n_up
        chunk_peak = float(np.max(np.abs(interpolated[valid_start:valid_end])))
        max_peak = max(max_peak, chunk_peak)

    # Intersample true-peak must never under-read the discrete sample peak (ITU-R BS.1770-4)
    sample_peak = float(np.max(np.abs(mono)))
    return float(max(max_peak, sample_peak))


def compute_true_peak_dbfs(
    audio: np.ndarray,
    sample_rate: int = FS,
    oversample: int = 4,
) -> float:
    """Computes inter-sample true-peak in dBFS."""
    tp = compute_true_peak(audio, sample_rate=sample_rate, oversample=oversample)
    return float(20.0 * math.log10(max(tp, 1e-12)))
