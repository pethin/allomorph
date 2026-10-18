#!/usr/bin/env python3
"""Allomorph - Darkglass Anagram Hardware Re-Amping CLI.

Plays dry excitation audio through the Darkglass Anagram USB audio interface
specifically on Output 9 (USB Reamp Send) while simultaneously recording the
wet return signal from Input 1 (or stereo Inputs 1 & 2) at 24-bit 48 kHz.

Features:
  - Closed-loop hardware auto-gain calibration probe on Output 9 to identify the
    optimal send gain that drives the Anagram return to target peak headroom (-3.0 dBFS)
    without clipping on either the USB send or wet return
  - Automatic sub-millisecond sample alignment via 2-stage coarse-to-fine envelope +
    impulse cross-correlation to compensate for USB roundtrip buffer latency
  - Blocking chunked streaming to avoid libffi/CFFI callback trampoline assertions on macOS
  - Lazy PortAudio initialization so CLI inspection and offline utilities work in headless/WSL environments.
"""

from __future__ import annotations

import importlib
import math
import sys
import time
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any

import numpy as np
from cyclopts import App, Parameter

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from allomorph.dsp.constants import CALIBRATION_PEAK_CEILING, FS, apply_cinf_fades
from allomorph.dsp.io import read_wav, write_wav_24bit

app = App(
    name="reamp-anagram",
    help="Allomorph - Darkglass Anagram Hardware Re-Amping CLI.",
    version="1.0.0",
)

DEFAULT_TARGET_PEAK_DBFS: float = -3.0
DEFAULT_GAIN_TOLERANCE_DB: float = 1.0
DEFAULT_MIN_SEND_GAIN_DB: float = -24.0
DEFAULT_MAX_SEND_GAIN_DB: float = 18.0
DEFAULT_MAX_CAL_ITERATIONS: int = 4
CLIP_DETECTION_DBFS: float = -0.2
SILENCE_FLOOR_DBFS: float = -60.0


@dataclass(frozen=True)
class GainCalibrationResult:
    """Telemetry from the closed-loop hardware output gain calibration pass."""

    output_gain_db: float
    output_gain_linear: float
    measured_peak_dbfs: float
    measured_rms_dbfs: float
    target_peak_dbfs: float
    max_unclipped_gain_db: float
    iterations: int
    converged: bool
    clipped: bool


def _get_sounddevice() -> Any:
    """Lazily imports sounddevice so CLI --help and offline functions work without PortAudio (e.g. on WSL)."""
    existing = sys.modules.get("sounddevice")
    if existing is not None:
        return existing
    try:
        return importlib.import_module("sounddevice")
    except OSError as exc:
        raise RuntimeError(
            f"PortAudio / sounddevice is unavailable in this environment ({exc}). "
            "Connect the Darkglass Anagram on a host OS with USB audio / PortAudio support."
        ) from exc


def find_anagram_device(
    device_name_pattern: str = "Anagram",
    min_in_channels: int = 1,
    min_out_channels: int = 1,
) -> tuple[int, dict[str, Any]]:
    """Locates the Anagram USB audio device index and property dictionary.

    Prefers full-duplex endpoints matching `device_name_pattern` that satisfy both
    `min_in_channels` and `min_out_channels`. Also accepts an explicit integer device index.
    """
    sd = _get_sounddevice()
    devices = list(sd.query_devices())

    if device_name_pattern.strip().isdigit():
        idx = int(device_name_pattern.strip())
        if 0 <= idx < len(devices):
            return idx, dict(devices[idx])

    pattern_lower = device_name_pattern.lower()
    matching: list[tuple[int, dict[str, Any]]] = []
    for idx, dev in enumerate(devices):
        dev_dict = dict(dev)
        name = str(dev_dict.get("name", ""))
        if pattern_lower in name.lower():
            matching.append((idx, dev_dict))

    # Prefer full-duplex devices that satisfy both input and output channel requirements
    for idx, dev_dict in matching:
        max_in = int(dev_dict.get("max_input_channels", 0))
        max_out = int(dev_dict.get("max_output_channels", 0))
        if max_in >= min_in_channels and max_out >= min_out_channels:
            return idx, dev_dict

    # Fallback to any full-duplex matching device
    for idx, dev_dict in matching:
        max_in = int(dev_dict.get("max_input_channels", 0))
        max_out = int(dev_dict.get("max_output_channels", 0))
        if max_in > 0 and max_out > 0:
            return idx, dev_dict

    if matching:
        return matching[0]

    raise RuntimeError(
        f"Audio device matching '{device_name_pattern}' not found.\n"
        f"Available devices:\n"
        + "\n".join(
            f"  [{i}] {d['name']} ({d['max_input_channels']} in, {d['max_output_channels']} out)"
            for i, d in enumerate(devices)
        )
    )


def compute_audio_levels(audio: np.ndarray) -> tuple[float, float]:
    """Computes (peak_dbfs, rms_dbfs) of mono or multi-channel audio."""
    if audio.size == 0:
        return -180.0, -180.0
    arr = np.asarray(audio, dtype=np.float64)
    peak = float(np.max(np.abs(arr)))
    rms = float(np.sqrt(np.mean(np.square(arr))))
    peak_dbfs = 20.0 * math.log10(max(peak, 1e-9))
    rms_dbfs = 20.0 * math.log10(max(rms, 1e-9))
    return peak_dbfs, rms_dbfs


def compute_max_unclipped_send_gain_db(
    dry_audio: np.ndarray,
    sample_rate: int = FS,
    peak_ceiling: float = CALIBRATION_PEAK_CEILING,
) -> float:
    """Computes the maximum output gain (in dB) that can be applied to `dry_audio`
    without clipping its musical body above `peak_ceiling` (-0.09 dBFS).

    Evaluates the peak across the active musical content (skipping the first 2.0s
    of 1-sample calibration blips on full-length training files).
    """
    if len(dry_audio) == 0:
        return 0.0
    skip_samples = int(2.0 * sample_rate) if len(dry_audio) > int(4.0 * sample_rate) else 0
    musical_region = dry_audio[skip_samples:]
    musical_peak = float(np.max(np.abs(musical_region)))
    if musical_peak <= 1e-9:
        musical_peak = float(np.max(np.abs(dry_audio)))
    if musical_peak <= 1e-9:
        return 0.0
    return float(20.0 * math.log10(peak_ceiling / musical_peak))


def extract_calibration_probe(
    dry_audio: np.ndarray,
    sample_rate: int = FS,
    window_sec: float = 1.2,
) -> np.ndarray:
    """Extracts a compact (~2.4s) high-energy calibration probe from `dry_audio`.

    Combines two key excitation windows from the musical body of `dry_audio`
    (skipping the leading 2.0s 1-sample Dirac calibration blips when `dry_audio` > 4.0s):
      1. The window with maximum broadband RMS energy (excites tube sag, power compression, and cab excursion)
      2. The distinct window with maximum transient peak amplitude (excites slap/pop and attack headroom)

    Each segment is shaped with C-infinity smoothstep boundary fades (`apply_cinf_fades`)
    to prevent discontinuity clicks during hardware probing.
    """
    if len(dry_audio) == 0:
        raise ValueError("Cannot extract calibration probe from empty input audio.")

    mono = np.asarray(dry_audio, dtype=np.float32)
    if mono.ndim > 1:
        mono = np.mean(mono, axis=0)

    win_len = max(256, int(window_sec * sample_rate))
    fade_len = min(max(16, int(0.010 * sample_rate)), win_len // 8)

    if len(mono) <= 2 * win_len:
        faded = apply_cinf_fades(mono.astype(np.float64), min(fade_len, len(mono) // 8))
        return faded.astype(np.float32)

    skip_start = int(2.0 * sample_rate) if len(mono) > int(4.0 * sample_rate) else 0
    active = mono[skip_start:]
    if len(active) <= win_len:
        faded = apply_cinf_fades(active.astype(np.float64), min(fade_len, len(active) // 8))
        return faded.astype(np.float32)

    hop_len = max(64, win_len // 4)
    starts = list(range(0, len(active) - win_len + 1, hop_len))
    if not starts:
        starts = [0]

    best_rms_start = starts[0]
    best_rms_val = -1.0
    for s in starts:
        seg = active[s : s + win_len]
        rms_sq = float(np.mean(np.square(seg, dtype=np.float64)))
        if rms_sq > best_rms_val:
            best_rms_val = rms_sq
            best_rms_start = s

    # Select highest-peak transient window distinct from best_rms_start (>= 50% non-overlapping)
    best_peak_start: int | None = None
    best_peak_val = -1.0
    min_separation = win_len // 2
    for s in starts:
        if abs(s - best_rms_start) < min_separation:
            continue
        seg = active[s : s + win_len]
        pk = float(np.max(np.abs(seg)))
        if pk > best_peak_val:
            best_peak_val = pk
            best_peak_start = s

    seg_rms = apply_cinf_fades(
        active[best_rms_start : best_rms_start + win_len].astype(np.float64),
        fade_len,
    ).astype(np.float32)

    if best_peak_start is None:
        return seg_rms

    seg_peak = apply_cinf_fades(
        active[best_peak_start : best_peak_start + win_len].astype(np.float64),
        fade_len,
    ).astype(np.float32)

    gap = np.zeros(int(0.08 * sample_rate), dtype=np.float32)
    # Order chronologically as they appear in the track
    if best_peak_start < best_rms_start:
        return np.concatenate([seg_peak, gap, seg_rms])
    return np.concatenate([seg_rms, gap, seg_peak])


def _stream_audio_pass(
    send_mono: np.ndarray,
    *,
    device_idx: int,
    output_ch: int,
    rec_channels: int,
    max_out_channels: int,
    sample_rate: int = FS,
    tail_samples: int = 0,
    monitor: bool = False,
    blocksize: int = 2048,
    show_progress: bool = False,
) -> np.ndarray:
    """Streams `send_mono` out on `output_ch` (1-indexed) while recording `rec_channels` inputs.

    Constructs per-block output buffers on the fly to avoid allocating multi-hundred-MB
    multi-channel arrays in host RAM.
    """
    sd = _get_sounddevice()
    n_send = len(send_mono)
    total_samples = n_send + max(0, tail_samples)
    recorded = np.zeros((total_samples, rec_channels), dtype=np.float32)
    chunk_out = np.zeros((blocksize, max_out_channels), dtype=np.float32)

    start_time = time.time()
    last_update = 0.0
    total_duration_s = total_samples / sample_rate

    with sd.Stream(
        device=device_idx,
        samplerate=sample_rate,
        blocksize=blocksize,
        channels=(rec_channels, max_out_channels),
        dtype="float32",
    ) as stream:
        for start_idx in range(0, total_samples, blocksize):
            end_idx = min(start_idx + blocksize, total_samples)
            n_cur = end_idx - start_idx
            chunk_out.fill(0.0)

            if start_idx < n_send:
                n_copy = min(n_cur, n_send - start_idx)
                dry_slice = send_mono[start_idx : start_idx + n_copy]
                if monitor:
                    if max_out_channels >= 1 and output_ch != 1:
                        chunk_out[:n_copy, 0] = dry_slice * 0.5
                    if max_out_channels >= 2 and output_ch != 2:
                        chunk_out[:n_copy, 1] = dry_slice * 0.5
                chunk_out[:n_copy, output_ch - 1] = dry_slice

            stream.write(chunk_out)
            chunk_in, _ = stream.read(blocksize)
            chunk_in_arr = np.asarray(chunk_in, dtype=np.float32)
            if chunk_in_arr.ndim == 1:
                chunk_in_arr = chunk_in_arr[:, np.newaxis]
            recorded[start_idx:end_idx] = chunk_in_arr[:n_cur, :rec_channels]

            if show_progress:
                now = time.time()
                if now - last_update >= 0.25 or end_idx >= total_samples:
                    last_update = now
                    elapsed = now - start_time
                    progress = min(1.0, end_idx / max(total_samples, 1))
                    bar_len = 30
                    filled = int(bar_len * progress)
                    bar = (
                        "=" * filled
                        + (">" if filled < bar_len else "")
                        + " " * (bar_len - filled - 1)
                    )
                    rem = max(0.0, total_duration_s - elapsed)
                    elapsed_str = f"{int(elapsed // 60):02d}:{int(elapsed % 60):02d}"
                    rem_str = f"{int(rem // 60):02d}:{int(rem % 60):02d}"
                    pct_str = f"{progress * 100.0:5.1f}%"
                    sys.stdout.write(
                        f"\r  [{bar[:bar_len]}] {pct_str} | {elapsed_str} elapsed (ETA -{rem_str})"
                    )
                    sys.stdout.flush()

    return recorded


def calibrate_output_gain(
    dry_audio: np.ndarray,
    *,
    device_idx: int,
    output_ch: int = 9,
    input_ch: int = 1,
    stereo: bool = False,
    max_in_channels: int = 3,
    max_out_channels: int = 9,
    sample_rate: int = FS,
    target_peak_dbfs: float = DEFAULT_TARGET_PEAK_DBFS,
    tolerance_db: float = DEFAULT_GAIN_TOLERANCE_DB,
    min_gain_db: float = DEFAULT_MIN_SEND_GAIN_DB,
    max_gain_db: float = DEFAULT_MAX_SEND_GAIN_DB,
    max_iterations: int = DEFAULT_MAX_CAL_ITERATIONS,
    initial_gain_db: float = 0.0,
    tail_sec: float = 0.6,
    monitor: bool = False,
) -> GainCalibrationResult:
    """Identifies the optimal Output 9 send gain for `dry_audio` via closed-loop hardware probing.

    Plays a compact high-energy probe extracted from `dry_audio` through the Anagram and measures
    the return peak on `input_ch` (or stereo Inputs 1 & 2), iteratively adjusting the send gain
    so the wet return hits `target_peak_dbfs` (within `±tolerance_db`) without clipping either
    the USB send or the Anagram return.
    """
    probe = extract_calibration_probe(dry_audio, sample_rate=sample_rate)
    max_unclipped_db = compute_max_unclipped_send_gain_db(dry_audio, sample_rate=sample_rate)
    effective_max_gain_db = min(float(max_gain_db), max_unclipped_db)
    effective_min_gain_db = float(min_gain_db)
    effective_max_gain_db = max(effective_max_gain_db, effective_min_gain_db)

    rec_channels = 2 if stereo else max(input_ch, min(max_in_channels, 3))
    tail_samples = max(0, int(tail_sec * sample_rate))

    current_gain_db = float(np.clip(initial_gain_db, effective_min_gain_db, effective_max_gain_db))
    prev_unclipped_gain_db: float | None = None
    prev_unclipped_peak_dbfs: float | None = None

    best_gain_db = current_gain_db
    best_peak_dbfs = -180.0
    best_rms_dbfs = -180.0
    best_err_db = float("inf")
    converged = False
    last_clipped = False
    iterations_run = 0

    print(
        f"\n[AUTO-GAIN] Probing Output {output_ch} -> "
        f"{'Inputs 1 & 2' if stereo else f'Input {input_ch}'} "
        f"(Target Peak: {target_peak_dbfs:+.1f} dBFS ±{tolerance_db:.1f} dB, "
        f"Max Clean Send: {effective_max_gain_db:+.2f} dB)...",
        flush=True,
    )

    for iteration in range(1, max(1, max_iterations) + 1):
        iterations_run = iteration
        gain_linear = 10.0 ** (current_gain_db / 20.0)
        scaled_probe = np.clip(
            probe * gain_linear,
            -CALIBRATION_PEAK_CEILING,
            CALIBRATION_PEAK_CEILING,
        ).astype(np.float32)

        rec = _stream_audio_pass(
            scaled_probe,
            device_idx=device_idx,
            output_ch=output_ch,
            rec_channels=rec_channels,
            max_out_channels=max_out_channels,
            sample_rate=sample_rate,
            tail_samples=tail_samples,
            monitor=monitor,
            show_progress=False,
        )

        ret_sig = rec[:, :2] if stereo else rec[:, input_ch - 1]
        ret_peak_dbfs, ret_rms_dbfs = compute_audio_levels(ret_sig)

        if ret_peak_dbfs < SILENCE_FLOOR_DBFS:
            raise RuntimeError(
                f"Auto-gain probe detected silence on return ({ret_peak_dbfs:+.1f} dBFS < {SILENCE_FLOOR_DBFS:+.1f} dBFS). "
                f"Verify that Anagram preset input is routed from USB Output {output_ch} and preset volume is unmuted."
            )

        is_clipped = ret_peak_dbfs >= CLIP_DETECTION_DBFS
        last_clipped = is_clipped
        status_tag = (
            "CLIP!"
            if is_clipped
            else ("OK" if abs(target_peak_dbfs - ret_peak_dbfs) <= tolerance_db else "adjusting")
        )
        print(
            f"  [Probe {iteration}/{max_iterations}] Send: {current_gain_db:+.2f} dB ({gain_linear:.3f}x) -> "
            f"Return Peak: {ret_peak_dbfs:+.2f} dBFS | RMS: {ret_rms_dbfs:+.2f} dBFS [{status_tag}]",
            flush=True,
        )

        if is_clipped:
            # Aggressive step-down to escape digital converter / return saturation quickly
            backoff_db = max(6.0, (ret_peak_dbfs - target_peak_dbfs) + 3.0)
            next_gain_db = max(effective_min_gain_db, current_gain_db - backoff_db)
            if best_err_db == float("inf"):
                best_gain_db = next_gain_db
                best_peak_dbfs = ret_peak_dbfs
                best_rms_dbfs = ret_rms_dbfs
            if abs(next_gain_db - current_gain_db) < 0.10:
                break
            current_gain_db = next_gain_db
            continue

        err_db = target_peak_dbfs - ret_peak_dbfs
        abs_err = abs(err_db)
        if abs_err < best_err_db:
            best_err_db = abs_err
            best_gain_db = current_gain_db
            best_peak_dbfs = ret_peak_dbfs
            best_rms_dbfs = ret_rms_dbfs

        if abs_err <= tolerance_db:
            converged = True
            break

        # Compute next gain adjustment (accounting for non-linear amp compression / limiting)
        if prev_unclipped_gain_db is not None and prev_unclipped_peak_dbfs is not None:
            d_in = current_gain_db - prev_unclipped_gain_db
            d_out = ret_peak_dbfs - prev_unclipped_peak_dbfs
            if d_in > 1.5 and d_out < 0.25 and ret_peak_dbfs > -15.0:
                # Internal preset brickwall limiter / hard saturation ceiling reached
                print(
                    f"  [Auto-Gain] Preset compression ceiling detected "
                    f"(+{d_in:.2f} dB send yielded only {d_out:+.2f} dB peak change). "
                    f"Locking previous unclipped send gain ({prev_unclipped_gain_db:+.2f} dB).",
                    flush=True,
                )
                best_gain_db = prev_unclipped_gain_db
                best_peak_dbfs = prev_unclipped_peak_dbfs
                converged = True
                break
            if abs(d_in) >= 0.25:
                slope = float(np.clip(d_out / d_in, 0.40, 1.50))
                step_db = err_db / slope
            else:
                step_db = err_db
        else:
            step_db = err_db

        prev_unclipped_gain_db = current_gain_db
        prev_unclipped_peak_dbfs = ret_peak_dbfs

        step_db = float(np.clip(step_db, -12.0, 9.0))
        next_gain_db = float(
            np.clip(current_gain_db + step_db, effective_min_gain_db, effective_max_gain_db)
        )

        if abs(next_gain_db - current_gain_db) < 0.15:
            if next_gain_db >= effective_max_gain_db - 1e-6 and err_db > tolerance_db:
                print(
                    f"  [Auto-Gain] Reached maximum unclipped Output {output_ch} digital ceiling "
                    f"({effective_max_gain_db:+.2f} dB; return peak {ret_peak_dbfs:+.2f} dBFS).",
                    flush=True,
                )
            break

        current_gain_db = next_gain_db

    final_linear = 10.0 ** (best_gain_db / 20.0)
    print(
        f"  ✅ Selected Output {output_ch} Send Gain: {best_gain_db:+.2f} dB ({final_linear:.3f}x) "
        f"[Return Peak: {best_peak_dbfs:+.2f} dBFS | RMS: {best_rms_dbfs:+.2f} dBFS]",
        flush=True,
    )
    return GainCalibrationResult(
        output_gain_db=best_gain_db,
        output_gain_linear=final_linear,
        measured_peak_dbfs=best_peak_dbfs,
        measured_rms_dbfs=best_rms_dbfs,
        target_peak_dbfs=target_peak_dbfs,
        max_unclipped_gain_db=effective_max_gain_db,
        iterations=iterations_run,
        converged=converged,
        clipped=last_clipped,
    )


def align_hardware_latency(
    dry: np.ndarray,
    wet: np.ndarray,
    sample_rate: int = FS,
    max_search_ms: float = 1000.0,
    causal_guard_samples: int = 4,
) -> tuple[int, np.ndarray]:
    """Estimates roundtrip hardware latency via 2-stage coarse-to-fine cross-correlation and returns (delay_samples, aligned_wet).

    Guarantees `len(aligned_wet) == len(dry)` by padding trailing zeros if `delay_samples + len(dry) > len(wet)`.
    """
    target_len = len(dry)
    if target_len == 0 or len(wet) == 0:
        return 0, np.zeros(target_len, dtype=wet.dtype if len(wet) > 0 else np.float32)

    max_delay_samples = min(
        int(sample_rate * (max_search_ms / 1000.0)),
        max(0, len(wet) - 16),
    )
    if max_delay_samples <= 0 or len(wet) <= 16:
        out = wet[:target_len]
        if len(out) < target_len:
            out = np.pad(out, (0, target_len - len(out)))
        return 0, out

    avail_dry = max(0, min(target_len, len(wet) - max_delay_samples))
    if avail_dry < min(target_len, int(sample_rate * 1.5)):
        max_delay_samples = max(0, min(max_delay_samples, len(wet) // 2))
        avail_dry = max(0, min(target_len, len(wet) - max_delay_samples))

    search_len = min(avail_dry, int(sample_rate * 60.0))
    if search_len <= 16 or len(wet) < search_len + max_delay_samples:
        out = wet[:target_len]
        if len(out) < target_len:
            out = np.pad(out, (0, target_len - len(out)))
        return 0, out

    # Stage 1: Rectified first-difference envelope FFT cross-correlation (invariant to polarity & 2-blip aliasing)
    rx = np.abs(np.diff(dry[:search_len].astype(np.float32)))
    ry = np.abs(np.diff(wet[: search_len + max_delay_samples].astype(np.float32)))
    rx -= float(np.mean(rx))
    ry -= float(np.mean(ry))

    n_fft = 1 << (len(rx) + len(ry) - 1).bit_length()
    cc_env = np.fft.irfft(
        np.fft.rfft(ry, n=n_fft) * np.conj(np.fft.rfft(rx, n=n_fft)),
        n=n_fft,
    )[: max_delay_samples + 1]
    coarse_delay = int(np.argmax(cc_env))

    # Stage 2: Fine waveform cross-correlation around coarse_delay (+/- 64 samples)
    fine_radius = 64
    w_lo = max(0, coarse_delay - fine_radius)
    w_hi = min(max_delay_samples, coarse_delay + fine_radius)

    blip_window = min(search_len, int(sample_rate * 1.5))
    if float(np.max(np.abs(dry[:blip_window]))) < 1e-3:
        blip_window = min(search_len, int(sample_rate * 6.0))

    xb = dry[:blip_window].astype(np.float32)
    yb = wet[w_lo : blip_window + w_hi].astype(np.float32)

    if len(yb) > len(xb) and float(np.max(np.abs(yb))) > 1e-6:
        corr_fine = np.correlate(yb, xb, mode="valid")
        peak_delay = w_lo + int(np.argmax(np.abs(corr_fine)))
    else:
        peak_delay = coarse_delay

    delay_samples = (
        peak_delay - causal_guard_samples if peak_delay >= causal_guard_samples else peak_delay
    )

    aligned_wet = wet[delay_samples : delay_samples + target_len]
    if len(aligned_wet) < target_len:
        aligned_wet = np.pad(aligned_wet, (0, target_len - len(aligned_wet)))
    return delay_samples, aligned_wet


def write_reamp_wav_24bit(
    filepath: str | Path,
    samples: np.ndarray,
    sample_rate: int = FS,
) -> None:
    """Writes a 24-bit 48 kHz mono or stereo WAV file."""
    arr = np.asarray(samples, dtype=np.float32)
    if arr.ndim == 1:
        write_wav_24bit(filepath, arr, sample_rate=sample_rate)
        return

    # 2D array shaped (n_samples, n_channels)
    n_samples, n_channels = arr.shape
    if n_channels == 1:
        write_wav_24bit(filepath, arr[:, 0], sample_rate=sample_rate)
        return

    try:
        from pedalboard.io import AudioFile

        ch_first = np.ascontiguousarray(arr.T, dtype=np.float32)
        with AudioFile(
            str(filepath),
            "w",
            samplerate=sample_rate,
            num_channels=n_channels,
            bit_depth=24,
        ) as f:
            f.write(ch_first)
    except ImportError, RuntimeError, OSError, ValueError:
        with wave.open(str(filepath), "wb") as wf:
            wf.setnchannels(n_channels)
            wf.setsampwidth(3)
            wf.setframerate(sample_rate)
            interleaved = arr.reshape(-1)
            scaled = np.clip(interleaved * 8388607.0, -8388608.0, 8388607.0).astype(np.int32)
            raw_bytes = (
                scaled.astype("<i4")
                .view(np.uint8)
                .reshape(n_samples * n_channels, 4)[:, :3]
                .tobytes()
            )
            wf.writeframes(raw_bytes)


def run_test_tone(
    device_idx: int,
    output_ch: int = 9,
    input_ch: int = 1,
    max_in_channels: int = 3,
    max_out_channels: int = 9,
    sample_rate: int = FS,
) -> bool:
    """Emits a 1-second 440 Hz test tone on Output 9 to verify signal path and levels."""
    print(f"\n[TEST] Emitting 1.0s 440 Hz test tone (-18.0 dBFS) through Output {output_ch}...")
    t = np.arange(sample_rate, dtype=np.float64) / sample_rate
    raw_tone = 0.125 * np.sin(2.0 * np.pi * 440.0 * t)
    tone = apply_cinf_fades(raw_tone, int(0.015 * sample_rate)).astype(np.float32)

    rec_channels = max(input_ch, min(max_in_channels, 3))
    rec = _stream_audio_pass(
        tone,
        device_idx=device_idx,
        output_ch=output_ch,
        rec_channels=rec_channels,
        max_out_channels=max_out_channels,
        sample_rate=sample_rate,
        tail_samples=int(0.5 * sample_rate),
        monitor=False,
        show_progress=False,
    )

    ret_signal = rec[:, input_ch - 1]
    peak_dbfs, rms_dbfs = compute_audio_levels(ret_signal)

    print(f"  Return Signal Peak: {peak_dbfs:+.1f} dBFS | RMS: {rms_dbfs:+.1f} dBFS")
    if peak_dbfs < -40.0:
        print("  ❌ WARNING: Return signal is extremely quiet (<-40 dBFS).")
        print(
            f"     Check that your preset in Anagram has USB Reamp (Output {output_ch}) assigned to block input"
        )
        print("     and that the master preset / block volume is turned up.")
        return False
    if peak_dbfs > -1.0:
        print(
            "  ⚠️ WARNING: Return signal is near clipping (> -1.0 dBFS). Back off block output level."
        )
    else:
        print("  ✅ Signal path verified cleanly!")
    return True


@app.default
def reamp(
    *,
    input_file: Annotated[
        Path,
        Parameter(
            name=["--input", "-i"],
            help="Path to dry excitation input WAV file.",
        ),
    ] = REPO_ROOT / "audio" / "input.wav",
    output_file: Annotated[
        Path,
        Parameter(
            name=["--output", "-o"],
            help="Path to save aligned wet audio WAV file.",
        ),
    ] = REPO_ROOT / "audio" / "reamped_wet.wav",
    output_channel: Annotated[
        int,
        Parameter(
            name=["--output-channel"],
            help="Anagram USB Output channel number (1-indexed; Output 9 is default Reamp Send).",
        ),
    ] = 9,
    input_channel: Annotated[
        int,
        Parameter(
            name=["--input-channel"],
            help="Anagram USB Input channel number (1-indexed; Input 1 is Mono/Left Rig Return).",
        ),
    ] = 1,
    output_gain: Annotated[
        float | str,
        Parameter(
            name=["--output-gain", "-g"],
            help="Output send gain in dB ('auto' for closed-loop hardware probe, or explicit dB value).",
        ),
    ] = "auto",
    target_peak: Annotated[
        float,
        Parameter(
            name=["--target-peak"],
            help="Target wet return peak in dBFS during automatic output gain calibration.",
        ),
    ] = DEFAULT_TARGET_PEAK_DBFS,
    no_auto_gain: Annotated[
        bool,
        Parameter(
            name=["--no-auto-gain"],
            help="Disable automatic output gain calibration and use 0.0 dB unity send gain.",
        ),
    ] = False,
    stereo: Annotated[
        bool,
        Parameter(
            name=["--stereo"],
            help="Record stereo return from Anagram Inputs 1 & 2 instead of mono Input 1.",
        ),
    ] = False,
    tail: Annotated[
        float,
        Parameter(
            name=["--tail"],
            help="Extra recording tail in seconds to capture USB latency and cab ring-out decay.",
        ),
    ] = 1.5,
    no_align: Annotated[
        bool,
        Parameter(
            name=["--no-align"],
            help="Disable automatic latency alignment (output raw recording).",
        ),
    ] = False,
    save_raw: Annotated[
        bool,
        Parameter(
            name=["--save-raw"],
            help="Save raw unaligned recording to <output>_raw.wav alongside aligned output.",
        ),
    ] = False,
    monitor: Annotated[
        bool,
        Parameter(
            name=["--monitor"],
            help="Mirror Output 9 audio to Outputs 1 & 2 for headphone/speaker monitoring.",
        ),
    ] = False,
    test: Annotated[
        bool,
        Parameter(
            name=["--test"],
            help="Run 1-second 440 Hz test tone to check levels before re-amping.",
        ),
    ] = False,
    device: Annotated[
        str,
        Parameter(
            name=["--device"],
            help="Audio device name search pattern or index.",
        ),
    ] = "Anagram",
) -> None:
    """Plays dry excitation audio through Anagram USB Output 9 and records the aligned wet return."""
    min_req_in = 2 if stereo else max(1, input_channel)
    min_req_out = max(1, output_channel)

    # 1. Locate Device & Validate Channels
    dev_idx, dev_info = find_anagram_device(
        device,
        min_in_channels=min_req_in,
        min_out_channels=min_req_out,
    )
    max_out = int(dev_info["max_output_channels"])
    max_in = int(dev_info["max_input_channels"])

    if output_channel < 1 or output_channel > max_out:
        raise ValueError(
            f"Requested output channel {output_channel} is outside device range (1..{max_out})."
        )
    if stereo and max_in < 2:
        raise ValueError(
            f"Stereo return requested (--stereo), but device only has {max_in} input channel(s)."
        )
    if not stereo and (input_channel < 1 or input_channel > max_in):
        raise ValueError(
            f"Requested input channel {input_channel} is outside device range (1..{max_in})."
        )

    print("==================================================================")
    print(" Allomorph Darkglass Anagram Hardware Re-Amper")
    print("==================================================================")
    print(f"  Device:         {dev_info['name']} (Index {dev_idx})")
    print(f"  Capabilities:   {max_in} Inputs, {max_out} Outputs @ 48 kHz")
    print(f"  Send Channel:   Output {output_channel} (USB Reamp Send)")
    in_desc = "Inputs 1 & 2 (Stereo)" if stereo else f"Input {input_channel} (Mono)"
    print(f"  Return Channel: {in_desc}")
    print("------------------------------------------------------------------")

    # 2. Optional Quick Test Tone Check
    if test:
        run_test_tone(
            dev_idx,
            output_ch=output_channel,
            input_ch=input_channel,
            max_in_channels=max_in,
            max_out_channels=max_out,
        )
        return

    # 3. Load Dry Audio
    if not input_file.exists():
        raise FileNotFoundError(f"Input excitation file '{input_file}' not found.")

    print(f"Loading dry input: {input_file.name}...")
    dry_audio, sr = read_wav(input_file, dtype=np.float32)
    if dry_audio.ndim > 1:
        dry_audio = np.mean(dry_audio, axis=0)

    if sr != FS:
        raise ValueError(f"Input audio sample rate must be {FS} Hz (got {sr} Hz).")

    n_samples = len(dry_audio)
    duration_s = n_samples / sr
    tail_samples = max(0, int(tail * sr))
    total_samples = n_samples + tail_samples
    total_duration_s = total_samples / sr
    dry_peak_dbfs, dry_rms_dbfs = compute_audio_levels(dry_audio)

    print(
        f"  Dry Audio:      {n_samples:,} samples ({duration_s:.1f}s / {duration_s / 60.0:.2f} min)"
    )
    print(f"  Dry Level:      Peak: {dry_peak_dbfs:+.2f} dBFS | RMS: {dry_rms_dbfs:+.2f} dBFS")
    print(f"  Tail Padding:   {tail:.1f}s ({tail_samples:,} samples)")
    print(f"  Total Session:  {total_duration_s:.1f}s ({total_duration_s / 60.0:.2f} min)")
    if monitor:
        print("  Monitoring:     Outputs 1 & 2 enabled (headphone mirror)")

    # 4. Resolve or Automatically Calibrate Output Send Gain
    if no_auto_gain and isinstance(output_gain, str) and output_gain.strip().lower() == "auto":
        send_gain_db = 0.0
        send_gain_linear = 1.0
        print("  Output Gain:    +0.00 dB (1.000x, --no-auto-gain)")
    elif isinstance(output_gain, (int, float)) or (
        isinstance(output_gain, str) and output_gain.strip().lower() != "auto"
    ):
        send_gain_db = float(output_gain)
        max_clean_db = compute_max_unclipped_send_gain_db(dry_audio, sample_rate=sr)
        if send_gain_db > max_clean_db:
            print(
                f"  ⚠️ WARNING: Requested --output-gain {send_gain_db:+.2f} dB exceeds clean digital ceiling "
                f"({max_clean_db:+.2f} dB); clamping peaks to -0.09 dBFS."
            )
        send_gain_linear = 10.0 ** (send_gain_db / 20.0)
        print(f"  Output Gain:    {send_gain_db:+.2f} dB ({send_gain_linear:.3f}x, manual)")
    else:
        cal_res = calibrate_output_gain(
            dry_audio,
            device_idx=dev_idx,
            output_ch=output_channel,
            input_ch=input_channel,
            stereo=stereo,
            max_in_channels=max_in,
            max_out_channels=max_out,
            sample_rate=sr,
            target_peak_dbfs=target_peak,
            monitor=monitor,
        )
        send_gain_db = cal_res.output_gain_db
        send_gain_linear = cal_res.output_gain_linear

    scaled_dry = np.clip(
        dry_audio * send_gain_linear,
        -CALIBRATION_PEAK_CEILING,
        CALIBRATION_PEAK_CEILING,
    ).astype(np.float32)

    # 5. Playback and Record via blocking Stream (bypasses CFFI callback trampolines)
    rec_channels = 2 if stereo else max(input_channel, min(max_in, 3))
    print("\nStarting real-time re-amping pass. Press Ctrl+C to abort...")

    try:
        recorded = _stream_audio_pass(
            scaled_dry,
            device_idx=dev_idx,
            output_ch=output_channel,
            rec_channels=rec_channels,
            max_out_channels=max_out,
            sample_rate=sr,
            tail_samples=tail_samples,
            monitor=monitor,
            show_progress=True,
        )
    except KeyboardInterrupt:
        print("\n\n[ABORTED] Re-amping cancelled by user.")
        return

    print("\n\nPlayback complete! Processing recorded return signal...")

    # 6. Channel Extraction
    if stereo:
        wet_raw = recorded[:total_samples, :2]
        ref_channel_audio = wet_raw[:, 0]
    else:
        wet_raw = recorded[:total_samples, input_channel - 1]
        ref_channel_audio = wet_raw

    raw_peak, raw_rms = compute_audio_levels(wet_raw)
    print(f"  Raw Peak:       {raw_peak:+.1f} dBFS | RMS: {raw_rms:+.1f} dBFS")

    # 7. Automatic Time-Alignment
    if not no_align:
        delay_samples, aligned_mono = align_hardware_latency(
            dry=scaled_dry,
            wet=ref_channel_audio,
            sample_rate=sr,
        )
        delay_ms = (delay_samples / sr) * 1000.0
        print(f"  Latency Offset: {delay_samples} samples ({delay_ms:.2f} ms) — aligned")

        if stereo:
            aligned_ch0 = wet_raw[delay_samples : delay_samples + n_samples, 0]
            if len(aligned_ch0) < n_samples:
                aligned_ch0 = np.pad(aligned_ch0, (0, n_samples - len(aligned_ch0)))
            aligned_ch1 = wet_raw[delay_samples : delay_samples + n_samples, 1]
            if len(aligned_ch1) < n_samples:
                aligned_ch1 = np.pad(aligned_ch1, (0, n_samples - len(aligned_ch1)))
            aligned_wet = np.column_stack([aligned_ch0, aligned_ch1])
        else:
            aligned_wet = aligned_mono
    else:
        print("  Latency Offset: Skipped (--no-align)")
        aligned_wet = wet_raw[:n_samples]

    # 8. Export Files
    output_file.parent.mkdir(parents=True, exist_ok=True)
    if save_raw:
        raw_path = output_file.parent / f"{output_file.stem}_raw.wav"
        write_reamp_wav_24bit(raw_path, wet_raw, sample_rate=sr)
        print(f"  Raw Wet Audio:  {raw_path.name}")

    write_reamp_wav_24bit(output_file, aligned_wet, sample_rate=sr)
    final_peak, final_rms = compute_audio_levels(aligned_wet)

    print("------------------------------------------------------------------")
    print(f"  Saved Output:   {output_file}")
    print(f"  Send Gain Used: {send_gain_db:+.2f} dB ({send_gain_linear:.3f}x)")
    print(f"  Final Level:    Peak: {final_peak:+.1f} dBFS | RMS: {final_rms:+.1f} dBFS")

    if final_peak >= -0.1:
        print(
            "  ⚠️ WARNING: Output signal is hard-clipping! Lower the amp/cab level in the Anagram preset."
        )
    elif final_peak < -24.0:
        print("  ℹ️ NOTICE: Output signal is fairly quiet. Check gain staging to ensure good SNR.")
    else:
        print("  ✅ Exported bit-exact 24-bit 48 kHz aligned stem ready for NAM training!")
    print("==================================================================\n")


if __name__ == "__main__":
    app()
