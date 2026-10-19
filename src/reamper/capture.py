"""Allomorph Reamper - Darkglass Anagram Hardware Re-Amping & Idle Noise Capture.

Plays dry excitation audio through the Darkglass Anagram USB audio interface
specifically on Output 9 (USB Reamp Send) while simultaneously recording the
wet return signal from Input 1 (or stereo Inputs 1 & 2) at 24-bit 48 kHz.
Also provides `capture_idle_noise` for recording real-world hardware hum and
SMPS noise profiles.
"""

from __future__ import annotations

import importlib
import json
import math
import sys
import time
import wave
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any

import numpy as np
from cyclopts import Parameter

from allomorph.dsp.constants import CALIBRATION_PEAK_CEILING, FS, apply_cinf_fades
from allomorph.dsp.io import read_wav, write_wav_24bit

REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_TARGET_PEAK_DBFS: float = -3.0
DEFAULT_GAIN_TOLERANCE_DB: float = 0.5
DEFAULT_MIN_SEND_GAIN_DB: float = -24.0
DEFAULT_MAX_SEND_GAIN_DB: float = 18.0
DEFAULT_MAX_CAL_ITERATIONS: int = 5
DEFAULT_DOWNWARD_PROBE_STEP_DB: float = -6.0
COMPRESSION_SLOPE_THRESHOLD: float = 0.40
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
    """Lazily imports sounddevice so CLI --help and offline functions work without PortAudio."""
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
    """Locates the Anagram USB audio device index and property dictionary."""
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

    for idx, dev_dict in matching:
        max_in = int(dev_dict.get("max_input_channels", 0))
        max_out = int(dev_dict.get("max_output_channels", 0))
        if max_in >= min_in_channels and max_out >= min_out_channels:
            return idx, dev_dict

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
    window_sec: float = 0.9,
) -> np.ndarray:
    """Extracts a compact (~2.8s) high-energy calibration probe from `dry_audio`."""
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

    freqs = np.fft.rfftfreq(len(active), 1.0 / sample_rate)
    cab_weight = np.exp(-0.5 * ((np.log10(np.maximum(freqs, 10.0) / 95.0)) / 0.25) ** 2)
    cab_filtered = np.fft.irfft(
        np.fft.rfft(active.astype(np.float64)) * cab_weight,
        n=len(active),
    ).astype(np.float32)

    hop_len = max(64, win_len // 4)
    starts = list(range(0, len(active) - win_len + 1, hop_len))
    if not starts:
        starts = [0]

    min_separation = win_len // 2
    selected_starts: list[int] = []

    best_cab_start = max(
        starts,
        key=lambda s: float(np.max(np.abs(cab_filtered[s : s + win_len]))),
    )
    selected_starts.append(best_cab_start)

    rms_candidates = [
        s for s in starts if all(abs(s - prev) >= min_separation for prev in selected_starts)
    ]
    if rms_candidates:
        best_rms_start = max(
            rms_candidates,
            key=lambda s: float(np.mean(np.square(active[s : s + win_len], dtype=np.float64))),
        )
        selected_starts.append(best_rms_start)

    peak_candidates = [
        s for s in starts if all(abs(s - prev) >= min_separation for prev in selected_starts)
    ]
    if peak_candidates:
        best_peak_start = max(
            peak_candidates,
            key=lambda s: float(np.max(np.abs(active[s : s + win_len]))),
        )
        selected_starts.append(best_peak_start)

    selected_starts.sort()
    gap = np.zeros(int(0.06 * sample_rate), dtype=np.float32)
    pieces: list[np.ndarray] = []
    for idx, s in enumerate(selected_starts):
        if idx > 0:
            pieces.append(gap)
        seg = apply_cinf_fades(
            active[s : s + win_len].astype(np.float64),
            fade_len,
        ).astype(np.float32)
        pieces.append(seg)

    return np.concatenate(pieces)


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
    """Streams `send_mono` out on `output_ch` (1-indexed) while recording `rec_channels` inputs."""
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
                if max_out_channels >= output_ch >= 1:
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
    """Identifies the optimal Output 9 send gain for `dry_audio` via closed-loop hardware probing."""
    probe = extract_calibration_probe(dry_audio, sample_rate=sample_rate)
    max_unclipped_db = compute_max_unclipped_send_gain_db(dry_audio, sample_rate=sample_rate)
    effective_max_gain_db = min(float(max_gain_db), max_unclipped_db)
    effective_min_gain_db = float(min_gain_db)
    effective_max_gain_db = max(effective_max_gain_db, effective_min_gain_db)

    rec_channels = 2 if stereo else max(input_ch, min(max_in_channels, 3))
    tail_samples = max(0, int(tail_sec * sample_rate))

    current_gain_db = float(np.clip(initial_gain_db, effective_min_gain_db, effective_max_gain_db))
    ref_gain_db: float | None = None
    ref_peak_dbfs: float | None = None
    ref_rms_dbfs: float | None = None

    best_gain_db = current_gain_db
    best_peak_dbfs = -180.0
    best_rms_dbfs = -180.0
    best_err_db = float("inf")
    compression_search_active = False
    probed_gain_reduction = False
    converged = False
    last_clipped = False
    iterations_run = 0

    print(
        f"\n[AUTO-GAIN] Probing Output {output_ch} -> "
        f"{'Inputs 1 & 2' if stereo else f'Input {input_ch}'} "
        f"(Target Peak: {target_peak_dbfs:+.1f} dBFS ±{tolerance_db:.1f} dB, "
        f"Range: [{effective_min_gain_db:+.1f}, {effective_max_gain_db:+.2f}] dB)...",
        flush=True,
    )

    total_iters = max(1, max_iterations)
    for iteration in range(1, total_iters + 1):
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
        err_db = target_peak_dbfs - ret_peak_dbfs
        abs_err = abs(err_db)

        measured_slope: float | None = None
        if ref_gain_db is not None and ref_peak_dbfs is not None:
            d_in = current_gain_db - ref_gain_db
            d_out = ret_peak_dbfs - ref_peak_dbfs
            if abs(d_in) >= 0.20:
                measured_slope = d_out / d_in

        slope_str = f" | Slope: {measured_slope:.2f} dB/dB" if measured_slope is not None else ""
        if is_clipped:
            status_tag = "CLIP! -> reducing gain"
        elif compression_search_active:
            status_tag = (
                "knee found"
                if (measured_slope is not None and measured_slope >= COMPRESSION_SLOPE_THRESHOLD)
                else "saturated -> reducing gain"
            )
        elif iteration == 1 and not probed_gain_reduction and total_iters > 1:
            status_tag = (
                "hot -> reducing gain"
                if err_db < -tolerance_db
                else "baseline -> probing reduced gain"
            )
        elif abs_err <= tolerance_db and (
            measured_slope is None or measured_slope >= COMPRESSION_SLOPE_THRESHOLD
        ):
            status_tag = "OK"
        else:
            status_tag = "adjusting"

        print(
            f"  [Probe {iteration}/{total_iters}] Send: {current_gain_db:+.2f} dB ({gain_linear:.3f}x) -> "
            f"Return Peak: {ret_peak_dbfs:+.2f} dBFS | RMS: {ret_rms_dbfs:+.2f} dBFS{slope_str} [{status_tag}]",
            flush=True,
        )

        if is_clipped:
            probed_gain_reduction = True
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

        if ref_gain_db is None or ref_peak_dbfs is None or ref_rms_dbfs is None:
            ref_gain_db = current_gain_db
            ref_peak_dbfs = ret_peak_dbfs
            ref_rms_dbfs = ret_rms_dbfs
            best_gain_db = current_gain_db
            best_peak_dbfs = ret_peak_dbfs
            best_rms_dbfs = ret_rms_dbfs
            best_err_db = abs_err

            if abs_err <= tolerance_db and (probed_gain_reduction or total_iters == 1):
                converged = True
                break

            probed_gain_reduction = True
            if err_db < -tolerance_db:
                step_down_db = float(np.clip(min(err_db / 0.65, -3.0), -12.0, -1.5))
                next_gain_db = max(effective_min_gain_db, current_gain_db + step_down_db)
            else:
                next_gain_db = max(
                    effective_min_gain_db,
                    current_gain_db + DEFAULT_DOWNWARD_PROBE_STEP_DB,
                )

            if abs(next_gain_db - current_gain_db) < 0.15:
                break
            current_gain_db = next_gain_db
            continue

        if compression_search_active:
            if measured_slope is not None and measured_slope >= COMPRESSION_SLOPE_THRESHOLD:
                print(
                    f"  [Auto-Gain] Dynamic response recovered below {ref_gain_db:+.2f} dB "
                    f"(slope {measured_slope:.2f} dB/dB). Locking compression knee at {ref_gain_db:+.2f} dB.",
                    flush=True,
                )
                best_gain_db = ref_gain_db
                best_peak_dbfs = ref_peak_dbfs
                best_rms_dbfs = ref_rms_dbfs
                converged = True
                break

            ref_gain_db = current_gain_db
            ref_peak_dbfs = ret_peak_dbfs
            ref_rms_dbfs = ret_rms_dbfs
            best_gain_db = current_gain_db
            best_peak_dbfs = ret_peak_dbfs
            best_rms_dbfs = ret_rms_dbfs
            best_err_db = abs_err

            next_gain_db = max(
                effective_min_gain_db,
                current_gain_db + DEFAULT_DOWNWARD_PROBE_STEP_DB,
            )
            if abs(next_gain_db - current_gain_db) < 0.15:
                break
            current_gain_db = next_gain_db
            continue

        d_in = current_gain_db - ref_gain_db
        slope_val = measured_slope if measured_slope is not None else 1.0

        if d_in < 0.0 and ref_peak_dbfs <= target_peak_dbfs + tolerance_db:
            if slope_val < COMPRESSION_SLOPE_THRESHOLD and ref_peak_dbfs > -24.0:
                print(
                    f"  [Auto-Gain] Internal preset compression ceiling detected between "
                    f"{current_gain_db:+.2f} dB and {ref_gain_db:+.2f} dB "
                    f"(slope {slope_val:.2f} dB/dB; peak changed by only {ret_peak_dbfs - ref_peak_dbfs:+.2f} dB). "
                    f"Probing further gain reduction...",
                    flush=True,
                )
                compression_search_active = True
                ref_gain_db = current_gain_db
                ref_peak_dbfs = ret_peak_dbfs
                ref_rms_dbfs = ret_rms_dbfs
                best_gain_db = current_gain_db
                best_peak_dbfs = ret_peak_dbfs
                best_rms_dbfs = ret_rms_dbfs
                best_err_db = abs_err

                next_gain_db = max(
                    effective_min_gain_db,
                    current_gain_db + DEFAULT_DOWNWARD_PROBE_STEP_DB,
                )
                if abs(next_gain_db - current_gain_db) < 0.15:
                    break
                current_gain_db = next_gain_db
                continue

            ref_err_db = target_peak_dbfs - ref_peak_dbfs
            if abs(ref_err_db) <= tolerance_db:
                best_gain_db = ref_gain_db
                best_peak_dbfs = ref_peak_dbfs
                best_rms_dbfs = ref_rms_dbfs
                best_err_db = abs(ref_err_db)
                converged = True
                break

            slope_eff = float(np.clip(slope_val, COMPRESSION_SLOPE_THRESHOLD, 1.50))
            step_up_db = float(np.clip(ref_err_db / slope_eff, -12.0, 12.0))
            next_gain_db = float(
                np.clip(ref_gain_db + step_up_db, effective_min_gain_db, effective_max_gain_db)
            )
            if abs(next_gain_db - ref_gain_db) < 0.15:
                if next_gain_db >= effective_max_gain_db - 1e-6 and ref_err_db > tolerance_db:
                    print(
                        f"  [Auto-Gain] Reached maximum unclipped Output {output_ch} digital ceiling "
                        f"({effective_max_gain_db:+.2f} dB; return peak {ref_peak_dbfs:+.2f} dBFS).",
                        flush=True,
                    )
                break
            current_gain_db = next_gain_db
            continue

        if d_in > 0.5 and slope_val < COMPRESSION_SLOPE_THRESHOLD and ret_peak_dbfs > -24.0:
            print(
                f"  [Auto-Gain] Preset compression ceiling reached above {ref_gain_db:+.2f} dB "
                f"(+{d_in:.2f} dB send yielded only {ret_peak_dbfs - ref_peak_dbfs:+.2f} dB peak change, "
                f"slope {slope_val:.2f} dB/dB). Backing off to {ref_gain_db:+.2f} dB.",
                flush=True,
            )
            best_gain_db = ref_gain_db
            best_peak_dbfs = ref_peak_dbfs
            best_rms_dbfs = ref_rms_dbfs
            converged = True
            break

        if abs_err < best_err_db:
            best_err_db = abs_err
            best_gain_db = current_gain_db
            best_peak_dbfs = ret_peak_dbfs
            best_rms_dbfs = ret_rms_dbfs

        if abs_err <= tolerance_db and slope_val >= 0.30:
            converged = True
            break

        if err_db < 0.0 and slope_val < COMPRESSION_SLOPE_THRESHOLD:
            step_db = min(err_db / max(slope_val, 0.20), DEFAULT_DOWNWARD_PROBE_STEP_DB)
        else:
            slope_eff = float(np.clip(slope_val, 0.35, 1.50))
            step_db = err_db / slope_eff

        ref_gain_db = current_gain_db
        ref_peak_dbfs = ret_peak_dbfs
        ref_rms_dbfs = ret_rms_dbfs

        step_db = float(np.clip(step_db, -12.0, 12.0))
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
    """Estimates roundtrip hardware latency via 2-stage coarse-to-fine cross-correlation and returns (delay_samples, aligned_wet)."""
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


def get_reamp_sidecar_path(wav_filepath: str | Path) -> Path:
    """Returns the JSON sidecar path corresponding to a re-amped WAV stem."""
    p = Path(wav_filepath)
    return p.with_suffix(".json")


def write_reamp_sidecar(
    wav_filepath: str | Path,
    *,
    send_gain_db: float,
    send_gain_linear: float,
    input_file: str | Path | None = None,
    target_peak_dbfs: float = DEFAULT_TARGET_PEAK_DBFS,
    measured_peak_dbfs: float | None = None,
    measured_rms_dbfs: float | None = None,
    latency_offset_samples: int = 0,
    sample_rate: int = FS,
    stereo: bool = False,
) -> Path:
    """Writes a JSON sidecar alongside `wav_filepath` recording `send_gain_db` and capture telemetry."""
    wav_path = Path(wav_filepath)
    sidecar_path = get_reamp_sidecar_path(wav_path)
    payload: dict[str, Any] = {
        "stem_file": wav_path.name,
        "input_file": str(input_file) if input_file is not None else None,
        "send_gain_db": round(float(send_gain_db), 6),
        "send_gain_linear": float(send_gain_linear),
        "target_peak_dbfs": float(target_peak_dbfs),
        "measured_return_peak_dbfs": (
            round(float(measured_peak_dbfs), 4) if measured_peak_dbfs is not None else None
        ),
        "measured_return_rms_dbfs": (
            round(float(measured_rms_dbfs), 4) if measured_rms_dbfs is not None else None
        ),
        "latency_offset_samples": int(latency_offset_samples),
        "latency_offset_ms": round(
            (float(latency_offset_samples) / max(1, sample_rate)) * 1000.0, 4
        ),
        "sample_rate": int(sample_rate),
        "stereo": bool(stereo),
    }
    sidecar_path.parent.mkdir(parents=True, exist_ok=True)
    sidecar_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return sidecar_path


def read_reamp_sidecar(wav_filepath: str | Path) -> dict[str, Any] | None:
    """Reads the JSON sidecar for a re-amped WAV stem if present."""
    sidecar_path = get_reamp_sidecar_path(wav_filepath)
    if not sidecar_path.is_file():
        return None
    try:
        data = json.loads(sidecar_path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else None
    except OSError, ValueError, TypeError:
        return None


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
        return False
    if peak_dbfs > -1.0:
        print(
            "  ⚠️ WARNING: Return signal is near clipping (> -1.0 dBFS). Back off block output level."
        )
    else:
        print("  ✅ Signal path verified cleanly!")
    return True


def capture_idle_noise(
    output_file: Annotated[
        Path,
        Parameter(
            name=["--output", "-o"],
            help="Output path to save captured 24-bit idle hardware noise WAV file.",
        ),
    ] = REPO_ROOT / "models" / "denoiser" / "captured_noise.wav",
    *,
    duration: Annotated[
        float,
        Parameter(
            name=["--duration", "-d"],
            help="Recording duration in seconds (default 10.0s).",
        ),
    ] = 10.0,
    input_channel: Annotated[
        int,
        Parameter(
            name=["--input-channel"],
            help="USB audio input channel (1-indexed).",
        ),
    ] = 1,
    device: Annotated[
        str,
        Parameter(
            name=["--device"],
            help="Audio device name search pattern or index.",
        ),
    ] = "Anagram",
    sample_rate: int = FS,
) -> Path:
    """Records idle hardware hum and SMPS power-supply noise from a USB audio interface into a 24-bit WAV file."""
    if duration <= 0.0:
        raise ValueError(f"duration must be positive (got {duration}).")

    dev_idx, dev_info = find_anagram_device(
        device,
        min_in_channels=max(1, input_channel),
        min_out_channels=1,
    )
    max_in = int(dev_info.get("max_input_channels", 1))
    max_out = max(1, int(dev_info.get("max_output_channels", 1)))
    if input_channel < 1 or input_channel > max_in:
        raise ValueError(
            f"Requested input channel {input_channel} is outside device range (1..{max_in})."
        )

    total_samples = int(duration * sample_rate)
    silence_send = np.zeros(total_samples, dtype=np.float32)
    rec_channels = input_channel

    print("==================================================================", flush=True)
    print(" Hardware Idle Hum & SMPS Noise Capture", flush=True)
    print("==================================================================", flush=True)
    print(f"  Device:         {dev_info['name']} (Index {dev_idx})", flush=True)
    print(f"  Input Channel:  Input {input_channel} @ {sample_rate} Hz", flush=True)
    print(f"  Duration:       {duration:.1f}s ({total_samples:,} samples)", flush=True)
    print("  Keep strings unplucked with volume/tone knobs wide open...", flush=True)

    rec = _stream_audio_pass(
        silence_send,
        device_idx=dev_idx,
        output_ch=1,
        rec_channels=rec_channels,
        max_out_channels=max_out,
        sample_rate=sample_rate,
        tail_samples=0,
        monitor=False,
        show_progress=True,
    )
    print(flush=True)

    noise_mono = rec[:total_samples, input_channel - 1].astype(np.float32)
    peak_dbfs, rms_dbfs = compute_audio_levels(noise_mono)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    write_wav_24bit(output_file, noise_mono, sample_rate=sample_rate)

    sidecar_path = output_file.with_suffix(".json")
    sidecar_payload = {
        "noise_file": output_file.name,
        "device_name": str(dev_info.get("name", device)),
        "input_channel": int(input_channel),
        "sample_rate": int(sample_rate),
        "duration_seconds": round(float(duration), 3),
        "measured_peak_dbfs": round(float(peak_dbfs), 4),
        "measured_rms_dbfs": round(float(rms_dbfs), 4),
        "captured_at": datetime.now(tz=UTC).isoformat(),
    }
    sidecar_path.write_text(json.dumps(sidecar_payload, indent=2) + "\n", encoding="utf-8")

    print(f"  Captured Levels: Peak: {peak_dbfs:+.2f} dBFS | RMS: {rms_dbfs:+.2f} dBFS", flush=True)
    print(f"  Saved WAV:       {output_file}", flush=True)
    print(f"  Saved Sidecar:   {sidecar_path}", flush=True)
    print("==================================================================\n", flush=True)
    return output_file


def reamp(
    *,
    input_file: Annotated[
        Path,
        Parameter(
            name=["--input", "-i"],
            help="Path to dry excitation input WAV file.",
        ),
    ] = REPO_ROOT / "reamp" / "input.wav",
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

    if test:
        run_test_tone(
            dev_idx,
            output_ch=output_channel,
            input_ch=input_channel,
            max_in_channels=max_in,
            max_out_channels=max_out,
        )
        return

    if not input_file.exists() and input_file == REPO_ROOT / "reamp" / "input.wav":
        fallback_input = REPO_ROOT / "audio" / "input.wav"
        if fallback_input.exists():
            input_file = fallback_input
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

    if no_auto_gain and isinstance(output_gain, str) and output_gain.strip().lower() == "auto":
        send_gain_db = 0.0
        send_gain_linear = 1.0
        print("  Output Gain:    +0.00 dB (1.000x, --no-auto-gain)")
    elif isinstance(output_gain, int | float) or (
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

    if stereo:
        wet_raw = recorded[:total_samples, :2]
        ref_channel_audio = wet_raw[:, 0]
    else:
        wet_raw = recorded[:total_samples, input_channel - 1]
        ref_channel_audio = wet_raw

    raw_peak, raw_rms = compute_audio_levels(wet_raw)
    print(f"  Raw Peak:       {raw_peak:+.1f} dBFS | RMS: {raw_rms:+.1f} dBFS")

    delay_samples = 0
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

    output_file.parent.mkdir(parents=True, exist_ok=True)
    if save_raw:
        raw_path = output_file.parent / f"{output_file.stem}_raw.wav"
        write_reamp_wav_24bit(raw_path, wet_raw, sample_rate=sr)
        print(f"  Raw Wet Audio:  {raw_path.name}")

    write_reamp_wav_24bit(output_file, aligned_wet, sample_rate=sr)
    final_peak, final_rms = compute_audio_levels(aligned_wet)
    sidecar_path = write_reamp_sidecar(
        output_file,
        send_gain_db=send_gain_db,
        send_gain_linear=send_gain_linear,
        input_file=input_file,
        target_peak_dbfs=target_peak,
        measured_peak_dbfs=final_peak,
        measured_rms_dbfs=final_rms,
        latency_offset_samples=delay_samples,
        sample_rate=sr,
        stereo=stereo,
    )

    print("------------------------------------------------------------------")
    print(f"  Saved Output:   {output_file}")
    print(f"  Gain Sidecar:   {sidecar_path.name} (send_gain_db={send_gain_db:+.2f} dB)")
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
