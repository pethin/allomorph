#!/usr/bin/env python3
"""Allomorph - Darkglass Anagram Hardware Re-Amping CLI.

Plays dry excitation audio through the Darkglass Anagram USB audio interface
specifically on Output 9 (USB Reamp Send) while simultaneously recording the
wet return signal from Input 1 (or stereo Inputs 1 & 2) at 24-bit 48 kHz.

Performs automatic sub-millisecond sample alignment via cross-correlation to
compensate for USB audio interface roundtrip buffer latency.
Uses blocking chunked streaming to avoid libffi/CFFI callback trampoline assertions on macOS.
"""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path
from typing import Annotated, Any

import numpy as np
import sounddevice as sd
from cyclopts import App, Parameter

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from allomorph.dsp.constants import FS
from allomorph.dsp.io import read_wav, write_wav_24bit

app = App(
    name="reamp-anagram",
    help="Allomorph - Darkglass Anagram Hardware Re-Amping CLI.",
    version="1.0.0",
)


def find_anagram_device(device_name_pattern: str = "Anagram") -> tuple[int, dict[str, Any]]:
    """Locates the Anagram USB audio device index and property dictionary."""
    devices = sd.query_devices()
    for idx, dev in enumerate(devices):
        name = str(dev.get("name", ""))
        if device_name_pattern.lower() in name.lower():
            return idx, dev
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
    peak = float(np.max(np.abs(audio)))
    rms = float(np.sqrt(np.mean(np.square(audio))))
    peak_dbfs = 20.0 * math.log10(max(peak, 1e-9))
    rms_dbfs = 20.0 * math.log10(max(rms, 1e-9))
    return peak_dbfs, rms_dbfs


def align_hardware_latency(
    dry: np.ndarray,
    wet: np.ndarray,
    sample_rate: int = FS,
    max_search_ms: float = 120.0,
) -> tuple[int, np.ndarray]:
    """Estimates roundtrip hardware latency via cross-correlation and returns (delay_samples, aligned_wet)."""
    max_delay_samples = int(sample_rate * (max_search_ms / 1000.0))
    search_len = min(len(dry), int(sample_rate * 3.0))  # Cross-correlate first 3.0s

    x = dry[:search_len].astype(np.float32)
    # Search window in wet signal up to search_len + max_delay_samples
    y_window = wet[: search_len + max_delay_samples].astype(np.float32)

    if len(y_window) <= len(x):
        return 0, wet[: len(dry)]

    # Compute valid cross-correlation: length = max_delay_samples + 1
    corr = np.correlate(y_window, x, mode="valid")
    delay_samples = int(np.argmax(np.abs(corr)))

    # Slice aligned wet audio matching exact dry length
    aligned_wet = wet[delay_samples : delay_samples + len(dry)]
    if len(aligned_wet) < len(dry):
        # Pad with silence if slightly short
        padded = np.zeros(len(dry), dtype=wet.dtype)
        padded[: len(aligned_wet)] = aligned_wet
        aligned_wet = padded

    return delay_samples, aligned_wet


def run_test_tone(
    device_idx: int,
    output_ch: int = 9,
    input_ch: int = 1,
    sample_rate: int = FS,
) -> bool:
    """Emits a 1-second 440 Hz test tone on Output 9 to verify signal path and levels."""
    print("\n[TEST] Emitting 1.0s 440 Hz test tone (-18.0 dBFS) through Output 9...")
    t = np.arange(sample_rate) / sample_rate
    tone = (0.125 * np.sin(2.0 * np.pi * 440.0 * t)).astype(np.float32)

    total_samples = sample_rate + 2400
    out_buf = np.zeros((total_samples, 9), dtype=np.float32)
    out_buf[:sample_rate, output_ch - 1] = tone

    blocksize = 2048
    rec = np.zeros((total_samples, 3), dtype=np.float32)

    # Use blocking Stream (no CFFI callbacks) for macOS stability
    with sd.Stream(
        device=device_idx,
        samplerate=sample_rate,
        blocksize=blocksize,
        channels=(3, 9),
        dtype="float32",
    ) as stream:
        for i in range(0, total_samples, blocksize):
            end = min(i + blocksize, total_samples)
            chunk = out_buf[i:end]
            if len(chunk) < blocksize:
                chunk = np.pad(chunk, ((0, blocksize - len(chunk)), (0, 0)))
            stream.write(chunk)
            data, _ = stream.read(blocksize)
            rec[i:end] = data[: end - i]

    ret_signal = rec[:, input_ch - 1]
    peak_dbfs, rms_dbfs = compute_audio_levels(ret_signal)

    print(f"  Return Signal Peak: {peak_dbfs:+.1f} dBFS | RMS: {rms_dbfs:+.1f} dBFS")
    if peak_dbfs < -40.0:
        print("  ❌ WARNING: Return signal is extremely quiet (<-40 dBFS).")
        print(
            "     Check that your preset in Anagram has USB Reamp (Output 9) assigned to block input"
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
            help="Extra recording tail in seconds to capture room/cab ring-out decay.",
        ),
    ] = 1.0,
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
            help="Audio device name search pattern.",
        ),
    ] = "Anagram",
) -> None:
    """Plays dry excitation audio through Anagram USB Output 9 and records the aligned wet return."""
    # 1. Locate Device
    dev_idx, dev_info = find_anagram_device(device)
    max_out = int(dev_info["max_output_channels"])
    max_in = int(dev_info["max_input_channels"])

    print("==================================================================")
    print(" Allomorph Darkglass Anagram Hardware Re-Amper")
    print("==================================================================")
    print(f"  Device:         {dev_info['name']} (Index {dev_idx})")
    print(f"  Capabilities:   {max_in} Inputs, {max_out} Outputs @ 48 kHz")
    print(f"  Send Channel:   Output {output_channel} (USB Reamp Send)")
    in_desc = "Inputs 1 & 2 (Stereo)" if stereo else f"Input {input_channel} (Mono)"
    print(f"  Return Channel: {in_desc}")
    print("------------------------------------------------------------------")

    if output_channel > max_out:
        raise ValueError(
            f"Requested output channel {output_channel} exceeds device max outputs ({max_out})."
        )

    # 2. Optional Quick Test Tone Check
    if test:
        run_test_tone(
            dev_idx,
            output_ch=output_channel,
            input_ch=input_channel,
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
    tail_samples = int(tail * sr)
    total_samples = n_samples + tail_samples
    total_duration_s = total_samples / sr

    print(
        f"  Dry Audio:      {n_samples:,} samples ({duration_s:.1f}s / {duration_s / 60.0:.2f} min)"
    )
    print(f"  Tail Padding:   {tail:.1f}s ({tail_samples:,} samples)")
    print(f"  Total Session:  {total_duration_s:.1f}s ({total_duration_s / 60.0:.2f} min)")

    # 4. Construct Multi-Channel Output Buffer
    out_buffer = np.zeros((total_samples, max_out), dtype=np.float32)
    out_buffer[:n_samples, output_channel - 1] = dry_audio

    if monitor:
        print("  Monitoring:     Outputs 1 & 2 enabled (headphone mirror)")
        out_buffer[:n_samples, 0] = dry_audio * 0.5
        out_buffer[:n_samples, 1] = dry_audio * 0.5

    # 5. Playback and Record via blocking Stream (bypasses CFFI callback trampolines)
    rec_channels = 2 if stereo else max_in
    recorded = np.zeros((total_samples, rec_channels), dtype=np.float32)
    blocksize = 2048

    print("\nStarting real-time re-amping pass. Press Ctrl+C to abort...")
    start_time = time.time()
    last_update = 0.0

    try:
        with sd.Stream(
            device=dev_idx,
            samplerate=sr,
            blocksize=blocksize,
            channels=(rec_channels, max_out),
            dtype="float32",
        ) as stream:
            for start_idx in range(0, total_samples, blocksize):
                end_idx = min(start_idx + blocksize, total_samples)
                chunk_out = out_buffer[start_idx:end_idx]
                if len(chunk_out) < blocksize:
                    chunk_out = np.pad(chunk_out, ((0, blocksize - len(chunk_out)), (0, 0)))

                stream.write(chunk_out)
                chunk_in, _ = stream.read(blocksize)
                recorded[start_idx:end_idx] = chunk_in[: end_idx - start_idx]

                # Update progress bar every ~0.25s
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
            dry=dry_audio,
            wet=ref_channel_audio,
            sample_rate=sr,
        )
        delay_ms = (delay_samples / sr) * 1000.0
        print(f"  Latency Offset: {delay_samples} samples ({delay_ms:.2f} ms) — aligned")

        if stereo:
            aligned_ch0 = wet_raw[delay_samples : delay_samples + n_samples, 0]
            aligned_ch1 = wet_raw[delay_samples : delay_samples + n_samples, 1]
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
        write_wav_24bit(raw_path, wet_raw if wet_raw.ndim == 1 else wet_raw[:, 0], sample_rate=sr)
        print(f"  Raw Wet Audio:  {raw_path.name}")

    export_audio = aligned_wet if aligned_wet.ndim == 1 else aligned_wet[:, 0]
    write_wav_24bit(output_file, export_audio, sample_rate=sr)
    final_peak, final_rms = compute_audio_levels(export_audio)

    print("------------------------------------------------------------------")
    print(f"  Saved Output:   {output_file}")
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
