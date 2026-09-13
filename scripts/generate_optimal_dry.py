"""
Allomorph - Optimal Bass Synthetic Dry Signal Generator
Synthesizes a 48 kHz / 24-bit PCM mono dry excitation track engineered specifically
for bass pickup and analog digital twin neural modeling on Tone3000 / NAM.
"""

import argparse
import math
from pathlib import Path

import numpy as np

from allomorph.dsp import (
    FS,
    ensure_optimal_dry_wav,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate high-fidelity synthetic dry excitation signal for bass pickup modeling."
    )
    parser.add_argument(
        "--out",
        "-o",
        type=Path,
        default=None,
        help="Target output WAV filepath (default: versioned audio/canonical/optimal_bass_dry_<version>.wav)",
    )
    parser.add_argument(
        "--version-tag",
        type=str,
        default=None,
        help="Semantic version token (e.g. 'v2.1.1' or 'v2'). Defaults to resolved tri-part version.",
    )
    parser.add_argument(
        "--no-manifest",
        action="store_true",
        help="Disable automatic manifest.json sidecar emission in output directory.",
    )
    parser.add_argument(
        "--duration",
        "-d",
        type=float,
        default=180.0,
        help="Audio duration in seconds (default: 180.0)",
    )
    parser.add_argument(
        "--sample-rate",
        "-sr",
        type=int,
        default=FS,
        help=f"Audio sample rate in Hz (default: {FS})",
    )
    parser.add_argument(
        "--peak-dbfs",
        type=float,
        default=-1.0,
        help="True Peak ceiling in dBFS (default: -1.0 dBFS / 0.891)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output file if present",
    )
    args = parser.parse_args()

    from allomorph.naming import get_optimal_dry_path

    out_path = Path(args.out) if args.out is not None else get_optimal_dry_path(version_tag=args.version_tag)
    if out_path.exists() and not args.overwrite:
        print(f"[Optimal Dry] Output file already exists: {out_path}")
        print("Use --overwrite to regenerate.")
        return

    print("==================================================================")
    print("  ALLOMORPH OPTIMAL BASS SYNTHETIC DRY SIGNAL GENERATOR")
    print(f"  Destination: {out_path}")
    print(f"  Duration:    {args.duration:.1f} s ({int(args.duration * args.sample_rate):,} samples)")
    print(f"  Sample Rate: {args.sample_rate} Hz (24-bit PCM Mono)")
    print(f"  Peak Ceiling:{args.peak_dbfs:+.2f} dBFS")
    print("==================================================================")

    print("\nSynthesizing excitation stages (Zero Artificial Dither Policy):")
    print("  [1/7] Latency calibration alignment double-blips (0.3s & 0.8s)...")
    print("  [2/7] Multi-tier full-range log chirps (15 Hz -> 22 kHz at -24, -12, -6, -1 dBFS + reverse)...")
    print("  [3/7] 5-step velocity ladder (pp -> ff) & modal plucks with pitch sag (Drop A0 27.5 Hz -> C3)...")
    print("  [4/7] Bass articulations (pick down/upstrokes, slap & pop, staccato, ghost notes, harmonics)...")
    print("  [5/7] Polyphonic dyads (power 5ths & octaves for IMD) & Schroeder multitone complex...")
    print("  [6/7] Continuous register glissandi across pickup comb nulls...")
    print("  [7/7] Shaped wideband pink noise bursts & clean silence boundary termination...")

    out_path = ensure_optimal_dry_wav(
        output_path=args.out,
        duration_sec=args.duration,
        sample_rate=args.sample_rate,
        peak_dbfs=args.peak_dbfs,
        overwrite=args.overwrite,
        version_tag=args.version_tag,
        no_manifest=args.no_manifest,
    )

    from allomorph.dsp import read_wav

    audio, _ = read_wav(out_path)
    peak_db = 20.0 * math.log10(max(float(np.max(np.abs(audio))), 1e-9))
    rms_db = 20.0 * math.log10(max(float(np.sqrt(np.mean(audio**2))), 1e-9))
    file_size_mb = out_path.stat().st_size / (1024 * 1024)

    print("\nSynthesis complete:")
    print(f"  File:      {out_path}")
    print(f"  Size:      {file_size_mb:.2f} MB")
    print(f"  True Peak: {peak_db:+.2f} dBFS")
    print(f"  RMS Level: {rms_db:+.2f} dBFS")
    print(f"  Crest:     {peak_db - rms_db:.2f} dB")
    print("\nReady for Tone3000 'Dry/Wet Pair' upload and Allomorph circuit simulation.")


if __name__ == "__main__":
    main()
