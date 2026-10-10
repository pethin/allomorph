#!/usr/bin/env python3
"""
Allomorph - Final Audio Stem Diagnostic CLI
Uses Fast Logarithmic Sine Sweep deconvolution to analyze rendered wet audio stems
for causal zero-latency, sub-bass transmission, EQ anchors, and non-linear THD.
"""

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from allomorph.circuit.stem_debug import debug_voicing_stem, format_stem_report_table
from allomorph.config.voices import VOICES


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Allomorph Final Audio Stem Diagnostic CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "voices",
        nargs="*",
        default=None,
        help="Optional positional voice ID(s) to debug (e.g. precision_vintage jazz_bridge_open)",
    )
    parser.add_argument(
        "--voice",
        "-v",
        type=str,
        default=None,
        help="Voice ID to debug (e.g. precision_vintage, jazz_bridge_open, stingray_parallel)",
    )
    parser.add_argument(
        "--instrument",
        "-i",
        type=str,
        default=None,
        help="Optional source instrument ID (e.g. 30in, 34in_standard_p, 34in_standard_jazz)",
    )
    parser.add_argument(
        "--drive-dbfs",
        type=float,
        default=-20.5,
        help="Excitation drive level in dBFS (-30.0 for small-signal, -20.5 nominal, -6.0 hot)",
    )
    parser.add_argument(
        "--no-gate",
        action="store_true",
        help="Disable 4,096-tap causal impulse gating to inspect full tail decay",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run diagnostics across all standard catalog target voices",
    )
    parser.add_argument(
        "--all-instruments",
        action="store_true",
        help="Run diagnostics across all 56 native voicings defined across all 16 catalog instruments",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        default=None,
        help="Optional directory to export deconvolved IR and test stem WAV files",
    )

    args = parser.parse_args()
    gate_taps = None if args.no_gate else 4096

    if args.all_instruments:
        from allomorph.config.instruments import INSTRUMENTS

        pass_count = 0
        total_count = 0
        for inst_id, inst in sorted(INSTRUMENTS.items()):
            for v_id in sorted(inst.voicings.keys()):
                total_count += 1
                report = debug_voicing_stem(
                    voice_id=v_id,
                    instrument=inst_id,
                    drive_dbfs=args.drive_dbfs,
                    gate_taps=gate_taps,
                    export_dir=args.output_dir,
                )
                print(format_stem_report_table(report))
                print()
                if report.is_causal_zero_latency and report.peak_dbfs <= -0.09:
                    pass_count += 1
        print(
            f"Summary: {pass_count}/{total_count} instrument voicings passed all telemetry invariants."
        )
        return

    target_voices: list[str] = []
    if args.all:
        target_voices = sorted(VOICES.keys())
    elif args.voices:
        target_voices = list(args.voices)
    elif args.voice:
        target_voices = [args.voice]
    else:
        target_voices = ["precision_vintage"]

    pass_count = 0
    for vid in target_voices:
        report = debug_voicing_stem(
            voice_id=vid,
            instrument=args.instrument,
            drive_dbfs=args.drive_dbfs,
            gate_taps=gate_taps,
            export_dir=args.output_dir,
        )
        print(format_stem_report_table(report))
        print()
        if report.is_causal_zero_latency:
            pass_count += 1

    if len(target_voices) > 1:
        print(
            f"Summary: {pass_count}/{len(target_voices)} voices passed causal zero-latency validation."
        )


if __name__ == "__main__":
    main()
