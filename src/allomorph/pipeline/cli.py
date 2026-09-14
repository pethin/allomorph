"""
Allomorph Pipeline - CLI Parser and Dispatcher
Command-line entrypoint coordinating full end-to-end simulation, export, and training workflows.
"""

import argparse
from collections.abc import Sequence
from pathlib import Path

from allomorph.config.geometry import (
    compute_effective_position,
    resolve_voice_coils,
    resolve_voice_pickups,
)
from allomorph.config.instruments import (
    INSTRUMENTS,
)
from allomorph.config.scales import REPO_ROOT
from allomorph.config.voices import VOICES
from allomorph.naming import (
    resolve_instruments,
    resolve_voices,
)
from allomorph.pipeline.schema import PipelineCliConfig
from allomorph.pipeline.stages import (
    run_training,
    run_visualization,
)


def list_instruments():
    """Lists all configured source instruments and their pickups."""
    print("Available Allomorph Source Instruments:")
    for iid, cfg in INSTRUMENTS.items():
        print(f'  - {iid}: {cfg.name} ({cfg.scale_length_in}")')
        pickups = cfg.pickups
        for pid, pcfg in pickups.items():
            pos_m = pcfg.position_from_bridge_m or 0.0
            print(
                f'      * [{pid}] {pcfg.name}: pos={pos_m * 1000:.1f}mm, w={pcfg.aperture_width_in:.2f}", d={pcfg.coil_spacing_in:.2f}"'
            )


def list_voices():
    """Lists all target pickup voices and their SPICE netlists."""
    print("Available Allomorph Target Pickup Voices (SPICE Digital Twins):")
    for vid, cfg in VOICES.items():
        print(f"  - {vid}: {cfg.name} ({cfg.topology})")
        coils = resolve_voice_coils(cfg)
        pickups = resolve_voice_pickups(cfg)
        eff_pos = compute_effective_position(coils)
        if len(pickups) > 1:
            print(
                f"      Circuit: {cfg.circuit} | Pickups={len(pickups)}, Coils={len(coils)} (Eff pos={eff_pos * 1000:.1f}mm) | Composite fr={cfg.fr}Hz (Q={cfg.Q})"
            )
            for p in pickups:
                print(
                    f"        * [{p.name}]: fr={p.fr:.0f}Hz (Q={p.Q:.1f}), weight={p.weight:.2f}, coils={len(p.coils)}"
                )
        else:
            print(
                f"      Circuit: {cfg.circuit} | Coils={len(coils)} (Eff pos={eff_pos * 1000:.1f}mm) | fr={cfg.fr}Hz (Q={cfg.Q})"
            )


def main(argv: Sequence[str] | None = None):
    """Main CLI entrypoint for Allomorph pipeline automation."""
    parser = argparse.ArgumentParser(description="Allomorph SPICE -> NAM Automation Pipeline")
    parser.add_argument(
        "--instrument",
        "-i",
        default="all",
        help="Source instrument configuration (ID, comma-separated list, 'all', path to .toml, or alias like 30in, 32in; default: 'all')",
    )
    parser.add_argument(
        "--stage",
        choices=[
            "all",
            "viz",
            "sim",
            "pack",
            "train",
        ],
        default="all",
        help="Pipeline stage to execute: 'viz' (interactive frequency charts & portal), 'sim' (direct forward simulation of instrument voicings), 'pack' (Tone3000 upload pack bundles), 'train' (train NAM neural models), or 'all' (sim + pack + viz; default: 'all').",
    )
    parser.add_argument(
        "--normalize",
        choices=["auto", "rms", "peak", "none"],
        default="auto",
        help="Output level normalization mode for target voice wet simulation (default: auto)",
    )
    parser.add_argument(
        "--target-dbfs",
        type=float,
        default=None,
        help="Explicit target level in dBFS for target voice wet simulation (default: auto-derived from input calibration sweep RMS, ~ -22.1 dBFS)",
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="Train NAM model locally with Apple Silicon Metal/MPS acceleration",
    )
    parser.add_argument(
        "--pickup",
        "-p",
        default=None,
        help="Physical pickup setting for source instrument ('auto' to resolve from pickup_mapping, or explicit pickup ID)",
    )
    parser.add_argument(
        "--voice",
        "-v",
        default="all",
        help="Target pickup voice for audio pre-filtering, simulation, and training (voice ID, comma-separated list, or 'all'; default: 'all')",
    )
    parser.add_argument(
        "--vol-pos",
        "--vol",
        type=float,
        default=None,
        dest="vol_pos",
        help="Volume pot wiper position (0.0 to 1.0, default 1.0 full open)",
    )
    parser.add_argument(
        "--tone-pos",
        "--tone",
        type=float,
        default=None,
        dest="tone_pos",
        help="Tone pot wiper position (0.0 to 1.0, default 1.0 full open/bright)",
    )
    parser.add_argument(
        "--cable-pf",
        type=float,
        default=None,
        help="Cable capacitance loading in pF (default: from circuit config, typically 750 pF)",
    )
    parser.add_argument(
        "--jobs",
        "-j",
        type=int,
        default=None,
        help="Number of parallel worker processes for batch simulation (default: min(4, CPU count))",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum audio sample frames to simulate (default: None for full file)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Force recompilation of cached dry and wet audio stems even if they already exist",
    )
    parser.add_argument(
        "--input-wav",
        default=None,
        help="Path to dry calibration audio file (default: auto-generates audio/canonical/optimal_bass_dry.wav)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=500,
        help="Maximum number of training epochs for NAM model (default: 500 for A2-Lite studio reference)",
    )
    parser.add_argument(
        "--goal-esr",
        type=float,
        default=0.0005,
        help="Goal validation ESR for early stopping (default: 0.0005 for A2-Lite studio reference; set to 0 to disable)",
    )
    parser.add_argument(
        "--no-goal-esr",
        action="store_true",
        help="Disable goal ESR early stopping and train for the exact number of epochs specified",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for model training (default: 32)",
    )
    parser.add_argument(
        "--a2-lite-only",
        action="store_true",
        help="Train A2-Lite channels_8 only instead of the full slimmable container",
    )
    parser.add_argument(
        "--fast-dev-run",
        action="store_true",
        help="Run 1-batch dry run for smoke testing NAM training",
    )
    parser.add_argument(
        "--version-tag",
        default="auto",
        help="Semantic version tag (default: 'auto' -> v[dsp].[inst].[voice], or explicit string, or 'none' to disable)",
    )
    parser.add_argument(
        "--no-manifest",
        action="store_true",
        help="Disable generating sidecar manifest.json",
    )
    parser.add_argument(
        "--list-instruments",
        action="store_true",
        help="List all configured source instruments and their pickups",
    )
    parser.add_argument(
        "--list-voices",
        action="store_true",
        help="List all target pickup voices and their SPICE netlists",
    )
    args = parser.parse_args(argv)

    PipelineCliConfig.model_validate(
        {
            "instrument": args.instrument or "all",
            "stage": args.stage,
            "pickup": args.pickup,
            "voice": args.voice or "all",
            "train": args.train,
            "vol_pos": args.vol_pos,
            "tone_pos": args.tone_pos,
            "cable_pf": args.cable_pf if args.cable_pf is not None else 750.0,
            "normalize": args.normalize,
            "target_dbfs": args.target_dbfs,
            "input_wav": args.input_wav,
            "version_tag": args.version_tag,
            "no_manifest": args.no_manifest,
        }
    )

    if args.list_instruments:
        list_instruments()
        return

    if args.list_voices:
        list_voices()
        return

    if args.jobs is not None and args.jobs < 1:
        parser.error("--jobs must be a positive integer >= 1")

    if args.max_samples is not None and args.max_samples < 1:
        parser.error("--max-samples must be a positive integer >= 1")

    effective_goal_esr = (
        None
        if args.no_goal_esr or (args.goal_esr is not None and args.goal_esr <= 0)
        else args.goal_esr
    )
    instruments_to_run = resolve_instruments(args.instrument or "all")
    voices_to_run = resolve_voices(args.voice or "all")

    samples_str = str(args.max_samples) if args.max_samples is not None else "full"

    print("========================================")
    print("  ALLOMORPH SPICE -> NAM PIPELINE")
    print(f"  Instruments ({len(instruments_to_run)}): {', '.join(instruments_to_run)}")
    print(f"  Stage:       {args.stage}")
    print(f"  Max Samples: {samples_str}")
    print(f"  Voices ({len(voices_to_run)}): {', '.join(voices_to_run)}")
    print("========================================")

    input_wav = args.input_wav
    if not input_wav or not (Path(input_wav).exists() or (REPO_ROOT / input_wav).exists()):
        from allomorph.circuit.audio import find_default_input_audio

        dry_p = find_default_input_audio(version_tag=args.version_tag)
        input_wav = str(dry_p) if dry_p else None
        dry_name = dry_p.name if dry_p else "None"
        print(f"  Dry Source:  Optimal Bass Synthetic ({dry_name})")
    else:
        actual_path = Path(input_wav) if Path(input_wav).exists() else (REPO_ROOT / input_wav)
        input_wav = str(actual_path)
        print(f"  Dry Source:  {actual_path.name}")

    if args.stage == "sim":
        from allomorph.circuit.forward import simulate_all_instrument_voicings

        for inst in instruments_to_run:
            print(f"\n[Simulation] Simulating all voicings for {inst}...")
            simulate_all_instrument_voicings(
                inst,
                max_samples=args.max_samples,
                jobs=args.jobs,
            )
        return

    if args.stage == "pack":
        from allomorph.pipeline.pack import export_tone_pack

        for inst in instruments_to_run:
            print(f"\n[Tone Pack] Exporting tone pack bundles for {inst}...")
            export_tone_pack(
                inst,
                max_samples=args.max_samples,
                jobs=args.jobs,
                overwrite=args.overwrite,
            )
        return

    if args.stage == "viz":
        if len(instruments_to_run) == 1 and args.instrument != "all":
            run_visualization(instrument=instruments_to_run[0])
        else:
            run_visualization(instrument="all")
        return

    if args.stage == "train":
        for inst in instruments_to_run:
            for idx, voice in enumerate(voices_to_run, 1):
                print(f"\n[{idx}/{len(voices_to_run)}] Training NAM A2 Model: {inst} -> {voice}...")
                run_training(
                    instrument=inst,
                    voice=voice,
                    input_wav=input_wav,
                    epochs=args.epochs,
                    goal_esr=effective_goal_esr,
                    fast_dev_run=args.fast_dev_run,
                    batch_size=args.batch_size,
                    a2_lite_only=args.a2_lite_only,
                    version_tag=args.version_tag,
                    no_manifest=args.no_manifest,
                )
        return

    if args.stage == "all":
        print("\n--- Step 1: Direct Forward Simulation of Voicings ---")
        from allomorph.circuit.forward import simulate_all_instrument_voicings

        for inst in instruments_to_run:
            simulate_all_instrument_voicings(
                inst,
                max_samples=args.max_samples,
                jobs=args.jobs,
            )

        print("\n--- Step 2: Tone Pack Bundles Export ---")
        from allomorph.pipeline.pack import export_tone_pack

        for inst in instruments_to_run:
            export_tone_pack(
                inst,
                max_samples=args.max_samples,
                jobs=args.jobs,
            )

        print("\n--- Step 3: Interactive Altair Frequency Visualization ---")
        if len(instruments_to_run) == 1 and args.instrument != "all":
            run_visualization(instrument=instruments_to_run[0])
        else:
            run_visualization(instrument="all")
        print("\n[Pipeline Complete: Direct Voicings + Tone Packs + Interactive Portal Ready]")
        return
