"""
Allomorph Pipeline - CLI Parser and Dispatcher
Command-line entrypoint coordinating full end-to-end simulation, export, and training workflows.
"""

import argparse
import os
from collections.abc import Sequence
from pathlib import Path

from allomorph.circuit import (
    AUDIO_DIR,
    MODELS_DIR,
    _simulate_voice_task,
    export_all_frontend_irs,
    export_all_frontend_wet_wavs,
    export_frontend_ir,
    export_frontend_wet_wav,
    generate_canonical_sweep,
    simulate_backend_targets,
    simulate_voice,
)
from allomorph.circuit.schema import SimulationConfig
from allomorph.config.geometry import (
    compute_effective_position,
    resolve_voice_coils,
    resolve_voice_pickups,
)
from allomorph.config.instruments import (
    INSTRUMENTS,
    get_source_pickup,
    load_instrument,
)
from allomorph.config.scales import REPO_ROOT
from allomorph.config.voices import VOICES
from allomorph.naming import (
    get_baked_basename,
    get_t3k_basename,
    resolve_instruments,
    resolve_voices,
)
from allomorph.pipeline.schema import PipelineCliConfig
from allomorph.pipeline.stages import (
    run_frontend_training,
    run_training,
    run_visualization,
)
from allomorph.version import (
    DSP_GENERATION,
    resolve_tri_part_version,
    write_manifest,
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
            "canonical",
            "frontends",
            "targets",
            "train",
            "bake",
        ],
        default="all",
        help="Pipeline stage to execute: 'viz' (interactive frequency charts & portal), 'canonical' (calibrated intermediate baseline sweep), 'frontends' (export frontend wet sweeps / IRs / train frontend NAM models), 'targets' (simulate 3-tier backend universal target sweeps), 'train' (train backend NAM A2 neural models), 'bake' (on-demand single-block monolithic model), or 'all' (canonical + frontends + targets + viz; default: 'all').",
    )
    parser.add_argument(
        "--frontend-format",
        choices=["wet", "ir", "both", "nam"],
        default="wet",
        help="Format for frontend deconvolution exports: 'wet' (wet sweep WAV convolved with FIR), 'ir' (FIR impulse response WAV), 'both' (both wet sweep and IR WAVs), 'nam' (train NAM neural model on wet WAV; default: wet)",
    )
    parser.add_argument(
        "--normalize-frontend",
        action="store_true",
        default=False,
        help="Enable full-scale peak normalization (0.9900) for frontend deconvolution (default: False for unnormalized unity gain)",
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
        "--gain-db",
        type=float,
        default=0.0,
        help="Optional manual gain trim in dB applied to frontend deconvolution filter (default: 0.0 dB)",
    )
    parser.add_argument(
        "--tier",
        choices=["clean", "standard", "std", "hotrod", "dynamic", "all"],
        default=None,
        help="Dynamic tier: 'standard' / 'std' (100%% nominal target saturation; default for Architecture C targets), 'clean' (0%% saturation), 'hotrod' (175%% overwound), 'dynamic' (differential source/target saturation; default when using --stage bake), 'all'.",
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="Train NAM model locally with Apple Silicon Metal/MPS acceleration after simulation (used with --stage bake)",
    )
    parser.add_argument(
        "--pickup",
        "-p",
        default=None,
        help="Physical pickup setting for source instrument ('auto' to resolve from pickup_mapping, or explicit pickup ID; default when using --stage bake is 'auto')",
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
        "--t3k-pack",
        action="store_true",
        help="Export baked model and audio files formatted as 'Tone Name [Pickup Position]' (max 34 chars)",
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
            "tier": args.tier,
            "pickup": args.pickup,
            "voice": args.voice or "all",
            "train": args.train,
            "vol_pos": args.vol_pos,
            "tone_pos": args.tone_pos,
            "cable_pf": args.cable_pf if args.cable_pf is not None else 750.0,
            "frontend_format": args.frontend_format,
            "normalize_frontend": args.normalize_frontend,
            "normalize": args.normalize,
            "target_dbfs": args.target_dbfs,
            "gain_db": args.gain_db,
            "input_wav": args.input_wav,
            "t3k_pack": args.t3k_pack,
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

    if args.stage == "bake":
        effective_tier = args.tier if args.tier is not None else "dynamic"
        pickup_setting = args.pickup if args.pickup is not None else "auto"

        print("\n========================================")
        print("  ALLOMORPH ON-DEMAND SINGLE-BLOCK BAKE")
        print(f"  Source Instruments ({len(instruments_to_run)}): {', '.join(instruments_to_run)}")
        print(f"  Pickup Switch:     {pickup_setting}")
        print(f"  Dynamic Tier:      {effective_tier}")
        print(f"  Voices ({len(voices_to_run)}): {', '.join(voices_to_run)}")
        print("========================================\n")

        total_bakes = len(instruments_to_run) * len(voices_to_run)
        tasks: list[tuple[str, SimulationConfig, str, str, Path, Path]] = []
        for inst in instruments_to_run:
            inst_cfg = load_instrument(inst)
            inst_id = inst_cfg.id

            for voice in voices_to_run:
                if pickup_setting == "auto":
                    src_pickup = get_source_pickup(inst_cfg, voice)
                    eff_pickup = src_pickup.id or "default"
                else:
                    eff_pickup = pickup_setting
                    if eff_pickup in inst_cfg.pickups:
                        src_pickup = inst_cfg.pickups[eff_pickup]
                    else:
                        raise KeyError(
                            f"Pickup '{eff_pickup}' not found on instrument '{inst_cfg.id}'."
                        )

                pickup_baked_audio_dir = AUDIO_DIR / "baked" / inst_id / eff_pickup
                pickup_baked_audio_dir.mkdir(parents=True, exist_ok=True)
                pickup_models_dir = MODELS_DIR / "baked" / inst_id / eff_pickup
                if args.train or args.stage == "train":
                    pickup_models_dir.mkdir(parents=True, exist_ok=True)

                # Dedicated distinguished dry file per instrument pickup (dry_<inst_id>_<pickup>.wav)
                if args.input_wav:
                    inst_input_wav = input_wav
                else:
                    from allomorph.circuit.staging import find_instrument_dry_wav

                    inst_input_wav = str(
                        find_instrument_dry_wav(
                            inst_id,
                            pickup_key=eff_pickup,
                            version_tag=args.version_tag,
                            audio_dir=pickup_baked_audio_dir,
                        )
                    )

                inst_ver = getattr(inst_cfg, "version", 1)
                vcfg = VOICES[voice]
                voice_ver = getattr(vcfg, "version", 1)
                tri_part = resolve_tri_part_version(DSP_GENERATION, inst_ver, voice_ver)
                ver_tag = (
                    tri_part
                    if (args.version_tag == "auto" or args.version_tag is True)
                    else (args.version_tag if args.version_tag not in (None, "none", False) else None)
                )

                if args.t3k_pack:
                    tone_name = vcfg.tone_name or vcfg.name
                    pos_name = (
                        None
                        if (len(inst_cfg.pickups) <= 1 or vcfg.preserve_aperture)
                        else (src_pickup.position_name or src_pickup.name)
                    )
                    basename = get_t3k_basename(
                        tone_name,
                        pos_name,
                        preserve_aperture=vcfg.preserve_aperture,
                        version_tag=ver_tag,
                    )
                else:
                    basename = get_baked_basename(
                        voice,
                        tier=effective_tier,
                        pickup=pickup_setting,
                        preserve_aperture=vcfg.preserve_aperture,
                        version_tag=ver_tag,
                    )
                baked_wav = pickup_baked_audio_dir / f"{basename}.wav"

                sim_cfg = SimulationConfig(
                    input_wav=inst_input_wav,
                    output_wav=baked_wav,
                    instrument=inst,
                    pickup=eff_pickup,
                    tier=effective_tier,
                    normalize=args.normalize,
                    target_dbfs=args.target_dbfs,
                    max_samples=args.max_samples,
                    vol_pos=args.vol_pos,
                    tone_pos=args.tone_pos,
                    cable_pf=args.cable_pf,
                    skip_identity=True,
                )
                tasks.append((voice, sim_cfg, inst, basename, pickup_models_dir, baked_wav))

        max_workers = args.jobs if args.jobs is not None else min(4, os.cpu_count() or 4)
        if len(tasks) > 1 and max_workers > 1:
            os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
            os.environ.setdefault("OMP_NUM_THREADS", "1")
            os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
            print(
                f"Simulating {len(tasks)} baked voices in parallel ({max_workers} workers)...\n"
            )
            from concurrent.futures import ProcessPoolExecutor

            sim_tasks = [(t[0], t[1]) for t in tasks]
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                results = list(executor.map(_simulate_voice_task, sim_tasks))
            for (voice, _cfg, inst, _base, _mdir, baked_wav), success in zip(tasks, results):
                if success and baked_wav.exists():
                    print(f"Baked simulation exported: {baked_wav}")
                else:
                    print(
                        f"Skipped bit-for-bit identity voice: {inst} -> {voice} ({baked_wav.name} not output)"
                    )
        else:
            for idx, (voice, sim_cfg, inst, _base, _mdir, baked_wav) in enumerate(tasks, 1):
                if total_bakes > 1:
                    print(f"\n--- [{idx}/{total_bakes}] Baking {inst} -> {voice} ---")
                success = simulate_voice(voice, config=sim_cfg)
                if success and baked_wav.exists():
                    print(f"Baked simulation exported: {baked_wav}")
                else:
                    print(
                        f"Skipped bit-for-bit identity voice: {inst} -> {voice} ({baked_wav.name} not output)"
                    )

        if not args.no_manifest:
            for inst in instruments_to_run:
                inst_cfg_m = load_instrument(inst)
                inst_tasks = [t for t in tasks if t[2] == inst and t[5].exists()]
                dirs_to_manifest: dict[Path, list[Path]] = {}
                for t in inst_tasks:
                    dirs_to_manifest.setdefault(t[5].parent, []).append(t[5])

                v_tag = resolve_tri_part_version(
                    DSP_GENERATION, getattr(inst_cfg_m, "version", 1), 1
                )
                for p_dir, files in dirs_to_manifest.items():
                    write_manifest(
                        output_dir=p_dir,
                        stage="bake",
                        files=files,
                        version_tag=v_tag,
                    )

        if args.train or args.stage == "train":
            for idx, (voice, _cfg, inst, basename, inst_models_dir, baked_wav) in enumerate(
                tasks, 1
            ):
                if not baked_wav.exists():
                    print(f"\nSkipping training for bit-for-bit identity voice: {inst} -> {voice}")
                    continue
                if total_bakes > 1:
                    print(f"\n--- [{idx}/{total_bakes}] Training {inst} -> {voice} ---")
                run_training(
                    instrument=inst,
                    voice=voice,
                    input_wav=str(_cfg.input_wav) if _cfg.input_wav else input_wav,
                    output_wav=baked_wav,
                    models_dir=inst_models_dir,
                    tier=effective_tier,
                    epochs=args.epochs,
                    goal_esr=effective_goal_esr,
                    fast_dev_run=args.fast_dev_run,
                    basename=basename,
                    batch_size=args.batch_size,
                    a2_lite_only=args.a2_lite_only,
                    version_tag=args.version_tag,
                    no_manifest=args.no_manifest,
                )
        return

    if args.stage == "canonical":
        generate_canonical_sweep(
            input_wav=input_wav,
            version_tag=args.version_tag,
            no_manifest=args.no_manifest,
        )
        return

    if args.stage == "frontends" or args.frontend_format == "nam":
        fmt = args.frontend_format
        if fmt == "nam":
            for inst in instruments_to_run:
                run_frontend_training(
                    instrument=inst,
                    pickup=args.pickup,
                    input_wav=input_wav,
                    epochs=args.epochs,
                    goal_esr=effective_goal_esr,
                    fast_dev_run=args.fast_dev_run,
                    normalize=args.normalize_frontend,
                    gain_db=args.gain_db,
                    batch_size=args.batch_size,
                    a2_lite_only=args.a2_lite_only,
                    version_tag=args.version_tag,
                    no_manifest=args.no_manifest,
                )
            return

        if instruments_to_run and args.instrument != "all":
            for inst in instruments_to_run:
                inst_cfg = load_instrument(inst)
                pickups_to_run = (
                    [args.pickup]
                    if (args.pickup and args.pickup != "auto")
                    else list(inst_cfg.pickups.keys())
                )
                for pkey in pickups_to_run:
                    if fmt in ["ir", "both"]:
                        ir_p = export_frontend_ir(
                            inst,
                            pkey,
                            normalize=args.normalize_frontend,
                            gain_db=args.gain_db,
                            version_tag=args.version_tag,
                            no_manifest=args.no_manifest,
                        )
                        rel_ir = ir_p.relative_to(REPO_ROOT) if ir_p.is_relative_to(REPO_ROOT) else ir_p
                        print(f" [Frontend IR] Exported {rel_ir}")
                    if fmt in ["wet", "both"]:
                        wet_p = export_frontend_wet_wav(
                            inst,
                            pkey,
                            input_wav=input_wav,
                            normalize=args.normalize_frontend,
                            gain_db=args.gain_db,
                            version_tag=args.version_tag,
                            no_manifest=args.no_manifest,
                        )
                        rel_wet = wet_p.relative_to(REPO_ROOT) if wet_p.is_relative_to(REPO_ROOT) else wet_p
                        print(f" [Frontend Wet WAV] Exported {rel_wet}")
            return

        if fmt in ["ir", "both"]:
            export_all_frontend_irs(
                jobs=args.jobs,
                normalize=args.normalize_frontend,
                gain_db=args.gain_db,
                version_tag=args.version_tag,
                no_manifest=args.no_manifest,
            )
        if fmt in ["wet", "both"]:
            export_all_frontend_wet_wavs(
                input_wav=input_wav,
                jobs=args.jobs,
                normalize=args.normalize_frontend,
                gain_db=args.gain_db,
                version_tag=args.version_tag,
                no_manifest=args.no_manifest,
            )
        return

    if args.stage == "targets":
        simulate_backend_targets(
            tier=args.tier or "standard",
            voice_id=args.voice,
            max_samples=args.max_samples,
            jobs=args.jobs,
            normalize=args.normalize,
            target_dbfs=args.target_dbfs,
            version_tag=args.version_tag,
            no_manifest=args.no_manifest,
        )
        return

    if args.stage == "viz":
        if len(instruments_to_run) == 1 and args.instrument != "all":
            run_visualization(instrument=instruments_to_run[0])
        else:
            run_visualization(instrument="all")
        return

    if args.stage == "train":
        tiers_to_train = (
            ["clean", "standard", "hotrod"] if args.tier == "all" else [args.tier or "standard"]
        )
        for inst in instruments_to_run:
            for t in tiers_to_train:
                for idx, voice in enumerate(voices_to_run, 1):
                    print(
                        f"\n[{idx}/{len(voices_to_run)}] Training NAM A2 Model: {inst} -> {voice} (Tier: {t})..."
                    )
                    run_training(
                        instrument=inst,
                        voice=voice,
                        input_wav=input_wav,
                        tier=t,
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
        print("\n--- Step 1: Canonical Intermediate Baseline Sweep ---")
        generate_canonical_sweep(
            input_wav=input_wav,
            version_tag=args.version_tag,
            no_manifest=args.no_manifest,
        )
        fmt = args.frontend_format
        if fmt in ["wet", "both"]:
            print("\n--- Step 2a: Export All 32 Frontend Deconvolution Wet Sweeps ---")
            export_all_frontend_wet_wavs(
                input_wav=input_wav,
                jobs=args.jobs,
                normalize=args.normalize_frontend,
                gain_db=args.gain_db,
                version_tag=args.version_tag,
                no_manifest=args.no_manifest,
            )
        if fmt in ["ir", "both"]:
            print("\n--- Step 2b: Export All 32 Frontend Deconvolution IRs ---")
            export_all_frontend_irs(
                jobs=args.jobs,
                normalize=args.normalize_frontend,
                gain_db=args.gain_db,
                version_tag=args.version_tag,
                no_manifest=args.no_manifest,
            )
        print("\n--- Step 3: Simulate Backend Targets ---")
        simulate_backend_targets(
            tier=args.tier or "standard",
            voice_id=args.voice,
            max_samples=args.max_samples,
            jobs=args.jobs,
            normalize=args.normalize,
            target_dbfs=args.target_dbfs,
            version_tag=args.version_tag,
            no_manifest=args.no_manifest,
        )
        print("\n--- Step 4: Interactive Altair Frequency Visualization ---")
        if len(instruments_to_run) == 1 and args.instrument != "all":
            run_visualization(instrument=instruments_to_run[0])
        else:
            run_visualization(instrument="all")
        print("\n[Pipeline Complete: 32 Frontend Sweeps/IRs + Backend Sweeps + Interactive Portal Ready]")
        return
