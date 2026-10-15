"""
Allomorph CLI - Stems Subcommand
Manages forward digital twin wet stem synthesis and orphan stem pruning.
"""

import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from allomorph.circuit.forward import simulate_instrument_voicing
from allomorph.cli.console import (
    log_error,
    log_info,
    log_step,
    log_success,
    log_warning,
    render_table,
)
from allomorph.cli.targets import resolve_targets
from allomorph.config.instruments import INSTRUMENTS, load_instrument, resolve_target_voicing
from allomorph.config.preamps import PREAMPS
from allomorph.config.scales import REPO_ROOT
from allomorph.config.schema import InstrumentConfig, VoicingConfig
from allomorph.config.strings import STRINGS
from allomorph.dsp import (
    check_dry_excitation_needs_regeneration,
    compute_lufs,
    ensure_input_audio_wav,
    read_wav,
)
from allomorph.naming import get_default_input_path
from allomorph.version import (
    DSP_GENERATION,
    check_stem_freshness,
    compute_recipe_token,
    resolve_tri_part_version,
)

STEMS_AUDIO_DIR = REPO_ROOT / "audio" / "stems"


def generate_stems(
    targets: tuple[str, ...] = ("all",),
    *,
    input_wav: Path | None = None,
    max_samples: int | None = None,
    force: bool = False,
    jobs: int | None = None,
) -> None:
    """Synthesizes target-filtered audio stems directly from dry string excitation."""
    resolved = resolve_targets(
        targets,
        default="all",
        allow_shorthand=True,
        shorthand_kind="instrument",
    )
    if not resolved:
        log_warning("No targets matched for stem generation.")
        return

    # 1. Check if input excitation needs regeneration (existence, token freshness, force)
    excitation_targets = [t for t in resolved if t.kind == "excitation"]
    instrument_targets = [t for t in resolved if t.kind == "instrument"]

    default_dry_path = get_default_input_path()

    if input_wav is None:
        candidates = [default_dry_path, REPO_ROOT / "input.wav"]
        existing_candidate = next((c for c in candidates if c.exists()), None)
        target_dry = existing_candidate or default_dry_path

        needs_regen, reason = check_dry_excitation_needs_regeneration(target_dry, force=force)
        if needs_regen:
            action_desc = "Re-synthesizing" if target_dry.exists() else "Synthesizing"
            log_step("DRY", f"{action_desc} dry excitation audio ({target_dry.name}): {reason}...")
            input_wav = ensure_input_audio_wav(target_dry, overwrite=True)
            log_success(f"Dry excitation audio ready: {input_wav.name}")
        else:
            log_step("DRY", f"Cached & fresh dry excitation audio: {target_dry.name}")
            input_wav = target_dry
    else:
        if not input_wav.exists():
            if input_wav.resolve() == default_dry_path.resolve() or input_wav.name == "input.wav":
                log_step("DRY", f"Synthesizing dry excitation audio ({input_wav.name})...")
                input_wav = ensure_input_audio_wav(input_wav, overwrite=True)
                log_success(f"Dry excitation audio ready: {input_wav.name}")
            else:
                log_error(f"Specified input audio '{input_wav}' does not exist.")
                sys.exit(1)
        else:
            if force and (
                input_wav.resolve() == default_dry_path.resolve() or input_wav.name == "input.wav"
            ):
                log_step(
                    "DRY",
                    f"Re-synthesizing dry excitation audio ({input_wav.name}) (--force)...",
                )
                input_wav = ensure_input_audio_wav(input_wav, overwrite=True)
                log_success(f"Dry excitation audio ready: {input_wav.name}")
            else:
                log_step("DRY", f"Using input excitation audio: {input_wav.name}")

    if not instrument_targets:
        if excitation_targets:
            log_success("Dry excitation audio generation completed.")
        return

    log_step(
        "STEMS",
        f"Synthesizing stems for {len(instrument_targets)} instrument target rule(s) (force={force})...",
    )

    # In-process token cache
    recipe_token_cache: dict[tuple[str, str], str] = {}

    tasks: list[tuple[InstrumentConfig, VoicingConfig]] = []
    for tgt in instrument_targets:
        try:
            inst = load_instrument(tgt.id)
        except Exception as e:  # noqa: BLE001
            log_error(f"Failed to load instrument '{tgt.id}': {e}")
            sys.exit(1)

        if tgt.sub_target is not None:
            # Specific voicing or tone
            try:
                tgt_inst, v_cfg = resolve_target_voicing(tgt.sub_target, instrument=inst)
                tasks.append((tgt_inst, v_cfg))
            except Exception as e:  # noqa: BLE001
                log_error(f"Failed to resolve voicing '{tgt.sub_target}' on '{tgt.id}': {e}")
                sys.exit(1)
        else:
            # All voicings on instrument
            for v_cfg in inst.voicings.values():
                tasks.append((inst, v_cfg))

    # Deduplicate tasks
    unique_tasks: list[tuple[InstrumentConfig, VoicingConfig]] = []
    seen_tasks: set[tuple[str, str]] = set()
    for inst, v_cfg in tasks:
        v_key = (inst.id, v_cfg.id or "")
        if v_key not in seen_tasks:
            seen_tasks.add(v_key)
            unique_tasks.append((inst, v_cfg))

    total = len(unique_tasks)
    log_info(f"Resolved {total} total voicing stem(s) across target(s).")

    cached_count = 0
    simulated_count = 0
    failures: list[tuple[str, str]] = []

    tasks_to_simulate: list[tuple[int, InstrumentConfig, VoicingConfig, str, Path, str]] = []

    for idx, (inst, v_cfg) in enumerate(unique_tasks, 1):
        stem_slug = v_cfg.id or ""
        target_path = STEMS_AUDIO_DIR / inst.id / f"{stem_slug}.wav"

        # Check in-process cache for recipe token
        cache_key = (inst.id, stem_slug)
        if cache_key in recipe_token_cache:
            recipe_tok = recipe_token_cache[cache_key]
        else:
            recipe_tok = compute_recipe_token(
                instrument=inst,
                voicing=v_cfg,
                base_dry_path=input_wav,
                dsp_generation=DSP_GENERATION,
                preamps=PREAMPS,
                strings=STRINGS,
            )
            recipe_token_cache[cache_key] = recipe_tok

        v_tag = resolve_tri_part_version(
            DSP_GENERATION,
            getattr(inst, "version", 1),
            getattr(v_cfg, "version", 1),
        )

        if force:
            is_fresh, reason = False, "force flag"
        else:
            is_fresh, reason = check_stem_freshness(
                target_path,
                expected_recipe_token=recipe_tok,
                base_dry_path=input_wav,
                expected_version=v_tag,
            )

        if is_fresh:
            log_step("SKIP", f"[{idx}/{total}] Cached & fresh: {inst.id}:{stem_slug} ({v_tag})")
            cached_count += 1
            continue

        log_step(
            "QUEUE",
            f"[{idx}/{total}] Queued for simulation ({reason}): {inst.id}:{stem_slug} ({v_tag})",
        )
        tasks_to_simulate.append((idx, inst, v_cfg, stem_slug, target_path, v_tag))

    if tasks_to_simulate:
        log_info(
            f"Stem audit: {cached_count} cached & fresh, {len(tasks_to_simulate)} queued for synthesis."
        )
        input_audio: np.ndarray | None = None
        ref_lufs: float | None = None
        if input_wav is not None and input_wav.exists():
            try:
                input_audio, input_sr = read_wav(
                    input_wav, max_samples=max_samples, dtype=np.float64
                )
                input_mono = input_audio[0] if input_audio.ndim > 1 else input_audio
                ref_lufs = compute_lufs(input_mono, sample_rate=input_sr)
            except Exception:  # noqa: BLE001
                input_audio = None
                ref_lufs = None

        eff_jobs = jobs if jobs is not None and jobs > 0 else min(os.cpu_count() or 4, 8)

        def _execute_task(
            t_item: tuple[int, InstrumentConfig, VoicingConfig, str, Path, str],
        ) -> tuple[str, str | None]:
            t_idx, t_inst, t_v_cfg, t_slug, t_path, t_v_tag = t_item
            log_step("SIM", f"[{t_idx}/{total}] Simulating {t_inst.id}:{t_slug} ({t_v_tag})...")
            try:
                simulate_instrument_voicing(
                    instrument=t_inst,
                    voicing=t_v_cfg,
                    input_wav=input_wav,
                    input_audio=input_audio,
                    output_wav=t_path,
                    max_samples=max_samples,
                    force=force,
                    preamps=PREAMPS,
                    strings=STRINGS,
                    ref_lufs=ref_lufs,
                )
                return (f"{t_inst.id}:{t_slug}", None)
            except Exception as e:  # noqa: BLE001
                log_error(f"Failed to simulate {t_inst.id}:{t_slug}: {e}")
                return (f"{t_inst.id}:{t_slug}", str(e))

        if eff_jobs > 1 and len(tasks_to_simulate) > 1:
            log_info(
                f"Synthesizing {len(tasks_to_simulate)} stems in parallel across {min(eff_jobs, len(tasks_to_simulate))} worker threads..."
            )
            with ThreadPoolExecutor(max_workers=eff_jobs) as executor:
                futures = [executor.submit(_execute_task, item) for item in tasks_to_simulate]
                for fut in as_completed(futures):
                    target_label, err = fut.result()
                    if err is None:
                        simulated_count += 1
                    else:
                        failures.append((target_label, err))
        else:
            for item in tasks_to_simulate:
                target_label, err = _execute_task(item)
                if err is None:
                    simulated_count += 1
                else:
                    failures.append((target_label, err))

    # Summary
    log_info("")
    log_info(
        f"Stem Generation Summary: {simulated_count} simulated, {cached_count} cached/fresh, {len(failures)} failed."
    )

    if failures:
        render_table(
            "Stem Simulation Failures",
            headers=["Target Voicing", "Error Reason"],
            rows=failures,
            border_style="red",
        )
        sys.exit(1)

    log_success("All requested stems generated successfully.")


def prune_stems(*, dry_run: bool = False) -> None:
    """Deletes orphan WAV files, stale JSON sidecars, and empty directories in audio/stems/."""
    log_step(
        "PRUNE", f"Scanning {STEMS_AUDIO_DIR} for orphan stems and sidecars (dry_run={dry_run})..."
    )

    if not STEMS_AUDIO_DIR.exists():
        log_info("Audio directory 'audio/stems/' does not exist. Nothing to prune.")
        return

    # Build index of valid stems across all configured instruments
    valid_stems_by_inst: dict[str, set[str]] = {}
    for inst_id, inst in INSTRUMENTS.items():
        valid_stems_by_inst[inst_id] = set()
        for v_id, v_cfg in inst.voicings.items():
            valid_stems_by_inst[inst_id].add(v_cfg.id or v_id)

    deleted_files: list[str] = []
    deleted_dirs: list[str] = []

    # Scan instrument subdirectories in audio/stems
    for item in sorted(STEMS_AUDIO_DIR.iterdir()):
        if not item.is_dir() or item.name.startswith("."):
            continue

        inst_id = item.name
        is_known_inst = inst_id in valid_stems_by_inst
        known_slugs = valid_stems_by_inst.get(inst_id, set())

        # Check files inside the directory
        for f in sorted(item.iterdir()):
            if f.name.startswith("."):
                continue
            if f.name == "manifest.json":
                continue

            stem_name = f.stem
            is_orphan = False

            if not is_known_inst:
                is_orphan = True
            elif f.suffix in [".wav", ".json"]:
                if stem_name not in known_slugs:
                    is_orphan = True
            else:
                is_orphan = True

            if is_orphan:
                action = "[DRY-RUN] Would delete" if dry_run else "Deleted"
                rel_desc = str(f.relative_to(REPO_ROOT)) if f.is_relative_to(REPO_ROOT) else str(f)
                log_step("PRUNE", f"{action} file: {rel_desc}")
                deleted_files.append(rel_desc)
                if not dry_run:
                    f.unlink(missing_ok=True)

        # Check if directory is now empty (or only has leftover manifest / empty)
        remaining = [f for f in item.iterdir() if not f.name.startswith(".")]
        if not remaining or (
            len(remaining) == 1 and remaining[0].name == "manifest.json" and not is_known_inst
        ):
            action = "[DRY-RUN] Would remove" if dry_run else "Removed"
            rel_dir = (
                str(item.relative_to(REPO_ROOT)) if item.is_relative_to(REPO_ROOT) else str(item)
            )
            log_step("PRUNE", f"{action} empty directory: {rel_dir}")
            deleted_dirs.append(rel_dir)
            if not dry_run:
                if (item / "manifest.json").exists():
                    (item / "manifest.json").unlink(missing_ok=True)
                item.rmdir()

    log_info("")
    if dry_run:
        log_info(
            f"Dry Run Complete: {len(deleted_files)} file(s) and {len(deleted_dirs)} directory(ies) would be pruned."
        )
    else:
        log_success(
            f"Prune Complete: {len(deleted_files)} file(s) and {len(deleted_dirs)} directory(ies) pruned."
        )
