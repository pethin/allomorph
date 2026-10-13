"""
Allomorph CLI - Stems Subcommand
Manages forward digital twin wet stem synthesis and orphan stem pruning.
"""

import sys
from pathlib import Path

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
from allomorph.version import (
    DSP_GENERATION,
    compute_recipe_token,
    is_wet_stem_valid,
    resolve_tri_part_version,
)

WET_AUDIO_DIR = REPO_ROOT / "audio" / "wet"


def generate_stems(
    targets: tuple[str, ...] = ("all",),
    *,
    input_wav: Path | None = None,
    max_samples: int | None = None,
    force: bool = False,
) -> None:
    """Synthesizes target-filtered wet audio stems directly from dry string excitation."""
    resolved = resolve_targets(targets, default="all", allow_shorthand=True)
    if not resolved:
        log_warning("No targets matched for stem generation.")
        return

    log_step("STEMS", f"Synthesizing stems for {len(resolved)} target rule(s) (force={force})...")

    # In-process token cache
    recipe_token_cache: dict[tuple[str, str], str] = {}

    tasks: list[tuple[InstrumentConfig, VoicingConfig]] = []
    for tgt in resolved:
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
        v_key = (inst.id, v_cfg.tone_name or v_cfg.id or "")
        if v_key not in seen_tasks:
            seen_tasks.add(v_key)
            unique_tasks.append((inst, v_cfg))

    total = len(unique_tasks)
    log_info(f"Resolved {total} total voicing stem(s) across target(s).")

    cached_count = 0
    simulated_count = 0
    failures: list[tuple[str, str]] = []

    for idx, (inst, v_cfg) in enumerate(unique_tasks, 1):
        v_label = v_cfg.tone_name or v_cfg.id or "default"
        stem_slug = v_label.lower().replace(" ", "_").replace("∕", "_").replace("/", "_")
        target_path = WET_AUDIO_DIR / inst.id / f"{stem_slug}.wav"

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

        is_fresh = False
        if not force and target_path.exists():
            is_fresh = is_wet_stem_valid(
                target_path,
                expected_recipe_token=recipe_tok,
                base_dry_path=input_wav,
                expected_version=v_tag,
            )

        if is_fresh:
            log_step("SKIP", f"[{idx}/{total}] Cached & fresh: {inst.id}:{stem_slug} ({v_tag})")
            cached_count += 1
            continue

        log_step("SIM", f"[{idx}/{total}] Simulating {inst.id}:{stem_slug} ({v_tag})...")
        try:
            simulate_instrument_voicing(
                instrument=inst,
                voicing=v_cfg,
                input_wav=input_wav,
                output_wav=target_path,
                max_samples=max_samples,
                force=force,
                preamps=PREAMPS,
                strings=STRINGS,
            )
            simulated_count += 1
        except Exception as e:  # noqa: BLE001
            log_error(f"Failed to simulate {inst.id}:{stem_slug}: {e}")
            failures.append((f"{inst.id}:{stem_slug}", str(e)))

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
    """Deletes orphan WAV files, stale JSON sidecars, and empty directories in audio/wet/."""
    log_step(
        "PRUNE", f"Scanning {WET_AUDIO_DIR} for orphan stems and sidecars (dry_run={dry_run})..."
    )

    if not WET_AUDIO_DIR.exists():
        log_info("Audio directory 'audio/wet/' does not exist. Nothing to prune.")
        return

    # Build index of valid stems across all configured instruments
    valid_stems_by_inst: dict[str, set[str]] = {}
    for inst_id, inst in INSTRUMENTS.items():
        valid_stems_by_inst[inst_id] = set()
        for v_cfg in inst.voicings.values():
            v_label = v_cfg.tone_name or v_cfg.id or "default"
            slug = v_label.lower().replace(" ", "_").replace("∕", "_").replace("/", "_")
            valid_stems_by_inst[inst_id].add(slug)

    deleted_files: list[str] = []
    deleted_dirs: list[str] = []

    # Scan instrument subdirectories in audio/wet
    for item in sorted(WET_AUDIO_DIR.iterdir()):
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
