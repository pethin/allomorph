"""
Allomorph CLI - Pack Subcommand
Packages Tone3000 upload packs with dry excitation stems, target wet stems, and metadata.
"""

import sys
from pathlib import Path

from allomorph.cli.console import log_error, log_info, log_step, log_success, log_warning
from allomorph.cli.targets import resolve_targets
from allomorph.config.instruments import load_instrument
from allomorph.pipeline.pack import check_pack_needs_repacking, export_tone_pack


def pack(
    targets: tuple[str, ...] = ("packs",),
    *,
    output_dir: Path | None = None,
    input_wav: Path | None = None,
    max_samples: int | None = None,
) -> None:
    """Packages Tone3000 upload bundles respecting the '1 Dry + Multiple Wet Stems' constraint."""
    resolved = resolve_targets(targets, default="packs", allow_shorthand=True)
    if not resolved:
        log_warning("No targets matched for tone packing.")
        return

    log_step("PACK", f"Packaging tone pack(s) for {len(resolved)} target(s)...")

    # Deduplicate instrument IDs
    target_inst_ids: list[str] = []
    seen: set[str] = set()
    for tgt in resolved:
        if tgt.id not in seen:
            seen.add(tgt.id)
            target_inst_ids.append(tgt.id)

    packed_count = 0
    for idx, inst_id in enumerate(target_inst_ids, 1):
        log_step("PACK", f"[{idx}/{len(target_inst_ids)}] Exporting tone pack for '{inst_id}'...")
        try:
            inst = load_instrument(inst_id)
            if not check_pack_needs_repacking(inst, output_dir=output_dir, input_wav=input_wav):
                log_step(
                    "SKIP",
                    f"[{idx}/{len(target_inst_ids)}] Pack '{inst_id}' already fresh.",
                )
                packed_count += 1
                continue

            pack_out = export_tone_pack(
                instrument=inst,
                output_dir=output_dir,
                input_wav=input_wav,
                max_samples=max_samples,
            )
            log_info(f"Pack ready at: [bold]{pack_out}[/bold]")
            packed_count += 1
        except Exception as e:  # noqa: BLE001
            log_error(f"Failed to export tone pack for '{inst_id}': {e}")
            sys.exit(1)

    log_success(f"Successfully exported {packed_count} tone pack(s).")
