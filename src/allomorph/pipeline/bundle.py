"""Bundle metadata and packaging primitives for Tone3000 upload packs.

Provides pure, testable functions for resolving target file descriptors,
generating upload instructions, and assembling bundle and pack manifests.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

from allomorph.naming import get_t3k_basename
from allomorph.version import (
    ALLOMORPH_VERSION,
    DSP_GENERATION,
    resolve_tri_part_version,
)

if TYPE_CHECKING:
    from allomorph.config.instruments import VoicingBundle
    from allomorph.config.schema import InstrumentConfig


def resolve_bundle_target_file_descriptors(
    bundle: VoicingBundle,
    is_multi_pickup: bool,
    dsp_generation: int = DSP_GENERATION,
) -> list[dict[str, Any]]:
    """Resolves target audio stem metadata and Tone3000 basenames for all targets in a bundle."""
    pos_label = bundle.position_name or bundle.bundle_name.capitalize()
    pos_tag = pos_label if is_multi_pickup else None

    descriptors: list[dict[str, Any]] = []
    for target_ref in bundle.targets:
        tone_name = target_ref.voicing.tone_name or target_ref.voicing.name
        target_v_tag = resolve_tri_part_version(
            dsp_generation,
            target_ref.instrument.version,
            target_ref.voicing.version,
        )
        stem_base = get_t3k_basename(
            tone_name=tone_name,
            position_name=pos_tag,
            version_tag=target_v_tag,
        )
        stem_filename = f"{stem_base}.wav"

        target_slug = (
            target_ref.voicing.tone_name.lower()
            .replace(" ", "_")
            .replace("∕", "_")
            .replace("/", "_")
            if target_ref.voicing.tone_name
            else (target_ref.voicing.id or "default_voicing")
        )

        descriptors.append(
            {
                "target_ref": target_ref,
                "tone_name": tone_name,
                "pos_tag": pos_tag,
                "target_v_tag": target_v_tag,
                "stem_base": stem_base,
                "stem_filename": stem_filename,
                "target_slug": target_slug,
            }
        )

    return descriptors


def generate_bundle_upload_instructions(
    bundle_name: str,
    position_name: str,
    instrument_name: str,
    dry_filename: str,
    stem_filenames: Sequence[str],
) -> str:
    """Generates the upload instructions text for a single Tone3000 bundle."""
    lines = [
        "=" * 78,
        f"TONE3000 UPLOAD BUNDLE: {bundle_name.upper()}",
        "=" * 78,
        f"Source Instrument: {instrument_name}",
        f"Pickup Position: {position_name}",
        f"Physical Switch Setting on Bass: {position_name}",
        "Tone Knob Setting: 100% (Wide Open)",
        "",
        "STEP-BY-STEP UPLOAD INSTRUCTIONS (Tone3000 Studio Trainer):",
        "1. Open Tone3000 Trainer in your browser or local batch tool.",
        f"2. DRAG DRY FILE: Drag '{dry_filename}' from this directory into the '1 Dry File' slot.",
        f"3. DRAG WET STEMS: Select and drag the following {len(stem_filenames)} .wav files into the 'Wet Stems' slot:",
    ]
    for fn in stem_filenames:
        lines.append(f"   • {fn}")
    lines.extend(
        [
            "",
            "4. START TRAINING: Tone3000 will batch-train all models against this single dry baseline.",
            "=" * 78,
        ]
    )
    return "\n".join(lines) + "\n"


def build_bundle_manifest_entry(
    bundle_name: str,
    position_name: str,
    source_voicing_id: str,
    dry_filename: str,
    dry_version: str,
    dry_sha256: str,
    instrument_id: str,
    instrument_version: int,
    source_voicing_version: int,
    bundle_stems: Sequence[dict[str, Any]],
    base_dry_file: str | None = None,
    base_dry_sha256: str | None = None,
) -> dict[str, Any]:
    """Constructs the authoritative manifest record for a single bundle."""
    return {
        "bundle": bundle_name,
        "source_voicing": source_voicing_id,
        "position_name": position_name,
        "base_dry_file": base_dry_file,
        "base_dry_sha256": base_dry_sha256,
        "dry_file": dry_filename,
        "dry_version": dry_version,
        "dry_sha256": dry_sha256,
        "instrument_id": instrument_id,
        "instrument_version": instrument_version,
        "source_voicing_version": source_voicing_version,
        "stem_count": len(bundle_stems),
        "stems": list(bundle_stems),
    }


def assemble_root_pack_manifest(
    inst: InstrumentConfig,
    bundle_entries: Mapping[str, Any],
    dsp_generation: int = DSP_GENERATION,
) -> dict[str, Any]:
    """Assembles the root pack manifest dictionary for an instrument."""
    return {
        "instrument_id": inst.id,
        "instrument_name": inst.name,
        "instrument_version": inst.version,
        "version": resolve_tri_part_version(dsp_generation, inst.version, 1),
        "allomorph_version": ALLOMORPH_VERSION,
        "dsp_generation": dsp_generation,
        "bundles": dict(bundle_entries),
    }
