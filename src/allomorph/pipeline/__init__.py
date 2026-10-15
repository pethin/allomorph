"""Allomorph Pipeline Subpackage.

Coordinates bundle packaging, storefront generation, and manifest provenance.
"""

from allomorph.pipeline.bundle import (
    assemble_root_pack_manifest,
    build_bundle_manifest_entry,
    generate_bundle_upload_instructions,
    resolve_bundle_target_file_descriptors,
)
from allomorph.pipeline.manifest import (
    assemble_manifest_document,
    build_dict_manifest_entry,
    build_file_manifest_entry,
    extract_wav_metrics,
    read_existing_manifest,
    write_manifest,
)
from allomorph.pipeline.pack import (
    check_pack_needs_repacking,
    export_tone_pack,
    train_tone_pack,
)

__all__ = [
    "assemble_manifest_document",
    "assemble_root_pack_manifest",
    "build_bundle_manifest_entry",
    "build_dict_manifest_entry",
    "build_file_manifest_entry",
    "check_pack_needs_repacking",
    "export_tone_pack",
    "extract_wav_metrics",
    "generate_bundle_upload_instructions",
    "read_existing_manifest",
    "resolve_bundle_target_file_descriptors",
    "train_tone_pack",
    "write_manifest",
]
