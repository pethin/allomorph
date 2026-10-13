"""Allomorph Pipeline Subpackage.

Coordinates bundle packaging, storefront generation, manifest provenance,
and batch SPICE circuit simulation.
"""

from allomorph.pipeline.batch import (
    _run_circuit_simulation_task,
    run_circuit_simulation,
    run_spice_batch,
)
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
from allomorph.pipeline.sim_cli import (
    build_sim_arg_parser,
    parse_and_validate_sim_args,
)

__all__ = [
    "_run_circuit_simulation_task",
    "assemble_manifest_document",
    "assemble_root_pack_manifest",
    "build_bundle_manifest_entry",
    "build_dict_manifest_entry",
    "build_file_manifest_entry",
    "build_sim_arg_parser",
    "check_pack_needs_repacking",
    "export_tone_pack",
    "extract_wav_metrics",
    "generate_bundle_upload_instructions",
    "parse_and_validate_sim_args",
    "read_existing_manifest",
    "resolve_bundle_target_file_descriptors",
    "run_circuit_simulation",
    "run_spice_batch",
    "train_tone_pack",
    "write_manifest",
]
