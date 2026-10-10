"""Allomorph Pipeline Subpackage.

Coordinates visualization, audio prefiltering, batch SPICE circuit simulation,
bundle packaging, storefront generation, manifest provenance, and NAM neural model training.
"""

from allomorph.pipeline.batch import (
    _run_circuit_simulation_task,
    run_spice_batch,
)
from allomorph.pipeline.bundle import (
    assemble_root_pack_manifest,
    build_bundle_manifest_entry,
    generate_bundle_upload_instructions,
    resolve_bundle_target_file_descriptors,
)
from allomorph.pipeline.cli import (
    build_pipeline_arg_parser,
    list_instruments,
    list_voices,
    main,
    parse_and_validate_pipeline_args,
    resolve_execution_targets,
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
    export_tone_pack,
    train_tone_pack,
)
from allomorph.pipeline.sim_cli import (
    build_sim_arg_parser,
    parse_and_validate_sim_args,
)
from allomorph.pipeline.stages import (
    run_circuit_simulation,
    run_training,
    run_visualization,
)
from allomorph.pipeline.storefront import (
    FAMILY_ORDER,
    VOICE_CATALOG_DESCRIPTIONS,
    VOICE_ORDER_IN_FAMILY,
    VOICE_SLUG_ALIASES,
    build_storefront_instrument_setup,
    build_storefront_legal_disclaimer,
    build_storefront_overview,
    build_storefront_signal_chain,
    build_storefront_voicings_catalog,
    generate_storefront_description,
)

__all__ = [
    "FAMILY_ORDER",
    "VOICE_CATALOG_DESCRIPTIONS",
    "VOICE_ORDER_IN_FAMILY",
    "VOICE_SLUG_ALIASES",
    "_run_circuit_simulation_task",
    "assemble_manifest_document",
    "assemble_root_pack_manifest",
    "build_bundle_manifest_entry",
    "build_dict_manifest_entry",
    "build_file_manifest_entry",
    "build_pipeline_arg_parser",
    "build_sim_arg_parser",
    "build_storefront_instrument_setup",
    "build_storefront_legal_disclaimer",
    "build_storefront_overview",
    "build_storefront_signal_chain",
    "build_storefront_voicings_catalog",
    "export_tone_pack",
    "extract_wav_metrics",
    "generate_bundle_upload_instructions",
    "generate_storefront_description",
    "list_instruments",
    "list_voices",
    "main",
    "parse_and_validate_pipeline_args",
    "parse_and_validate_sim_args",
    "read_existing_manifest",
    "resolve_bundle_target_file_descriptors",
    "resolve_execution_targets",
    "run_circuit_simulation",
    "run_spice_batch",
    "run_training",
    "run_visualization",
    "train_tone_pack",
    "write_manifest",
]
