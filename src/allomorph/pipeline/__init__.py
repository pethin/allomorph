"""Allomorph Pipeline Subpackage.

Coordinates visualization, audio prefiltering, batch SPICE circuit simulation,
bundle packaging, storefront generation, and NAM neural model training.
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
    list_instruments,
    list_voices,
    main,
)
from allomorph.pipeline.pack import (
    export_tone_pack,
    train_tone_pack,
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
    "assemble_root_pack_manifest",
    "build_bundle_manifest_entry",
    "build_storefront_instrument_setup",
    "build_storefront_legal_disclaimer",
    "build_storefront_overview",
    "build_storefront_signal_chain",
    "build_storefront_voicings_catalog",
    "export_tone_pack",
    "generate_bundle_upload_instructions",
    "generate_storefront_description",
    "list_instruments",
    "list_voices",
    "main",
    "resolve_bundle_target_file_descriptors",
    "run_circuit_simulation",
    "run_spice_batch",
    "run_training",
    "run_visualization",
    "train_tone_pack",
]
