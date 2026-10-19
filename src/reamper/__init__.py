"""Allomorph Hardware Reamper & Native NAM A2 Trainer Package."""

from __future__ import annotations

from reamper.capture import (
    GainCalibrationResult,
    align_hardware_latency,
    calibrate_output_gain,
    capture_idle_noise,
    compute_max_unclipped_send_gain_db,
    extract_calibration_probe,
    find_anagram_device,
    read_reamp_sidecar,
    reamp,
    write_reamp_sidecar,
)
from reamper.generate import (
    generate_optimal_reamp_dry,
    generate_reamp_input,
    synth_reamp_alignment_pulses,
)
from reamper.train import (
    CANONICAL_SIGNAL_CHAINS,
    AnagramSignalChain,
    apply_model_io_scale_compensation,
    load_signal_chains,
    resolve_signal_chain_for_stem,
    resolve_stem_send_gain_db,
    select_warm_start_offset,
    train_reamp,
    train_single_reamp_pair,
    verify_or_align_stems,
)

__all__ = [
    "CANONICAL_SIGNAL_CHAINS",
    "AnagramSignalChain",
    "GainCalibrationResult",
    "align_hardware_latency",
    "apply_model_io_scale_compensation",
    "calibrate_output_gain",
    "capture_idle_noise",
    "compute_max_unclipped_send_gain_db",
    "extract_calibration_probe",
    "find_anagram_device",
    "generate_optimal_reamp_dry",
    "generate_reamp_input",
    "load_signal_chains",
    "read_reamp_sidecar",
    "reamp",
    "resolve_signal_chain_for_stem",
    "resolve_stem_send_gain_db",
    "select_warm_start_offset",
    "synth_reamp_alignment_pulses",
    "train_reamp",
    "train_single_reamp_pair",
    "verify_or_align_stems",
    "write_reamp_sidecar",
]
