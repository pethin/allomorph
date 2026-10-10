"""Allomorph Circuit Simulation Engine.

Analytical closed-form nodal RLC solver and state-space non-linear saturation.
"""

from allomorph.circuit.audio import (
    find_default_input_audio,
)
from allomorph.circuit.audit import (
    AudioAuditRecord,
    AudioAuditReport,
    audit_audio_file,
    audit_wet_audio_catalog,
)
from allomorph.circuit.diagnostics import (
    CALIBRATION_PEAK_CEILING,
    DEFAULT_FREQUENCY_ANCHORS,
    aggregate_audit_report,
    compute_stem_level_metrics,
    evaluate_causal_onset_and_peak,
    extract_frequency_anchors,
    is_manifest_entry_fresh,
)
from allomorph.circuit.forward import (
    AUDIO_DIR,
    simulate_instrument_voicing,
    simulate_pickup_transducer_branch,
    simulate_voicing_dsp,
    synthesize_multi_pickup_spatial_blend,
)
from allomorph.circuit.parser import (
    MAGNET_PROPERTIES,
    eval_pot_taper,
    parse_spice_val,
)
from allomorph.circuit.saturation import (
    _HAS_NUMBA,
    _dahl_core,
    _lenz_envelope_core,
    _lenz_velocity_drag_core,
    _slew_limit_core,
    apply_active_pickup_dynamics,
    apply_algebraic_rail_limiter,
    apply_dahl_hysteresis,
    apply_elliptical_orbit_projection,
    apply_oversampled_saturation,
)
from allomorph.circuit.solver import (
    DisjointSet,
    compute_active_preamp_eq,
    compute_core_impedance,
    smooth_soft_knee_db,
    solve_mna_harness,
)
from allomorph.circuit.sweeps import (
    ParametricSweepResult,
    apply_sweep_parameter_override,
    compute_parametric_sweep,
    compute_sweep_curve_metrics,
    extract_channel_response,
)
from allomorph.config.scales import REPO_ROOT
from allomorph.dsp import (
    DEFAULT_INPUT_PATH,
    ensure_input_audio_wav,
)

MODELS_DIR = REPO_ROOT / "models"

__all__ = [
    "AUDIO_DIR",
    "CALIBRATION_PEAK_CEILING",
    "DEFAULT_FREQUENCY_ANCHORS",
    "DEFAULT_INPUT_PATH",
    "MAGNET_PROPERTIES",
    "MODELS_DIR",
    "REPO_ROOT",
    "_HAS_NUMBA",
    "AudioAuditRecord",
    "AudioAuditReport",
    "DisjointSet",
    "ParametricSweepResult",
    "_dahl_core",
    "_lenz_envelope_core",
    "_lenz_velocity_drag_core",
    "_slew_limit_core",
    "aggregate_audit_report",
    "apply_active_pickup_dynamics",
    "apply_algebraic_rail_limiter",
    "apply_dahl_hysteresis",
    "apply_elliptical_orbit_projection",
    "apply_oversampled_saturation",
    "apply_sweep_parameter_override",
    "audit_audio_file",
    "audit_wet_audio_catalog",
    "compute_active_preamp_eq",
    "compute_core_impedance",
    "compute_parametric_sweep",
    "compute_stem_level_metrics",
    "compute_sweep_curve_metrics",
    "ensure_input_audio_wav",
    "eval_pot_taper",
    "evaluate_causal_onset_and_peak",
    "extract_channel_response",
    "extract_frequency_anchors",
    "find_default_input_audio",
    "is_manifest_entry_fresh",
    "parse_spice_val",
    "simulate_instrument_voicing",
    "simulate_pickup_transducer_branch",
    "simulate_voicing_dsp",
    "smooth_soft_knee_db",
    "solve_mna_harness",
    "synthesize_multi_pickup_spatial_blend",
]
