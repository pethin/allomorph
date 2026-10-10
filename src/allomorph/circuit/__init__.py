"""
Allomorph Circuit Simulation Engine
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
from allomorph.circuit.forward import (
    AUDIO_DIR,
    CALIBRATION_PEAK_CEILING,
    simulate_instrument_voicing,
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
from allomorph.config.scales import REPO_ROOT

MODELS_DIR = REPO_ROOT / "models"
from allomorph.circuit.solver import (
    DisjointSet,
    compute_active_preamp_eq,
    compute_core_impedance,
    smooth_soft_knee_db,
    solve_mna_harness,
)
from allomorph.circuit.sweeps import (
    ParametricSweepResult,
    compute_parametric_sweep,
)
from allomorph.dsp import (
    DEFAULT_INPUT_PATH,
    ensure_input_audio_wav,
)

__all__ = [
    "AUDIO_DIR",
    "CALIBRATION_PEAK_CEILING",
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
    "apply_active_pickup_dynamics",
    "apply_algebraic_rail_limiter",
    "apply_dahl_hysteresis",
    "apply_elliptical_orbit_projection",
    "apply_oversampled_saturation",
    "audit_audio_file",
    "audit_wet_audio_catalog",
    "compute_active_preamp_eq",
    "compute_core_impedance",
    "compute_parametric_sweep",
    "ensure_input_audio_wav",
    "eval_pot_taper",
    "find_default_input_audio",
    "parse_spice_val",
    "simulate_instrument_voicing",
    "smooth_soft_knee_db",
    "solve_mna_harness",
]
