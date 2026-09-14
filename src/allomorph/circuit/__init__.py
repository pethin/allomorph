"""
Allomorph Circuit Simulation Engine
Analytical closed-form nodal RLC solver and state-space non-linear saturation.
"""

from allomorph.circuit.audio import (
    find_default_input_audio,
)
from allomorph.circuit.forward import (
    AUDIO_DIR,
    CALIBRATION_PEAK_CEILING,
    resolve_target_voicing,
    simulate_all_instrument_voicings,
    simulate_circuit_audio,
    simulate_instrument_voicing,
    simulate_voice,
)
from allomorph.circuit.parser import (
    MAGNET_PROPERTIES,
    CircuitModel,
    eval_pot_taper,
    load_circuit,
    parse_netlist,
    parse_spice_val,
)
from allomorph.circuit.saturation import (
    _HAS_NUMBA,
    _dahl_core,
    _lenz_envelope_core,
    _lenz_velocity_drag_core,
    _slew_limit_core,
    apply_dahl_hysteresis,
    apply_elliptical_orbit_projection,
    apply_oversampled_saturation,
)
from allomorph.config.scales import REPO_ROOT

MODELS_DIR = REPO_ROOT / "models"
from allomorph.circuit.solver import (
    apply_magnet_properties_to_model,
    compute_active_preamp_eq,
    compute_circuit_transfer_functions,
    compute_core_impedance,
    compute_differential_circuit_transfer_functions,
    smooth_soft_knee_db,
)
from allomorph.circuit.staging import (
    export_instrument_pickup_wav,
    main,
)
from allomorph.circuit.sweeps import (
    ParametricSweepResult,
    compute_parametric_sweep,
)
from allomorph.dsp import (
    OPTIMAL_DRY_PATH,
    ensure_optimal_dry_wav,
)

__all__ = [
    "AUDIO_DIR",
    "CALIBRATION_PEAK_CEILING",
    "MAGNET_PROPERTIES",
    "MODELS_DIR",
    "OPTIMAL_DRY_PATH",
    "REPO_ROOT",
    "_HAS_NUMBA",
    "CircuitModel",
    "ParametricSweepResult",
    "_dahl_core",
    "_lenz_envelope_core",
    "_lenz_velocity_drag_core",
    "_slew_limit_core",
    "apply_dahl_hysteresis",
    "apply_elliptical_orbit_projection",
    "apply_magnet_properties_to_model",
    "apply_oversampled_saturation",
    "compute_active_preamp_eq",
    "compute_circuit_transfer_functions",
    "compute_core_impedance",
    "compute_differential_circuit_transfer_functions",
    "compute_parametric_sweep",
    "ensure_optimal_dry_wav",
    "eval_pot_taper",
    "export_instrument_pickup_wav",
    "find_default_input_audio",
    "load_circuit",
    "main",
    "parse_netlist",
    "parse_spice_val",
    "resolve_target_voicing",
    "simulate_all_instrument_voicings",
    "simulate_circuit_audio",
    "simulate_instrument_voicing",
    "simulate_voice",
    "smooth_soft_knee_db",
]
