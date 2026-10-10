"""
Allomorph MNA Circuit Engine Subpackage
Vectorized Modified Nodal Analysis (MNA), analytical Cramer linear solvers,
Foster ladder core impedance, Cole-Davidson dielectric relaxation,
and bilinear transform biquad synthesis.
"""

from allomorph.circuit.mna.biquads import (
    compute_active_preamp_biquads,
    compute_active_preamp_eq,
    compute_active_preamp_transfer,
    evaluate_analog_band,
)
from allomorph.circuit.mna.impedance import (
    _get_cached_s_ratio_power,
    compute_core_impedance,
    compute_core_impedance_jacobians,
)
from allomorph.circuit.mna.linear_solve import solve_mna_linear_system
from allomorph.circuit.mna.smooth import smooth_soft_knee_db
from allomorph.circuit.mna.stamping import DisjointSet, _CoilBranch

__all__ = [
    "DisjointSet",
    "_CoilBranch",
    "_get_cached_s_ratio_power",
    "compute_active_preamp_biquads",
    "compute_active_preamp_eq",
    "compute_active_preamp_transfer",
    "compute_core_impedance",
    "compute_core_impedance_jacobians",
    "evaluate_analog_band",
    "smooth_soft_knee_db",
    "solve_mna_linear_system",
]
