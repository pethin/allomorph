"""
Allomorph - MNA Inductive Core & Skin Impedance Engine
Computes Foster 2-stage ladder impedance, Jordan after-effect permeability dispersion,
and solid pole eddy skin-effect dispersion.
"""

import functools
import math

import numpy as np


@functools.lru_cache(maxsize=16)
def _get_cached_s_ratio_power(alpha: float, n_points: int) -> np.ndarray:
    """Caches normalized s_ratio ** alpha vectors for standard frequency grids."""
    freqs_arr = np.linspace(0.0, 24000.0, n_points)
    w = 2.0 * np.pi * freqs_arr
    w0 = 2.0 * np.pi * 1000.0
    s_ratio = np.where(w > 0.0, w / w0, 0.0)
    return s_ratio**alpha


def compute_core_impedance(
    s: complex | np.ndarray,
    L: float,
    L_core: float = 0.0,
    R_core: float = 0.0,
    chi_mu: float = 0.0,
    omega_mu: float = 2.0 * math.pi * 1200.0,
    k_skin: float = 0.0,
    omega_skin: float = 2.0 * math.pi * 3200.0,
    Rdc: float = 8000.0,
) -> complex | np.ndarray:
    """
    Computes Foster 2-stage ladder impedance of the coil inductor with
    Jordan after-effect complex magnetic permeability dispersion and
    solid pole eddy skin-effect dispersion:
    mu_rel(s) = 1.0 - chi_mu * ln(1.0 + s / omega_mu)
    Z_L(s) = mu_rel(s) * [s * L_inf + (s * L_core * R_core) / (s * L_core + R_core)] + Z_skin(s)
    where Z_skin(s) = Rdc * k_skin * (sqrt(1.0 + s / omega_skin) - 1.0)
    where L_inf = max(L - L_core, 0.0).
    Captures high-frequency magnetic flux expulsion from conductive pole pieces (skin effect),
    complex permeability dispersion, and eddy damping losses.
    """
    if chi_mu > 0.0:
        mu_rel = 1.0 - chi_mu * np.log(1.0 + s / omega_mu)
    else:
        mu_rel = 1.0

    if k_skin > 0.0 and omega_skin > 0.0:
        R_skin = Rdc * k_skin
        Z_skin = R_skin * (np.sqrt(1.0 + s / omega_skin) - 1.0)
    else:
        Z_skin = 0.0

    if L_core <= 0.0 or R_core <= 0.0:
        return s * L * mu_rel + Z_skin
    L_inf = max(L - L_core, 0.0)
    num = s * L_core * R_core
    den = s * L_core + R_core
    return (s * L_inf + (num / den)) * mu_rel + Z_skin


def compute_core_impedance_jacobians(
    s: complex | np.ndarray,
    L: float,
    L_core: float = 0.0,
    R_core: float = 0.0,
    chi_mu: float = 0.0,
    omega_mu: float = 2.0 * math.pi * 1200.0,
    k_skin: float = 0.0,
    omega_skin: float = 2.0 * math.pi * 3200.0,
    Rdc: float = 8000.0,
) -> dict[str, complex | np.ndarray]:
    """
    Computes exact closed-form partial derivatives (Jacobians) of Z_L(s) with respect
    to physical parameters (L, chi_mu, k_skin). Provides instantaneous sensitivity
    gradients for SPICE netlist parameter estimation without finite-difference noise.
    """
    if chi_mu > 0.0:
        mu_rel = 1.0 - chi_mu * np.log(1.0 + s / omega_mu)
        if L_core <= 0.0 or R_core <= 0.0:
            z_ind = s * L
        else:
            z_ind = s * max(L - L_core, 0.0) + (s * L_core * R_core) / (s * L_core + R_core)
        dZ_dchi_mu = -np.log(1.0 + s / omega_mu) * z_ind
    else:
        mu_rel = 1.0
        dZ_dchi_mu = 0.0

    dZ_dL = s * mu_rel

    if k_skin > 0.0 and omega_skin > 0.0:
        dZ_dk_skin = Rdc * (np.sqrt(1.0 + s / omega_skin) - 1.0)
    else:
        dZ_dk_skin = 0.0

    return {"dZ_dL": dZ_dL, "dZ_dchi_mu": dZ_dchi_mu, "dZ_dk_skin": dZ_dk_skin}
