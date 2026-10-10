"""
Allomorph - SPICE Value Parser & Pot Taper Modeling
Provides SPICE engineering unit suffix parsing, authentic magnet metallurgy specifications,
and continuous C^inf potentiometer taper evaluations.
"""

import math
from pathlib import Path

from allomorph.base import parse_spice_unit
from allomorph.circuit.schema import MagnetPropertiesConfig

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent


def parse_spice_val(val_str: str) -> float:
    """Parses standard SPICE engineering suffix notation (k, Meg, p, n, u, m, g)."""
    return float(parse_spice_unit(val_str))


MAGNET_PROPERTIES: dict[str, MagnetPropertiesConfig] = {
    "alnico_v": MagnetPropertiesConfig(
        k_core=0.08,
        f_core=2500.0,
        k_skin=0.10,
        f_skin=3200.0,
        lambda_L=0.05,
        k_emf=0.04,
        eta_hyst=0.06,
        alpha=0.26,
        alpha3=0.10,
        k_sag=0.08,
        vsat=1.00,
        k_eddy=0.16,
        kappa_orbit=0.06,
        k_body=0.08,
        beta_curv=0.035,
        k_pull=0.040,
        tau_touch=0.045,
        chi_mu=0.035,
        k_dist=0.18,
        kappa_geom=0.20,
        k_stein=0.030,
        kappa_ap=0.035,
    ),
    "alnico_ii": MagnetPropertiesConfig(
        k_core=0.10,
        f_core=1800.0,
        k_skin=0.12,
        f_skin=2800.0,
        lambda_L=0.07,
        k_emf=0.05,
        eta_hyst=0.09,
        alpha=0.32,
        alpha3=0.14,
        k_sag=0.12,
        vsat=0.90,
        k_eddy=0.20,
        kappa_orbit=0.07,
        k_body=0.10,
        beta_curv=0.050,
        k_pull=0.025,
        tau_touch=0.035,
        chi_mu=0.050,
        k_dist=0.22,
        kappa_geom=0.24,
        k_stein=0.040,
        kappa_ap=0.045,
    ),
    "alnico_iii": MagnetPropertiesConfig(
        k_core=0.09,
        f_core=2000.0,
        k_skin=0.11,
        f_skin=2900.0,
        lambda_L=0.06,
        k_emf=0.045,
        eta_hyst=0.08,
        alpha=0.30,
        alpha3=0.12,
        k_sag=0.10,
        vsat=0.96,
        k_eddy=0.18,
        kappa_orbit=0.065,
        k_body=0.09,
        beta_curv=0.040,
        k_pull=0.020,
        tau_touch=0.035,
        chi_mu=0.040,
        k_dist=0.20,
        kappa_geom=0.22,
        k_stein=0.035,
        kappa_ap=0.040,
    ),
    "ceramic": MagnetPropertiesConfig(
        k_core=0.02,
        f_core=6500.0,
        k_skin=0.00,
        f_skin=0.0,
        lambda_L=0.02,
        k_emf=0.02,
        eta_hyst=0.02,
        alpha=0.12,
        alpha3=0.04,
        k_sag=0.03,
        vsat=1.40,
        k_eddy=0.03,
        kappa_orbit=0.02,
        k_body=0.03,
        beta_curv=0.010,
        k_pull=0.015,
        tau_touch=0.025,
        chi_mu=0.010,
        k_dist=0.12,
        kappa_geom=0.15,
        k_stein=0.015,
        kappa_ap=0.020,
    ),
    "ceramic_steel": MagnetPropertiesConfig(
        k_core=0.035,
        f_core=5500.0,
        k_skin=0.025,
        f_skin=5500.0,
        lambda_L=0.025,
        k_emf=0.025,
        eta_hyst=0.030,
        alpha=0.15,
        alpha3=0.055,
        k_sag=0.045,
        vsat=1.28,
        k_eddy=0.060,
        kappa_orbit=0.030,
        k_body=0.035,
        beta_curv=0.015,
        k_pull=0.018,
        tau_touch=0.028,
        chi_mu=0.018,
        k_dist=0.14,
        kappa_geom=0.16,
        k_stein=0.020,
        kappa_ap=0.025,
    ),
    "ceramic_alnico_hybrid": MagnetPropertiesConfig(
        k_core=0.05,
        f_core=4500.0,
        k_skin=0.05,
        f_skin=4500.0,
        lambda_L=0.03,
        k_emf=0.03,
        eta_hyst=0.04,
        alpha=0.18,
        alpha3=0.07,
        k_sag=0.05,
        vsat=1.04,
        k_eddy=0.08,
        kappa_orbit=0.04,
        k_body=0.05,
        beta_curv=0.025,
        k_pull=0.050,
        tau_touch=0.035,
        chi_mu=0.020,
        k_dist=0.15,
        kappa_geom=0.18,
        k_stein=0.025,
        kappa_ap=0.030,
    ),
    "neodymium": MagnetPropertiesConfig(
        k_core=0.01,
        f_core=8500.0,
        k_skin=0.02,
        f_skin=8000.0,
        lambda_L=0.01,
        k_emf=0.01,
        eta_hyst=0.01,
        alpha=0.08,
        alpha3=0.02,
        k_sag=0.01,
        vsat=1.80,
        k_eddy=0.01,
        kappa_orbit=0.01,
        k_body=0.02,
        beta_curv=0.005,
        k_pull=0.010,
        tau_touch=0.015,
        chi_mu=0.005,
        k_dist=0.10,
        kappa_geom=0.10,
        k_stein=0.008,
        kappa_ap=0.010,
    ),
    "piezo": MagnetPropertiesConfig(
        k_core=0.00,
        f_core=0.0,
        k_skin=0.00,
        f_skin=0.0,
        lambda_L=0.00,
        k_emf=0.00,
        eta_hyst=0.00,
        alpha=0.00,
        alpha3=0.00,
        k_sag=0.00,
        vsat=2.00,
        k_eddy=0.00,
        kappa_orbit=0.00,
        k_body=0.00,
        beta_curv=0.000,
        k_pull=0.000,
        tau_touch=0.000,
        chi_mu=0.000,
        k_dist=0.00,
        kappa_geom=0.00,
        k_stein=0.000,
        kappa_ap=0.000,
    ),
    "active": MagnetPropertiesConfig(
        k_core=0.00,
        f_core=0.0,
        k_skin=0.00,
        f_skin=0.0,
        lambda_L=0.00,
        k_emf=0.00,
        eta_hyst=0.00,
        alpha=0.00,
        alpha3=0.00,
        k_sag=0.00,
        vsat=2.40,
        k_eddy=0.00,
        kappa_orbit=0.00,
        k_body=0.00,
        beta_curv=0.000,
        k_pull=0.000,
        tau_touch=0.000,
        chi_mu=0.000,
        k_dist=0.00,
        kappa_geom=0.00,
        k_stein=0.000,
        kappa_ap=0.010,
    ),
    "ideal": MagnetPropertiesConfig(
        k_core=0.00,
        f_core=0.0,
        k_skin=0.00,
        f_skin=0.0,
        lambda_L=0.00,
        k_emf=0.00,
        eta_hyst=0.00,
        alpha=0.00,
        alpha3=0.00,
        k_sag=0.00,
        vsat=20.00,
        k_eddy=0.00,
        kappa_orbit=0.00,
        k_body=0.00,
        beta_curv=0.000,
        k_pull=0.000,
        tau_touch=0.000,
        chi_mu=0.000,
        k_dist=0.00,
        kappa_geom=0.00,
        k_stein=0.000,
        kappa_ap=0.000,
    ),
}
MAGNET_PROPERTIES["hybrid"] = MAGNET_PROPERTIES["ceramic_alnico_hybrid"]
MAGNET_PROPERTIES["ceramic_and_steel"] = MAGNET_PROPERTIES["ceramic_steel"]
MAGNET_PROPERTIES["ceramic/steel"] = MAGNET_PROPERTIES["ceramic_steel"]
MAGNET_PROPERTIES["cs"] = MAGNET_PROPERTIES["ceramic_steel"]
MAGNET_PROPERTIES["ideal_passive"] = MAGNET_PROPERTIES["ideal"]
MAGNET_PROPERTIES["linear"] = MAGNET_PROPERTIES["ideal"]


_AUDIO10_GAMMA: float = 4.394449154672439  # ln(81) = 2 * ln(9) -> 10% at 50% rotation
_AUDIO10_DENOM: float = 80.0
_AUDIO15_GAMMA: float = 3.4689389547514337  # 2 * ln(1/0.15 - 1) -> 15% at 50% rotation
_AUDIO15_DENOM: float = math.expm1(_AUDIO15_GAMMA)


def eval_pot_taper(pos: float, taper: str = "audio") -> float:
    """
    Evaluates potentiometer electrical resistance fraction (0.0 to 1.0) given mechanical wiper rotation pos (0.0 to 1.0).
    Supported tapers:
      - 'linear': f(theta) = theta
      - 'audio' / 'audio10': Standard CTS 10% audio taper (f(0.5) = 0.10).
      - 'audio15': Standard Bourns 15% audio taper (f(0.5) = 0.15).
      - 'reverse_audio': Standard reverse log taper.
      - 'mn_blend': Bourns MN blend pot taper.
    Satisfies Guardrail 5.2: C^inf smooth, strictly monotonic, zero slope kinks, exact (0,0) and (1,1) endpoints.
    Formula: f(theta; gamma) = (exp(gamma * theta) - 1.0) / (exp(gamma) - 1.0)
    where gamma = 2 * ln(1/k - 1).
    """
    theta = min(max(float(pos), 0.0), 1.0)
    if theta <= 0.0:
        return 0.0
    if theta >= 1.0:
        return 1.0

    t = (
        taper.lower().strip()
        if isinstance(taper, str) and taper.lower().strip() not in ("", "none")
        else "audio"
    )
    if t == "linear":
        return theta
    elif t in ("audio", "audio10", "audio_10"):
        return min(max(math.expm1(_AUDIO10_GAMMA * theta) / _AUDIO10_DENOM, 0.0), 1.0)
    elif t in ("audio15", "audio_15"):
        return min(max(math.expm1(_AUDIO15_GAMMA * theta) / _AUDIO15_DENOM, 0.0), 1.0)
    elif t == "reverse_audio":
        return min(max(1.0 - math.expm1(_AUDIO10_GAMMA * (1.0 - theta)) / _AUDIO10_DENOM, 0.0), 1.0)
    elif t == "mn_blend":
        return theta
    else:
        raise ValueError(
            f"Unknown pot taper '{taper}'. Supported tapers: 'audio', 'audio10', 'audio15', 'linear', 'reverse_audio', 'mn_blend'."
        )


