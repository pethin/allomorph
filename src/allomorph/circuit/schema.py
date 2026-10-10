"""
Allomorph Circuit - Pydantic Configuration Schemas.

Defines formal, strictly-typed Pydantic v2 schemas for RLC circuit branches,
potentiometer controls, dynamic magnetic metallurgy, saturation, and simulation configurations.
"""

from typing import Literal, Self

from pydantic import Field

from allomorph.base import AllomorphBaseModel

__all__ = [
    "CircuitMetricsRecord",
    "MagnetPropertiesConfig",
    "SaturationConfig",
]


class MagnetPropertiesConfig(AllomorphBaseModel):
    """Physical non-linear metallurgy and dynamic magnetic parameters."""

    k_core: float = 0.0
    f_core: float = 0.0
    k_skin: float = 0.0
    f_skin: float = 0.0
    lambda_L: float = 0.0
    k_emf: float = 0.0
    eta_hyst: float = 0.0
    alpha: float = 0.20
    alpha3: float = 0.08
    k_sag: float = 0.08
    vsat: float = 0.50
    k_eddy: float = 0.0
    kappa_orbit: float = 0.0
    k_body: float = 0.0
    beta_curv: float = 0.0
    k_pull: float = 0.0
    tau_touch: float = 0.0
    chi_mu: float = 0.0
    k_dist: float = 0.0
    kappa_geom: float = 0.0
    k_stein: float = 0.0
    kappa_ap: float = 0.0

    def diff(self, source: Self) -> MagnetPropertiesConfig:
        """
        Computes differential softening parameters satisfying Guardrail 5.4.1:
        Δparam = max(voice_param - source_param, 0.0).
        """
        return MagnetPropertiesConfig(
            k_core=max(self.k_core - source.k_core, 0.0),
            f_core=self.f_core,
            k_skin=max(self.k_skin - source.k_skin, 0.0),
            f_skin=self.f_skin,
            lambda_L=max(self.lambda_L - source.lambda_L, 0.0),
            k_emf=max(self.k_emf - source.k_emf, 0.0),
            eta_hyst=max(self.eta_hyst - source.eta_hyst, 0.0),
            alpha=max(self.alpha - source.alpha, 0.0),
            alpha3=max(self.alpha3 - source.alpha3, 0.0),
            k_sag=max(self.k_sag - source.k_sag, 0.0),
            vsat=self.vsat,
            k_eddy=max(self.k_eddy - source.k_eddy, 0.0),
            kappa_orbit=max(self.kappa_orbit - source.kappa_orbit, 0.0),
            k_body=max(self.k_body - source.k_body, 0.0),
            beta_curv=max(self.beta_curv - source.beta_curv, 0.0),
            k_pull=max(self.k_pull - source.k_pull, 0.0),
            tau_touch=max(self.tau_touch - source.tau_touch, 0.0),
            chi_mu=max(self.chi_mu - source.chi_mu, 0.0),
            k_dist=max(self.k_dist - source.k_dist, 0.0),
            kappa_geom=max(self.kappa_geom - source.kappa_geom, 0.0),
            k_stein=max(self.k_stein - source.k_stein, 0.0),
            kappa_ap=max(self.kappa_ap - source.kappa_ap, 0.0),
        )


class SaturationConfig(AllomorphBaseModel):
    """Consolidated configuration for oversampled dynamic magnetic saturation."""

    vsat: float = Field(default=0.50, gt=0.0)
    alpha: float = Field(default=0.20, ge=0.0)
    alpha3: float = Field(default=0.08, ge=0.0)
    eta_hyst: float = Field(default=0.0, ge=0.0, le=1.0)
    k_sag: float = Field(default=0.08, ge=0.0)
    k_eddy: float = Field(default=0.0, ge=0.0)
    kappa_orbit: float = Field(default=0.0, ge=0.0)
    beta_curv: float = Field(default=0.0, ge=0.0)
    k_pull: float = Field(default=0.0, ge=0.0)
    tau_touch: float = Field(default=0.0, ge=0.0)
    kappa_geom: float = Field(default=0.0, ge=0.0)
    k_stein: float = Field(default=0.0, ge=0.0)
    k_emf: float = Field(default=0.0, ge=0.0)
    lambda_L: float = Field(default=0.0, ge=0.0)
    k_core: float = Field(default=0.0, ge=0.0)
    kappa_ap: float = Field(default=0.0, ge=0.0)
    k_level: float = Field(default=0.0, ge=0.0, le=1.0)
    active_variant: Literal["x_series", "classic"] | None = None
    slew_limit: bool = True
    f_slew: float = Field(default=16000.0, gt=0.0)
    oversample: Literal[1, 2, 4] = 2
    displacement_weighting: bool = True
    magnet_drag: bool = True
    mix: float = Field(default=1.0, ge=0.0, le=1.0)


class CircuitMetricsRecord(AllomorphBaseModel):
    """Analytical resonance and bandwidth metrics for a single swept frequency curve."""

    param: str
    param_value: float
    label: str
    f_res_hz: float | None = None
    peak_db: float
    insertion_loss_db: float
    peak_boost_db: float
    q_loaded: float | None = None
    bandwidth_hz: float | None = None
    cutoff_3db_hz: float | None = None
    hf_slope_db_oct: float
