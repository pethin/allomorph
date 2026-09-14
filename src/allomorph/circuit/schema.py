"""
Allomorph Circuit - Pydantic Configuration Schemas.

Defines formal, strictly-typed Pydantic v2 schemas for RLC circuit branches,
potentiometer controls, dynamic magnetic metallurgy, saturation, and simulation configurations.
"""

from pathlib import Path
from typing import Any, Literal, Self

from pydantic import Field, model_validator

from allomorph.base import AllomorphBaseModel, SpiceFloat

__all__ = [
    "CircuitBranchConfig",
    "CircuitConfig",
    "CircuitMetricsRecord",
    "HarnessControls",
    "MagnetPropertiesConfig",
    "SaturationConfig",
    "SimulationConfig",
]


class CircuitBranchConfig(AllomorphBaseModel):
    """RLC parameters for a single pickup coil branch in dual-branch circuits."""

    L: SpiceFloat = Field(..., gt=0.0)
    L_core: SpiceFloat | None = None
    R_core: SpiceFloat | None = None
    Rdc: SpiceFloat = Field(..., gt=0.0)
    Reddy: SpiceFloat = Field(..., gt=0.0)
    Ccoil: SpiceFloat = Field(..., ge=0.0)
    vsat: float | None = None


class CircuitConfig(AllomorphBaseModel):
    """
    SPICE RLC circuit configuration for passive and active pickups.
    Models single-branch or dual-branch (neck/bridge) pickups with volume/tone pot tapers.
    """

    topology: str = "single"
    L: SpiceFloat | None = None
    L_core: SpiceFloat | None = None
    R_core: SpiceFloat | None = None
    Rdc: SpiceFloat | None = None
    Reddy: SpiceFloat | None = None
    Ccoil: SpiceFloat | None = None
    vsat: float | None = None
    vsat_n: float | None = None
    vsat_b: float | None = None
    L_b: SpiceFloat | None = None
    L_core_b: SpiceFloat | None = None
    R_core_b: SpiceFloat | None = None
    Rdc_b: SpiceFloat | None = None
    Reddy_b: SpiceFloat | None = None
    Ccoil_b: SpiceFloat | None = None
    Rtop: SpiceFloat | None = None
    Rbot: SpiceFloat | None = None
    Rvol: SpiceFloat | None = None
    Rtone: SpiceFloat | None = None
    Ctone: SpiceFloat | None = None
    Crick: SpiceFloat | None = None
    series_hpf_cap: SpiceFloat | None = None
    series_hpf_cap_nf: SpiceFloat | None = None
    Ctb: SpiceFloat | None = None
    Rtb_par: SpiceFloat | None = None
    Rtb_ser: SpiceFloat | None = None
    Rpot_n: SpiceFloat | None = None
    Rpot_b: SpiceFloat | None = None
    active: bool | None = None
    has_active_buffer: bool | None = None
    preamp: str | None = None
    preamp_type: str | None = None
    preamp_preset: str | None = None
    preamp_gain: float | None = None
    R_preamp_in: SpiceFloat | None = None
    C_preamp_in: SpiceFloat | None = None
    Rin: SpiceFloat | None = None
    Cin: SpiceFloat | None = None
    R_out: SpiceFloat | None = None
    Rout: SpiceFloat | None = None
    no_eq: bool | None = None
    preamp_bands: list[Any] | None = None
    Ccable: SpiceFloat | None = None
    tan_delta: float | None = None
    tan_delta_coil: float | None = None
    Ranagram: SpiceFloat | None = None
    Canagram: SpiceFloat | None = None
    k_mutual: float | None = None
    C_mutual: SpiceFloat | None = None
    alpha_dielectric_tone: float | None = None
    alpha_tone: float | None = None
    alpha_dielectric_cable: float | None = None
    alpha_cable: float | None = None
    chi_mu: float | None = None
    chi_mu_b: float | None = None
    omega_mu: float | None = None
    k_dist: float | None = None
    k_dist_b: float | None = None
    omega_dist: float | None = None
    k_skin: float | None = None
    f_skin: float | None = None
    k_skin_b: float | None = None
    f_skin_b: float | None = None
    Rblend: SpiceFloat | None = None
    Rblend_total: SpiceFloat | None = None
    vol_pos: float | None = None
    tone_pos: float | None = None
    blend_pos: float | None = None
    pot_taper: str | None = None
    neck: CircuitBranchConfig | None = None
    bridge: CircuitBranchConfig | None = None

    @model_validator(mode="after")
    def validate_circuit_branches(self) -> Self:
        if (
            self.topology in ("parallel", "series")
            and self.neck is None
            and self.bridge is None
            and (self.L is None or self.Rdc is None or self.Reddy is None)
        ):
            raise ValueError(
                f"Multi-coil circuit with topology '{self.topology}' must specify either "
                "'neck' and 'bridge' branches or top-level L/Rdc/Reddy parameters."
            )
        return self


class HarnessControls(AllomorphBaseModel):
    """Interactive volume, tone, and blend potentiometer wiper state with cable load."""

    vol_pos: float = Field(default=1.0, ge=0.0, le=1.0)
    tone_pos: float = Field(default=1.0, ge=0.0, le=1.0)
    blend_pos: float = Field(default=0.5, ge=0.0, le=1.0)
    pot_taper: Literal["audio", "audio10", "audio15", "linear", "reverse_audio", "mn_blend"] = (
        "audio"
    )
    cable_pf: float = Field(default=750.0, ge=0.0, le=20000.0)


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
    slew_limit: bool = True
    f_slew: float = Field(default=16000.0, gt=0.0)
    oversample: Literal[1, 2, 4] = 2
    displacement_weighting: bool = True
    magnet_drag: bool = True
    mix: float = Field(default=1.0, ge=0.0, le=1.0)


class SimulationConfig(AllomorphBaseModel):
    """Configuration for native Virtual Analog circuit simulation."""

    input_wav: Path | str | None = None
    output_wav: Path | str | None = None
    instrument: str | None = "30in"
    pickup: str | None = None
    cir_path: Path | str | None = None
    normalize: Literal["auto", "peak", "rms", "none"] = "auto"
    target_dbfs: float | None = None
    oversample: int = Field(default=2, ge=1)
    displacement_weighting: bool = True
    magnet_drag: bool = True
    alpha: float | None = None
    alpha3: float | None = None
    eta_hyst: float | None = None
    k_sag: float | None = None
    k_eddy: float | None = None
    kappa_orbit: float | None = None
    beta_curv: float | None = None
    k_pull: float | None = None
    tau_touch: float | None = None
    kappa_geom: float | None = None
    k_stein: float | None = None
    k_emf: float | None = None
    lambda_L: float | None = None
    vol_pos: float | None = Field(default=None, ge=0.0, le=1.0)
    tone_pos: float | None = Field(default=None, ge=0.0, le=1.0)
    blend_pos: float | None = Field(default=None, ge=0.0, le=1.0)
    pot_taper: (
        Literal["audio", "audio10", "audio15", "linear", "reverse_audio", "mn_blend"] | None
    ) = None
    cable_pf: float | None = Field(default=None, ge=0.0, le=20000.0)
    slew_limit: bool = True
    f_slew: float = 16000.0
    noise_dither: bool = True
    eddy_diffusion: bool = True
    dc_block: bool = True
    max_samples: int | None = None
    skip_identity: bool = False
    harness_controls: HarnessControls | None = None
    saturation_config: SaturationConfig | None = None

    def to_sim_kwargs(self) -> dict[str, Any]:
        """Converts configuration into a keyword arguments dictionary for simulation execution."""
        h = self.harness_controls
        s = self.saturation_config
        return {
            "input_wav": self.input_wav,
            "output_wav": self.output_wav,
            "instrument": self.instrument,
            "pickup": self.pickup,
            "cir_path": self.cir_path,
            "normalize": self.normalize,
            "target_dbfs": self.target_dbfs,
            "oversample": self.oversample if s is None else s.oversample,
            "displacement_weighting": (
                self.displacement_weighting if not s else s.displacement_weighting
            ),
            "magnet_drag": self.magnet_drag if not s else s.magnet_drag,
            "alpha": self.alpha if self.alpha is not None else (s.alpha if s else None),
            "alpha3": self.alpha3 if self.alpha3 is not None else (s.alpha3 if s else None),
            "eta_hyst": self.eta_hyst if self.eta_hyst is not None else (s.eta_hyst if s else None),
            "k_sag": self.k_sag if self.k_sag is not None else (s.k_sag if s else None),
            "k_eddy": self.k_eddy if self.k_eddy is not None else (s.k_eddy if s else None),
            "kappa_orbit": (
                self.kappa_orbit if self.kappa_orbit is not None else (s.kappa_orbit if s else None)
            ),
            "beta_curv": (
                self.beta_curv if self.beta_curv is not None else (s.beta_curv if s else None)
            ),
            "k_pull": self.k_pull if self.k_pull is not None else (s.k_pull if s else None),
            "tau_touch": (
                self.tau_touch if self.tau_touch is not None else (s.tau_touch if s else None)
            ),
            "kappa_geom": (
                self.kappa_geom if self.kappa_geom is not None else (s.kappa_geom if s else None)
            ),
            "k_stein": self.k_stein if self.k_stein is not None else (s.k_stein if s else None),
            "k_emf": self.k_emf if self.k_emf is not None else (s.k_emf if s else None),
            "lambda_L": self.lambda_L if self.lambda_L is not None else (s.lambda_L if s else None),
            "vol_pos": self.vol_pos if self.vol_pos is not None else (h.vol_pos if h else None),
            "tone_pos": self.tone_pos if self.tone_pos is not None else (h.tone_pos if h else None),
            "blend_pos": self.blend_pos
            if self.blend_pos is not None
            else (h.blend_pos if h else None),
            "pot_taper": self.pot_taper
            if self.pot_taper is not None
            else (h.pot_taper if h else None),
            "cable_pf": self.cable_pf if self.cable_pf is not None else (h.cable_pf if h else None),
            "slew_limit": self.slew_limit if not s else s.slew_limit,
            "f_slew": self.f_slew if not s else s.f_slew,
            "noise_dither": self.noise_dither,
            "eddy_diffusion": self.eddy_diffusion,
            "dc_block": self.dc_block,
            "max_samples": self.max_samples,
            "skip_identity": self.skip_identity,
            "vsat": s.vsat if s else None,
        }


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
