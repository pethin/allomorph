"""
Allomorph - Circuit Model & Declarative Netlist Parser
Parses Allomorph declarative circuit configurations into structured CircuitModel instances
with engineering unit suffixes, continuous pot tapers, and wiper positioning.
"""

import math
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
from pydantic import Field, field_validator

from allomorph.base import AllomorphBaseModel, SpiceFloat, parse_spice_unit
from allomorph.circuit.schema import CircuitConfig, MagnetPropertiesConfig
from allomorph.config.schema import PickupConfig, PreampBandConfig, VoiceConfig

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
        vsat=0.50,
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
        vsat=0.45,
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
        vsat=0.48,
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
        vsat=0.70,
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
        vsat=0.52,
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
        vsat=0.90,
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
        vsat=1.00,
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
        vsat=1.20,
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
        vsat=10.00,
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
    ),
}
MAGNET_PROPERTIES["hybrid"] = MAGNET_PROPERTIES["ceramic_alnico_hybrid"]
MAGNET_PROPERTIES["ideal_passive"] = MAGNET_PROPERTIES["ideal"]
MAGNET_PROPERTIES["canonical_ideal"] = MAGNET_PROPERTIES["ideal"]
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
    t = (
        taper.lower().strip()
        if isinstance(taper, str) and taper.lower().strip() not in ("", "none")
        else "audio"
    )
    if t == "linear":
        return theta
    elif t in ("audio", "audio10"):
        return math.expm1(_AUDIO10_GAMMA * theta) / _AUDIO10_DENOM
    elif t == "audio15":
        return math.expm1(_AUDIO15_GAMMA * theta) / _AUDIO15_DENOM
    elif t == "reverse_audio":
        return 1.0 - math.expm1(_AUDIO10_GAMMA * (1.0 - theta)) / _AUDIO10_DENOM
    elif t == "mn_blend":
        return theta
    else:
        raise ValueError(
            f"Unknown pot taper '{taper}'. Supported tapers: 'audio', 'audio10', 'audio15', 'linear', 'reverse_audio', 'mn_blend'."
        )


class CircuitModel(AllomorphBaseModel):
    """Represents a parsed RLC guitar circuit digital twin."""

    topology: Literal["single", "parallel", "series"] = "single"
    vsat: float = 0.50
    vsat_n: float = 0.50
    vsat_b: float = 0.50

    # Branch parameters (single or neck)
    L: SpiceFloat = Field(default=4.8, gt=0.0)
    L_core: SpiceFloat = Field(default=0.0, ge=0.0)
    R_core: SpiceFloat = Field(default=0.0, ge=0.0)
    Rdc: SpiceFloat = Field(default=9500.0, gt=0.0)
    Reddy: SpiceFloat = Field(default=110000.0, gt=0.0)
    Ccoil: SpiceFloat = Field(default=80e-12, ge=0.0)

    # Bridge branch (for parallel or series)
    L_b: SpiceFloat = Field(default=3.6, gt=0.0)
    L_core_b: SpiceFloat = Field(default=0.0, ge=0.0)
    R_core_b: SpiceFloat = Field(default=0.0, ge=0.0)
    Rdc_b: SpiceFloat = Field(default=7800.0, gt=0.0)
    Reddy_b: SpiceFloat = Field(default=125000.0, gt=0.0)
    Ccoil_b: SpiceFloat = Field(default=70e-12, ge=0.0)

    # Optional tone / HPF
    Ctone: SpiceFloat = Field(default=0.0, ge=0.0)
    Rtone: SpiceFloat = Field(default=0.0, ge=0.0)
    Crick: SpiceFloat = Field(default=0.0, ge=0.0)

    # Volume pot & treble bleed (disabled by default unless specified in netlist)
    Rtop: SpiceFloat = Field(default=10.0, ge=0.0)
    Rbot: SpiceFloat = Field(default=500000.0, ge=0.0)
    Ctb: SpiceFloat = Field(default=0.0, ge=0.0)
    Rtb_par: SpiceFloat = Field(default=0.0, ge=0.0)
    Rtb_ser: SpiceFloat = Field(default=0.0, ge=0.0)

    # Active preamp buffer & EQ
    has_active_buffer: bool = False
    preamp_type: str = "none"  # "sadowsky_2band", "stingray_2band", or "none"
    preamp_bands: list[PreampBandConfig] | None = None
    preamp_gain: float = 1.0
    preamp_gain_db: float = 0.0
    R_preamp_in: SpiceFloat = Field(default=1.0e6, gt=0.0)
    C_preamp_in: SpiceFloat = Field(default=25e-12, ge=0.0)
    R_out: SpiceFloat = Field(default=100.0, ge=0.0)

    # Transparent zero-EQ mode
    no_eq: bool = False

    # Cable & pedalboard load
    Ccable: SpiceFloat = Field(default=750e-12, ge=0.0)
    tan_delta: float = 0.025
    tan_delta_coil: float = 0.025  # Enameled magnet wire dissipation factor
    Ranagram: SpiceFloat = Field(default=1.0e6, gt=0.0)
    Canagram: SpiceFloat = Field(default=30e-12, ge=0.0)

    # Individual pickup volume pot decoupling (e.g. rolled-off neck pot for Jaco growl)
    Rpot_n: SpiceFloat = Field(default=0.0, ge=0.0)
    Rpot_b: SpiceFloat = Field(default=0.0, ge=0.0)

    # Dielectric absorption (Cole-Davidson fractional-order relaxation)
    alpha_dielectric_tone: float = 0.988
    alpha_dielectric_cable: float = 0.994

    # Inter-coil mutual inductive & capacitive coupling for multi-pickup configurations
    k_mutual: float = 0.0
    C_mutual: SpiceFloat = Field(default=0.0, ge=0.0)

    # Complex magnetic permeability dispersion (Jordan after-effect)
    chi_mu: float = 0.0
    chi_mu_b: float = 0.0
    omega_mu: float = 2.0 * math.pi * 1200.0

    # Distributed inter-winding transmission line capacitance
    k_dist: float = 0.0
    k_dist_b: float = 0.0
    omega_dist: float = 2.0 * math.pi * 10000.0

    # Solid core eddy skin-effect fractional dispersion
    k_skin: float = 0.0
    f_skin: float = 3200.0
    k_skin_b: float = 0.0
    f_skin_b: float = 3200.0

    # Potentiometer wiper positions (1.0 = full open/bright baseline, 0.5 = center for blend)
    vol_pos: float = Field(default=1.0, ge=0.0, le=1.0)
    tone_pos: float = Field(default=1.0, ge=0.0, le=1.0)
    blend_pos: float = Field(default=0.5, ge=0.0, le=1.0)
    pot_taper: str = "audio"
    Rvol_total: SpiceFloat = Field(default=500000.0, ge=0.0)
    Rtone_total: SpiceFloat = Field(default=250000.0, ge=0.0)
    Rblend_total: SpiceFloat = Field(default=250000.0, ge=0.0)

    Rtop_default: SpiceFloat = Field(default=10.0, ge=0.0)
    Rbot_default: SpiceFloat = Field(default=500000.0, ge=0.0)
    Rtone_default: SpiceFloat = Field(default=0.0, ge=0.0)
    Rpot_n_default: SpiceFloat = Field(default=0.0, ge=0.0)
    Rpot_b_default: SpiceFloat = Field(default=0.0, ge=0.0)

    @field_validator("topology", mode="before")
    @classmethod
    def _validate_topology(cls, v: Any) -> str:
        if isinstance(v, str):
            v_clean = v.lower().strip()
            if v_clean in ("single", "parallel", "series"):
                return v_clean
        raise ValueError(f"Invalid topology '{v}'. Must be 'single', 'parallel', or 'series'.")

    def apply_pot_positions(
        self,
        vol_pos: float | None = None,
        tone_pos: float | None = None,
        blend_pos: float | None = None,
        pot_taper: str | None = None,
    ) -> None:
        """
        Dynamically positions Volume, Tone, and Blend pot wipers.
        - vol_pos: 0.0 (muted) to 1.0 (full open). Splits volume pot into series Rtop and shunt Rbot.
        - tone_pos: 0.0 (dark / max cap shunting) to 1.0 (open / bright). Scales series Rtone.
        - blend_pos: 0.0 (Neck 100%, Bridge muted) to 0.5 (Center detent 100%/100%) to 1.0 (Neck muted, Bridge 100%).
        - pot_taper: 'audio' (10% CTS), 'audio15' (15% Bourns), or 'linear'. Default is self.pot_taper or 'audio'.
        When wipers are at default positions (vol=1.0, tone=1.0, blend=0.5), preserves exact netlist defaults.
        """
        taper = (
            pot_taper.lower().strip()
            if pot_taper is not None and pot_taper.lower().strip() not in ("", "none")
            else self.pot_taper
        )
        if taper in ("", "none", None):
            taper = "audio"

        if vol_pos is not None:
            self.vol_pos = float(np.clip(vol_pos, 0.0, 1.0))
            if self.vol_pos >= 0.9999:
                self.Rtop = self.Rtop_default
                self.Rbot = self.Rbot_default
            else:
                eff_vol = eval_pot_taper(self.vol_pos, taper)
                r_total = self.Rvol_total if self.Rvol_total > 0.0 else (self.Rtop + self.Rbot)
                self.Rtop = max(r_total * (1.0 - eff_vol), self.Rtop_default)
                self.Rbot = max(r_total * eff_vol, 1.0)

        if tone_pos is not None:
            self.tone_pos = float(np.clip(tone_pos, 0.0, 1.0))
            if self.tone_pos >= 0.9999:
                self.Rtone = self.Rtone_default
            else:
                eff_tone = eval_pot_taper(self.tone_pos, taper)
                r_tone_tot = (
                    self.Rtone_total
                    if self.Rtone_total > 0.0
                    else (self.Rtone if self.Rtone > 0.0 else 250000.0)
                )
                self.Rtone = max(r_tone_tot * eff_tone, 0.0)

        if blend_pos is not None:
            self.blend_pos = float(np.clip(blend_pos, 0.0, 1.0))
            r_blend = self.Rblend_total if self.Rblend_total > 0.0 else 250000.0
            if abs(self.blend_pos - 0.5) < 1e-4:
                # Center detent: unattenuated 100%/100% (0 dB insertion loss)
                self.Rpot_n = self.Rpot_n_default
                self.Rpot_b = self.Rpot_b_default
            elif self.blend_pos < 0.5:
                # Turning toward Neck (Neck 100%, Bridge attenuated)
                self.Rpot_n = self.Rpot_n_default
                norm_atten = (0.5 - self.blend_pos) / 0.5  # 0.0 at center to 1.0 at full Neck
                eff_atten = eval_pot_taper(norm_atten, taper)
                self.Rpot_b = self.Rpot_b_default + r_blend * eff_atten
            else:
                # Turning toward Bridge (Bridge 100%, Neck attenuated)
                self.Rpot_b = self.Rpot_b_default
                norm_atten = (self.blend_pos - 0.5) / 0.5  # 0.0 at center to 1.0 at full Bridge
                eff_atten = eval_pot_taper(norm_atten, taper)
                self.Rpot_n = self.Rpot_n_default + r_blend * eff_atten

    @classmethod
    def from_circuit_config(cls, cfg: CircuitConfig) -> CircuitModel:
        """Creates a CircuitModel directly from a validated CircuitConfig model."""
        model = cls()
        model.topology = cast(
            Literal["single", "parallel", "series"], str(cfg.topology or "single").lower()
        )

        # Dynamic saturation limit
        model.vsat = cfg.vsat if cfg.vsat is not None else 0.50
        model.vsat_n = (
            cfg.neck.vsat
            if (cfg.neck is not None and cfg.neck.vsat is not None)
            else (cfg.vsat_n if cfg.vsat_n is not None else model.vsat)
        )
        model.vsat_b = (
            cfg.bridge.vsat
            if (cfg.bridge is not None and cfg.bridge.vsat is not None)
            else (cfg.vsat_b if cfg.vsat_b is not None else model.vsat)
        )

        # Neck / single pickup branch
        if cfg.neck is not None:
            model.L = cfg.neck.L
            if cfg.neck.L_core is not None:
                model.L_core = cfg.neck.L_core
            if cfg.neck.R_core is not None:
                model.R_core = cfg.neck.R_core
            model.Rdc = cfg.neck.Rdc
            model.Reddy = cfg.neck.Reddy
            model.Ccoil = cfg.neck.Ccoil
        else:
            if cfg.L is not None:
                model.L = cfg.L
            if cfg.L_core is not None:
                model.L_core = cfg.L_core
            if cfg.R_core is not None:
                model.R_core = cfg.R_core
            if cfg.Rdc is not None:
                model.Rdc = cfg.Rdc
            if cfg.Reddy is not None:
                model.Reddy = cfg.Reddy
            if cfg.Ccoil is not None:
                model.Ccoil = cfg.Ccoil

        # Bridge pickup branch
        if cfg.bridge is not None:
            model.L_b = cfg.bridge.L
            if cfg.bridge.L_core is not None:
                model.L_core_b = cfg.bridge.L_core
            if cfg.bridge.R_core is not None:
                model.R_core_b = cfg.bridge.R_core
            model.Rdc_b = cfg.bridge.Rdc
            model.Reddy_b = cfg.bridge.Reddy
            model.Ccoil_b = cfg.bridge.Ccoil
        else:
            if cfg.L_b is not None:
                model.L_b = cfg.L_b
            if cfg.L_core_b is not None:
                model.L_core_b = cfg.L_core_b
            if cfg.R_core_b is not None:
                model.R_core_b = cfg.R_core_b
            if cfg.Rdc_b is not None:
                model.Rdc_b = cfg.Rdc_b
            if cfg.Reddy_b is not None:
                model.Reddy_b = cfg.Reddy_b
            if cfg.Ccoil_b is not None:
                model.Ccoil_b = cfg.Ccoil_b

        # Volume Pot
        if cfg.Rtop is not None or cfg.Rbot is not None:
            model.Rtop = cfg.Rtop if cfg.Rtop is not None else 10.0
            model.Rbot = cfg.Rbot if cfg.Rbot is not None else 500000.0
        elif cfg.Rvol is not None:
            model.Rtop = 10.0
            model.Rbot = cfg.Rvol
        else:
            model.Rtop = 10.0
            model.Rbot = 500000.0

        # Tone Pot
        if cfg.Rtone is not None:
            model.Rtone = cfg.Rtone
        if cfg.Ctone is not None:
            model.Ctone = cfg.Ctone

        # HPF
        if cfg.Crick is not None:
            model.Crick = cfg.Crick
        elif cfg.series_hpf_cap is not None:
            model.Crick = cfg.series_hpf_cap
        elif cfg.series_hpf_cap_nf is not None:
            model.Crick = cfg.series_hpf_cap_nf * 1e-9

        # Treble bleed
        if cfg.Ctb is not None:
            model.Ctb = cfg.Ctb
        if cfg.Rtb_par is not None:
            model.Rtb_par = cfg.Rtb_par
        if cfg.Rtb_ser is not None:
            model.Rtb_ser = cfg.Rtb_ser

        # Individual pickup volume pot decoupling
        if cfg.Rpot_n is not None:
            model.Rpot_n = cfg.Rpot_n
        if cfg.Rpot_b is not None:
            model.Rpot_b = cfg.Rpot_b

        # Active preamp & buffer
        active = bool(cfg.active or cfg.has_active_buffer)
        preamp_val = cfg.preamp or cfg.preamp_type or "none"
        if preamp_val != "none" or active:
            model.has_active_buffer = True
            model.preamp_type = preamp_val

        if cfg.preamp_gain is not None:
            model.preamp_gain = cfg.preamp_gain
        r_pre = cfg.R_preamp_in if cfg.R_preamp_in is not None else cfg.Rin
        if r_pre is not None:
            model.R_preamp_in = r_pre
        c_pre = cfg.C_preamp_in if cfg.C_preamp_in is not None else cfg.Cin
        if c_pre is not None:
            model.C_preamp_in = c_pre
        r_out = cfg.R_out if cfg.R_out is not None else cfg.Rout
        if r_out is not None:
            model.R_out = r_out
        if cfg.no_eq is not None:
            model.no_eq = bool(cfg.no_eq)
        if cfg.preamp_bands:
            model.preamp_bands = [
                b if isinstance(b, PreampBandConfig) else PreampBandConfig.model_validate(b)
                for b in cfg.preamp_bands
            ]

        # Cable & load
        if cfg.Ccable is not None:
            model.Ccable = cfg.Ccable
        if cfg.tan_delta is not None:
            model.tan_delta = cfg.tan_delta
        if cfg.tan_delta_coil is not None:
            model.tan_delta_coil = cfg.tan_delta_coil
        if cfg.Ranagram is not None:
            model.Ranagram = cfg.Ranagram
        if cfg.Canagram is not None:
            model.Canagram = cfg.Canagram

        # Coupling & Dielectrics
        if model.topology in ("parallel", "series"):
            model.k_mutual = cfg.k_mutual if cfg.k_mutual is not None else 0.05
            model.C_mutual = cfg.C_mutual if cfg.C_mutual is not None else 20e-12
        else:
            model.k_mutual = cfg.k_mutual if cfg.k_mutual is not None else 0.0
            model.C_mutual = cfg.C_mutual if cfg.C_mutual is not None else 0.0

        a_tone = (
            cfg.alpha_dielectric_tone if cfg.alpha_dielectric_tone is not None else cfg.alpha_tone
        )
        if a_tone is not None:
            model.alpha_dielectric_tone = a_tone

        a_cable = (
            cfg.alpha_dielectric_cable
            if cfg.alpha_dielectric_cable is not None
            else cfg.alpha_cable
        )
        if a_cable is not None:
            model.alpha_dielectric_cable = a_cable

        # Jordan / skin-effect dispersion
        if cfg.chi_mu is not None:
            model.chi_mu = cfg.chi_mu
        if cfg.chi_mu_b is not None:
            model.chi_mu_b = cfg.chi_mu_b
        if cfg.omega_mu is not None:
            model.omega_mu = cfg.omega_mu
        if cfg.k_dist is not None:
            model.k_dist = cfg.k_dist
        if cfg.k_dist_b is not None:
            model.k_dist_b = cfg.k_dist_b
        if cfg.omega_dist is not None:
            model.omega_dist = cfg.omega_dist
        if cfg.k_skin is not None:
            model.k_skin = cfg.k_skin
        if cfg.f_skin is not None:
            model.f_skin = cfg.f_skin
        if cfg.k_skin_b is not None:
            model.k_skin_b = cfg.k_skin_b
        if cfg.f_skin_b is not None:
            model.f_skin_b = cfg.f_skin_b

        # Pot defaults
        model.Rvol_total = model.Rtop + model.Rbot
        model.Rtone_total = model.Rtone if model.Rtone > 0.0 else 250000.0
        r_blend = cfg.Rblend if cfg.Rblend is not None else cfg.Rblend_total
        model.Rblend_total = r_blend if r_blend is not None else 250000.0
        taper_cfg = cfg.pot_taper
        model.pot_taper = (
            str(taper_cfg).lower().strip()
            if taper_cfg is not None and str(taper_cfg).lower().strip() not in ("", "none")
            else "audio"
        )
        model.blend_pos = cfg.blend_pos if cfg.blend_pos is not None else 0.5
        model.Rtop_default = model.Rtop
        model.Rbot_default = model.Rbot
        model.Rtone_default = model.Rtone
        model.Rpot_n_default = model.Rpot_n
        model.Rpot_b_default = model.Rpot_b

        has_vol = cfg.vol_pos is not None
        has_tone = cfg.tone_pos is not None
        has_blend = cfg.blend_pos is not None
        if has_vol or has_tone or has_blend:
            model.apply_pot_positions(
                cfg.vol_pos,
                cfg.tone_pos,
                cfg.blend_pos,
                model.pot_taper,
            )

        return model

    @classmethod
    def from_dict(cls, cfg: dict[str, Any]) -> CircuitModel:
        """Creates a CircuitModel from a declarative configuration dictionary."""
        return cls.from_circuit_config(CircuitConfig.model_validate(cfg))


def load_circuit(
    source: CircuitModel | CircuitConfig | VoiceConfig | PickupConfig | str | Path,
) -> CircuitModel:
    """Loads a CircuitModel from a CircuitModel, CircuitConfig, VoiceConfig, PickupConfig, or file path (.toml), voice ID, or instrument ID."""
    if isinstance(source, CircuitModel):
        return source.model_copy()
    if isinstance(source, CircuitConfig):
        return CircuitModel.from_circuit_config(source)
    if isinstance(source, VoiceConfig):
        return CircuitModel.from_circuit_config(source.circuit)
    if isinstance(source, PickupConfig):
        if source.circuit is None:
            raise ValueError(f"Pickup '{source.name}' has no embedded circuit configuration")
        return CircuitModel.from_circuit_config(source.circuit)

    if isinstance(source, (str, Path)):
        p = Path(source)
        if p.exists() and p.is_file() and p.suffix == ".cir":
            raise ValueError(
                f"Legacy SPICE ASCII netlists (.cir) are deprecated and no longer supported. "
                f"Circuits must be defined as declarative TOML files or tables. "
                f"Attempted to load: {source}"
            )

        if p.exists() and p.is_file() and p.suffix == ".toml":
            import tomllib

            with open(p, "rb") as f:
                data = tomllib.load(f)
            circuit_data = data.get("circuit", data)
            return CircuitModel.from_circuit_config(CircuitConfig.model_validate(circuit_data))

        # Try relative to REPO_ROOT
        repo_rel = REPO_ROOT / source
        if repo_rel.exists() and repo_rel.is_file():
            return load_circuit(repo_rel)

        if p.suffix == ".cir" or str(source).endswith(".cir"):
            raise ValueError(
                f"Legacy SPICE ASCII netlists (.cir) are deprecated and no longer supported. "
                f"Circuits must be defined as declarative TOML files or tables. "
                f"Attempted to load: {source}"
            )

        # Check voices and instruments registry by string ID
        try:
            from allomorph.config.voices import VOICES

            if str(source) in VOICES:
                return load_circuit(VOICES[str(source)])
        except ImportError:
            pass

        try:
            from allomorph.config.instruments import INSTRUMENTS

            if str(source) in INSTRUMENTS:
                inst = INSTRUMENTS[str(source)]
                default_p = inst.default_pickup
                if default_p and default_p in inst.pickups:
                    p = inst.pickups[default_p]
                    if p.circuit is not None:
                        return load_circuit(p.circuit)
                for p in inst.pickups.values():
                    if p.circuit is not None:
                        return load_circuit(p.circuit)
        except ImportError:
            pass

    raise ValueError(f"Could not load circuit from: {source}")


def parse_netlist(
    source: CircuitModel | CircuitConfig | VoiceConfig | PickupConfig | str | Path,
) -> CircuitModel:
    """Parses a netlist or declarative circuit configuration into a CircuitModel."""
    return load_circuit(source)
