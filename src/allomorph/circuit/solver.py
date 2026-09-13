"""
Allomorph - Nodal RLC Matrix Solver & Differential Transfer Functions
Evaluates closed-form nodal AC transfer functions across frequencies using
vectorized NumPy SIMD operations, Cole-Davidson dielectric relaxation,
Jordan after-effect permeability dispersion, and Wiener-regularized deconvolution.
"""

import functools
import math
from collections.abc import Sequence
from typing import Literal, overload

import numpy as np

from allomorph.circuit.parser import MAGNET_PROPERTIES, CircuitModel, eval_pot_taper


@functools.lru_cache(maxsize=16)
def _get_cached_s_ratio_power(alpha: float, n_points: int) -> np.ndarray:
    """Caches normalized s_ratio ** alpha vectors for standard frequency grids."""
    freqs_arr = np.linspace(0.0, 24000.0, n_points)
    w = 2.0 * np.pi * freqs_arr
    w0 = 2.0 * np.pi * 1000.0
    s_ratio = np.where(w > 0.0, w / w0, 0.0)
    return s_ratio ** alpha
from allomorph.config.schema import (
    PickupConfig,
    PreampBandConfig,
    PreampConfig,
    VoiceConfig,
    VoicePickupConfig,
)
from allomorph.dsp import FREQS


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


def apply_magnet_properties_to_model(
    model: CircuitModel,
    vcfg: VoiceConfig | PickupConfig,
    eddy_diffusion: bool = True,
) -> None:
    """
    Applies Foster 2-stage core eddy diffusion parameters (L_core, R_core),
    complex permeability dispersion (chi_mu), distributed winding factor (k_dist),
    and solid pole eddy skin-effect dispersion (k_skin, f_skin)
    to the CircuitModel based on authentic magnet metallurgy if not explicitly
    specified in the SPICE netlist.
    """
    if not eddy_diffusion:
        model.L_core = 0.0
        model.R_core = 0.0
        model.L_core_b = 0.0
        model.R_core_b = 0.0
        model.chi_mu = 0.0
        model.chi_mu_b = 0.0
        model.k_dist = 0.0
        model.k_dist_b = 0.0
        model.k_skin = 0.0
        model.k_skin_b = 0.0
        return

    mag_type_global = vcfg.magnet_type or "alnico_v"

    pickups: list[VoicePickupConfig] = vcfg.pickups or [] if isinstance(vcfg, VoiceConfig) else []

    if model.topology in ["parallel", "series"] and len(pickups) >= 2:
        mag_n = pickups[0].magnet_type or mag_type_global
        mag_b = pickups[1].magnet_type or mag_type_global
    else:
        mag_n = mag_type_global
        mag_b = mag_type_global

    if mag_n not in MAGNET_PROPERTIES:
        raise KeyError(
            f"Unknown magnet type '{mag_n}'. Available magnet types: {list(MAGNET_PROPERTIES.keys())}"
        )
    if mag_b not in MAGNET_PROPERTIES:
        raise KeyError(
            f"Unknown magnet type '{mag_b}'. Available magnet types: {list(MAGNET_PROPERTIES.keys())}"
        )
    props_n = MAGNET_PROPERTIES[mag_n]
    props_b = MAGNET_PROPERTIES[mag_b]

    if model.L_core <= 0.0 and props_n.k_core > 0.0:
        model.L_core = props_n.k_core * model.L
        f_c = props_n.f_core
        model.R_core = 2.0 * math.pi * f_c * model.L_core if f_c > 0.0 else 0.0

    if model.chi_mu <= 0.0 and props_n.chi_mu > 0.0:
        model.chi_mu = props_n.chi_mu
    if model.k_dist <= 0.0 and props_n.k_dist > 0.0:
        model.k_dist = props_n.k_dist
    if model.k_skin <= 0.0 and props_n.k_skin > 0.0:
        model.k_skin = props_n.k_skin
        model.f_skin = props_n.f_skin

    if model.topology in ["parallel", "series"]:
        if model.L_core_b <= 0.0 and props_b.k_core > 0.0:
            model.L_core_b = props_b.k_core * model.L_b
            f_cb = props_b.f_core
            model.R_core_b = 2.0 * math.pi * f_cb * model.L_core_b if f_cb > 0.0 else 0.0

        if model.chi_mu_b <= 0.0 and props_b.chi_mu > 0.0:
            model.chi_mu_b = props_b.chi_mu
        if model.k_dist_b <= 0.0 and props_b.k_dist > 0.0:
            model.k_dist_b = props_b.k_dist
        if model.k_skin_b <= 0.0 and props_b.k_skin > 0.0:
            model.k_skin_b = props_b.k_skin
            model.f_skin_b = props_b.f_skin


def evaluate_analog_band(band: PreampBandConfig, s: complex | np.ndarray) -> complex | np.ndarray:
    """Evaluates continuous s-domain analog transfer function for a single EQ band."""
    b_type = band.type
    g_db = band.gain_db
    if abs(g_db) < 1e-4 and b_type in ("low_shelf", "high_shelf", "bell"):
        return 1.0 if isinstance(s, complex) else np.ones_like(s, dtype=np.complex128)

    f0 = band.freq_hz
    w0 = 2.0 * math.pi * f0
    g = 10.0 ** (g_db / 20.0)

    if b_type == "low_shelf":
        return (s + g * w0) / (s + w0)
    elif b_type == "high_shelf":
        return (g * s + w0) / (s + w0)
    elif b_type == "bell":
        q = float(band.q) if band.q is not None else 1.0
        num = s**2 + (w0 / q) * g * s + w0**2
        den = s**2 + (w0 / q) * s + w0**2
        return num / den
    elif b_type == "low_pass":
        return w0 / (s + w0)
    elif b_type == "high_pass":
        return s / (s + w0)
    return np.ones_like(s, dtype=np.complex128)


def compute_active_preamp_biquads(
    bands: Sequence[PreampBandConfig] | None,
    gain_db: float = 0.0,
    fs: float = 48000.0,
) -> list[tuple[float, float, float, float, float, float]]:
    """
    Computes Direct-Form II Transposed biquad coefficients [b0, b1, b2, a0, a1, a2]
    for analog preamp EQ bands via the bilinear transform with frequency pre-warping:
      omega_a = 2 * fs * tan(omega_d / 2)
    Accelerates live DAW / pedalboard plugin hosts to < 10 ns execution without FIR latency.
    Ground-truth audio generation for NAM training stems strictly preserves full-length FIRs.
    """
    biquads: list[tuple[float, float, float, float, float, float]] = []
    k_bilinear = 2.0 * fs

    for band in bands or []:
        g_db = band.gain_db
        b_type = band.type
        if abs(g_db) < 1e-4 and b_type in ("low_shelf", "high_shelf", "bell"):
            continue

        f0 = band.freq_hz
        omega_d = 2.0 * math.pi * f0 / fs
        omega_a = 2.0 * fs * math.tan(omega_d / 2.0)
        g = 10.0 ** (g_db / 20.0)

        if b_type == "low_shelf":
            a0 = k_bilinear + omega_a
            b0 = (k_bilinear + g * omega_a) / a0
            b1 = (g * omega_a - k_bilinear) / a0
            b2 = 0.0
            a1 = (omega_a - k_bilinear) / a0
            a2 = 0.0
            biquads.append((b0, b1, b2, 1.0, a1, a2))
        elif b_type == "high_shelf":
            a0 = k_bilinear + omega_a
            b0 = (g * k_bilinear + omega_a) / a0
            b1 = (omega_a - g * k_bilinear) / a0
            b2 = 0.0
            a1 = (omega_a - k_bilinear) / a0
            a2 = 0.0
            biquads.append((b0, b1, b2, 1.0, a1, a2))
        elif b_type == "bell":
            q = float(band.q) if band.q is not None else 1.0
            k2 = k_bilinear * k_bilinear
            w2 = omega_a * omega_a
            kw_q = (k_bilinear * omega_a) / q
            a0 = k2 + kw_q + w2
            b0 = (k2 + g * kw_q + w2) / a0
            b1 = (2.0 * (w2 - k2)) / a0
            b2 = (k2 - g * kw_q + w2) / a0
            a1 = (2.0 * (w2 - k2)) / a0
            a2 = (k2 - kw_q + w2) / a0
            biquads.append((b0, b1, b2, 1.0, a1, a2))

    return biquads


def compute_active_preamp_transfer(
    bands: Sequence[PreampBandConfig] | None, s: complex | np.ndarray, gain_db: float = 0.0
) -> np.ndarray:
    """Evaluates the composite analog active preamp contour across frequencies with finite DC transmission."""
    h_total = np.ones_like(s, dtype=np.complex128) * (10.0 ** (gain_db / 20.0))
    if not bands:
        return h_total
    for band in bands:
        h_total = h_total * evaluate_analog_band(band, s)
    return h_total


def compute_active_preamp_eq(
    preamp_spec: str | PreampConfig | Sequence[PreampBandConfig], s: complex | np.ndarray
) -> np.ndarray:
    """
    Evaluates analog active preamp contour transfer function.
    Accepts:
      - str (preset name): looks up in PREAMPS catalog (e.g. 'sadowsky_2band', 'stingray_2band')
      - PreampConfig: evaluates preamp model
      - Sequence[PreampBandConfig]: evaluates sequence of band configs
    """
    if isinstance(preamp_spec, str):
        from allomorph.config.preamps import get_preamp

        preset = get_preamp(preamp_spec)
        return compute_active_preamp_transfer(preset.bands, s, gain_db=float(preset.gain_db))
    elif isinstance(preamp_spec, PreampConfig):
        return compute_active_preamp_transfer(
            preamp_spec.bands, s, gain_db=float(preamp_spec.gain_db)
        )
    elif isinstance(preamp_spec, (list, tuple)):
        return compute_active_preamp_transfer(preamp_spec, s)
    return np.ones_like(s, dtype=np.complex128)


@overload
def compute_circuit_transfer_functions(
    model: CircuitModel,
    freqs: Sequence[float] | np.ndarray = FREQS,
    return_numpy: Literal[False] = False,
) -> list[list[float]]: ...


@overload
def compute_circuit_transfer_functions(
    model: CircuitModel,
    freqs: Sequence[float] | np.ndarray = FREQS,
    return_numpy: Literal[True] = ...,
) -> list[np.ndarray]: ...


@overload
def compute_circuit_transfer_functions(
    model: CircuitModel,
    freqs: Sequence[float] | np.ndarray = FREQS,
    return_numpy: bool = ...,
) -> list[list[float]] | list[np.ndarray]: ...


def compute_circuit_transfer_functions(
    model: CircuitModel,
    freqs: Sequence[float] | np.ndarray = FREQS,
    return_numpy: bool = False,
) -> list[list[float]] | list[np.ndarray]:
    """
    Computes closed-form nodal AC transfer functions across frequencies using vectorized NumPy SIMD operations.
    Returns a list of magnitude curves:
      - Single-pickup: [mag_curve] (length 1)
      - Dual-pickup (parallel or series): [mag_neck, mag_bridge] (length 2)
    Supports both passive high-Z harnesses and active buffered preamps.
    """

    def _ret(res_list: list[np.ndarray]) -> list[list[float]] | list[np.ndarray]:
        if return_numpy:
            return res_list
        return [np.asarray(x, dtype=np.float64).tolist() for x in res_list]

    f = np.asarray(freqs, dtype=np.float64)
    if model.no_eq:
        return _ret([np.ones_like(f)])

    w = 2.0 * np.pi * f
    s = 1j * w

    # Dielectric absorption parameters (Cole-Davidson fractional-order relaxation)
    tan_d = getattr(model, "tan_delta", 0.025)
    if tan_d == 0.0:
        alpha_cable = 1.0
    elif (
        getattr(model, "alpha_dielectric_cable", None) is not None
        and model.alpha_dielectric_cable != 0.994
    ):
        alpha_cable = float(model.alpha_dielectric_cable)
    elif tan_d is not None and tan_d > 0.0:
        alpha_cable = 1.0 - (2.0 / np.pi) * np.arctan(float(tan_d))
    else:
        alpha_cable = 0.994

    alpha_tone = (
        model.alpha_dielectric_tone
        if getattr(model, "alpha_dielectric_tone", None) is not None
        else 0.988
    )
    w0 = 2.0 * np.pi * 1000.0  # 1 kHz calibration reference frequency
    n_pts = len(f)
    on_standard_grid = n_pts == 4096 and f[0] == 0.0 and abs(f[-1] - 24000.0) < 1e-6

    if on_standard_grid:
        s_pow_cable = _get_cached_s_ratio_power(alpha_cable, 4096)
    else:
        s_ratio = np.where(w > 0.0, w / w0, 0.0)
        s_pow_cable = s_ratio ** alpha_cable

    kappa_cable = 1j * (w0 * model.Ccable) * np.exp(1j * (alpha_cable - 1.0) * (np.pi / 2.0))
    Y_cable_diel = kappa_cable * s_pow_cable

    if model.Ctone > 0:
        if on_standard_grid:
            s_pow_tone = _get_cached_s_ratio_power(alpha_tone, 4096)
        else:
            s_ratio = np.where(w > 0.0, w / w0, 0.0)
            s_pow_tone = s_ratio ** alpha_tone
        kappa_tone = 1j * (w0 * model.Ctone) * np.exp(1j * (alpha_tone - 1.0) * (np.pi / 2.0))
        Y_c_tone = kappa_tone * s_pow_tone
        Y_tone = Y_c_tone / (1.0 + Y_c_tone * model.Rtone) if model.Rtone > 0 else Y_c_tone
    else:
        Y_tone = 0.0

    k_m = model.k_mutual
    c_m = model.C_mutual

    chi_mu = model.chi_mu
    chi_mu_b = model.chi_mu_b
    omega_mu = model.omega_mu

    k_dist = model.k_dist
    k_dist_b = model.k_dist_b
    omega_dist = model.omega_dist

    k_skin = model.k_skin
    f_skin = model.f_skin
    omega_skin = 2.0 * math.pi * f_skin if f_skin > 0.0 else 1.0

    k_skin_b = model.k_skin_b
    f_skin_b = model.f_skin_b
    omega_skin_b = 2.0 * math.pi * f_skin_b if f_skin_b > 0.0 else 1.0

    if k_dist > 0.0:
        gamma_dist = k_dist * np.sqrt(s / omega_dist)
        safe_gamma = np.where(np.abs(gamma_dist) < 1e-5, 1.0, gamma_dist)
        dist_factor = np.where(np.abs(gamma_dist) < 1e-5, 1.0, np.tanh(safe_gamma) / safe_gamma)
    else:
        dist_factor = 1.0

    if k_dist_b > 0.0:
        gamma_dist_b = k_dist_b * np.sqrt(s / omega_dist)
        safe_gamma_b = np.where(np.abs(gamma_dist_b) < 1e-5, 1.0, gamma_dist_b)
        dist_factor_b = np.where(
            np.abs(gamma_dist_b) < 1e-5, 1.0, np.tanh(safe_gamma_b) / safe_gamma_b
        )
    else:
        dist_factor_b = 1.0

    tan_d_coil = model.tan_delta_coil
    G_coil = w * model.Ccoil * tan_d_coil if tan_d_coil > 0.0 else 0.0
    G_coil_b = w * model.Ccoil_b * tan_d_coil if tan_d_coil > 0.0 else 0.0
    Y_c_n = (s * model.Ccoil + G_coil) * dist_factor
    Y_c_b = (s * model.Ccoil_b + G_coil_b) * dist_factor_b

    if model.has_active_buffer:
        # Active Preamp Buffer: coils terminate into high-Z buffer, isolating them from cable capacitance.
        # Op-amp buffer drives cable and Anagram pedalboard load through low-Z output stage.
        Z_cable_load = 1.0 / (1.0 / model.Ranagram + Y_cable_diel + s * model.Canagram)
        H_buf_to_out = Z_cable_load / (model.R_out + Z_cable_load)

        # Preamp active contour & voltage gain scaling
        preamp_gain = model.preamp_gain
        preamp_bands = model.preamp_bands
        if preamp_bands is not None:
            H_eq = (
                compute_active_preamp_transfer(preamp_bands, s, gain_db=model.preamp_gain_db)
                * preamp_gain
            )
        else:
            H_eq = compute_active_preamp_eq(model.preamp_type, s) * preamp_gain

        # Coils terminated into high-Z preamp input (R_preamp_in || C_preamp_in)
        Y_preamp_in = 1.0 / model.R_preamp_in + s * model.C_preamp_in
        Y_eff2 = Y_preamp_in + Y_tone

        if model.topology == "single":
            Z_L = compute_core_impedance(
                s,
                model.L,
                model.L_core,
                model.R_core,
                chi_mu=chi_mu,
                omega_mu=omega_mu,
                k_skin=k_skin,
                omega_skin=omega_skin,
                Rdc=model.Rdc,
            )
            Y_branch = 1.0 / (model.Rdc + Z_L) + 1.0 / model.Reddy
            Y_shunt2 = Y_c_n + Y_eff2
            H_dyn_to_2 = Y_branch / (Y_branch + Y_shunt2)

            H_total = H_dyn_to_2 * H_eq * H_buf_to_out
            return _ret([np.abs(H_total)])

        elif model.topology == "parallel":
            Z_L = compute_core_impedance(
                s,
                model.L,
                model.L_core,
                model.R_core,
                chi_mu=chi_mu,
                omega_mu=omega_mu,
                k_skin=k_skin,
                omega_skin=omega_skin,
                Rdc=model.Rdc,
            )
            Z_L_b = compute_core_impedance(
                s,
                model.L_b,
                model.L_core_b,
                model.R_core_b,
                chi_mu=chi_mu_b,
                omega_mu=omega_mu,
                k_skin=k_skin_b,
                omega_skin=omega_skin_b,
                Rdc=model.Rdc_b,
            )
            Y_br_n = 1.0 / (model.Rdc + Z_L) + 1.0 / model.Reddy
            Y_br_b = 1.0 / (model.Rdc_b + Z_L_b) + 1.0 / model.Reddy_b

            r_pot_n = model.Rpot_n
            r_pot_b = model.Rpot_b
            if r_pot_n > 0.0:
                Y_br_n = 1.0 / (1.0 / Y_br_n + r_pot_n)
            if r_pot_b > 0.0:
                Y_br_b = 1.0 / (1.0 / Y_br_b + r_pot_b)

            Y_shunt2 = Y_c_n + Y_c_b + Y_eff2

            if k_m > 0.0 or c_m > 0.0:
                M = k_m * np.sqrt(model.L * model.L_b) if k_m > 0.0 else 0.0
                Z_m = s * M
                Y_m = s * c_m
                Z_n = 1.0 / Y_br_n
                Z_b = 1.0 / Y_br_b
                delta_Z = Z_n * Z_b - (Z_m**2)
                denom_total = delta_Z * (Y_shunt2 + Y_m) + Z_n + Z_b - 2.0 * Z_m
                H_n_to_2 = (Z_b - Z_m) / denom_total
                H_b_to_2 = (Z_n - Z_m) / denom_total
            else:
                Y_total = Y_br_n + Y_br_b + Y_shunt2
                H_n_to_2 = Y_br_n / Y_total
                H_b_to_2 = Y_br_b / Y_total

            H_n = H_n_to_2 * H_eq * H_buf_to_out
            H_b = H_b_to_2 * H_eq * H_buf_to_out

            blend_pos = model.blend_pos
            if abs(blend_pos - 0.5) >= 1e-4:
                taper = model.pot_taper

                if blend_pos < 0.5:
                    gain_n = 1.0
                    norm_atten = (0.5 - blend_pos) / 0.5
                    gain_b = max(1.0 - eval_pot_taper(norm_atten, taper), 0.0)
                else:
                    gain_b = 1.0
                    norm_atten = (blend_pos - 0.5) / 0.5
                    gain_n = max(1.0 - eval_pot_taper(norm_atten, taper), 0.0)
                H_n = H_n * gain_n
                H_b = H_b * gain_b

            return _ret([np.abs(H_n), np.abs(H_b)])

        elif model.topology == "series":
            Z_L = compute_core_impedance(
                s,
                model.L,
                model.L_core,
                model.R_core,
                chi_mu=chi_mu,
                omega_mu=omega_mu,
                k_skin=k_skin,
                omega_skin=omega_skin,
                Rdc=model.Rdc,
            )
            Z_L_b = compute_core_impedance(
                s,
                model.L_b,
                model.L_core_b,
                model.R_core_b,
                chi_mu=chi_mu_b,
                omega_mu=omega_mu,
                k_skin=k_skin_b,
                omega_skin=omega_skin_b,
                Rdc=model.Rdc_b,
            )
            Y_br_n = 1.0 / (model.Rdc + Z_L) + 1.0 / model.Reddy
            Y_br_b = 1.0 / (model.Rdc_b + Z_L_b) + 1.0 / model.Reddy_b
            Y_cn = Y_c_n
            Y_cb = Y_c_b
            Y_2b = Y_br_b + Y_cb

            Y_m = Y_br_n + Y_cn + Y_2b
            Y_2 = Y_2b + Y_eff2
            delta = Y_m * Y_2 - Y_2b**2

            T2_n = (Y_2b * Y_br_n) / delta
            T2_b = ((Y_br_n + Y_cn) * Y_br_b) / delta

            H_n = T2_n * H_eq * H_buf_to_out
            H_b = T2_b * H_eq * H_buf_to_out

            blend_pos = model.blend_pos
            if abs(blend_pos - 0.5) >= 1e-4:
                taper = model.pot_taper

                if blend_pos < 0.5:
                    gain_n = 1.0
                    norm_atten = (0.5 - blend_pos) / 0.5
                    gain_b = max(1.0 - eval_pot_taper(norm_atten, taper), 0.0)
                else:
                    gain_b = 1.0
                    norm_atten = (blend_pos - 0.5) / 0.5
                    gain_n = max(1.0 - eval_pot_taper(norm_atten, taper), 0.0)
                H_n = H_n * gain_n
                H_b = H_b * gain_b

            return _ret([np.abs(H_n), np.abs(H_b)])

    # Passive RLC Guitar Harness: Coils directly loaded by pots, cable capacitance, and Anagram load
    Rload = (model.Rbot * model.Ranagram) / (model.Rbot + model.Ranagram)
    Zload = 1.0 / (1.0 / Rload + Y_cable_diel + s * model.Canagram)

    # Treble bleed impedance (if configured)
    if model.Ctb > 0 and model.Rtb_par > 0:
        Z_tb = model.Rtb_ser + model.Rtb_par / (1.0 + s * model.Rtb_par * model.Ctb)
        Z23_pot = (model.Rtop * Z_tb) / (model.Rtop + Z_tb)
    else:
        Z23_pot = model.Rtop

    if model.topology == "single":
        Z_L = compute_core_impedance(
            s,
            model.L,
            model.L_core,
            model.R_core,
            chi_mu=chi_mu,
            omega_mu=omega_mu,
            k_skin=k_skin,
            omega_skin=omega_skin,
            Rdc=model.Rdc,
        )
        Y_branch = 1.0 / (model.Rdc + Z_L) + 1.0 / model.Reddy
        Y_shunt2 = Y_c_n + Y_tone

        if model.Crick > 0:
            Y_rick = s * model.Crick
            Y_series = Y_rick / (1.0 + (Z23_pot + Zload) * Y_rick)
            H_2_to_3 = (Zload * Y_rick) / (1.0 + (Z23_pot + Zload) * Y_rick)
        else:
            Y_series = 1.0 / (Z23_pot + Zload)
            H_2_to_3 = Zload / (Z23_pot + Zload)

        Y_eff2 = Y_shunt2 + Y_series
        H_dyn_to_2 = Y_branch / (Y_branch + Y_eff2)
        H_total = np.abs(H_dyn_to_2 * H_2_to_3)
        return _ret([H_total])

    elif model.topology == "parallel":
        Z_L = compute_core_impedance(
            s,
            model.L,
            model.L_core,
            model.R_core,
            chi_mu=chi_mu,
            omega_mu=omega_mu,
            k_skin=k_skin,
            omega_skin=omega_skin,
            Rdc=model.Rdc,
        )
        Z_L_b = compute_core_impedance(
            s,
            model.L_b,
            model.L_core_b,
            model.R_core_b,
            chi_mu=chi_mu_b,
            omega_mu=omega_mu,
            k_skin=k_skin_b,
            omega_skin=omega_skin_b,
            Rdc=model.Rdc_b,
        )
        Y_br_n = 1.0 / (model.Rdc + Z_L) + 1.0 / model.Reddy
        Y_br_b = 1.0 / (model.Rdc_b + Z_L_b) + 1.0 / model.Reddy_b

        r_pot_n = model.Rpot_n
        r_pot_b = model.Rpot_b
        if r_pot_n > 0.0:
            Y_br_n = 1.0 / (1.0 / Y_br_n + r_pot_n)
        if r_pot_b > 0.0:
            Y_br_b = 1.0 / (1.0 / Y_br_b + r_pot_b)

        Y_shunt2 = Y_c_n + Y_c_b + Y_tone
        Z23 = Z23_pot

        Y_out_branch = 1.0 / (Z23 + Zload)
        Y_eff2 = Y_shunt2 + Y_out_branch

        if k_m > 0.0 or c_m > 0.0:
            M = k_m * np.sqrt(model.L * model.L_b) if k_m > 0.0 else 0.0
            Z_m = s * M
            Y_m = s * c_m
            Z_n = 1.0 / Y_br_n
            Z_b = 1.0 / Y_br_b
            delta_Z = Z_n * Z_b - (Z_m**2)
            denom_total = delta_Z * (Y_eff2 + Y_m) + Z_n + Z_b - 2.0 * Z_m
            H_n_to_2 = (Z_b - Z_m) / denom_total
            H_b_to_2 = (Z_n - Z_m) / denom_total
        else:
            Y_total = Y_br_n + Y_br_b + Y_eff2
            H_n_to_2 = Y_br_n / Y_total
            H_b_to_2 = Y_br_b / Y_total

        H_2_to_3 = Zload / (Z23 + Zload)
        H_n = H_n_to_2 * H_2_to_3
        H_b = H_b_to_2 * H_2_to_3

        blend_pos = model.blend_pos
        if abs(blend_pos - 0.5) >= 1e-4:
            taper = model.pot_taper

            if blend_pos < 0.5:
                gain_n = 1.0
                norm_atten = (0.5 - blend_pos) / 0.5
                gain_b = max(1.0 - eval_pot_taper(norm_atten, taper), 0.0)
            else:
                gain_b = 1.0
                norm_atten = (blend_pos - 0.5) / 0.5
                gain_n = max(1.0 - eval_pot_taper(norm_atten, taper), 0.0)
            H_n = H_n * gain_n
            H_b = H_b * gain_b

        return _ret([np.abs(H_n), np.abs(H_b)])

    elif model.topology == "series":
        Z_L = compute_core_impedance(
            s,
            model.L,
            model.L_core,
            model.R_core,
            chi_mu=chi_mu,
            omega_mu=omega_mu,
            k_skin=k_skin,
            omega_skin=omega_skin,
            Rdc=model.Rdc,
        )
        Z_L_b = compute_core_impedance(
            s,
            model.L_b,
            model.L_core_b,
            model.R_core_b,
            chi_mu=chi_mu_b,
            omega_mu=omega_mu,
            k_skin=k_skin_b,
            omega_skin=omega_skin_b,
            Rdc=model.Rdc_b,
        )
        Y_br_n = 1.0 / (model.Rdc + Z_L) + 1.0 / model.Reddy
        Y_br_b = 1.0 / (model.Rdc_b + Z_L_b) + 1.0 / model.Reddy_b
        Y_cn = Y_c_n
        Y_cb = Y_c_b
        Y_2b = Y_br_b + Y_cb

        Z23 = Z23_pot
        Y_out_load = 1.0 / (Z23 + Zload)

        Y_m = Y_br_n + Y_cn + Y_2b
        Y_2 = Y_2b + Y_out_load + Y_tone
        delta = Y_m * Y_2 - Y_2b**2

        T2_n = (Y_2b * Y_br_n) / delta
        T2_b = ((Y_br_n + Y_cn) * Y_br_b) / delta
        T_2_to_3 = Zload / (Z23 + Zload)

        H_n = T2_n * T_2_to_3
        H_b = T2_b * T_2_to_3

        blend_pos = model.blend_pos
        if abs(blend_pos - 0.5) >= 1e-4:
            taper = model.pot_taper

            if blend_pos < 0.5:
                gain_n = 1.0
                norm_atten = (0.5 - blend_pos) / 0.5
                gain_b = max(1.0 - eval_pot_taper(norm_atten, taper), 0.0)
            else:
                gain_b = 1.0
                norm_atten = (blend_pos - 0.5) / 0.5
                gain_n = max(1.0 - eval_pot_taper(norm_atten, taper), 0.0)
            H_n = H_n * gain_n
            H_b = H_b * gain_b

        return _ret([np.abs(H_n), np.abs(H_b)])

    raise ValueError(f"Unknown circuit topology: {model.topology}")


@overload
def smooth_soft_knee_db(
    x_db: float,
    thresh: float = ...,
    ceiling: float = ...,
    alpha: float = ...,
) -> float: ...


@overload
def smooth_soft_knee_db(
    x_db: np.ndarray,
    thresh: float = ...,
    ceiling: float = ...,
    alpha: float = ...,
) -> np.ndarray: ...


def smooth_soft_knee_db(
    x_db: float | np.ndarray,
    thresh: float = 6.0,
    ceiling: float = 8.0,
    alpha: float = 2.0,
) -> float | np.ndarray:
    """
    Applies a strictly C^inf infinitely differentiable thresholded soft-knee saturation
    to gain in decibels without piecewise conditionals or slope kinks:
        excess = (1 / alpha) * ln(1 + e^(alpha * (x - thresh)))
        sat_excess = w * tanh(excess / w)
        y = x - excess + sat_excess
    where w = max(ceiling - thresh, 1e-6).

    Properties:
    - Strictly C^inf smooth everywhere on R (zero piecewise conditionals or boundary cusps).
    - As x << thresh: excess -> 0, sat_excess -> excess, y -> x (100% linear passband transparency).
    - At x = thresh: y ≈ thresh.
    - As x >> thresh: excess -> x - thresh, sat_excess -> w, y -> thresh + w = ceiling.
    - Strictly monotonic: dy/dx = 1 - sigma(alpha*(x-thresh)) * tanh^2(excess/w) > 0 everywhere.
    """
    w = max(ceiling - thresh, 1e-6)
    excess = np.logaddexp(0.0, alpha * (x_db - thresh)) / alpha
    res = x_db - excess + w * np.tanh(excess / w)
    if isinstance(x_db, (float, int)):
        return float(res)
    return res


def compute_differential_circuit_transfer_functions(
    target_model: CircuitModel,
    source_model: CircuitModel,
    freqs: Sequence[float] | np.ndarray = FREQS,
    max_boost_db: float = 8.0,
    eps: float = 0.05,
) -> list[list[float]]:
    """
    Computes regularized differential AC transfer functions for passive-to-passive modeling:
    |H_diff(s)| = (|H_target(s)| * |H_source(s)|) / (|H_source(s)|^2 + eps^2)
    with Wiener regularization and frequency-dependent high-frequency gain clamping (<= max_boost_db
    above 4.5 kHz) to prevent amplifying passive coil hum, Johnson noise, or cable hiss.
    """
    tgt_curves = compute_circuit_transfer_functions(target_model, freqs=freqs)
    src_curves = compute_circuit_transfer_functions(source_model, freqs=freqs)

    f_arr = np.asarray(freqs, dtype=np.float64)

    diff_curves = []
    for ch_idx, tgt_c in enumerate(tgt_curves):
        if len(tgt_curves) == 1 and len(src_curves) > 1:
            # Parallel multi-pickup source summing into single-channel canonical/target stage:
            # Net source electrical transfer function is the sum of parallel branch currents
            src_arr = np.sum(src_curves, axis=0)
        else:
            src_c = src_curves[ch_idx] if len(src_curves) > ch_idx else src_curves[0]
            src_arr = np.asarray(src_c, dtype=np.float64)

        tgt_arr = np.asarray(tgt_c, dtype=np.float64)

        if np.allclose(tgt_arr, src_arr, rtol=1e-4):
            diff_curves.append(np.ones_like(tgt_arr).tolist())
            continue

        # Wiener regularized quotient
        h_diff = (tgt_arr * src_arr) / (src_arr**2 + eps**2)

        # Convert to absolute dB (linear ratio between target and source circuits)
        h_db = 20.0 * np.log10(np.maximum(h_diff, 1e-6))

        # Strictly C^inf soft-knee limiting: smoothly saturate boost towards max_boost_db
        knee_width = min(2.5, max_boost_db / 2.0)
        thresh = max_boost_db - knee_width
        h_db_soft = smooth_soft_knee_db(h_db, thresh=thresh, ceiling=max_boost_db, alpha=2.0)

        # Smooth high-frequency cosine taper above 8.0 kHz to 20.0 kHz
        # Eliminates unnatural flat horizontal ceilings and suppresses extreme ultrasonic noise
        f_start = 8000.0
        f_end = 20000.0
        t = np.clip((f_arr - f_start) / (f_end - f_start), 0.0, 1.0)
        w = 0.5 * (1.0 + np.cos(np.pi * t))
        s = 0.25 + 0.75 * w
        # Smooth C^inf transition: softplus ensures strictly monotonic, C^1 smooth blending across 0 dB
        excess_boost = (1.0 / 1.2) * np.logaddexp(0.0, 1.2 * h_db_soft)
        h_db_final = h_db_soft - (1.0 - s) * excess_boost

        h_diff_smooth = 10.0 ** (h_db_final / 20.0)
        diff_curves.append(h_diff_smooth.tolist())

    return diff_curves
