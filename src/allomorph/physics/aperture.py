"""
Allomorph - Acoustic Sensing Aperture & Spatial Boundary Engine
Computes 2D cylindrical rod Bessel apertures, 1D blade slit sinc apertures,
saddle witness-point boundary stiffness, body microphonics, and multi-coil
spatial responses across the continuous wave-speed continuum.
"""

import math
from collections.abc import Sequence

import numpy as np

from allomorph.circuit.parser import MAGNET_PROPERTIES
from allomorph.config.geometry import resolve_pickup_coils, resolve_voice_coils
from allomorph.config.instruments import get_source_pickup, load_instrument
from allomorph.config.scales import SCALES
from allomorph.config.schema import (
    CoilConfig,
    InstrumentConfig,
    PickupConfig,
    ScaleConfig,
    VoiceCoilConfig,
    VoiceConfig,
)
from allomorph.config.voices import VOICES
from allomorph.dsp import cinf_smoothstep
from allomorph.physics.schema import WaveSpeedContinuumPoint
from allomorph.physics.strings import (
    generate_wave_speed_continuum,
    get_inharmonicity_for_f0,
    resolve_scale_range,
)


def compute_body_microphonic_coupling(
    freqs: Sequence[float] | np.ndarray,
    src_pickup: PickupConfig,
    tgt_voice: VoiceConfig,
    inst: InstrumentConfig | None = None,
) -> np.ndarray:
    """
    Computes diffuse mechanical body-pickup microphonic coupling transfer curve.
    Unpotted and lightly potted vintage passive pickups exhibit subtle mechanical
    coupling to body vibrations around 6.2 kHz, damped above 9.5 kHz.
    Active epoxy-potted and sealed modern pickups have near-zero microphonic coupling.
    Evaluated differentially: Δk_body = max(k_tgt - k_src, 0.0).
    """
    freqs = np.asarray(freqs, dtype=np.float64)
    if inst is not None and inst.electronics == "active":
        src_mag = "active"
    else:
        src_mag = src_pickup.magnet_type or "active"

    tgt_mag = tgt_voice.magnet_type or "alnico_v"

    k_src = MAGNET_PROPERTIES.get(src_mag, MAGNET_PROPERTIES["active"]).k_body
    k_tgt = MAGNET_PROPERTIES.get(tgt_mag, MAGNET_PROPERTIES["alnico_v"]).k_body

    delta_k = max(k_tgt - k_src, 0.0)
    if delta_k <= 0.0:
        return np.ones_like(freqs)

    fb = 6200.0
    Qb = 1.8
    fdamp = 9500.0

    fn = freqs / fb
    denom = Qb * np.sqrt((1.0 - fn**2) ** 2 + (fn / Qb) ** 2)
    resonance = fn / np.maximum(denom, 1e-9)
    damping = np.exp(-((freqs / fdamp) ** 2))

    return 1.0 + delta_k * resonance * damping


def compute_coil_aperture(
    freqs: Sequence[float] | np.ndarray,
    v_disp: float | np.ndarray,
    w_m: float,
    pole_type: str = "rod",
) -> np.ndarray:
    """
    Computes spatial sensing aperture response across frequencies:
    - 'rod': 2D cylindrical pole piece (Airy / Bessel J1(x)/x algebraic approximation)
      ap(f) = 1 / sqrt(1 + 0.25 * (2*pi*r_p*f / v)^2) where r_p = w_m / 2.0
    - 'blade': 1D continuous bar/blade rectangular slit
      ap(f) = 1 / sqrt(1 + (1/3) * (pi*w_m*f / v)^2)
    Evaluates with C^inf smoothness, exact 1.000 at f=0, and 0.00 dB identity.
    """
    f = np.asarray(freqs, dtype=np.float64)
    beta = 1.0 / 3.0 if pole_type == "blade" else 0.25
    arg = (math.pi * w_m * f) / v_disp
    return 1.0 / np.sqrt(1.0 + beta * (arg**2))


def compute_saddle_boundary_coupling(
    freqs: Sequence[float] | np.ndarray,
    pos_m: float,
    scale_m: float = 0.8636,
) -> np.ndarray:
    """
    Models the exponential boundary layer (l_b ≈ sqrt(B_s) * L) of flexural rigidity
    at the bridge saddle witness point for pickups situated close to the bridge (pos_m < 0.075 m).
    Smoothly transitions to 1.000 (0.00 dB) as distance increases to >= 75 mm using a real-analytic
    C^infinity smoothstep mollifier.
    """
    f = np.asarray(freqs, dtype=np.float64)
    _ = scale_m
    if pos_m >= 0.075 or pos_m <= 0.0:
        return np.ones_like(f)
    t = (pos_m - 0.035) / (0.075 - 0.035)
    s = float(cinf_smoothstep(t))
    shelf_db = -4.0 * (1.0 - s)
    g = 10.0 ** (shelf_db / 20.0)
    f0 = 4500.0
    return np.sqrt((1.0 + (g**2) * (f / f0) ** 2) / (1.0 + (f / f0) ** 2))


def soft_clamp_displacement_ratio(
    delta_g: float | np.ndarray,
    g_pos: float = 12.0,
    g_neg: float = 16.0,
    k: float = 4.0,
) -> float | np.ndarray:
    """
    Applies an asymmetric C^inf order-4 algebraic limiter ('alg4') with a smooth
    partition-of-unity sigmoid blend to scale-normalized displacement ratios.

    Eliminates premature O(x^3) compression on standard pickup positions (|ΔG| <= 6 dB)
    while strictly bounding positive excursion boost (<= +12.0 dB) and negative
    proximity thinning (>= -16.0 dB) to ensure full fidelity on Mudbucker and Rickenbacker
    bridge voicings while guaranteeing compliance with Guardrail 5.3.6 (20 Hz DC transmission
    in [-12.0 dB, +12.0 dB]).
    """
    dg = np.asarray(delta_g, dtype=np.float64)
    sigma = 0.5 * (1.0 + np.tanh(dg / k))
    g_eff = sigma * g_pos + (1.0 - sigma) * g_neg
    soft = dg / (1.0 + (dg / g_eff) ** 4) ** 0.25
    if np.isscalar(delta_g):
        return float(soft)
    return soft


UNIVERSAL_DATUM_POS_M: float = 0.0935  # 93.5mm datum
UNIVERSAL_DATUM_SCALE_M: float = 0.8636  # 34.0" scale
UNIVERSAL_DATUM_ETA: float = UNIVERSAL_DATUM_POS_M / UNIVERSAL_DATUM_SCALE_M  # 0.1082677...


def compute_displacement_proximity_shelf(
    freqs: Sequence[float] | np.ndarray,
    pos_m: float,
    scale_m: float = UNIVERSAL_DATUM_SCALE_M,
    ref_eta: float = UNIVERSAL_DATUM_ETA,
) -> np.ndarray:
    """
    Computes the scale-normalized bridge proximity low-shelf filter H_pos(f)
    governed by the standing-wave fractional displacement ratio (eta = pos_m / scale_m)
    relative to a reference fractional coordinate (default: universal datum eta_datum = 10.83%),
    bounded by the asymmetric C^inf order-4 algebraic limiter ('alg4').
    Corner frequency fc = 220.0 Hz.
    """
    f = np.asarray(freqs, dtype=np.float64)
    if pos_m <= 0.0 or scale_m <= 0.0:
        return np.ones_like(f)
    eta = pos_m / scale_m
    delta_g = 20.0 * np.log10(max(eta / max(ref_eta, 1e-4), 1e-6))
    delta_g_soft = soft_clamp_displacement_ratio(delta_g)
    g_0 = 10.0 ** (delta_g_soft / 20.0)
    return np.sqrt((g_0**2 + (f / 220.0) ** 2) / (1.0 + (f / 220.0) ** 2))


def compute_pickup_isolation_leveling(
    pos_m: float,
    scale_m: float = UNIVERSAL_DATUM_SCALE_M,
    ref_pos_m: float | None = None,
    max_boost_db: float = 6.0,
    alpha: float = 2.0,
) -> float:
    """Computes the luthier setup pickup height compensation factor in isolation.

    On physical instruments, bridge pickups are mounted closer to the strings
    and wound hotter to compensate for the smaller string displacement envelope (eta = x / L).
    Returns a C^inf smooth linear gain multiplier (1.0 for neck/middle pickups; up to +6.0 dB for bridge pickups).
    Evaluates with exact 1.0000 (0.00 dB) identity when pos_m >= ref_pos_m.
    """
    if pos_m <= 0.0 or scale_m <= 0.0:
        return 1.0
    eta = pos_m / scale_m
    ref_eta = (
        (ref_pos_m / scale_m)
        if (ref_pos_m is not None and ref_pos_m > 0.0)
        else UNIVERSAL_DATUM_ETA
    )
    deficit_db = 20.0 * math.log10(max(ref_eta / eta, 1e-4))
    if deficit_db <= 0.0:
        return 1.0
    excess_db = float((np.logaddexp(0.0, alpha * deficit_db) - math.log(2.0)) / alpha)
    boost_db = max_boost_db * math.tanh(max(excess_db, 0.0) / max_boost_db)
    return float(10.0 ** (boost_db / 20.0))


def is_voice_matching_source(
    instrument: InstrumentConfig | str,
    voice_id: str,
    voice_cfg: VoiceConfig | None = None,
) -> bool:
    """
    Determines if a target voice matches the source instrument's physical scale and pickup geometry,
    meaning zero spatial or acoustic transfer is required (identity transformation).
    Tuning- and string-count-agnostic: matches on physical scale length and coil geometry.
    """
    inst = load_instrument(instrument) if isinstance(instrument, str) else instrument
    vcfg = voice_cfg or VOICES.get(voice_id)
    if vcfg is None:
        return False

    if vcfg.preserve_aperture:
        return bool(getattr(vcfg.circuit, "no_eq", False))

    src_range = resolve_scale_range(inst)
    tgt_scale = vcfg.scale
    tgt_scale_info = SCALES.get(tgt_scale)
    tgt_range = resolve_scale_range(tgt_scale_info)

    # Scale match based on physical vibrating length range (within 1.2 cm)
    if abs(src_range[0] - tgt_range[0]) > 0.012 or abs(src_range[1] - tgt_range[1]) > 0.012:
        return False

    src_p = get_source_pickup(inst, voice_id)
    src_coils = resolve_pickup_coils(src_p, inst)
    tgt_coils = resolve_voice_coils(vcfg)

    if len(src_coils) != len(tgt_coils):
        return False

    s_sort = sorted(src_coils, key=lambda c: c.position_from_bridge_m)
    t_sort = sorted(tgt_coils, key=lambda c: c.position_from_bridge_m)

    for sc, tc in zip(s_sort, t_sort):
        if abs(sc.position_from_bridge_m - tc.position_from_bridge_m) > 0.005:
            return False
        if abs(sc.aperture_width_in - tc.aperture_width_in) > 0.15:
            return False

    return True


def get_coil_register(coil: CoilConfig | VoiceCoilConfig) -> str:
    """
    Identifies whether a coil half is 'lower' (bass strings register),
    'upper' (treble strings register), or 'all' across the string bed.
    """
    strings = coil.strings
    if "all" in strings:
        return "all"

    strings_set = set(strings)
    lower_markers = {"E", "A", "B", "low", "lower", "bass", 3, 4, 5, 6, "3", "4", "5", "6"}
    upper_markers = {"D", "G", "C", "high", "upper", "treble", 1, 2, "1", "2"}

    has_lower = bool(strings_set & lower_markers)
    has_upper = bool(strings_set & upper_markers)

    if has_lower and not has_upper:
        return "lower"
    elif has_upper and not has_lower:
        return "upper"
    return "all"


def numpy_pickup_acoustic_response(
    freqs: Sequence[float] | np.ndarray,
    coils: Sequence[CoilConfig | VoiceCoilConfig],
    scale_length_m: float
    | tuple[float, float]
    | list[float]
    | Sequence[float]
    | ScaleConfig
    | InstrumentConfig
    | None = None,
    string_speeds: Sequence[float] | None = None,
    string_names: Sequence[str | int] | None = None,
) -> np.ndarray:
    """
    Computes compound spatial aperture and multi-coil response for an arbitrary
    array of N physical coils across the continuous wave-speed continuum of the instrument.
    Completely tuning-agnostic, gauge-agnostic, and string-count-agnostic.
    Respects geometric register half bindings for split-coil pickups (e.g. P-Bass).
    """
    f = np.asarray(freqs, dtype=np.float64)

    if (
        isinstance(scale_length_m, (list, tuple, np.ndarray))
        and len(scale_length_m) > 0
        and any(float(v) > 10.0 for v in scale_length_m)
    ):
        string_speeds = scale_length_m
        scale_length_m = None

    scale_range = resolve_scale_range(scale_length_m)
    l_eff = (scale_range[0] + scale_range[1]) / 2.0

    if string_speeds is not None and len(string_speeds) > 0 and len(string_speeds) != 24:
        continuum: list[WaveSpeedContinuumPoint] = []
        n_str = len(string_speeds)
        half = n_str // 2 if n_str > 2 else 1
        for s_idx, v in enumerate(string_speeds):
            f0 = max(v / (2.0 * l_eff), 15.0)
            if string_names and s_idx < len(string_names):
                s_name = string_names[s_idx]
                if s_name in [1, 2, "1", "2", "D", "G", "C", "high", "upper", "treble"]:
                    reg = "upper"
                elif s_name in [
                    3,
                    4,
                    5,
                    6,
                    "3",
                    "4",
                    "5",
                    "6",
                    "E",
                    "A",
                    "B",
                    "low",
                    "lower",
                    "bass",
                ]:
                    reg = "lower"
                else:
                    reg = "lower" if s_idx < half else "upper"
            else:
                reg = "lower" if s_idx < half else "upper"
            continuum.append(
                WaveSpeedContinuumPoint(
                    f0=f0,
                    v0=float(v),
                    scale_m=l_eff,
                    register=reg,
                    weight=1.0 / n_str,
                )
            )
    else:
        continuum = generate_wave_speed_continuum(scale_range, num_points=24)

    # Partition continuum points by register half to enable 2D tensor broadcasting
    partitions: dict[str, list[WaveSpeedContinuumPoint]] = {}
    for pt in continuum:
        partitions.setdefault(str(pt.register), []).append(pt)

    acc = np.zeros_like(f, dtype=np.float64)
    total_pt_weight = 0.0

    for reg, pts in partitions.items():
        active: list[CoilConfig | VoiceCoilConfig] = []
        for c in coils:
            coil_reg = get_coil_register(c)
            if coil_reg == "all" or coil_reg == reg:
                active.append(c)

        if not active:
            active = list(coils)

        K = len(pts)
        f0_arr = np.array([pt.f0 for pt in pts], dtype=np.float64)
        v0_arr = np.array([pt.v0 for pt in pts], dtype=np.float64)
        w_arr = np.array([pt.weight for pt in pts], dtype=np.float64)

        b_s = get_inharmonicity_for_f0(f0_arr)
        f_disp_max = 3500.0
        disp_factor = 1.0 + b_s[:, None] * ((f[None, :] / f0_arr[:, None]) ** 2) / (
            1.0 + (f[None, :] / f_disp_max) ** 2
        )
        v_disp = v0_arr[:, None] * np.sqrt(disp_factor)

        total_w = sum(abs(c.weight) for c in active) or 1.0
        center_pos = sum(c.position_from_bridge_m * abs(c.weight) for c in active) / total_w

        coil_sum = np.zeros((K, len(f)), dtype=np.complex128)
        p_incoh = np.zeros((K, len(f)), dtype=np.float64)

        for c in active:
            pos_m = c.position_from_bridge_m
            w_m = c.aperture_width_in * 0.0254
            weight = c.weight
            polarity = c.polarity
            delta_x = pos_m - center_pos
            phase = 2.0 * math.pi * f[None, :] * delta_x / v_disp
            c_pole = c.pole_type or "rod"
            beta = 1.0 / 3.0 if c_pole == "blade" else 0.25
            arg = (math.pi * w_m * f[None, :]) / v_disp
            ap_w = 1.0 / np.sqrt(1.0 + beta * (arg**2))
            w_eff = weight * ap_w
            coil_sum += (w_eff * polarity) * np.exp(-1j * phase)
            p_incoh += w_eff**2

        p_coh = np.abs(coil_sum) ** 2

        if len(active) > 1:
            delta_x_span = max(c.position_from_bridge_m for c in active) - min(
                c.position_from_bridge_m for c in active
            )
            if delta_x_span > 0.002:
                eps_quad = 0.18
                p_coh_reg = p_coh + (eps_quad**2) * p_incoh
                dc_incoh = sum(abs(c.weight) ** 2 for c in active)
                dc_norm = math.sqrt(total_w**2 + (eps_quad**2) * dc_incoh) / total_w
                f_mid = 1.4 * v0_arr[:, None] / delta_x_span
                f_sigma = np.maximum(0.4 * v0_arr[:, None] / delta_x_span, 1.0)
                gamma = 0.5 * (1.0 - np.tanh((f[None, :] - f_mid) / f_sigma))
                m_blend = np.sqrt(gamma * p_coh_reg + (1.0 - gamma) * p_incoh) / dc_norm
            else:
                m_blend = np.abs(coil_sum)
        else:
            m_blend = np.abs(coil_sum)

        acc += np.sum(w_arr[:, None] * m_blend, axis=0)
        total_pt_weight += float(np.sum(w_arr))

    return acc / total_pt_weight if total_pt_weight > 0 else acc


def numpy_pickup_macro_aperture(
    freqs: Sequence[float] | np.ndarray,
    coils: Sequence[CoilConfig | VoiceCoilConfig],
    scale_length_m: float
    | tuple[float, float]
    | list[float]
    | Sequence[float]
    | ScaleConfig
    | InstrumentConfig
    | None = None,
    string_speeds: Sequence[float] | None = None,
    string_names: Sequence[str | int] | None = None,
) -> np.ndarray:
    """
    Computes the macro sensing aperture response (smooth spatial low-pass envelope
    of the individual coil aperture) averaged across the continuous wave-speed continuum,
    without inter-coil phase cancellation nulls or unphysical sinc sidelobes.
    Used for safe, non-inverting deconvolution of multi-coil source pickups.
    """
    f = np.asarray(freqs, dtype=np.float64)
    w_in = coils[0].aperture_width_in if coils else 0.75
    w_m = w_in * 0.0254

    if (
        isinstance(scale_length_m, (list, tuple, np.ndarray))
        and len(scale_length_m) > 0
        and any(float(v) > 10.0 for v in scale_length_m)
    ):
        string_speeds = scale_length_m
        scale_length_m = None

    scale_range = resolve_scale_range(scale_length_m)
    l_eff = (scale_range[0] + scale_range[1]) / 2.0

    if string_speeds is not None and len(string_speeds) > 0 and len(string_speeds) != 24:
        continuum = [
            WaveSpeedContinuumPoint(
                f0=max(v / (2.0 * l_eff), 15.0),
                v0=float(v),
                scale_m=l_eff,
                register="lower" if i < len(string_speeds) // 2 else "upper",
                weight=1.0 / len(string_speeds),
            )
            for i, v in enumerate(string_speeds)
        ]
    else:
        continuum = generate_wave_speed_continuum(scale_range, num_points=24)

    c_pole = (coils[0].pole_type or "rod") if coils else "rod"
    beta = 1.0 / 3.0 if c_pole == "blade" else 0.25
    f0_arr = np.array([pt.f0 for pt in continuum], dtype=np.float64)
    v0_arr = np.array([pt.v0 for pt in continuum], dtype=np.float64)
    w_arr = np.array([pt.weight for pt in continuum], dtype=np.float64)
    b_s = get_inharmonicity_for_f0(f0_arr)
    f_disp_max = 3500.0
    disp_factor = 1.0 + b_s[:, None] * ((f[None, :] / f0_arr[:, None]) ** 2) / (
        1.0 + (f[None, :] / f_disp_max) ** 2
    )
    v_disp = v0_arr[:, None] * np.sqrt(disp_factor)
    arg = (math.pi * w_m * f[None, :]) / v_disp
    ap_w = 1.0 / np.sqrt(1.0 + beta * (arg**2))
    total_w = float(np.sum(w_arr))
    return np.sum(w_arr[:, None] * ap_w, axis=0) / total_w if total_w > 0 else np.zeros_like(f)


def numpy_aperture(
    freqs: Sequence[float] | np.ndarray,
    w_in: float,
    d_in: float,
    speeds: Sequence[float] | None = None,
    scale_length_m: float = 0.8636,
) -> np.ndarray:
    """Computes multi-string aperture sinc + dual-coil comb using NumPy across the wave-speed continuum."""
    f = np.asarray(freqs, dtype=np.float64)
    w_m = w_in * 0.0254
    d_m = d_in * 0.0254
    if speeds is None:
        continuum = generate_wave_speed_continuum(scale_length_m)
        speeds = [pt.v0 for pt in continuum]
    v_arr = np.asarray(speeds, dtype=np.float64)[:, None]
    sinc_v = np.abs(np.sinc((w_m * f[None, :]) / v_arr))
    if d_in > 0:
        comb_v = np.abs(np.cos((np.pi * d_m * f[None, :]) / v_arr))
        sinc_v *= comb_v
    return np.mean(sinc_v, axis=0)


def numpy_position(
    freqs: Sequence[float] | np.ndarray,
    pos_m: float,
    speeds: Sequence[float] | None = None,
    scale_length_m: float = 0.8636,
) -> np.ndarray:
    """Computes spatial standing wave envelope using NumPy across the wave-speed continuum."""
    f = np.asarray(freqs, dtype=np.float64)
    if speeds is None:
        continuum = generate_wave_speed_continuum(scale_length_m)
        speeds = [pt.v0 for pt in continuum]
    v_arr = np.asarray(speeds, dtype=np.float64)[:, None]
    arg_p = f[None, :] * (2.0 * math.pi * pos_m / v_arr)
    return np.mean(np.abs(np.sin(arg_p)), axis=0)


pickup_acoustic_response = numpy_pickup_acoustic_response
aperture_response = numpy_aperture
position_envelope = numpy_position
