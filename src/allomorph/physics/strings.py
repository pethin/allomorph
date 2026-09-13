"""
Allomorph - String Mechanics & Wave Dispersion Modeling
Computes differential string damping and bloom, longitudinal clank resonance,
inharmonicity B_s interpolation, scale-length conversions, and dispersive wave speeds.
"""

import math
from collections.abc import Sequence
from typing import overload

import numpy as np

from allomorph.circuit.solver import smooth_soft_knee_db
from allomorph.config.scales import SCALES, resolve_scale_range
from allomorph.config.schema import InstrumentConfig, ScaleConfig, StringPresetConfig
from allomorph.config.strings import get_voice_string
from allomorph.physics.schema import WaveSpeedContinuumPoint

__all__ = [
    "INHARMONICITY_ANCHORS_BS",
    "INHARMONICITY_ANCHORS_F0",
    "MEAN_BASS_F0",
    "NOTE_NAMES",
    "STRING_FUNDAMENTALS",
    "WaveSpeedContinuumPoint",
    "compute_differential_longitudinal_transfer",
    "compute_differential_string_transfer",
    "compute_dispersive_wave_speed",
    "generate_wave_speed_continuum",
    "get_inharmonicity_for_f0",
    "get_voice_string",
    "infer_string_names",
    "pitch_to_note_name",
    "resolve_scale_range",
]

# Standard open string fundamentals (EADG 4-string bass)
STRING_FUNDAMENTALS = [41.20, 55.00, 73.42, 98.00]
NOTE_NAMES = ["E", "A", "D", "G"]

# Empirical inharmonicity coefficient anchors across bass registers
INHARMONICITY_ANCHORS_F0 = np.array(
    [27.50, 30.87, 41.20, 55.00, 73.42, 98.00, 130.81, 196.00], dtype=np.float64
)
INHARMONICITY_ANCHORS_BS = np.array(
    [0.000028, 0.000025, 0.000020, 0.000012, 0.000006, 0.000003, 0.0000015, 0.0000008],
    dtype=np.float64,
)

# Precomputed Gaussian RBF solver (C^inf globally analytic anchor interpolator)
_LOG_F0_ANCHORS = np.log2(INHARMONICITY_ANCHORS_F0)
_LOG_BS_ANCHORS = np.log2(INHARMONICITY_ANCHORS_BS)
_RBF_EPSILON = 0.5
_RBF_D = np.abs(_LOG_F0_ANCHORS[:, None] - _LOG_F0_ANCHORS[None, :])
_RBF_A = np.exp(-(_RBF_EPSILON * _RBF_D) ** 2)
_RBF_WEIGHTS = np.linalg.solve(_RBF_A, _LOG_BS_ANCHORS)
MEAN_BASS_F0 = (
    66.9045  # Mean open-string fundamental frequency (E1=41.203, A1=55.000, D2=73.416, G2=97.999)
)


def compute_differential_string_transfer(
    freqs: Sequence[float] | np.ndarray,
    src_string: StringPresetConfig,
    tgt_string: StringPresetConfig,
) -> np.ndarray:
    """
    Computes differential transfer function between source instrument strings
    and target voicing goal strings using NumPy:
      H_string_transfer(f) = H_damp_ratio(f) * H_bloom_diff(f)
    Prevents double-damping when source bass already uses flatwounds, while
    providing authentic acoustic upright/fanned-fret damping and bloom.
    """
    f = np.asarray(freqs, dtype=np.float64)

    f_damp_src = float(src_string.damping_cutoff_hz)
    n_src = float(src_string.damping_order)

    f_damp_tgt = float(tgt_string.damping_cutoff_hz)
    n_tgt = float(tgt_string.damping_order)

    # Calculate fused magnitude damping curve ratio
    u_src = (f / f_damp_src) ** (2.0 * n_src)
    u_tgt = (f / f_damp_tgt) ** (2.0 * n_tgt)
    ratio = np.sqrt((1.0 + u_src) / (1.0 + u_tgt))
    r_db = 20.0 * np.log10(np.maximum(ratio, 1e-6))
    g_max_db = 8.0
    g_min_db = -36.0
    sigma = 0.5 * (1.0 + np.tanh(0.5 * r_db))
    f_pos = smooth_soft_knee_db(r_db, thresh=5.0, ceiling=g_max_db, alpha=2.0)
    f_neg = -smooth_soft_knee_db(-r_db, thresh=24.0, ceiling=abs(g_min_db), alpha=2.0)
    r_soft_db = sigma * f_pos + (1.0 - sigma) * f_neg
    h_damp_ratio = 10.0 ** (r_soft_db / 20.0)

    bloom_src = float(src_string.bloom_db)
    bloom_tgt = float(tgt_string.bloom_db)
    delta_bloom_db = bloom_tgt - bloom_src

    g_bloom = 10.0 ** (delta_bloom_db / 20.0)
    h_bloom = np.sqrt((g_bloom**2 + (f / 90.0) ** 2) / (1.0 + (f / 90.0) ** 2))

    return h_damp_ratio * h_bloom


def compute_differential_longitudinal_transfer(
    freqs: Sequence[float] | np.ndarray,
    src_string: StringPresetConfig,
    tgt_string: StringPresetConfig,
    scale_length_inches: float = 34.0,
) -> np.ndarray:
    """
    Computes differential longitudinal wave transmission and core percussion (H_long(f)).
    Steel core longitudinal compression waves (cL ≈ 5100 m/s) produce an instantaneous
    resonant clank peak around f_L = cL / (2 * L) (≈ 2.7 - 3.3 kHz).
    When target voicing has higher longitudinal clank than source, injects regularized
    percussive clank resonance. Returns 1.0 when matching source or delta <= 0.
    """
    f = np.asarray(freqs, dtype=np.float64)
    k_long_src = float(src_string.k_long)
    k_long_tgt = float(tgt_string.k_long)
    delta_k_long = max(k_long_tgt - k_long_src, 0.0)
    if delta_k_long <= 0.0:
        return np.ones_like(f)

    L_meters = float(scale_length_inches) * 0.0254
    c_L = 5100.0
    f_L = c_L / (2.0 * max(L_meters, 0.50))
    Q_L = 8.0
    denom_L = Q_L * np.sqrt((1.0 - (f / f_L) ** 2) ** 2 + (f / (Q_L * f_L)) ** 2)
    h_long = 1.0 + delta_k_long * (f / f_L) / np.maximum(denom_L, 1e-6) * np.exp(
        -((f / 6000.0) ** 2)
    )
    return h_long


def pitch_to_note_name(f0: float) -> str:
    """Converts a fundamental frequency in Hz to closest standard note name."""
    semitones = round(12.0 * math.log2(max(f0, 10.0) / 440.0)) + 69
    names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    return names[semitones % 12]


@overload
def get_inharmonicity_for_f0(f0: float) -> float: ...


@overload
def get_inharmonicity_for_f0(f0: np.ndarray) -> np.ndarray: ...


def get_inharmonicity_for_f0(f0: float | np.ndarray) -> float | np.ndarray:
    """Interpolates empirical string stiffness / inharmonicity constant B_s for a given f0
    using an infinitely differentiable (C^inf) Gaussian Radial Basis Function (RBF)."""
    if isinstance(f0, np.ndarray):
        log_f0 = np.log2(np.maximum(f0, 15.0))
        d = np.abs(log_f0[:, None] - _LOG_F0_ANCHORS[None, :])
        basis = np.exp(-(_RBF_EPSILON * d) ** 2)
        return 2.0 ** (basis @ _RBF_WEIGHTS)
    log_f0 = math.log2(max(f0, 15.0))
    d = np.abs(log_f0 - _LOG_F0_ANCHORS)
    basis = np.exp(-(_RBF_EPSILON * d) ** 2)
    return float(2.0 ** (basis @ _RBF_WEIGHTS))


# resolve_scale_range is imported from allomorph.config to maintain a single source of truth


def generate_wave_speed_continuum(
    scale_length_m: float
    | tuple[float, float]
    | list[float]
    | ScaleConfig
    | InstrumentConfig
    | str
    | None = 0.8636,
    num_points: int = 24,
) -> list[WaveSpeedContinuumPoint]:
    """
    Generates a dense, continuous log-spaced continuum of wave speeds spanning
    the full operating register of an electric bass for a given scale length or multi-scale range:
    v(f0) = 2 * L(f0) * f0
    From f_min = 30.87 Hz (Low B) to f_max = 100.00 Hz (High G).
    On multi-scale instruments, L(f0) smoothly interpolates from L_max at Low B to L_min at High G.
    """
    f_min = 30.87
    f_max = 100.00
    log_f = np.linspace(np.log2(f_min), np.log2(f_max), num_points)
    f0_arr = 2.0**log_f

    if isinstance(scale_length_m, (tuple, list)) and len(scale_length_m) == 2:
        l_min_m = float(min(scale_length_m))
        l_max_m = float(max(scale_length_m))
    elif isinstance(scale_length_m, (ScaleConfig, InstrumentConfig)) or (
        isinstance(scale_length_m, str) and scale_length_m in SCALES
    ):
        l_min_m, l_max_m = resolve_scale_range(scale_length_m)
    else:
        val = float(scale_length_m) if isinstance(scale_length_m, (int, float, str)) else 0.8636
        l_min_m = val
        l_max_m = val

    if abs(l_max_m - l_min_m) > 1e-4:
        t = (log_f - np.log2(f_min)) / (np.log2(f_max) - np.log2(f_min))
        l_arr = l_max_m - t * (l_max_m - l_min_m)
    else:
        l_arr = np.full_like(f0_arr, l_max_m)

    v0_arr = 2.0 * l_arr * f0_arr

    continuum: list[WaveSpeedContinuumPoint] = []
    half = num_points // 2
    for i, (f0, v0, l_eff) in enumerate(zip(f0_arr, v0_arr, l_arr)):
        reg = "lower" if i < half else "upper"
        continuum.append(
            WaveSpeedContinuumPoint(
                f0=float(f0),
                v0=float(v0),
                scale_m=float(l_eff),
                register=reg,
                weight=1.0 / num_points,
            )
        )
    return continuum


def resolve_scale_length(
    string_speeds: Sequence[float],
    scale_length_m: float | tuple[float, float] | list[float] | None = None,
) -> float:
    """Resolves the effective vibrating scale length in meters."""
    if scale_length_m is not None:
        if isinstance(scale_length_m, (tuple, list)) and len(scale_length_m) == 2:
            return float(sum(scale_length_m)) / 2.0
        if isinstance(scale_length_m, (int, float)) and scale_length_m > 0:
            return float(scale_length_m)
    for s_info in SCALES.values():
        speeds = s_info.speeds
        if len(speeds) == len(string_speeds) and np.allclose(speeds, string_speeds, rtol=0.005):
            return s_info.scale_m
    if len(string_speeds) == 5:
        sc_30 = SCALES.get("30in")
        if sc_30 and np.allclose(string_speeds[:4], sc_30.speeds, rtol=0.005):
            return 0.762
        sc_32 = SCALES.get("32in")
        if sc_32 and np.allclose(string_speeds[:4], sc_32.speeds, rtol=0.005):
            return 0.8128
    return 0.8636


def infer_string_names(
    string_speeds: Sequence[float], scale_length_m: float | None = None
) -> list[str]:
    """Infers note names for each string in string_speeds based on physical tuning physics."""
    n = len(string_speeds)
    if scale_length_m is not None and scale_length_m > 0:
        l_eff = scale_length_m
    else:
        if n == 4:
            for s_key in ["30in", "32in", "34in", "multiscale", "upright"]:
                sc = SCALES.get(s_key)
                if sc and np.allclose(string_speeds, sc.speeds, rtol=0.005):
                    return ["E", "A", "D", "G"]
        elif n == 5:
            if np.allclose(string_speeds, [53.28, 71.16, 95.0, 126.81, 169.27], rtol=0.005):
                return ["B", "E", "A", "D", "G"]
            if np.allclose(string_speeds, [58.02, 75.88, 99.19, 129.60, 169.27], rtol=0.005):
                return ["B", "E", "A", "D", "G"]
            sc_multi = SCALES.get("multiscale_super")
            if sc_multi and np.allclose(string_speeds, sc_multi.speeds, rtol=0.005):
                return ["B", "E", "A", "D", "G"]
            if np.allclose(string_speeds, [71.16, 95.0, 126.81, 169.27, 225.69], rtol=0.005):
                return ["E", "A", "D", "G", "C"]
            if np.allclose(string_speeds, [62.79, 83.82, 111.89, 149.35, 199.36], rtol=0.005):
                return ["E", "A", "D", "G", "C"]
        elif n == 6 and np.allclose(
            string_speeds, [53.28, 71.16, 95.0, 126.81, 169.27, 225.69], rtol=0.005
        ):
            return ["B", "E", "A", "D", "G", "C"]

        l_eff = resolve_scale_length(string_speeds, scale_length_m)

    names: list[str] = []
    for v in string_speeds:
        f0 = v / (2.0 * l_eff)
        names.append(pitch_to_note_name(f0))
    return names


def compute_dispersive_wave_speed(
    freqs: Sequence[float] | np.ndarray,
    v0: float,
    string_name: str | None = None,
    f0: float | None = None,
    scale_length_m: float | None = None,
) -> np.ndarray:
    """
    Computes frequency-dependent transverse wave speed v(f) accounting for flexural bending stiffness:
    v(f) = v0 * sqrt(1 + B_s * (f / f0)^2 / (1 + (f / 3500)^2))
    """
    f = np.asarray(freqs, dtype=np.float64)
    if f0 is None or f0 <= 0:
        l_eff = (
            scale_length_m if (scale_length_m is not None and scale_length_m > 0) else 0.8636
        )
        f0 = max(v0 / (2.0 * l_eff), 15.0)

    b_s = get_inharmonicity_for_f0(f0)
    f_disp_max = 3500.0
    disp_factor = 1.0 + b_s * ((f / f0) ** 2) / (1.0 + (f / f_disp_max) ** 2)
    return v0 * np.sqrt(disp_factor)
