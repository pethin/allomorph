"""
Allomorph - String Mechanics & Wave Dispersion Modeling
Computes differential string damping and tension compliance, longitudinal clank resonance,
inharmonicity B_s interpolation, scale-length conversions, and dispersive wave speeds.
"""

import math
from collections.abc import Sequence
from typing import overload

import numpy as np

from allomorph.config.scales import resolve_scale_range
from allomorph.config.schema import InstrumentConfig, ScaleConfig, StringPresetConfig
from allomorph.config.strings import get_voice_string
from allomorph.physics.schema import WaveSpeedContinuumPoint

__all__ = [
    "INHARMONICITY_ANCHORS_BS",
    "INHARMONICITY_ANCHORS_F0",
    "MEAN_BASS_F0",
    "NOTE_NAMES",
    "SENSOR_SCALE_EXPONENTS",
    "STRING_FUNDAMENTALS",
    "WaveSpeedContinuumPoint",
    "compute_dispersive_wave_speed",
    "compute_forward_string_transfer",
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
_RBF_A = np.exp(-((_RBF_EPSILON * _RBF_D) ** 2))
_RBF_WEIGHTS = np.linalg.solve(_RBF_A, _LOG_BS_ANCHORS)
MEAN_BASS_F0 = (
    66.9045  # Mean open-string fundamental frequency (E1=41.203, A1=55.000, D2=73.416, G2=97.999)
)


# Sensor scale and compliance coupling exponents derived from spatial derivative order:
# (kappa_exc, kappa_snap) = (1.0, 1.0) for transverse displacement/velocity (magnetic)
# (0.0, 0.0) for boundary shear force (bridge piezo) and direct (DI)
SENSOR_SCALE_EXPONENTS: dict[str, tuple[float, float]] = {
    "magnetic": (1.0, 1.0),
    "bridge_force": (0.0, 0.0),
    "direct": (0.0, 0.0),
}


def compute_forward_string_transfer(
    freqs: Sequence[float] | np.ndarray,
    string_cfg: StringPresetConfig,
    scale_length_inches: float = 34.0,
    sensor_type: str = "magnetic",
) -> np.ndarray:
    """Computes absolute forward viscoelastic string wrap damping transfer function H_wrap(f).

    Direct DI sensors define flat unity transfer (H_wrap = 1.0).
    Longitudinal core compression clank is an instantaneous mechanical attack transient,
    not a static linear frequency-domain filter, preserving pure aperture comb filtering.
    """
    f = np.asarray(freqs, dtype=np.float64)
    if sensor_type == "direct":
        return np.ones_like(f)

    # Forward Wrap Damping
    f_damp = float(string_cfg.damping_cutoff_hz)
    n_order = float(string_cfg.damping_order)
    return 1.0 / np.sqrt(1.0 + (f / max(f_damp, 100.0)) ** (2.0 * n_order))


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
        basis = np.exp(-((_RBF_EPSILON * d) ** 2))
        return 2.0 ** (basis @ _RBF_WEIGHTS)
    log_f0 = math.log2(max(f0, 15.0))
    d = np.abs(log_f0 - _LOG_F0_ANCHORS)
    basis = np.exp(-((_RBF_EPSILON * d) ** 2))
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
    elif isinstance(scale_length_m, (ScaleConfig, InstrumentConfig, str)):
        l_min_m, l_max_m = resolve_scale_range(scale_length_m)
    else:
        val = float(scale_length_m) if isinstance(scale_length_m, (int, float)) else 0.8636
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

    n = len(string_speeds)
    if n == 4:
        f0_cand = [41.2034, 55.0000, 73.4162, 97.9989]  # EADG
    elif n == 5:
        # Low B: (30.87, 41.20, ...) vs High C: (41.20, ..., 130.81)
        if string_speeds[0] < 60.0:
            f0_cand = [30.8677, 41.2034, 55.0000, 73.4162, 97.9989]  # BEADG
        else:
            f0_cand = [41.2034, 55.0000, 73.4162, 97.9989, 130.8128]  # EADGC
    elif n == 6:
        f0_cand = [30.8677, 41.2034, 55.0000, 73.4162, 97.9989, 130.8128]  # BEADGC
    else:
        return 0.8636

    l_estimates = [v / (2.0 * f) for v, f in zip(string_speeds, f0_cand)]
    median_l = float(np.median(l_estimates))
    for std_l in (0.762, 0.8128, 0.8636, 0.889):
        if abs(median_l - std_l) < 0.005:
            return std_l
    return 0.8636


def infer_string_names(
    string_speeds: Sequence[float],
    scale_length_m: float | tuple[float, float] | list[float] | None = None,
) -> list[str]:
    """Infers note names for each string in string_speeds based on physical tuning physics."""
    n = len(string_speeds)
    if n == 0:
        return []

    l_per_string: list[float] | None = None
    if isinstance(scale_length_m, (tuple, list)) and len(scale_length_m) == 2:
        l_min_m = float(min(scale_length_m))
        l_max_m = float(max(scale_length_m))
        if l_max_m - l_min_m > 1e-4 and n > 1:
            l_per_string = [l_max_m - (i / (n - 1)) * (l_max_m - l_min_m) for i in range(n)]

    if l_per_string is None:
        if n == 4:
            f0_cand = [41.2034, 55.0000, 73.4162, 97.9989]
        elif n == 5:
            f0_cand = (
                [30.8677, 41.2034, 55.0000, 73.4162, 97.9989]
                if string_speeds[0] < 60.0
                else [41.2034, 55.0000, 73.4162, 97.9989, 130.8128]
            )
        elif n == 6:
            f0_cand = [30.8677, 41.2034, 55.0000, 73.4162, 97.9989, 130.8128]
        else:
            f0_cand = None

        if f0_cand is not None:
            l_est = [v / (2.0 * f) for v, f in zip(string_speeds, f0_cand)]
            if l_est[0] - l_est[-1] > 0.03 and all(
                l_est[i] - l_est[i + 1] > 0.005 for i in range(n - 1)
            ):
                l_per_string = l_est

    if l_per_string is None:
        l_eff = resolve_scale_length(string_speeds, scale_length_m)
        l_per_string = [l_eff] * n

    names: list[str] = []
    for v, l_str in zip(string_speeds, l_per_string):
        f0 = v / (2.0 * l_str)
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
        l_eff = scale_length_m if (scale_length_m is not None and scale_length_m > 0) else 0.8636
        f0 = max(v0 / (2.0 * l_eff), 15.0)

    b_s = get_inharmonicity_for_f0(f0)
    f_disp_max = 3500.0
    disp_factor = 1.0 + b_s * ((f / f0) ** 2) / (1.0 + (f / f_disp_max) ** 2)
    return v0 * np.sqrt(disp_factor)
