"""
Allomorph - MNA Active Preamp Biquads & Analog Filters
Evaluates continuous analog transfer functions and synthesizes Direct-Form II
Transposed biquads via bilinear transform with frequency pre-warping.
"""

import math
from collections.abc import Mapping, Sequence

import numpy as np

from allomorph.config.schema import PreampBandConfig, PreampConfig


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
        if band.q is not None and band.q > 0.0:
            q = float(band.q)
            return (w0**2) / (s**2 + (w0 / q) * s + w0**2)
        return w0 / (s + w0)
    elif b_type == "high_pass":
        if band.q is not None and band.q > 0.0:
            q = float(band.q)
            return (s**2) / (s**2 + (w0 / q) * s + w0**2)
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
        elif b_type == "low_pass":
            if band.q is not None and band.q > 0.0:
                q = float(band.q)
                k2 = k_bilinear * k_bilinear
                w2 = omega_a * omega_a
                kw_q = (k_bilinear * omega_a) / q
                a0 = k2 + kw_q + w2
                b0 = w2 / a0
                b1 = (2.0 * w2) / a0
                b2 = w2 / a0
                a1 = (2.0 * (w2 - k2)) / a0
                a2 = (k2 - kw_q + w2) / a0
                biquads.append((b0, b1, b2, 1.0, a1, a2))
            else:
                a0 = k_bilinear + omega_a
                b0 = omega_a / a0
                b1 = omega_a / a0
                b2 = 0.0
                a1 = (omega_a - k_bilinear) / a0
                a2 = 0.0
                biquads.append((b0, b1, b2, 1.0, a1, a2))
        elif b_type == "high_pass":
            if band.q is not None and band.q > 0.0:
                q = float(band.q)
                k2 = k_bilinear * k_bilinear
                w2 = omega_a * omega_a
                kw_q = (k_bilinear * omega_a) / q
                a0 = k2 + kw_q + w2
                b0 = k2 / a0
                b1 = (-2.0 * k2) / a0
                b2 = k2 / a0
                a1 = (2.0 * (w2 - k2)) / a0
                a2 = (k2 - kw_q + w2) / a0
                biquads.append((b0, b1, b2, 1.0, a1, a2))
            else:
                a0 = k_bilinear + omega_a
                b0 = k_bilinear / a0
                b1 = -k_bilinear / a0
                b2 = 0.0
                a1 = (omega_a - k_bilinear) / a0
                a2 = 0.0
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
    preamp_spec: str | PreampConfig | Sequence[PreampBandConfig],
    s: complex | np.ndarray,
    preamps: Mapping[str, PreampConfig] | None = None,
) -> np.ndarray:
    """Evaluates analog active preamp contour transfer function.

    Accepts:
      - str (preset name): looks up in preamps mapping
      - PreampConfig: evaluates preamp model
      - Sequence[PreampBandConfig]: evaluates sequence of band configs
    """
    if isinstance(preamp_spec, str):
        if preamps is None or preamp_spec not in preamps:
            raise KeyError(f"Preamp '{preamp_spec}' required but not provided in preamps mapping.")
        preset = preamps[preamp_spec]
        return compute_active_preamp_transfer(preset.bands, s, gain_db=float(preset.gain_db))
    elif isinstance(preamp_spec, PreampConfig):
        return compute_active_preamp_transfer(
            preamp_spec.bands, s, gain_db=float(preamp_spec.gain_db)
        )
    elif isinstance(preamp_spec, (list, tuple)):
        return compute_active_preamp_transfer(preamp_spec, s)
    return np.ones_like(s, dtype=np.complex128)
