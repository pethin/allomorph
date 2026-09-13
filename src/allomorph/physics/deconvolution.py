"""
Allomorph - Pickup Electrical Response & Deconvolution Engine
Computes 2nd-order electrical RLC frequency responses, anti-resonance
flattening biquads, and composite pickup deconvolution using NumPy.
"""

from collections.abc import Sequence

import numpy as np

from allomorph.config.schema import InstrumentConfig, PickupConfig


def numpy_pickup_electrical_response(
    freqs: Sequence[float] | np.ndarray, fr: float, q: float
) -> np.ndarray:
    """
    Computes 2nd-order electrical low-pass magnitude response using NumPy.
    |H_elec(f)| = 1 / sqrt((1 - (f/fr)^2)^2 + (f / (q * fr))^2)
    """
    f = np.asarray(freqs, dtype=np.float64)
    if fr is None or fr <= 0.0 or q is None or q <= 0.0:
        return np.ones_like(f, dtype=np.float64)
    x = f / float(fr)
    denom = np.sqrt((1.0 - x**2) ** 2 + (x / float(q)) ** 2)
    return 1.0 / denom


def numpy_pickup_anti_resonance(
    freqs: Sequence[float] | np.ndarray,
    fr: float,
    q_src: float,
    q_target: float = 1.0,
) -> np.ndarray:
    """
    Computes 2nd-order electrical anti-resonance filter using NumPy:
    H_anti(f) = sqrt((1 - (f/fr)^2)^2 + (f/(q_src*fr))^2) / sqrt((1 - (f/fr)^2)^2 + (f/(q_target*fr))^2)
    """
    f = np.asarray(freqs, dtype=np.float64)
    if fr is None or fr <= 0.0 or q_src is None or q_src <= 0.0:
        return np.ones_like(f, dtype=np.float64)
    x = f / float(fr)
    num = np.sqrt((1.0 - x**2) ** 2 + (x / float(q_src)) ** 2)
    den = np.sqrt((1.0 - x**2) ** 2 + (x / float(q_target)) ** 2)
    return num / den


def resolve_pickup_electrical_response_np(
    freqs: Sequence[float] | np.ndarray,
    pickup_cfg: PickupConfig,
    inst_cfg: InstrumentConfig,
) -> np.ndarray:
    """Resolves electrical frequency response for a source pickup using NumPy."""
    f = np.asarray(freqs, dtype=np.float64)
    if pickup_cfg.type == "composite":
        components = pickup_cfg.components
        if not components:
            return np.ones_like(f, dtype=np.float64)
        total_w = sum(c.weight for c in components)
        if total_w <= 0.0:
            return np.ones_like(f, dtype=np.float64)
        acc = np.zeros_like(f, dtype=np.float64)
        for comp in components:
            sub_id = comp.pickup
            if not sub_id or sub_id not in inst_cfg.pickups:
                raise KeyError(
                    f"Composite pickup '{pickup_cfg.name}' references component '{sub_id}' "
                    f"which is not defined in instrument '{inst_cfg.id}' pickups"
                )
            sub_w = comp.weight
            sub_p = inst_cfg.pickups[sub_id]
            sub_elec = resolve_pickup_electrical_response_np(f, sub_p, inst_cfg)
            acc += sub_elec * (sub_w / total_w)
        return acc

    fr = float(pickup_cfg.resonant_frequency_hz or 3000.0)
    q = float(pickup_cfg.q_factor or 1.35)
    return numpy_pickup_electrical_response(f, fr, q)


def resolve_pickup_electrical_deconvolution_np(
    freqs: Sequence[float] | np.ndarray,
    pickup_cfg: PickupConfig,
    inst_cfg: InstrumentConfig,
    q_target: float = 1.0,
) -> np.ndarray:
    """Resolves anti-resonance flattening filter for a source pickup using NumPy."""
    f = np.asarray(freqs, dtype=np.float64)
    if pickup_cfg.type == "composite":
        components = pickup_cfg.components
        for comp in components:
            if not comp.pickup or comp.pickup not in inst_cfg.pickups:
                raise KeyError(
                    f"Composite pickup '{pickup_cfg.name}' references component '{comp.pickup}' "
                    f"which is not defined in instrument '{inst_cfg.id}' pickups"
                )
        has_fr = any(
            bool(inst_cfg.pickups[comp.pickup].resonant_frequency_hz)
            for comp in components
            if comp.pickup
        )
        if not has_fr:
            return np.ones_like(f, dtype=np.float64)
        total_w = sum(c.weight for c in components)
        if total_w <= 0.0:
            return np.ones_like(f, dtype=np.float64)
        acc = np.zeros_like(f, dtype=np.float64)
        for comp in components:
            sub_id = comp.pickup
            if not sub_id or sub_id not in inst_cfg.pickups:
                raise KeyError(
                    f"Composite pickup '{pickup_cfg.name}' references component '{sub_id}' "
                    f"which is not defined in instrument '{inst_cfg.id}' pickups"
                )
            sub_w = comp.weight
            sub_p = inst_cfg.pickups[sub_id]
            sub_deconv = resolve_pickup_electrical_deconvolution_np(
                f, sub_p, inst_cfg, q_target=q_target
            )
            acc += sub_deconv * (sub_w / total_w)
        return acc

    fr = pickup_cfg.resonant_frequency_hz
    if fr is None or fr <= 0.0:
        return np.ones_like(f, dtype=np.float64)
    q_src = pickup_cfg.q_factor or 1.35
    return numpy_pickup_anti_resonance(f, fr, q_src, q_target=q_target)


pickup_electrical_response = numpy_pickup_electrical_response
pickup_anti_resonance = numpy_pickup_anti_resonance
resolve_pickup_electrical_response = resolve_pickup_electrical_response_np
resolve_pickup_electrical_deconvolution = resolve_pickup_electrical_deconvolution_np
