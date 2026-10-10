"""
Allomorph - MNA Smooth Real-Analytic Mathematical Primitives
Provides C^inf smooth soft-knee limiters and algebraic bounds.
"""

from typing import overload

import numpy as np


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
