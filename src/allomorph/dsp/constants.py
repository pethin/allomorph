"""
Allomorph DSP Constants, Numerical Utilities, and Real-Analytic Mollifiers.
"""

import math
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any, overload

import numpy as np

if TYPE_CHECKING:
    prange = range

    def njit(*args: Any, **kwargs: Any) -> Callable[[Any], Any]:
        def decorator(func: Any) -> Any:
            return func

        return decorator

    _HAS_NUMBA = False
else:
    try:
        from numba import njit, prange

        _HAS_NUMBA = True
    except ImportError:

        def njit(*args: Any, **kwargs: Any) -> Callable[[Any], Any]:
            def decorator(func: Any) -> Any:
                return func

            return decorator

        def prange(*args: Any) -> Any:
            return range(*args)

        _HAS_NUMBA = False

FS = 48000
NUM_TAPS = 4096
NYQ = FS / 2.0
FREQS = [i * (NYQ / (NUM_TAPS - 1)) for i in range(NUM_TAPS)]

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
AUDIO_DIR = REPO_ROOT / "audio"
DEFAULT_INPUT_PATH = AUDIO_DIR / "input.wav"


@njit(fastmath=True, nogil=True)
def _cinf_smoothstep_kernel(t: np.ndarray, out: np.ndarray) -> None:
    n = len(t)
    for i in range(n):
        ti = t[i]
        if ti <= 0.0:
            out[i] = 0.0
        elif ti >= 1.0:
            out[i] = 1.0
        else:
            arg = (1.0 - 2.0 * ti) / (ti * (1.0 - ti))
            if arg > 80.0:
                arg = 80.0
            elif arg < -80.0:
                arg = -80.0
            out[i] = 1.0 / (1.0 + np.exp(arg))


@overload
def cinf_smoothstep(t: float) -> float: ...


@overload
def cinf_smoothstep(t: np.ndarray) -> np.ndarray: ...


def cinf_smoothstep(t: float | np.ndarray) -> float | np.ndarray:
    """Computes the real-analytic C^infinity smoothstep mollifier transition function S(t).

    All derivatives of all orders are identically zero at t <= 0 and t >= 1, eliminating
    high-order spectral boundary leakage into digital silence.
    """
    if isinstance(t, (int, float, np.floating, np.integer)):
        tf = float(t)
        if tf <= 0.0:
            return 0.0
        elif tf >= 1.0:
            return 1.0
        arg = (1.0 - 2.0 * tf) / (tf * (1.0 - tf))
        arg = min(max(arg, -80.0), 80.0)
        return 1.0 / (1.0 + math.exp(arg))

    t_arr = np.asarray(t, dtype=np.float64)
    out = np.zeros_like(t_arr, dtype=np.float64)
    _cinf_smoothstep_kernel(t_arr, out)
    return out


_cinf_smoothstep = cinf_smoothstep


def apply_cinf_fades(sig: np.ndarray, fade_len: int) -> np.ndarray:
    """Applies C^infinity smooth mollifier fade-in and fade-out to guarantee zero high-order
    spectral leakage and infinitely differentiable boundary transitions into digital silence.
    """
    if len(sig) < 2 * fade_len or fade_len <= 0:
        return sig
    out = sig.copy()
    t_in = np.linspace(0.0, 1.0, fade_len, endpoint=False)
    w_in = _cinf_smoothstep(t_in)
    out[:fade_len] *= w_in
    out[-fade_len:] *= w_in[::-1]
    return out


_apply_cinf_fades = apply_cinf_fades


@njit(fastmath=True, nogil=True)
def _biquad_filter_kernel(
    x: np.ndarray,
    b0: float,
    b1: float,
    b2: float,
    a1: float,
    a2: float,
    y: np.ndarray,
) -> None:
    """Direct Form II Transposed biquad filter implementation:
    y[n] = b0 * x[n] + s1
    s1   = b1 * x[n] - a1 * y[n] + s2
    s2   = b2 * x[n] - a2 * y[n]
    """
    s1 = 0.0
    s2 = 0.0
    n = len(x)
    for i in range(n):
        xi = x[i]
        yi = b0 * xi + s1
        s1 = b1 * xi - a1 * yi + s2
        s2 = b2 * xi - a2 * yi
        y[i] = yi


def apply_biquad(
    x: np.ndarray,
    b: tuple[float, float, float] | list[float] | np.ndarray,
    a: tuple[float, float, float] | list[float] | np.ndarray,
) -> np.ndarray:
    """Applies a 2nd-order IIR biquad filter using Transposed Direct Form II."""
    x_arr = np.asarray(x, dtype=np.float64)
    y = np.empty_like(x_arr)
    a0 = float(a[0])
    b0 = float(b[0]) / a0
    b1 = float(b[1]) / a0
    b2 = float(b[2]) / a0
    a1 = float(a[1]) / a0
    a2 = float(a[2]) / a0
    _biquad_filter_kernel(x_arr, b0, b1, b2, a1, a2, y)
    return y
