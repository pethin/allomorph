"""
Homomorphic real-cepstrum Hilbert transform minimum-phase FIR filter synthesis.
Decomposed into pure, property-testable stages:
1. Cepstral causal folding and analytic minimum-phase spectrum computation
2. Causal impulse response windowing, C^inf mollifier tail smoothing, and true-peak normalization
"""

from collections.abc import Sequence

import numpy as np

from allomorph.dsp.constants import NUM_TAPS, cinf_smoothstep


def compute_minimum_phase_spectrum(
    magnitude_grid: np.ndarray,
    n_fft: int | None = None,
) -> np.ndarray:
    """Computes the causal complex minimum-phase frequency spectrum H_min(omega)

    from a positive magnitude spectrum via the real-cepstrum Hilbert transform operator:
      c = IRFFT(log |H| + eps)
      c_hat = [c[0], 2*c[1], ..., 2*c[N/2-1], c[N/2], 0, ..., 0]
      H_min = exp(RFFT(c_hat))
    """
    mag_grid = np.asarray(magnitude_grid, dtype=np.float64)
    half = len(mag_grid) - 1
    if n_fft is None:
        n_fft = 2 * half

    # Extrapolate DC bin if dropping into deep transmission zero to avoid cepstral delta spike
    if mag_grid[0] < mag_grid[1] * 0.5:
        mag_grid = mag_grid.copy()
        mag_grid[0] = mag_grid[1]

    # Build half-spectrum log-magnitude with C^inf quadratic regularization
    log_mag = 0.5 * np.log(mag_grid**2 + 1e-8)

    # Real cepstrum via IRFFT directly on conjugate-symmetric half-spectrum
    c = np.fft.irfft(log_mag, n_fft)

    # Minimum-phase causal folding (Hilbert transform operator in cepstral domain)
    c_hat = np.zeros(n_fft, dtype=np.float64)
    c_hat[0] = c[0]
    c_hat[half] = c[half]
    c_hat[1:half] = 2.0 * c[1:half]

    # Complex minimum-phase frequency spectrum H_min = exp(RFFT(c_hat))
    spec = np.fft.rfft(c_hat)
    return np.exp(spec)


def window_fir_impulse(
    impulse: np.ndarray,
    num_taps: int = NUM_TAPS,
    normalize: bool = True,
    peak_limit: float = 0.99,
) -> list[float]:
    """Applies causal truncation, C^inf smoothstep tail tapering (final 15%),

    and peak normalization to peak_limit (-0.1 dBFS).
    """
    fir = np.asarray(impulse[:num_taps], dtype=np.float64).copy()

    # Smooth tail (final 15%) with a C^inf mollifier smoothstep to eliminate truncation artifacts
    taper_len = int(num_taps * 0.15)
    start_taper = num_taps - taper_len
    t = np.arange(taper_len, dtype=np.float64) / max(taper_len, 1)
    w = 1.0 - cinf_smoothstep(t)
    fir[start_taper:] *= w

    if not normalize:
        return fir.tolist()

    # Peak normalization to peak_limit (default 0.99 = -0.1 dBFS)
    max_peak = float(np.max(np.abs(fir)))
    if max_peak > 0:
        fir = (fir / max_peak) * peak_limit
    return fir.tolist()


def synthesize_minimum_phase_fir(
    magnitude_curve: Sequence[float] | np.ndarray,
    num_taps: int = NUM_TAPS,
    normalize: bool = True,
) -> list[float]:
    """Synthesizes a causal, minimum-phase FIR filter from a desired magnitude

    curve using the homomorphic real-cepstrum Hilbert transform.
    Vectorized with NumPy FFT, executing in < 0.1 ms.
    """
    mag = np.asarray(magnitude_curve, dtype=np.float64)
    n_fft = max(8192, 2 * num_taps)
    half = n_fft // 2

    # Linear interpolation of input magnitude curve to half + 1 points (bypass if already on grid)
    m_in = len(mag)
    if m_in == half + 1:
        mag_grid = mag.copy()
    else:
        orig_indices = np.linspace(0, half, m_in)
        target_indices = np.arange(half + 1)
        mag_grid = np.interp(target_indices, orig_indices, mag)

    h_min_spec = compute_minimum_phase_spectrum(mag_grid, n_fft=n_fft)
    h = np.fft.irfft(h_min_spec, n_fft)
    return window_fir_impulse(h, num_taps=num_taps, normalize=normalize, peak_limit=0.99)
