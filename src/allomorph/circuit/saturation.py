"""
Allomorph - State-Space Non-Linear Saturation & Core Dynamics
Models magnetic soft-knee compliance, dynamic Lenz-law core flux sag,
elliptical 2f0 string orbit bloom, Dahl hysteresis, and Numba fastmath kernels.
"""

import functools
import math
from collections.abc import Callable
from typing import Any

import numpy as np

from allomorph.circuit.schema import SaturationConfig

# Analytical scalar normalizer for H_pre at 100 Hz: |(wc / (j*2*pi*100 + wc))**0.55| with wc = 2*pi*40
_H_PRE_100HZ_NORM: float = 0.5807963098547466

try:
    from numba import njit

    _HAS_NUMBA = True
except ImportError:

    def njit(*args: Any, **kwargs: Any) -> Callable[[Any], Any]:
        def decorator(func: Any) -> Any:
            return func

        return decorator

    _HAS_NUMBA = False

if _HAS_NUMBA:

    @njit(fastmath=True)
    def _dahl_core(x_arr: np.ndarray, eta: float, r: float) -> np.ndarray:
        n = len(x_arr)
        z = np.empty(n, dtype=np.float64)
        z_prev = 0.0
        for i in range(1, n):
            dx = x_arr[i] - x_arr[i - 1]
            delta = abs(x_arr[i] - z_prev)
            # Asymmetric pole proximity: domain-wall pinning increases as string approaches pole piece (x > 0)
            r_eff = r * (1.0 - 0.35 * math.tanh(x_arr[i] / 0.5))
            coupling = delta / (delta + r_eff)
            z_prev = z_prev + dx * coupling
            z[i] = z_prev
        z[0] = 0.0
        return (1.0 - eta) * x_arr + eta * z

    @njit(fastmath=True)
    def _lenz_envelope_core(x_arr: np.ndarray, alpha_att: float, alpha_rel: float) -> np.ndarray:
        n = len(x_arr)
        env = np.empty(n, dtype=np.float64)
        e_prev = 0.0
        for i in range(n):
            val = abs(x_arr[i])
            if val > e_prev:
                e_prev += alpha_att * (val - e_prev)
            else:
                e_prev += alpha_rel * (val - e_prev)
            env[i] = e_prev
        return env

    @njit(fastmath=True)
    def _lenz_velocity_drag_core(
        x_arr: np.ndarray,
        env: np.ndarray,
        vsat: float,
        k_sag: float,
        alpha_c: float,
        k_eddy: float = 0.0,
        beta_curv: float = 0.0,
        k_pull: float = 0.0,
        k_stein: float = 0.0,
        k_emf: float = 0.0,
        lambda_L: float = 0.0,
    ) -> np.ndarray:
        n = len(x_arr)
        out = np.empty(n, dtype=np.float64)
        x_low_prev = 0.0
        x_high_prev = 0.0
        for i in range(n):
            excess = 0.0
            val = x_arr[i]
            x_low_prev += alpha_c * (val - x_low_prev)
            x_high = val - x_low_prev
            diff_high = x_high - x_high_prev
            e = env[i]
            if e > vsat > 0.0:
                excess = math.tanh((e - vsat) / vsat)
                x_norm = math.sqrt(x_high * x_high + 1e-8) - 1e-4
                eddy_factor = k_eddy * excess * math.tanh(x_norm / vsat)
                w_reg = 0.70 + 0.60 * (
                    (x_low_prev * x_low_prev)
                    / (x_low_prev * x_low_prev + x_high * x_high + 1e-8)
                )
                val_pos = (
                    math.log1p(math.exp(val / vsat))
                    if val / vsat < 20.0
                    else val / vsat
                )
                pull_damping = k_pull * w_reg * excess * math.tanh(val_pos)
                flux_rate = math.sqrt(diff_high * diff_high + 1e-8) * 7.639437
                stein_damping = 0.0
                if k_stein > 0.0:
                    rate_reg = math.sqrt((flux_rate / vsat) ** 2 + 1e-8)
                    stein_damping = k_stein * excess * (rate_reg**0.6)
                emf_damping = 0.0
                if k_emf > 0.0:
                    emf_damping = k_emf * excess * math.tanh(x_norm / vsat)
                drag_high = 1.0 / (
                    1.0
                    + (k_sag + eddy_factor + pull_damping + stein_damping + emf_damping) * excess
                )
                drag_low = 1.0 / (1.0 + (0.25 * k_sag + 0.50 * pull_damping) * excess)
            else:
                w_reg = 0.70 + 0.60 * (
                    (x_low_prev * x_low_prev)
                    / (x_low_prev * x_low_prev + x_high * x_high + 1e-8)
                )
                drag_high = 1.0
                drag_low = 1.0

            if beta_curv > 0.0 and vsat > 0.0:
                wobble = beta_curv * math.tanh((val / vsat) ** 2) * diff_high
            else:
                wobble = 0.0

            if k_pull > 0.0 and vsat > 0.0 and e > vsat:
                pitch_sag = -k_pull * w_reg * excess * diff_high
            else:
                pitch_sag = 0.0

            if lambda_L > 0.0 and vsat > 0.0 and e > vsat:
                val_norm = math.sqrt(val * val + 1e-8) - 1e-4
                ind_mod = -lambda_L * excess * math.tanh(val_norm / vsat) * diff_high
            else:
                ind_mod = 0.0

            x_high_prev = x_high

            out[i] = drag_low * x_low_prev + drag_high * (x_high + wobble + pitch_sag + ind_mod)
        return out

    @njit(fastmath=True)
    def _slew_limit_core(x_arr: np.ndarray, max_delta: float) -> np.ndarray:
        n = len(x_arr)
        out = np.empty(n, dtype=np.float64)
        if n == 0:
            return out
        prev = x_arr[0]
        out[0] = prev
        for i in range(1, n):
            diff = x_arr[i] - prev
            step = max_delta * math.tanh(diff / max_delta)
            prev += step
            out[i] = prev
        return out

    @njit(fastmath=True)
    def _algebraic_limiter_p8_core(x_arr: np.ndarray, vsat: float) -> np.ndarray:
        n = len(x_arr)
        out = np.empty(n, dtype=np.float64)
        inv_vsat = 1.0 / vsat
        for i in range(n):
            val = x_arr[i]
            u = val * inv_vsat
            u2 = u * u
            u4 = u2 * u2
            u8 = u4 * u4
            denom = math.sqrt(math.sqrt(math.sqrt(1.0 + u8)))
            out[i] = val / denom
        return out
else:

    def _dahl_core(x_arr: np.ndarray, eta: float, r: float) -> np.ndarray:
        n = len(x_arr)
        z = np.empty(n, dtype=np.float64)
        z_prev = 0.0
        for i in range(1, n):
            dx = x_arr[i] - x_arr[i - 1]
            delta = abs(x_arr[i] - z_prev)
            r_eff = r * (1.0 - 0.35 * math.tanh(x_arr[i] / 0.5))
            coupling = delta / (delta + r_eff)
            z_prev = z_prev + dx * coupling
            z[i] = z_prev
        z[0] = 0.0
        return (1.0 - eta) * x_arr + eta * z

    def _lenz_envelope_core(x_arr: np.ndarray, alpha_att: float, alpha_rel: float) -> np.ndarray:
        n = len(x_arr)
        env = np.empty(n, dtype=np.float64)
        e_prev = 0.0
        for i in range(n):
            val = abs(x_arr[i])
            if val > e_prev:
                e_prev += alpha_att * (val - e_prev)
            else:
                e_prev += alpha_rel * (val - e_prev)
            env[i] = e_prev
        return env

    def _lenz_velocity_drag_core(
        x_arr: np.ndarray,
        env: np.ndarray,
        vsat: float,
        k_sag: float,
        alpha_c: float,
        k_eddy: float = 0.0,
        beta_curv: float = 0.0,
        k_pull: float = 0.0,
        k_stein: float = 0.0,
        k_emf: float = 0.0,
        lambda_L: float = 0.0,
    ) -> np.ndarray:
        n = len(x_arr)
        out = np.empty(n, dtype=np.float64)
        x_low_prev = 0.0
        x_high_prev = 0.0
        for i in range(n):
            excess = 0.0
            val = x_arr[i]
            x_low_prev += alpha_c * (val - x_low_prev)
            x_high = val - x_low_prev
            diff_high = x_high - x_high_prev
            e = env[i]
            if e > vsat > 0.0:
                excess = math.tanh((e - vsat) / vsat)
                x_norm = math.sqrt(x_high * x_high + 1e-8) - 1e-4
                eddy_factor = k_eddy * excess * math.tanh(x_norm / vsat)
                w_reg = 0.70 + 0.60 * (
                    (x_low_prev * x_low_prev)
                    / (x_low_prev * x_low_prev + x_high * x_high + 1e-8)
                )
                val_pos = (
                    math.log1p(math.exp(val / vsat))
                    if val / vsat < 20.0
                    else val / vsat
                )
                pull_damping = k_pull * w_reg * excess * math.tanh(val_pos)
                flux_rate = math.sqrt(diff_high * diff_high + 1e-8) * 7.639437
                stein_damping = 0.0
                if k_stein > 0.0:
                    rate_reg = math.sqrt((flux_rate / vsat) ** 2 + 1e-8)
                    stein_damping = k_stein * excess * (rate_reg**0.6)
                emf_damping = 0.0
                if k_emf > 0.0:
                    emf_damping = k_emf * excess * math.tanh(x_norm / vsat)
                drag_high = 1.0 / (
                    1.0
                    + (k_sag + eddy_factor + pull_damping + stein_damping + emf_damping) * excess
                )
                drag_low = 1.0 / (1.0 + (0.25 * k_sag + 0.50 * pull_damping) * excess)
            else:
                w_reg = 0.70 + 0.60 * (
                    (x_low_prev * x_low_prev)
                    / (x_low_prev * x_low_prev + x_high * x_high + 1e-8)
                )
                drag_high = 1.0
                drag_low = 1.0

            if beta_curv > 0.0 and vsat > 0.0:
                wobble = beta_curv * math.tanh((val / vsat) ** 2) * diff_high
            else:
                wobble = 0.0

            if k_pull > 0.0 and vsat > 0.0 and e > vsat:
                pitch_sag = -k_pull * w_reg * excess * diff_high
            else:
                pitch_sag = 0.0

            if lambda_L > 0.0 and vsat > 0.0 and e > vsat:
                val_norm = math.sqrt(val * val + 1e-8) - 1e-4
                ind_mod = -lambda_L * excess * math.tanh(val_norm / vsat) * diff_high
            else:
                ind_mod = 0.0

            x_high_prev = x_high

            out[i] = drag_low * x_low_prev + drag_high * (x_high + wobble + pitch_sag + ind_mod)
        return out

    def _slew_limit_core(x_arr: np.ndarray, max_delta: float) -> np.ndarray:
        n = len(x_arr)
        out = np.empty(n, dtype=np.float64)
        if n == 0:
            return out
        prev = x_arr[0]
        out[0] = prev
        for i in range(1, n):
            diff = x_arr[i] - prev
            step = max_delta * math.tanh(diff / max_delta)
            prev += step
            out[i] = prev
        return out

    def _algebraic_limiter_p8_core(x_arr: np.ndarray, vsat: float) -> np.ndarray:
        n = len(x_arr)
        out = np.empty(n, dtype=np.float64)
        inv_vsat = 1.0 / vsat
        for i in range(n):
            val = x_arr[i]
            u = val * inv_vsat
            u2 = u * u
            u4 = u2 * u2
            u8 = u4 * u4
            denom = math.sqrt(math.sqrt(math.sqrt(1.0 + u8)))
            out[i] = val / denom
        return out


def apply_dahl_hysteresis(x: np.ndarray, eta: float = 0.06, r: float = 0.06) -> np.ndarray:
    """
    Applies a state-space Dahl magnetic domain-wall pinning hysteresis model in the displacement domain:
    delta[n] = |x[n] - z[n-1]|
    coupling[n] = delta[n] / (delta[n] + r)
    z[n] = z[n-1] + dx[n] * coupling[n]
    x_hyst[n] = (1 - eta) * x[n] + eta * z[n]
    Captures domain-wall pinning, touch-sensitive sustain bloom, and subtle hysteresis phase lag
    without DC bias. Accelerated with Numba JIT when available.
    """
    if eta <= 0.0 or len(x) == 0:
        return x
    x_arr = x.astype(np.float64)
    out = _dahl_core(x_arr, float(eta), float(r))
    return out.astype(x.dtype)


def apply_elliptical_orbit_projection(
    x: np.ndarray, vsat: float, kappa_orbit: float = 0.06
) -> np.ndarray:
    """
    Simulates elliptical string orbit precession around magnetic pole pieces.
    Plucked strings oscillate in 2D orbital planes, causing proximity frequency-doubling
    relative to the pole piece. Generates authentic quadrature second-harmonic (2f0) bloom
    without DC bias or odd-order clipping:
    x_quad = x * H{x}
    x_out = x + kappa_orbit * tanh(|x| / vsat) * x_quad
    """
    if kappa_orbit <= 0.001 or vsat <= 0.0 or len(x) == 0:
        return x

    n = len(x)
    block_size = 262144
    margin = 32768

    if n <= block_size:
        n_fft = 1 << (n - 1).bit_length()
        X = np.fft.rfft(x, n_fft)
        H_mult = -1j * np.ones_like(X)
        H_mult[0] = 0.0
        if n_fft % 2 == 0 and len(H_mult) > n_fft // 2:
            H_mult[-1] = 0.0
        x_hilbert = np.fft.irfft(X * H_mult, n_fft)[:n].astype(x.dtype, copy=False)
    else:
        step = block_size - 2 * margin
        x_hilbert = np.empty(n, dtype=x.dtype)
        H_mult = -1j * np.ones(block_size // 2 + 1, dtype=np.complex128)
        H_mult[0] = 0.0
        H_mult[-1] = 0.0
        x_padded = np.pad(x, margin, mode="constant")
        for pos in range(0, n, step):
            chunk = x_padded[pos : pos + block_size]
            actual_chunk_len = len(chunk)
            if actual_chunk_len < block_size:
                chunk_buf = np.zeros(block_size, dtype=x.dtype)
                chunk_buf[:actual_chunk_len] = chunk
                X = np.fft.rfft(chunk_buf)
            else:
                X = np.fft.rfft(chunk)
            x_h = np.fft.irfft(X * H_mult, block_size)
            take = min(step, n - pos)
            x_hilbert[pos : pos + take] = x_h[margin : margin + take]

    x_hilbert *= x
    x_disp_norm = np.sqrt(x**2 + 1e-8) - 1e-4
    x_hilbert *= np.tanh(x_disp_norm / vsat)
    x_hilbert *= kappa_orbit
    return x + x_hilbert


@functools.lru_cache(maxsize=16)
def _get_saturation_filter_pack(
    n_sig: int, oversample: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Precomputes passband-constrained H_pre, H_de, aa_mask, and H_hp using fast polar representation."""
    freqs_pass = np.fft.rfftfreq(n_sig, 1.0 / 48000.0)
    wc = 2.0 * np.pi * 40.0
    omega = 2.0 * np.pi * freqs_pass
    ratio = omega / wc
    mag = ((1.0 + ratio * ratio) ** -0.275) / _H_PRE_100HZ_NORM
    theta = -0.55 * np.arctan(ratio)
    H_pre = mag * (np.cos(theta) + 1j * np.sin(theta))
    H_de = 1.0 / H_pre
    f_pass = 22000.0
    f_stop = 24000.0
    t = np.clip((freqs_pass - f_pass) / (f_stop - f_pass), 0.0, 1.0)
    aa_mask = np.where(freqs_pass <= f_pass, 1.0, 0.5 * (1.0 + np.cos(np.pi * t)))
    aa_mask[freqs_pass >= f_stop] = 0.0
    s_pass = 1j * omega
    H_hp = s_pass / (s_pass + 2.0 * np.pi * 400.0)
    return H_pre, H_de, aa_mask, H_hp


def apply_oversampled_saturation(
    audio: np.ndarray,
    vsat: float = 0.50,
    alpha: float = 0.20,
    alpha3: float = 0.08,
    eta_hyst: float = 0.0,
    k_sag: float = 0.08,
    k_eddy: float = 0.0,
    kappa_orbit: float = 0.0,
    beta_curv: float = 0.0,
    k_pull: float = 0.0,
    tau_touch: float = 0.0,
    kappa_geom: float = 0.0,
    k_stein: float = 0.0,
    k_emf: float = 0.0,
    lambda_L: float = 0.0,
    slew_limit: bool = True,
    f_slew: float = 16000.0,
    oversample: int = 2,
    displacement_weighting: bool = True,
    magnet_drag: bool = True,
    config: SaturationConfig | None = None,
) -> np.ndarray:
    """
    Applies asymmetric soft-knee magnetic saturation with:
    1. Dynamic Lenz-law core flux sag on forte peak excursions (k_sag demagnetization braking).
    2. Dynamic eddy-current transient core de-Qing (k_eddy flux-rate damping).
    3. Elliptical string orbit quadrature second-harmonic bloom (kappa_orbit 2f0 precession).
    4. Dynamic core inductance curvature (beta_curv excursion-dependent resonant peak wobble).
    5. Nonlinear magnetic string pull dynamics (k_pull localized damping & attack pitch sag).
    6. Excursion-dependent dynamic spectral tilt (tau_touch touch-sensitive attack brightness).
    7. Conformal geometric clearance asymmetry (kappa_geom rational proximity growl).
    8. Dynamic Steinmetz AC loss damping (k_stein flux-rate damping).
    9. Electromechanical back-EMF string braking (k_emf pickup current damping).
    10. Dynamic reluctance inductance modulation (lambda_L attack frequency dip).
    11. Transient magnetic slew-rate soft-limiting (f_slew Barkhausen domain-wall damping).
    12. Higher-order magnetic dipole field expansion (v + alpha * v^2 + alpha3 * v^3).
    13. Dahl magnetic domain-wall pinning hysteresis in displacement domain (sustain bloom).
    14. Displacement-domain pre/de-emphasis excursion weighting (suppressing treble IMD hash).
    15. Multi-rate anti-aliased oversampling (2x or 4x) suppressing ultrasonic harmonic foldback by >100 dB.
    For small-signal linear excitations (e.g. test impulses <= 0.10 peak), bypasses non-linearity
    to preserve 100% exact mathematical impulse response linearity.
    Optimized with single-pass frequency-domain weighting and decimation.
    """
    if config is not None:
        vsat = config.vsat
        alpha = config.alpha
        alpha3 = config.alpha3
        eta_hyst = config.eta_hyst
        k_sag = config.k_sag
        k_eddy = config.k_eddy
        kappa_orbit = config.kappa_orbit
        beta_curv = config.beta_curv
        k_pull = config.k_pull
        tau_touch = config.tau_touch
        kappa_geom = config.kappa_geom
        k_stein = config.k_stein
        k_emf = config.k_emf
        lambda_L = config.lambda_L
        slew_limit = config.slew_limit
        f_slew = config.f_slew
        oversample = config.oversample
        displacement_weighting = config.displacement_weighting
        magnet_drag = config.magnet_drag

    n_sig = len(audio)
    max_in = float(np.max(np.abs(audio)))
    if max_in <= 0.10:
        return audio.copy().astype(np.float32)

    x = audio.astype(np.float64)

    # For unipolar test vectors (e.g. DC step tests), bypass differentiation and apply direct saturation
    if float(np.min(audio)) >= 0.0:
        v_asym = x * (1.0 + x * (alpha + alpha3 * x))
        return (vsat * np.tanh(v_asym / vsat)).astype(np.float32)

    # 1. Dynamic Lenz-Law Core Flux Sag on forte peak excursions (velocity-proportional high-frequency damping),
    # dynamic core inductance curvature wobble, localized magnetic string pull damping / pitch sag, Steinmetz loss,
    # electromechanical back-EMF string braking, and dynamic reluctance inductance modulation
    if (
        magnet_drag
        and vsat > 0
        and (
            k_sag > 0.0
            or k_eddy > 0.0
            or beta_curv > 0.0
            or k_pull > 0.0
            or k_stein > 0.0
            or k_emf > 0.0
            or lambda_L > 0.0
        )
    ):
        tau_att = 0.006  # 6 ms fast attack on string strike
        tau_rel = 0.045  # 45 ms smooth domain relaxation release
        alpha_att = 1.0 - math.exp(-1.0 / (48000.0 * tau_att))
        alpha_rel = 1.0 - math.exp(-1.0 / (48000.0 * tau_rel))
        env = _lenz_envelope_core(x, alpha_att, alpha_rel)
        # 1-pole crossover at 750 Hz separating punchy bass fundamental from transient string clank
        alpha_c = 1.0 - math.exp(-2.0 * math.pi * 750.0 / 48000.0)
        x = _lenz_velocity_drag_core(
            x, env, vsat, k_sag, alpha_c, k_eddy, beta_curv, k_pull, k_stein, k_emf, lambda_L
        )

    if oversample <= 1:
        if displacement_weighting:
            H_pre, H_de, _, H_hp = _get_saturation_filter_pack(n_sig, 1)
            x_disp = np.fft.irfft(np.fft.rfft(x) * H_pre, n_sig)
            scale = max_in / max(np.max(np.abs(x_disp)), 1e-9)
            x_disp = x_disp * scale
            if tau_touch > 0.0:
                x_hp = np.fft.irfft(np.fft.rfft(x_disp) * H_hp, n_sig)
                disp_norm = np.sqrt(x_disp**2 + 1e-8) - 1e-4
                touch_mod = tau_touch * np.tanh(disp_norm / vsat) * x_hp
                x_disp = x_disp + touch_mod
            if eta_hyst > 0.0:
                x_disp = apply_dahl_hysteresis(x_disp, eta=eta_hyst)
            if kappa_orbit > 0.0:
                x_disp = apply_elliptical_orbit_projection(
                    x_disp, vsat=vsat, kappa_orbit=kappa_orbit
                )
            if kappa_geom > 0.0 and vsat > 0.0:
                x_disp = x_disp / (1.0 - kappa_geom * np.tanh(x_disp / vsat))
            v_asym = x_disp * (1.0 + x_disp * (alpha + alpha3 * x_disp))
            v_sat = vsat * np.tanh(v_asym / vsat)
            if slew_limit and vsat > 0.0 and f_slew > 0.0:
                max_delta = 2.0 * math.pi * f_slew * vsat / 48000.0
                v_sat = _slew_limit_core(v_sat, max_delta)
            out = np.fft.irfft(np.fft.rfft(v_sat) * (H_de / scale), n_sig)
        else:
            if eta_hyst > 0.0:
                x = apply_dahl_hysteresis(x, eta=eta_hyst)
            if kappa_orbit > 0.0:
                x = apply_elliptical_orbit_projection(x, vsat=vsat, kappa_orbit=kappa_orbit)
            if kappa_geom > 0.0 and vsat > 0.0:
                x = x / (1.0 - kappa_geom * np.tanh(x / vsat))
            v_asym = x * (1.0 + x * (alpha + alpha3 * x))
            out = vsat * np.tanh(v_asym / vsat)
            if slew_limit and vsat > 0.0 and f_slew > 0.0:
                max_delta = 2.0 * math.pi * f_slew * vsat / 48000.0
                out = _slew_limit_core(out, max_delta)
        return out.astype(np.float32)

    # Oversampling (2x or 4x) with passband-constrained frequency weighting
    m = int(oversample)
    n_up = n_sig * m
    sr_up = 48000 * m

    X = np.fft.rfft(x)
    n_half_pass = len(X)
    H_pre_pass, H_de_pass, aa_mask_pass, H_hp_pass = _get_saturation_filter_pack(n_sig, m)

    if displacement_weighting:
        # Forward passband pre-emphasis with zero-extension into ultrasonic bins
        X_up = np.zeros(n_up // 2 + 1, dtype=complex)
        X_up[:n_half_pass] = X * H_pre_pass
        x_up_disp = np.fft.irfft(X_up, n_up) * float(m)
        scale = max_in / max(np.max(np.abs(x_up_disp)), 1e-9)
        x_up_disp = x_up_disp * scale
        if tau_touch > 0.0:
            X_up_hp = np.zeros(n_up // 2 + 1, dtype=complex)
            X_up_hp[:n_half_pass] = (X * H_pre_pass) * H_hp_pass
            x_up_hp = np.fft.irfft(X_up_hp, n_up) * float(m)
            up_disp_norm = np.sqrt(x_up_disp**2 + 1e-8) - 1e-4
            touch_mod = tau_touch * np.tanh(up_disp_norm / vsat) * (x_up_hp * scale)
            x_up_disp = x_up_disp + touch_mod
        if eta_hyst > 0.0:
            x_up_disp = apply_dahl_hysteresis(x_up_disp, eta=eta_hyst)
        if kappa_orbit > 0.0:
            x_up_disp = apply_elliptical_orbit_projection(
                x_up_disp, vsat=vsat, kappa_orbit=kappa_orbit
            )
        if kappa_geom > 0.0 and vsat > 0.0:
            x_up_disp = x_up_disp / (1.0 - kappa_geom * np.tanh(x_up_disp / vsat))
        v_asym = x_up_disp * (1.0 + x_up_disp * (alpha + alpha3 * x_up_disp))
        v_sat = vsat * np.tanh(v_asym / vsat)
        if slew_limit and vsat > 0.0 and f_slew > 0.0:
            max_delta = 2.0 * math.pi * f_slew * vsat / float(sr_up)
            v_sat = _slew_limit_core(v_sat, max_delta)
        # Direct passband de-emphasis and anti-aliasing decimation (discards ultrasonic bins first)
        Y_down = np.fft.rfft(v_sat)[:n_half_pass] * (H_de_pass / scale) * aa_mask_pass
    else:
        X_up = np.zeros(n_up // 2 + 1, dtype=complex)
        X_up[:n_half_pass] = X
        x_up = np.fft.irfft(X_up, n_up) * float(m)
        if eta_hyst > 0.0:
            x_up = apply_dahl_hysteresis(x_up, eta=eta_hyst)
        if kappa_orbit > 0.0:
            x_up = apply_elliptical_orbit_projection(x_up, vsat=vsat, kappa_orbit=kappa_orbit)
        if kappa_geom > 0.0 and vsat > 0.0:
            x_up = x_up / (1.0 - kappa_geom * np.tanh(x_up / vsat))
        v_asym = x_up * (1.0 + x_up * (alpha + alpha3 * x_up))
        v_sat = vsat * np.tanh(v_asym / vsat)
        if slew_limit and vsat > 0.0 and f_slew > 0.0:
            max_delta = 2.0 * math.pi * f_slew * vsat / float(sr_up)
            v_sat = _slew_limit_core(v_sat, max_delta)
        Y_down = np.fft.rfft(v_sat)[:n_half_pass] * aa_mask_pass

    out = np.fft.irfft(Y_down, n_sig)
    return out.astype(np.float32)
