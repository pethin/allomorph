"""
Physical bass string pluck and articulation synthesis engine.
Decomposed into pure, property-testable stages:
1. Inharmonic modal frequency series calculation
2. Dynamic tension attack pitch sag phase modulation
3. Nonlinear fret contact buzz soft knee
4. Ringout exponential decay envelopes
5. Physical bass articulations (plucks, harmonics, dyads, ghost notes, glissandi, vibrato)
"""

import math

import numpy as np

from allomorph.dsp.constants import FS, apply_cinf_fades, njit, prange


@njit(fastmath=True, parallel=True)
def _accumulate_pluck_modes_simd(
    sig: np.ndarray,
    t: np.ndarray,
    phase_sag: np.ndarray,
    fns: np.ndarray,
    hws: np.ndarray,
    drvs: np.ndarray,
    drhs: np.ndarray,
    hs: np.ndarray,
    split_hz: float,
) -> None:
    n = len(t)
    num_modes = len(fns)
    two_pi = 2.0 * np.pi
    two_pi_split = two_pi * split_hz
    for i in prange(n):
        ti = t[i]
        sag_i = phase_sag[i]
        acc = 0.0
        for m in range(num_modes):
            fn = fns[m]
            hw = hws[m]
            drv = drvs[m]
            drh = drhs[m]
            h = hs[m]
            phi_base = two_pi * (fn * ti + h * sag_i)
            decay_v = np.exp(-ti * drv)
            decay_h = np.exp(-ti * drh)
            acc += hw * (
                0.65 * np.sin(phi_base) * decay_v
                + 0.35 * np.sin(phi_base + two_pi_split * ti) * decay_h
            )
        sig[i] += acc


@njit(fastmath=True)
def _accumulate_ringout_phasors_simd(
    sig: np.ndarray,
    n: int,
    dt: float,
    fns: np.ndarray,
    hws: np.ndarray,
    drvs: np.ndarray,
    drhs: np.ndarray,
    split_hz: float,
) -> None:
    num_modes = len(fns)
    two_pi = 2.0 * np.pi
    for m in range(num_modes):
        fn = fns[m]
        hw = hws[m]
        drv = drvs[m]
        drh = drhs[m]

        w_v = two_pi * fn
        mult_v = np.exp(-drv * dt) * (np.cos(w_v * dt) + 1j * np.sin(w_v * dt))
        w_h = two_pi * (fn + split_hz)
        mult_h = np.exp(-drh * dt) * (np.cos(w_h * dt) + 1j * np.sin(w_h * dt))

        z_v = 1.0 + 0.0j
        z_h = 1.0 + 0.0j
        c_v = 0.65 * hw
        c_h = 0.35 * hw
        for i in range(n):
            sig[i] += c_v * z_v.imag + c_h * z_h.imag
            z_v *= mult_v
            z_h *= mult_h


@njit(fastmath=True, parallel=True)
def _accumulate_vibrato_modes_simd(
    sig: np.ndarray,
    t: np.ndarray,
    base_phase: np.ndarray,
    fns: np.ndarray,
    hws: np.ndarray,
    drs: np.ndarray,
    hs: np.ndarray,
) -> None:
    n = len(t)
    num_modes = len(fns)
    for i in prange(n):
        ti = t[i]
        bp_i = base_phase[i]
        acc = 0.0
        for m in range(num_modes):
            hw = hws[m]
            dr = drs[m]
            h = hs[m]
            decay = np.exp(-ti * dr)
            acc += hw * np.sin(h * bp_i) * decay
        sig[i] += acc


@njit(fastmath=True, parallel=True)
def _accumulate_glissando_modes_simd(
    sig: np.ndarray,
    base_phase: np.ndarray,
    ks: np.ndarray,
    k_weights: np.ndarray,
) -> None:
    n = len(base_phase)
    num_k = len(ks)
    for i in prange(n):
        bp_i = base_phase[i]
        acc = 0.0
        for m in range(num_k):
            k = ks[m]
            kw = k_weights[m]
            acc += kw * np.sin(k * bp_i)
        sig[i] += acc


def compute_modal_frequencies(
    f0: float,
    inharmonicity_b: float,
    num_harmonics: int,
) -> np.ndarray:
    """Computes physical string inharmonic modal frequencies: fn = n * f0 * sqrt(1 + B * n^2)."""
    if num_harmonics <= 0:
        return np.empty(0, dtype=np.float64)
    ns = np.arange(1, num_harmonics + 1, dtype=np.float64)
    return ns * f0 * np.sqrt(np.maximum(1.0 + inharmonicity_b * (ns**2), 1e-12))


def apply_attack_pitch_sag(
    t: np.ndarray,
    pitch_sag_hz: float = 2.5,
    tau_sag: float = 0.08,
) -> np.ndarray:
    """Computes dynamic tension attack pitch sag phase modulation:

    phi_sag(t) = -sag * tau * (exp(-t / tau) - 1.0)
    """
    return -pitch_sag_hz * tau_sag * (np.exp(-t / max(tau_sag, 1e-4)) - 1.0)


def apply_fret_buzz_nonlinearity(
    t: np.ndarray,
    f0: float,
    f_buzz_hz: float = 2600.0,
    buzz_tau: float = 0.018,
) -> np.ndarray:
    """Computes C^inf softplus kinematic contact force for string striking fretwire on negative excursions."""
    burst = np.sin(2.0 * np.pi * f_buzz_hz * t) * np.exp(-t / 0.004)
    contact_force = np.logaddexp(0.0, 12.0 * (-np.sin(2.0 * np.pi * f0 * t))) / 12.0
    collision = (contact_force**3) * np.exp(-t / max(buzz_tau, 1e-4))
    return 0.10 * burst + 0.10 * collision


def generate_exponential_ringout_envelope(
    duration_sec: float,
    sample_rate: int = FS,
    t60_sec: float = 4.0,
) -> np.ndarray:
    """Generates an authentic exponential ring-out decay envelope falling by 60 dB over t60_sec."""
    n = int(duration_sec * sample_rate)
    if n <= 0:
        return np.empty(0, dtype=np.float64)
    t = np.linspace(0.0, duration_sec, n, endpoint=False)
    decay_rate = math.log(1000.0) / max(t60_sec, 1e-3)
    return np.exp(-t * decay_rate)


def _calc_inharmonicity_b(f0: float) -> float:
    """Calculates physical string inharmonicity coefficient B as a function of fundamental frequency f0."""
    return float(0.00008 + 0.00018 * np.exp(-f0 / 75.0))


def _synth_pluck(
    f0: float,
    amp: float,
    dur: float,
    sample_rate: int = FS,
    b_inharm: float | None = None,
    pitch_sag_hz: float = 2.5,
    tau_sag: float = 0.08,
    clank: bool = True,
    technique: str = "finger",
) -> np.ndarray:
    """Synthesizes an authentic physical bass string pluck."""
    n = int(dur * sample_rate)
    if n <= 0:
        return np.empty(0, dtype=np.float64)
    t = np.linspace(0.0, dur, n, endpoint=False)
    phase_sag = apply_attack_pitch_sag(t, pitch_sag_hz=pitch_sag_hz, tau_sag=tau_sag)
    decay_mult = (
        6.0
        if technique == "palm_mute"
        else (3.5 if technique == "staccato" else (1.3 if technique == "slap" else 1.0))
    )

    b_coeff = _calc_inharmonicity_b(f0) if b_inharm is None else b_inharm
    sig = np.zeros(n, dtype=np.float64)
    max_h = min(128, int((sample_rate / 2.0 - 200.0) / f0))
    split_hz = 0.18

    gamma = (
        0.95
        if technique == "slap"
        else (1.10 if technique == "pick" else (1.80 if technique == "palm_mute" else 1.45))
    )

    fns: list[float] = []
    hws: list[float] = []
    drvs: list[float] = []
    drhs: list[float] = []
    hs: list[float] = []
    for h in range(1, max_h):
        fn = h * f0 * np.sqrt(1.0 + b_coeff * (h**2))
        if fn >= (sample_rate / 2.0) - 200.0:
            break
        geo_pos = max(float(np.sin(h * np.pi * 0.14)), 0.25)
        h_weight = (1.0 / (h**gamma)) * geo_pos
        if technique == "finger":
            arg_tip = h * np.pi * 0.016
            f_tip = abs(np.sin(arg_tip) / arg_tip) if arg_tip > 1e-4 else 1.0
            h_weight *= max(float(f_tip), 0.15)
        if technique == "palm_mute" and fn > 600.0:
            h_weight *= float(np.exp(-(fn - 600.0) / 300.0))
        d_rate_v = (0.7 + 0.10 * h + 0.00025 * (h**2)) * decay_mult
        d_rate_h = (0.35 + 0.05 * h + 0.00012 * (h**2)) * decay_mult
        fns.append(fn)
        hws.append(h_weight)
        drvs.append(d_rate_v)
        drhs.append(d_rate_h)
        hs.append(float(h))

    if fns:
        _accumulate_pluck_modes_simd(
            sig,
            t,
            phase_sag,
            np.asarray(fns, dtype=np.float64),
            np.asarray(hws, dtype=np.float64),
            np.asarray(drvs, dtype=np.float64),
            np.asarray(drhs, dtype=np.float64),
            np.asarray(hs, dtype=np.float64),
            split_hz,
        )

    # Initial physical string displacement asymmetry
    if amp >= 0.70 and technique != "palm_mute":
        asym = 0.12 * amp * np.exp(-t / 0.012)
        sig += asym

    # Attack transients based on technique
    if clank or technique in ("pick", "slap"):
        if technique == "slap":
            burst = np.sin(2.0 * np.pi * 3200.0 * t) * np.exp(-t / 0.004)
            if amp >= 0.75:
                contact_force = np.logaddexp(0.0, 12.0 * (-np.sin(2.0 * np.pi * f0 * t))) / 12.0
                collision = (contact_force**3) * np.exp(-t / 0.025)
                sig += 0.35 * burst + 0.20 * collision
            else:
                sig += 0.20 * burst
        elif technique == "pick":
            burst = np.sin(2.0 * np.pi * 4200.0 * t) * np.exp(-t / 0.0035)
            sig += (0.25 if amp >= 0.75 else 0.12) * burst
        elif technique == "palm_mute":
            pass
        else:
            burst = np.sin(2.0 * np.pi * 2600.0 * t) * np.exp(-t / 0.004)
            if amp >= 0.75:
                contact_force = np.logaddexp(0.0, 12.0 * (-np.sin(2.0 * np.pi * f0 * t))) / 12.0
                collision = (contact_force**3) * np.exp(-t / 0.018)
                sig += 0.10 * burst + 0.10 * collision
            else:
                sig += 0.05 * burst

    sig -= np.mean(sig)
    p_max = float(np.max(np.abs(sig)))
    if p_max > 0:
        sig = (sig / p_max) * amp
    return apply_cinf_fades(sig, min(n // 4, int(0.005 * sample_rate)))


def _synth_ghost_note(
    dur: float,
    amp: float,
    sample_rate: int = FS,
    seed: int = 101,
) -> np.ndarray:
    """Synthesizes an unpitched dead-string percussive thump."""
    n = int(dur * sample_rate)
    if n <= 0:
        return np.empty(0, dtype=np.float64)
    t = np.linspace(0.0, dur, n, endpoint=False)
    m1 = np.sin(2.0 * np.pi * 115.0 * t) * np.exp(-t / 0.016)
    m2 = np.sin(2.0 * np.pi * 230.0 * t) * np.exp(-t / 0.012)
    m3 = np.sin(2.0 * np.pi * 380.0 * t) * np.exp(-t / 0.008)
    body = 0.50 * m1 + 0.35 * m2 + 0.25 * m3

    rng = np.random.default_rng(seed)
    noise = rng.standard_normal(n)
    scrape = np.diff(noise, prepend=noise[0])
    scrape_click = np.sin(2.0 * np.pi * 3200.0 * t) * np.exp(-t / 0.004)
    transient = (0.20 * scrape + 0.12 * scrape_click) * np.exp(-t / 0.006)

    sig = body + transient
    sig -= np.mean(sig)
    p_max = float(np.max(np.abs(sig)))
    if p_max > 0:
        sig = (sig / p_max) * amp
    return apply_cinf_fades(sig, min(n // 4, int(0.003 * sample_rate)))


def _synth_natural_harmonic(
    f_harmonic: float,
    amp: float,
    dur: float,
    sample_rate: int = FS,
    num_partials: int = 5,
) -> np.ndarray:
    """Synthesizes a pure crystalline natural harmonic overtone cascade."""
    n = int(dur * sample_rate)
    if n <= 0:
        return np.empty(0, dtype=np.float64)
    t = np.linspace(0.0, dur, n, endpoint=False)
    sig = np.zeros(n, dtype=np.float64)
    for k in range(1, num_partials + 1):
        fk = k * f_harmonic
        if fk >= (sample_rate / 2.0) - 200.0:
            break
        decay_rate = 0.25 + 0.10 * k
        decay = np.exp(-t * decay_rate)
        weight = 1.0 / (k**1.25)
        sig += weight * np.sin(2.0 * np.pi * fk * t) * decay

    chime = np.sin(2.0 * np.pi * 4800.0 * t) * np.exp(-t / 0.003)
    sig += 0.08 * chime

    sig -= np.mean(sig)
    p_max = float(np.max(np.abs(sig)))
    if p_max > 0:
        sig = (sig / p_max) * amp
    return apply_cinf_fades(sig, min(n // 4, int(0.005 * sample_rate)))


def _synth_dyad(
    f1: float,
    f2: float,
    amp: float,
    dur: float,
    sample_rate: int = FS,
    stagger_ms: float = 12.0,
) -> np.ndarray:
    """Synthesizes a two-note chord with micro-staggered string plucks."""
    stagger_samples = int((stagger_ms / 1000.0) * sample_rate)
    dur2 = max(0.2, dur - (stagger_ms / 1000.0))
    p1 = _synth_pluck(f1, 0.55, dur, sample_rate, clank=True, technique="finger")
    p2 = _synth_pluck(f2, 0.45, dur2, sample_rate, clank=True, technique="finger")
    total_len = max(len(p1), stagger_samples + len(p2))
    sig = np.zeros(total_len, dtype=np.float64)
    sig[: len(p1)] += p1
    sig[stagger_samples : stagger_samples + len(p2)] += p2
    sig -= np.mean(sig)
    p_max = float(np.max(np.abs(sig)))
    if p_max > 0:
        sig = (sig / p_max) * amp
    return apply_cinf_fades(sig, min(len(sig) // 4, int(0.005 * sample_rate)))


def _synth_glissando(
    f_start: float,
    f_end: float,
    amp: float,
    dur: float,
    sample_rate: int = FS,
    num_harmonics: int = 8,
) -> np.ndarray:
    """Synthesizes an authentic continuous string glissando."""
    n = int(dur * sample_rate)
    if n <= 0:
        return np.empty(0, dtype=np.float64)
    t = np.linspace(0.0, dur, n, endpoint=False)
    gamma = np.log(f_end / f_start)
    base_phase = 2.0 * np.pi * f_start * (dur / gamma) * ((f_end / f_start) ** (t / dur) - 1.0)
    sig = np.zeros(n, dtype=np.float64)
    ks: list[float] = []
    kws: list[float] = []
    for k in range(1, num_harmonics + 1):
        max_fk = k * max(f_start, f_end)
        if max_fk >= (sample_rate / 2.0) - 200.0:
            break
        ks.append(float(k))
        kws.append(1.0 / (k**1.30))

    if ks:
        _accumulate_glissando_modes_simd(
            sig,
            base_phase,
            np.asarray(ks, dtype=np.float64),
            np.asarray(kws, dtype=np.float64),
        )

    num_frets = abs(12.0 * np.log2(f_end / f_start))
    if num_frets > 1.0:
        fret_clicks = np.sin(2.0 * np.pi * num_frets * (t / dur)) ** 16
        sig += 0.03 * fret_clicks * np.sin(2.0 * np.pi * 3200.0 * t)

    sig -= np.mean(sig)
    p_max = float(np.max(np.abs(sig)))
    if p_max > 0:
        sig = (sig / p_max) * amp
    return apply_cinf_fades(sig, min(n // 4, int(0.005 * sample_rate)))


def _synth_ghost_rake(
    f0: float,
    amp: float,
    dur: float,
    sample_rate: int = FS,
) -> np.ndarray:
    """Synthesizes a percussive funk dead-note rake."""
    c1 = _synth_ghost_note(0.040, amp * 0.70, sample_rate, seed=101)
    c2 = _synth_ghost_note(0.040, amp * 0.75, sample_rate, seed=102)
    c3 = _synth_ghost_note(0.045, amp * 0.85, sample_rate, seed=103)
    p_dur = max(0.2, dur - 0.125)
    p = _synth_pluck(f0, amp, p_dur, sample_rate, clank=True, technique="finger")
    return np.concatenate([c1, c2, c3, p])


def _synth_groove_burst(
    f0: float,
    amp: float,
    bpm: float,
    count: int,
    sample_rate: int = FS,
    technique: str = "finger",
    rest_ms: float = 25.0,
) -> np.ndarray:
    """Synthesizes rapid repeated plucks at a given tempo with short micro-rests."""
    interval_sec = 60.0 / (bpm * 4.0) if bpm > 0 else 0.125
    rest_sec = rest_ms / 1000.0
    pluck_dur = max(0.04, interval_sec - rest_sec)
    p = _synth_pluck(f0, amp, pluck_dur, sample_rate, clank=True, technique=technique)
    rest_samples = max(10, int(rest_sec * sample_rate))
    rest = np.zeros(rest_samples, dtype=np.float64)
    pieces: list[np.ndarray] = []
    for i in range(count):
        stroke_amp = 1.0 if (i % 4 == 0) else (0.86 if i % 2 == 0 else 0.78)
        pieces.append(p * stroke_amp)
        if i < count - 1:
            pieces.append(rest)
    return np.concatenate(pieces)


def _synth_slap_pop_pair(
    f_slap: float,
    f_pop: float,
    amp: float,
    gap_ms: float = 70.0,
    dur: float = 1.4,
    sample_rate: int = FS,
) -> np.ndarray:
    """Synthesizes an authentic funk slap-and-pop pair: thumb slap followed by octave pop."""
    slap_part = _synth_pluck(f_slap, amp, dur, sample_rate, clank=True, technique="slap")
    pop_dur = max(0.4, dur - gap_ms / 1000.0)
    pop_part = _synth_pluck(f_pop, amp * 0.90, pop_dur, sample_rate, clank=True, technique="pick")
    gap_samples = int((gap_ms / 1000.0) * sample_rate)
    total_len = max(len(slap_part), gap_samples + len(pop_part))
    composite = np.zeros(total_len, dtype=np.float64)
    composite[: len(slap_part)] += slap_part
    composite[gap_samples : gap_samples + len(pop_part)] += pop_part
    composite -= np.mean(composite)
    p_max = float(np.max(np.abs(composite)))
    if p_max > 0:
        composite = (composite / p_max) * amp
    return apply_cinf_fades(composite, min(len(composite) // 4, int(0.005 * sample_rate)))


def _synth_vibrato_pluck(
    f0: float,
    amp: float,
    dur: float,
    mod_rate: float = 5.0,
    mod_depth_cents: float = 25.0,
    sample_rate: int = FS,
) -> np.ndarray:
    """Synthesizes a sustained bass pluck with finger vibrato."""
    n = int(dur * sample_rate)
    if n <= 0:
        return np.empty(0, dtype=np.float64)
    t = np.linspace(0.0, dur, n, endpoint=False)
    delta = 2.0 ** (mod_depth_cents / 1200.0) - 1.0
    phase_mod = -delta * (f0 / mod_rate) * np.cos(2.0 * np.pi * mod_rate * t)
    base_phase = 2.0 * np.pi * f0 * t + 2.0 * np.pi * phase_mod
    sig = np.zeros(n, dtype=np.float64)
    max_h = min(110, int((sample_rate / 2.0 - 200.0) / f0))
    fns: list[float] = []
    hws: list[float] = []
    drs: list[float] = []
    hs: list[float] = []
    for h in range(1, max_h):
        fn = h * f0
        if fn >= (sample_rate / 2.0) - 200.0:
            break
        geo_pos = max(float(np.sin(h * np.pi * 0.14)), 0.25)
        h_weight = (1.0 / (h**1.40)) * geo_pos
        decay_rate = 0.5 + 0.07 * h + 0.00018 * (h**2)
        fns.append(fn)
        hws.append(h_weight)
        drs.append(decay_rate)
        hs.append(float(h))

    if fns:
        _accumulate_vibrato_modes_simd(
            sig,
            t,
            base_phase,
            np.asarray(fns, dtype=np.float64),
            np.asarray(hws, dtype=np.float64),
            np.asarray(drs, dtype=np.float64),
            np.asarray(hs, dtype=np.float64),
        )
    sig -= np.mean(sig)
    p_max = float(np.max(np.abs(sig)))
    if p_max > 0:
        sig = (sig / p_max) * amp
    return apply_cinf_fades(sig, min(n // 4, int(0.005 * sample_rate)))


def _synth_long_ringout(
    f0: float,
    amp: float,
    dur: float,
    sample_rate: int = FS,
    b_inharm: float | None = None,
) -> np.ndarray:
    """Synthesizes an extended uninterrupted bass ring-out decaying across > 75 dB."""
    n = int(dur * sample_rate)
    if n <= 0:
        return np.empty(0, dtype=np.float64)
    b_coeff = _calc_inharmonicity_b(f0) if b_inharm is None else b_inharm
    sig = np.zeros(n, dtype=np.float64)
    max_h = min(128, int((sample_rate / 2.0 - 200.0) / f0))
    split_hz = 0.16
    fns: list[float] = []
    hws: list[float] = []
    drvs: list[float] = []
    drhs: list[float] = []
    for h in range(1, max_h):
        fn = h * f0 * np.sqrt(1.0 + b_coeff * (h**2))
        if fn >= (sample_rate / 2.0) - 200.0:
            break
        geo_pos = max(float(np.sin(h * np.pi * 0.14)), 0.25)
        h_weight = (1.0 / (h**1.40)) * geo_pos
        decay_v_rate = 0.35 + 0.05 * h + 0.00012 * (h**2)
        decay_h_rate = 0.20 + 0.03 * h + 0.00006 * (h**2)
        fns.append(fn)
        hws.append(h_weight)
        drvs.append(decay_v_rate)
        drhs.append(decay_h_rate)

    if fns:
        _accumulate_ringout_phasors_simd(
            sig,
            n,
            1.0 / sample_rate,
            np.asarray(fns, dtype=np.float64),
            np.asarray(hws, dtype=np.float64),
            np.asarray(drvs, dtype=np.float64),
            np.asarray(drhs, dtype=np.float64),
            split_hz,
        )
    sig -= np.mean(sig)
    p_max = float(np.max(np.abs(sig)))
    if p_max > 0:
        sig = (sig / p_max) * amp
    fade_samples = min(n // 8, int(0.05 * sample_rate))
    return apply_cinf_fades(sig, fade_samples)


def _synth_two_tone_probe(
    f1: float,
    f2: float,
    amp: float,
    dur: float,
    sample_rate: int = FS,
) -> np.ndarray:
    """Synthesizes a precision CCIF/DIN two-tone intermodulation probe burst."""
    n = int(dur * sample_rate)
    if n <= 0:
        return np.empty(0, dtype=np.float64)
    t = np.linspace(0.0, dur, n, endpoint=False)
    sig = 0.50 * np.sin(2.0 * np.pi * f1 * t) + 0.50 * np.sin(2.0 * np.pi * f2 * t)
    sig -= np.mean(sig)
    p_max = float(np.max(np.abs(sig)))
    if p_max > 0:
        sig = (sig / p_max) * amp
    return apply_cinf_fades(sig, min(n // 4, int(0.005 * sample_rate)))
