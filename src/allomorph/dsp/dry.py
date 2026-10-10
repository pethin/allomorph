"""
High-fidelity synthetic dry excitation signal generation for bass pickup modeling.
Decomposed into 8 pure, property-testable segment generators:
1. Latency calibration double-blips
2. Multi-tier full log sweeps (slew-rate diverse)
3. 7-step dynamic velocity ladder on open E1
4. Long-decay continuous ring-outs (> 75 dB gate-free linearity)
5. Inharmonic modal plucks across registers
6. Bass playing techniques & articulations
7. Polyphony, dyads, CCIF probes, and Schroeder multitone
8. Continuous glissandi slides
"""

from pathlib import Path

import numpy as np

from allomorph.dsp.constants import FS, apply_cinf_fades
from allomorph.dsp.io import write_wav_24bit
from allomorph.dsp.plucks import (
    _synth_dyad,
    _synth_ghost_note,
    _synth_ghost_rake,
    _synth_glissando,
    _synth_groove_burst,
    _synth_long_ringout,
    _synth_natural_harmonic,
    _synth_pluck,
    _synth_slap_pop_pair,
    _synth_two_tone_probe,
    _synth_vibrato_pluck,
)
from allomorph.dsp.sweeps import _synth_log_chirp


def synth_calibration_blips(sample_rate: int = FS, scale: float = 1.0) -> np.ndarray:
    """Generates Tone3000 calibration double-blips (+0.89 and -0.89) ensuring zero detected latency."""
    dur = max(0.8, 1.2 * scale)
    total = int(dur * sample_rate)
    blips = np.zeros(total, dtype=np.float64)
    blips[0] = 0.89
    b2 = max(100, int(0.5 * sample_rate * scale))
    if b2 < total:
        blips[b2] = -0.89
    return blips


def synth_log_chirps(sample_rate: int = FS, scale: float = 1.0) -> list[tuple[np.ndarray, float]]:
    """Generates multi-tier full sweeps (15 Hz -> 22 kHz) with slew-rate diversity."""
    chirp_tiers = [
        (15.0, 22000.0, 0.050, 12.0),
        (15.0, 22000.0, 0.200, 1.8),
        (15.0, 22000.0, 0.400, 7.0),
        (15.0, 22000.0, 0.650, 1.8),
        (22000.0, 15.0, 0.500, 5.0),
    ]
    results: list[tuple[np.ndarray, float]] = []
    for f_s, f_e, amp, dur_base in chirp_tiers:
        dur_c = max(0.6, dur_base * scale)
        c = _synth_log_chirp(dur_c, f_s, f_e, amp, sample_rate, string_tilt=True, f_corner=1200.0)
        results.append((c, 0.4))
    return results


def synth_velocity_ladder(
    sample_rate: int = FS, scale: float = 1.0
) -> list[tuple[np.ndarray, float]]:
    """Generates 7-step dynamic velocity ladder on open E1 (pp -> fff)."""
    v_dur = max(0.4, 1.8 * scale)
    velocity_tiers = [0.04, 0.10, 0.22, 0.40, 0.62, 0.82, 0.95]
    results: list[tuple[np.ndarray, float]] = []
    for v_amp in velocity_tiers:
        p = _synth_pluck(41.20, v_amp, v_dur, sample_rate, technique="finger")
        results.append((p, 0.4))
    return results


def synth_long_ringouts(
    sample_rate: int = FS, scale: float = 1.0
) -> list[tuple[np.ndarray, float]]:
    """Generates continuous uninterrupted bass ring-outs decaying across > 75 dB."""
    r_dur_e = max(1.0, 5.5 * scale)
    r_dur_a = max(1.0, 5.0 * scale)
    results: list[tuple[np.ndarray, float]] = []
    for f_ring, dur_ring in [(41.20, r_dur_e), (55.00, r_dur_a)]:
        r = _synth_long_ringout(f_ring, 0.85, dur_ring, sample_rate)
        results.append((r, 0.5))
    return results


def synth_modal_plucks(sample_rate: int = FS, scale: float = 1.0) -> list[tuple[np.ndarray, float]]:
    """Generates modal plucks with attack pitch sag and heavy-string inharmonicity across registers."""
    m_dur = max(0.4, 1.3 * scale)
    notes = [27.50, 30.87, 41.20, 55.00, 73.42, 98.00, 130.81, 164.81]
    results: list[tuple[np.ndarray, float]] = []
    for n_f in notes:
        sag = 4.5 if n_f < 35.0 else (2.5 if n_f < 60.0 else 1.2)
        for v_amp in [0.45, 0.82]:
            p = _synth_pluck(n_f, v_amp, m_dur, sample_rate, pitch_sag_hz=sag, technique="finger")
            results.append((p, 0.35))
    return results


def synth_articulation_bursts(
    sample_rate: int = FS, scale: float = 1.0
) -> list[tuple[np.ndarray, float]]:
    """Generates comprehensive bass playing techniques (groove bursts, slap/pop, ghost notes, mutes, harmonics)."""
    results: list[tuple[np.ndarray, float]] = []
    # Rapid groove bursts
    gb1 = _synth_groove_burst(
        41.20, 0.82, bpm=120.0, count=6, sample_rate=sample_rate, technique="finger"
    )
    results.append((gb1, 0.4))
    gb2 = _synth_groove_burst(
        55.00, 0.80, bpm=140.0, count=8, sample_rate=sample_rate, technique="pick"
    )
    results.append((gb2, 0.4))

    # Slap-and-pop pairs
    for f_slap, f_pop in [(41.20, 82.41), (55.00, 110.00)]:
        sp = _synth_slap_pop_pair(
            f_slap, f_pop, 0.89, gap_ms=70.0, dur=max(0.4, 1.4 * scale), sample_rate=sample_rate
        )
        results.append((sp, 0.4))

    # Funk ghost rake & ghost clicks
    rake = _synth_ghost_rake(41.20, 0.85, max(0.4, 1.0 * scale), sample_rate)
    results.append((rake, 0.4))
    for _ in range(2):
        g = _synth_ghost_note(max(0.15, 0.30 * scale), 0.75, sample_rate)
        results.append((g, 0.35))

    # Palm-muted Motown thuds
    for n_f in [41.20, 55.00, 73.42]:
        pm = _synth_pluck(n_f, 0.85, max(0.3, 0.7 * scale), sample_rate, technique="palm_mute")
        results.append((pm, 0.4))

    # Sustained vibrato plucks
    for v_f in [55.00, 73.42]:
        vib = _synth_vibrato_pluck(
            v_f,
            0.78,
            max(0.6, 2.5 * scale),
            mod_rate=5.0,
            mod_depth_cents=25.0,
            sample_rate=sample_rate,
        )
        results.append((vib, 0.4))

    # Pick strikes
    for _ in range(2):
        p = _synth_pluck(41.20, 0.78, max(0.4, 1.4 * scale), sample_rate, technique="pick")
        results.append((p, 0.4))

    # Natural harmonics
    for h_f in [82.41, 123.6, 164.8]:
        h = _synth_natural_harmonic(h_f, 0.70, max(0.4, 1.6 * scale), sample_rate)
        results.append((h, 0.4))

    return results


def synth_polyphony_and_probes(
    sample_rate: int = FS, scale: float = 1.0, max_multitone_samples: int = 0
) -> list[tuple[np.ndarray, float]]:
    """Generates dyads, upper tenths, CCIF intermodulation probes, and Schroeder multitone."""
    results: list[tuple[np.ndarray, float]] = []
    d_dur = max(0.5, 2.0 * scale)
    low_dyads = [(27.50, 41.25), (30.87, 46.31), (41.20, 61.74), (55.00, 82.50)]
    for f1, f2 in low_dyads:
        d = _synth_dyad(f1, f2, 0.75, d_dur, sample_rate)
        results.append((d, 0.4))

    tenths = [(82.41, 207.65), (110.00, 277.18), (146.83, 369.99)]
    for f1, f2 in tenths:
        dt = _synth_dyad(f1, f2, 0.72, max(0.4, 1.8 * scale), sample_rate)
        results.append((dt, 0.4))

    ccif_probes = [
        (3000.0, 3200.0, 41.20),
        (4000.0, 4250.0, 30.87),
        (2000.0, 2150.0, 55.00),
    ]
    probe_dur = max(0.4, 2.0 * scale)
    for f1, f2, f_drone in ccif_probes:
        drone = _synth_pluck(f_drone, 0.65, probe_dur, sample_rate, clank=False, technique="finger")
        pr = _synth_two_tone_probe(f1, f2, 0.05, probe_dur, sample_rate)
        n_p = min(len(drone), len(pr))
        embedded = drone[:n_p] + pr[:n_p]
        embedded -= np.mean(embedded)
        p_max = float(np.max(np.abs(embedded)))
        if p_max > 0:
            embedded = (embedded / p_max) * 0.70
        results.append((embedded, 0.4))

    if max_multitone_samples > int(2.0 * sample_rate * scale):
        dur_m = min(14.0 * scale, max_multitone_samples / sample_rate * 0.35)
        nm = int(dur_m * sample_rate)
        if nm > 100:
            tm = np.linspace(0.0, dur_m, nm, endpoint=False)
            clusters = [
                27.50,
                30.87,
                41.20,
                55.00,
                82.41,
                110.0,
                220.0,
                440.0,
                880.0,
                1250.0,
                1800.0,
                2400.0,
                3100.0,
                4200.0,
                6000.0,
            ]
            kc = len(clusters)
            sig_m = np.zeros(nm, dtype=np.float64)
            for k, fk in enumerate(clusters):
                th = (np.pi * (k**2)) / kc
                weight = 1.0 / np.sqrt(1.0 + (fk / 800.0) ** 1.8)
                sig_m += weight * np.sin(2.0 * np.pi * fk * tm + th)
            sig_m -= np.mean(sig_m)
            sig_m = (sig_m / np.max(np.abs(sig_m))) * 0.80
            am_env = 0.575 + 0.325 * np.sin(2.0 * np.pi * 0.25 * tm)
            faded = apply_cinf_fades(sig_m * am_env, min(nm // 4, int(0.01 * sample_rate)))
            results.append((faded, 0.4))

    return results


def synth_glissandi_slides(
    sample_rate: int = FS, scale: float = 1.0
) -> list[tuple[np.ndarray, float]]:
    """Generates continuous fretboard glissandi slides up to 24th fret (392 Hz)."""
    g_dur = max(0.5, 3.2 * scale)
    slides = [
        (27.50, 41.20),
        (41.20, 73.42),
        (73.42, 146.83),
        (146.83, 293.66),
        (293.66, 392.00),
        (392.00, 41.20),
    ]
    results: list[tuple[np.ndarray, float]] = []
    for fs, fe in slides:
        gl_dur = max(0.6, 4.0 * scale) if fe < fs else g_dur
        gl = _synth_glissando(fs, fe, 0.80, gl_dur, sample_rate)
        results.append((gl, 0.4))
    return results


def generate_optimal_bass_dry(
    duration_sec: float = 240.0,
    sample_rate: int = FS,
    peak_dbfs: float = -1.0,
    target_rms_dbfs: float | None = None,
    seed: int = 42,
) -> np.ndarray:
    """Synthesizes a 48 kHz high-fidelity synthetic dry excitation signal tailored for bass modeling."""
    total_samples = int(duration_sec * sample_rate)
    audio = np.zeros(total_samples, dtype=np.float64)
    scale = min(1.0, duration_sec / 240.0)

    lead_silence = min(int(0.3 * sample_rate), int(0.05 * total_samples))
    trail_silence = min(int(0.3 * sample_rate), int(0.05 * total_samples))
    cur = lead_silence

    # 1. Calibration blips
    blips = synth_calibration_blips(sample_rate=sample_rate, scale=scale)
    n_blips = min(len(blips), total_samples - trail_silence - cur)
    if n_blips > 0:
        audio[cur : cur + n_blips] = blips[:n_blips]
        cur += n_blips + max(100, int(0.3 * sample_rate * scale))

    def append_segment(seg: np.ndarray, pause_dur: float) -> None:
        nonlocal cur
        if len(seg) == 0:
            return
        end_idx = cur + len(seg)
        limit = total_samples - trail_silence
        if end_idx >= limit:
            avail = max(0, limit - cur)
            if avail > 0:
                audio[cur : cur + avail] = seg[:avail]
                cur += avail
            return
        audio[cur:end_idx] = seg
        p_samples = max(50, int(pause_dur * sample_rate * scale))
        cur = min(end_idx + p_samples, limit)

    # 2. Log sweeps
    for seg, pause in synth_log_chirps(sample_rate=sample_rate, scale=scale):
        if cur >= total_samples - trail_silence:
            break
        append_segment(seg, pause)

    # 3. Velocity ladder
    for seg, pause in synth_velocity_ladder(sample_rate=sample_rate, scale=scale):
        if cur >= total_samples - trail_silence:
            break
        append_segment(seg, pause)

    # 4. Long ring-outs
    for seg, pause in synth_long_ringouts(sample_rate=sample_rate, scale=scale):
        if cur >= total_samples - trail_silence:
            break
        append_segment(seg, pause)

    # 5. Modal plucks
    for seg, pause in synth_modal_plucks(sample_rate=sample_rate, scale=scale):
        if cur >= total_samples - trail_silence:
            break
        append_segment(seg, pause)

    # 6. Articulations
    for seg, pause in synth_articulation_bursts(sample_rate=sample_rate, scale=scale):
        if cur >= total_samples - trail_silence:
            break
        append_segment(seg, pause)

    # 7. Polyphony and multitone
    m_rem = max(0, total_samples - trail_silence - cur)
    for seg, pause in synth_polyphony_and_probes(
        sample_rate=sample_rate, scale=scale, max_multitone_samples=m_rem
    ):
        if cur >= total_samples - trail_silence:
            break
        append_segment(seg, pause)

    # 8. Glissandi slides
    for seg, pause in synth_glissandi_slides(sample_rate=sample_rate, scale=scale):
        if cur >= total_samples - trail_silence:
            break
        append_segment(seg, pause)

    # 9. Extended ring-outs and cascades
    for f_deep in [30.87, 27.50]:
        if cur >= total_samples - trail_silence:
            break
        r_deep = _synth_long_ringout(f_deep, 0.85, max(1.5, 6.0 * scale), sample_rate)
        append_segment(r_deep, 0.5)

    for f_trem in [41.20, 55.00]:
        if cur >= total_samples - trail_silence:
            break
        trem = _synth_groove_burst(
            f_trem, 0.80, bpm=140.0, count=12, sample_rate=sample_rate, technique="pick"
        )
        append_segment(trem, 0.4)

    slap_cascades = [
        (41.20, 82.41),
        (55.00, 110.00),
        (73.42, 146.83),
        (98.00, 196.00),
    ]
    for f_s, f_p in slap_cascades:
        if cur >= total_samples - trail_silence:
            break
        sp_pair = _synth_slap_pop_pair(
            f_s, f_p, 0.88, gap_ms=65.0, dur=max(0.4, 1.4 * scale), sample_rate=sample_rate
        )
        append_segment(sp_pair, 0.4)

    # 10. Pre-validation training fill
    val_samples = 432_000
    val_start = total_samples - val_samples
    pre_val_pause_samples = int(0.5 * sample_rate)
    train_fill_limit = val_start - pre_val_pause_samples

    if cur < train_fill_limit - int(0.5 * sample_rate):
        dur_fill = (train_fill_limit - cur) / sample_rate
        c_fill = _synth_log_chirp(
            dur_fill, 15.0, 22000.0, 0.35, sample_rate, string_tilt=True, f_corner=1200.0
        )
        append_segment(c_fill, 0.0)

    cur = max(cur, val_start)

    # 11. Representative validation suite
    v_ring = _synth_long_ringout(30.87, 0.85, max(1.0, 2.8 * scale), sample_rate)
    append_segment(v_ring, 0.35)

    v_groove = _synth_groove_burst(
        41.20, 0.80, bpm=140.0, count=8, sample_rate=sample_rate, technique="pick"
    )
    append_segment(v_groove, 0.35)

    v_slap = _synth_slap_pop_pair(
        41.20, 82.41, 0.88, gap_ms=65.0, dur=max(0.4, 1.4 * scale), sample_rate=sample_rate
    )
    append_segment(v_slap, 0.35)

    v_harm = _synth_natural_harmonic(123.6, 0.75, max(0.4, 1.6 * scale), sample_rate)
    append_segment(v_harm, 0.35)

    v_pm = _synth_pluck(55.00, 0.82, max(0.3, 0.6 * scale), sample_rate, technique="palm_mute")
    append_segment(v_pm, 0.0)

    # Zero-DC centering on active regions
    active_mask = audio != 0.0
    if np.any(active_mask):
        active_mean = float(np.mean(audio[active_mask]))
        audio[active_mask] -= active_mean

    # Level scaling
    if target_rms_dbfs is not None:
        target_rms = 10.0 ** (target_rms_dbfs / 20.0)
        current_rms = float(np.sqrt(np.mean(audio**2)))
        if current_rms > 0:
            audio = audio * (target_rms / current_rms)
        current_peak = float(np.max(np.abs(audio)))
        ceiling = 10.0 ** (peak_dbfs / 20.0)
        if current_peak > ceiling:
            audio = audio * (ceiling / current_peak)
    else:
        target_peak = 10.0 ** (peak_dbfs / 20.0)
        current_peak = float(np.max(np.abs(audio)))
        if current_peak > 0:
            audio = audio * (target_peak / current_peak)

    return audio.astype(np.float32)


generate_input_audio = generate_optimal_bass_dry


def ensure_input_audio_wav(
    output_path: Path | str | None = None,
    duration_sec: float = 240.0,
    sample_rate: int = FS,
    peak_dbfs: float = -1.0,
    target_rms_dbfs: float | None = -20.50,
    overwrite: bool = False,
    version_tag: str | None = None,
    no_manifest: bool = False,
) -> Path:
    """Ensures that the synthesized dry input string excitation signal exists on disk."""
    if output_path is not None:
        p = Path(output_path)
    else:
        from allomorph.naming import get_default_input_path

        p = get_default_input_path()

    if p.exists() and not overwrite:
        return p

    p.parent.mkdir(parents=True, exist_ok=True)
    audio = generate_optimal_bass_dry(
        duration_sec=duration_sec,
        sample_rate=sample_rate,
        peak_dbfs=peak_dbfs,
        target_rms_dbfs=target_rms_dbfs,
    )
    write_wav_24bit(p, audio, sample_rate)
    return p
