#!/usr/bin/env python3
"""Allomorph - Hardware Re-Amping Dry Excitation Generator CLI.

Synthesizes a 24-bit / 48 kHz dry bass excitation signal (`reamp/input.wav`)
purpose-built for hardware re-amping through non-linear bass amplifiers,
drive pedals, and speaker cabinets on the Darkglass Anagram:

  1. Asymmetric Band-Limited Alignment Pulses (0.00s, 0.23s, 0.61s):
     Combines sub-sample impulses with C-infinity windowed bass/midband sync wavelets
     (80 Hz - 3.5 kHz) and non-uniform spacing (+0.92, -0.65, +0.82) for >15 dB higher
     through-cabinet SNR and zero polarity or dual-blip aliasing ambiguity.
  2. Cabinet-Focused Multi-Tier Log Sweeps (18 Hz - 8.5 kHz):
     Concentrates sweep energy inside the active passband and cone breakup region of
     8x10, 4x10, and 1x15 bass cabinets across 5 slew-rate and drive tiers (forward & reverse),
     eliminating the 15.5-second dead ultrasonic (5-22 kHz) tail of pickup-only sweeps.
  3. Extended Dynamic Velocity Ladders & Continuous Swell Envelopes:
     8-step velocity ladders on E1 (41.2 Hz) and A1 (55.0 Hz) plus C-infinity crescendo/decrescendo
     tremolo swells that continuously trace tube/FET grid conduction and power-supply sag knees.
  4. High-Density Nonlinear Amp & Cab Conditioning:
     Low-register power fifths/dyads, aggressive 140-160 BPM palm-mute-to-open pick chugs,
     slap/pop octave bursts, modal plucks (B0 30.87 Hz - G3 196 Hz), glissandi, and
     bass-band Schroeder multitone.
  5. Calibrated Headroom (-3.0 dBFS Peak / -18.5 dBFS RMS):
     Places raw peaks directly inside the Darkglass Anagram's optimal -6.0 to -3.0 dBFS
     hardware input window even at 0.0 dB unity USB send gain.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Annotated

import numpy as np
from cyclopts import App, Parameter

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from allomorph.dsp.constants import FS, apply_cinf_fades, cinf_smoothstep
from allomorph.dsp.dry import (
    synth_articulation_bursts,
    synth_glissandi_slides,
    synth_long_ringouts,
    synth_modal_plucks,
)
from allomorph.dsp.io import write_wav_24bit
from allomorph.dsp.plucks import (
    _synth_dyad,
    _synth_ghost_rake,
    _synth_groove_burst,
    _synth_long_ringout,
    _synth_natural_harmonic,
    _synth_pluck,
    _synth_slap_pop_pair,
    _synth_two_tone_probe,
)
from allomorph.dsp.sweeps import _synth_log_chirp

app = App(
    name="generate-reamp-input",
    help="Allomorph - Generate Optimal Dry Excitation WAV for Hardware Amp+Cab Re-Amping.",
    version="1.0.0",
)

DEFAULT_REAMP_INPUT_PATH = REPO_ROOT / "reamp" / "input.wav"
DEFAULT_REAMP_DURATION_SEC: float = 240.0
DEFAULT_REAMP_PEAK_DBFS: float = -3.0
DEFAULT_REAMP_RMS_DBFS: float = -18.5


def synth_reamp_alignment_pulses(
    sample_rate: int = FS,
    scale: float = 1.0,
) -> np.ndarray:
    """Synthesizes a 3-pulse asymmetric band-limited alignment burst.

    Pulse timings (0.00s, 0.23s, 0.61s for scale=1.0) and signed amplitudes
    (+0.92, -0.65, +0.82) are non-uniform, guaranteeing a unique global
    cross-correlation peak under arbitrary latency (0-1000 ms) and amp polarity inversion.
    Each pulse blends a sharp 1-sample center spike with a 1.5 ms zero-phase
    bass/midband wavelet (80 Hz - 3.5 kHz) that penetrates steep speaker cabinet roll-offs.
    """
    dur = max(0.8, 1.0 * scale)
    total = int(dur * sample_rate)
    out = np.zeros(total, dtype=np.float64)

    half_w = max(8, int(0.00075 * sample_rate))  # 36 samples (1.5 ms total wavelet width)
    t_w = np.arange(-half_w, half_w + 1, dtype=np.float64) / sample_rate
    # Band-pass zero-phase sinc difference (80 Hz to 3500 Hz)
    f_hi = 3500.0
    f_lo = 80.0
    sinc_hi = 2.0 * f_hi * np.sinc(2.0 * f_hi * t_w)
    sinc_lo = 2.0 * f_lo * np.sinc(2.0 * f_lo * t_w)
    bp_wavelet = sinc_hi - sinc_lo
    # Smooth Blackman window
    n_w = len(t_w)
    win = (
        0.42
        - 0.50 * np.cos(2.0 * np.pi * np.arange(n_w) / max(1, n_w - 1))
        + 0.08 * np.cos(4.0 * np.pi * np.arange(n_w) / max(1, n_w - 1))
    )
    bp_wavelet = bp_wavelet * win
    bp_peak = float(np.max(np.abs(bp_wavelet)))
    if bp_peak > 0.0:
        bp_wavelet = bp_wavelet / bp_peak
    # Blend sub-sample center impulse (40%) + cabinet-penetrating wavelet (60%)
    bp_wavelet = 0.60 * bp_wavelet
    bp_wavelet[half_w] += 0.40

    pulse_specs = [
        (int(0.02 * sample_rate), 0.92),
        (int(max(0.15, 0.25 * scale) * sample_rate), -0.65),
        (int(max(0.38, 0.63 * scale) * sample_rate), 0.82),
    ]
    for center_idx, amp in pulse_specs:
        i0 = max(0, center_idx - half_w)
        i1 = min(total, center_idx + half_w + 1)
        w0 = i0 - (center_idx - half_w)
        w1 = w0 + (i1 - i0)
        if i1 > i0:
            out[i0:i1] += amp * bp_wavelet[w0:w1]

    return out


def synth_reamp_cab_sweeps(
    sample_rate: int = FS,
    scale: float = 1.0,
) -> list[tuple[np.ndarray, float]]:
    """Synthesizes multi-tier logarithmic sweeps concentrated in the active bass amp + cab passband (18 Hz - 8.5 kHz)."""
    sweep_tiers = [
        (18.0, 8500.0, 0.12, 5.5),
        (18.0, 8500.0, 0.35, 3.0),
        (18.0, 8500.0, 0.68, 5.5),
        (18.0, 8500.0, 0.92, 3.0),
        (8500.0, 18.0, 0.72, 4.5),
    ]
    results: list[tuple[np.ndarray, float]] = []
    for f_s, f_e, amp, dur_base in sweep_tiers:
        dur_c = max(0.5, dur_base * scale)
        c = _synth_log_chirp(
            dur_c,
            f_s,
            f_e,
            amp,
            sample_rate,
            string_tilt=True,
            f_corner=2400.0,
        )
        results.append((c, 0.35))
    return results


def synth_reamp_velocity_and_swells(
    sample_rate: int = FS,
    scale: float = 1.0,
) -> list[tuple[np.ndarray, float]]:
    """Synthesizes an 8-step velocity ladder on E1/A1 plus continuous crescendo/decrescendo dynamic swells."""
    results: list[tuple[np.ndarray, float]] = []
    v_dur = max(0.4, 1.5 * scale)
    velocity_tiers = [0.04, 0.09, 0.18, 0.32, 0.50, 0.68, 0.84, 0.95]
    for idx, v_amp in enumerate(velocity_tiers):
        f0 = 41.20 if idx % 2 == 0 else 55.00
        tech = "finger" if v_amp < 0.75 else "pick"
        p = _synth_pluck(f0, v_amp, v_dur, sample_rate, technique=tech)
        results.append((p, 0.30))

    # Continuous dynamic envelope swell (ppp -> fff -> ppp) on E1 (41.2 Hz) and D2 (73.42 Hz)
    for f_swell, bpm, tech in [(41.20, 135.0, "pick"), (73.42, 125.0, "finger")]:
        raw_burst = _synth_groove_burst(
            f_swell,
            0.92,
            bpm=bpm,
            count=max(4, int(16 * scale)),
            sample_rate=sample_rate,
            technique=tech,
            rest_ms=18.0,
        )
        n_b = len(raw_burst)
        if n_b > 0:
            u = np.linspace(0.0, 1.0, n_b, endpoint=False)
            # Smooth C-infinity arch envelope from 0.08 -> 1.0 -> 0.08
            up_ramp = np.asarray(cinf_smoothstep(np.clip(u * 2.0, 0.0, 1.0)), dtype=np.float64)
            down_ramp = np.asarray(
                cinf_smoothstep(np.clip((1.0 - u) * 2.0, 0.0, 1.0)), dtype=np.float64
            )
            env = 0.08 + 0.92 * np.minimum(up_ramp, down_ramp)
            results.append((raw_burst * env, 0.35))

    return results


def synth_reamp_drive_conditioning(
    sample_rate: int = FS,
    scale: float = 1.0,
    max_samples_budget: int = 0,
) -> list[tuple[np.ndarray, float]]:
    """Synthesizes high-density nonlinear amp/cab conditioning phrases:
    power fifths, octaves, palm-to-open pick chugs, CCIF probes, and bass-band Schroeder multitone.
    """
    results: list[tuple[np.ndarray, float]] = []

    # 1. Power chords (root + fifth) across low/mid bass registers (intermodulation conditioning)
    power_fifths = [
        (30.87, 46.25, 0.85),  # B0 + F#1
        (41.20, 61.74, 0.88),  # E1 + B1
        (55.00, 82.41, 0.90),  # A1 + E2
        (73.42, 110.00, 0.88),  # D2 + A2
        (98.00, 146.83, 0.85),  # G2 + D3
    ]
    d_dur = max(0.4, 1.6 * scale)
    for f1, f2, amp in power_fifths:
        dyad = _synth_dyad(f1, f2, amp, d_dur, sample_rate, stagger_ms=8.0)
        results.append((dyad, 0.30))

    # 2. Upper tenths and double-stops
    tenths = [(65.41, 164.81), (82.41, 207.65), (110.00, 277.18)]
    for f1, f2 in tenths:
        dt = _synth_dyad(f1, f2, 0.80, max(0.4, 1.5 * scale), sample_rate, stagger_ms=10.0)
        results.append((dt, 0.30))

    # 3. Alternating palm-mute chugs to wide-open pick accents (speaker excursion + rail sag recovery)
    chug_roots = [30.87, 41.20, 49.00, 55.00, 65.41, 73.42]
    for f_root in chug_roots:
        pm1 = _synth_pluck(
            f_root, 0.78, max(0.10, 0.16 * scale), sample_rate, technique="palm_mute"
        )
        pm2 = _synth_pluck(
            f_root, 0.82, max(0.10, 0.16 * scale), sample_rate, technique="palm_mute"
        )
        accent = _synth_pluck(f_root, 0.94, max(0.35, 0.85 * scale), sample_rate, technique="pick")
        gap = np.zeros(max(8, int(0.015 * sample_rate)), dtype=np.float64)
        phrase = np.concatenate([pm1, gap, pm2, gap, accent])
        results.append((phrase, 0.30))

    # 4. Two-tone CCIF intermodulation probes embedded on low bass fundamentals
    ccif_probes = [
        (800.0, 920.0, 41.20),
        (1600.0, 1780.0, 55.00),
        (2800.0, 3050.0, 30.87),
    ]
    probe_dur = max(0.4, 1.8 * scale)
    for f1, f2, f_drone in ccif_probes:
        drone = _synth_pluck(f_drone, 0.72, probe_dur, sample_rate, clank=False, technique="finger")
        pr = _synth_two_tone_probe(f1, f2, 0.12, probe_dur, sample_rate)
        n_p = min(len(drone), len(pr))
        embedded = drone[:n_p] + pr[:n_p]
        embedded -= np.mean(embedded)
        p_max = float(np.max(np.abs(embedded)))
        if p_max > 0:
            embedded = (embedded / p_max) * 0.82
        results.append((embedded, 0.30))

    # 5. Bass-band Schroeder-phased multitone (25 Hz - 5500 Hz) for dense cabinet transfer conditioning
    if max_samples_budget > int(2.0 * sample_rate * scale):
        dur_m = min(10.0 * scale, (max_samples_budget / sample_rate) * 0.28)
        nm = int(dur_m * sample_rate)
        if nm > 256:
            t_m = np.linspace(0.0, dur_m, nm, endpoint=False)
            n_tones = 64
            freqs = np.geomspace(25.0, 5500.0, n_tones)
            mt = np.zeros(nm, dtype=np.float64)
            for k_idx, fk in enumerate(freqs, 1):
                # Schroeder quadratic phase minimizes peak-to-average ratio
                phi_k = -math.pi * k_idx * (k_idx - 1) / n_tones
                weight = 1.0 / math.sqrt(1.0 + (fk / 1800.0) ** 2)
                mt += weight * np.cos(2.0 * np.pi * fk * t_m + phi_k)
            mt -= float(np.mean(mt))
            mt_max = float(np.max(np.abs(mt)))
            if mt_max > 0.0:
                mt = (mt / mt_max) * 0.78
            mt = apply_cinf_fades(mt, min(nm // 8, int(0.025 * sample_rate)))
            results.append((mt, 0.35))

    return results


def generate_optimal_reamp_dry(
    duration_sec: float = DEFAULT_REAMP_DURATION_SEC,
    sample_rate: int = FS,
    peak_dbfs: float = DEFAULT_REAMP_PEAK_DBFS,
    target_rms_dbfs: float | None = None,
) -> np.ndarray:
    """Synthesizes a 48 kHz high-SNR dry excitation signal optimized for hardware amp+cab re-amping."""
    total_samples = int(duration_sec * sample_rate)
    if total_samples <= 0:
        raise ValueError(f"duration_sec must be positive (got {duration_sec}).")

    audio = np.zeros(total_samples, dtype=np.float64)
    scale = min(1.0, duration_sec / 240.0)

    lead_silence = min(int(0.3 * sample_rate), int(0.04 * total_samples))
    trail_silence = min(int(0.3 * sample_rate), int(0.04 * total_samples))
    cur = lead_silence
    limit = total_samples - trail_silence

    # Reserve final 432,000 samples (9.0s) for the held-out validation suite when duration >= 30s
    val_samples = 432_000 if total_samples >= int(30.0 * sample_rate) else int(0.10 * total_samples)
    val_start = max(cur, total_samples - val_samples)
    pre_val_pause_samples = min(int(0.4 * sample_rate), max(64, val_samples // 20))
    train_limit = max(cur, val_start - pre_val_pause_samples)

    # 1. Asymmetric cabinet-penetrating alignment pulses
    pulses = synth_reamp_alignment_pulses(sample_rate=sample_rate, scale=scale)
    n_pulses = min(len(pulses), train_limit - cur)
    if n_pulses > 0:
        audio[cur : cur + n_pulses] = pulses[:n_pulses]
        cur += n_pulses + max(100, int(0.25 * sample_rate * scale))

    def append_segment(seg: np.ndarray, pause_dur: float, seg_limit: int) -> None:
        nonlocal cur
        if len(seg) == 0 or cur >= seg_limit:
            return
        end_idx = cur + len(seg)
        if end_idx >= seg_limit:
            avail = max(0, seg_limit - cur)
            if avail > 0:
                faded_tail = apply_cinf_fades(
                    seg[:avail].astype(np.float64),
                    min(avail // 4, int(0.010 * sample_rate)),
                )
                audio[cur : cur + avail] = faded_tail
                cur += avail
            return
        audio[cur:end_idx] = seg
        p_samples = max(48, int(pause_dur * sample_rate * scale))
        cur = min(end_idx + p_samples, seg_limit)

    # 2. Cabinet-focused log sweeps (18 Hz - 8.5 kHz)
    for seg, pause in synth_reamp_cab_sweeps(sample_rate=sample_rate, scale=scale):
        if cur >= train_limit:
            break
        append_segment(seg, pause, train_limit)

    # 3. Extended velocity ladder & continuous dynamic swells
    for seg, pause in synth_reamp_velocity_and_swells(sample_rate=sample_rate, scale=scale):
        if cur >= train_limit:
            break
        append_segment(seg, pause, train_limit)

    # 4. Long uninterrupted ring-outs (> 75 dB decay)
    for seg, pause in synth_long_ringouts(sample_rate=sample_rate, scale=scale):
        if cur >= train_limit:
            break
        append_segment(seg, pause, train_limit)

    # 5. Inharmonic modal plucks across registers (A0 27.5 Hz - E3 164.8 Hz)
    for seg, pause in synth_modal_plucks(sample_rate=sample_rate, scale=scale):
        if cur >= train_limit:
            break
        append_segment(seg, pause, train_limit)

    # 6. Bass playing techniques & articulations
    for seg, pause in synth_articulation_bursts(sample_rate=sample_rate, scale=scale):
        if cur >= train_limit:
            break
        append_segment(seg, pause, train_limit)

    # 7. High-density nonlinear amp/drive conditioning (power fifths, chugs, CCIF, multitone)
    rem_budget = max(0, train_limit - cur)
    for seg, pause in synth_reamp_drive_conditioning(
        sample_rate=sample_rate,
        scale=scale,
        max_samples_budget=rem_budget,
    ):
        if cur >= train_limit:
            break
        append_segment(seg, pause, train_limit)

    # 8. Continuous glissandi slides
    for seg, pause in synth_glissandi_slides(sample_rate=sample_rate, scale=scale):
        if cur >= train_limit:
            break
        append_segment(seg, pause, train_limit)

    # 9. Sub-bass ring-outs, aggressive pick grooves, slap cascades, and active riff fill
    for f_deep in [30.87, 27.50, 36.71]:
        if cur >= train_limit:
            break
        r_deep = _synth_long_ringout(f_deep, 0.88, max(1.2, 4.8 * scale), sample_rate)
        append_segment(r_deep, 0.35, train_limit)

    for f_riff, bpm, tech in [
        (41.20, 145.0, "pick"),
        (55.00, 130.0, "finger"),
        (30.87, 135.0, "pick"),
        (73.42, 150.0, "pick"),
    ]:
        if cur >= train_limit:
            break
        riff = _synth_groove_burst(
            f_riff, 0.88, bpm=bpm, count=12, sample_rate=sample_rate, technique=tech
        )
        append_segment(riff, 0.30, train_limit)

    slap_cascades = [
        (41.20, 82.41),
        (55.00, 110.00),
        (73.42, 146.83),
        (98.00, 196.00),
        (30.87, 61.74),
    ]
    for f_s, f_p in slap_cascades:
        if cur >= train_limit:
            break
        sp_pair = _synth_slap_pop_pair(
            f_s, f_p, 0.90, gap_ms=65.0, dur=max(0.4, 1.3 * scale), sample_rate=sample_rate
        )
        append_segment(sp_pair, 0.30, train_limit)

    # Fill any remaining training budget with dynamic musical phrases & bass-passband sweeps (no dead >8.5 kHz tail)
    fill_notes = [41.20, 49.00, 55.00, 61.74, 65.41, 73.42, 82.41, 98.00]
    fill_idx = 0
    while cur < train_limit - int(0.25 * sample_rate):
        rem_sec = (train_limit - cur) / sample_rate
        if rem_sec <= 3.5:
            c_fill = _synth_log_chirp(
                rem_sec,
                22.0,
                6500.0,
                0.70,
                sample_rate,
                string_tilt=True,
                f_corner=2200.0,
            )
            append_segment(c_fill, 0.0, train_limit)
            break
        fn = fill_notes[fill_idx % len(fill_notes)]
        if fill_idx % 3 == 0:
            seg_fill = _synth_dyad(fn, fn * 1.5, 0.86, min(1.4, rem_sec * 0.5), sample_rate)
        elif fill_idx % 3 == 1:
            seg_fill = _synth_ghost_rake(fn, 0.88, min(1.1, rem_sec * 0.5), sample_rate)
        else:
            seg_fill = _synth_groove_burst(
                fn, 0.85, bpm=138.0, count=8, sample_rate=sample_rate, technique="pick"
            )
        append_segment(seg_fill, 0.25, train_limit)
        fill_idx += 1

    # 10. Representative Held-Out Validation Suite (final 432,000 samples = 9.0s)
    cur = max(cur, val_start)
    if cur < limit:
        v_ring = _synth_long_ringout(30.87, 0.85, max(0.8, 2.4 * scale), sample_rate)
        append_segment(v_ring, 0.30, limit)
    if cur < limit:
        v_groove = _synth_groove_burst(
            41.20, 0.84, bpm=140.0, count=8, sample_rate=sample_rate, technique="pick"
        )
        append_segment(v_groove, 0.30, limit)
    if cur < limit:
        v_slap = _synth_slap_pop_pair(
            41.20, 82.41, 0.90, gap_ms=65.0, dur=max(0.4, 1.3 * scale), sample_rate=sample_rate
        )
        append_segment(v_slap, 0.30, limit)
    if cur < limit:
        v_dyad = _synth_dyad(55.00, 82.41, 0.86, max(0.4, 1.2 * scale), sample_rate)
        append_segment(v_dyad, 0.30, limit)
    if cur < limit:
        v_harm = _synth_natural_harmonic(123.6, 0.78, max(0.4, 1.2 * scale), sample_rate)
        append_segment(v_harm, 0.25, limit)
    if cur < limit:
        v_pm = _synth_pluck(55.00, 0.84, max(0.3, 0.6 * scale), sample_rate, technique="palm_mute")
        append_segment(v_pm, 0.0, limit)

    # Zero-DC centering on active regions
    active_mask = audio != 0.0
    if np.any(active_mask):
        audio[active_mask] -= float(np.mean(audio[active_mask]))

    # Calibrate RMS and Peak Headroom using smooth C-infinity soft-knee limiting if needed
    target_peak = 10.0 ** (peak_dbfs / 20.0)
    if target_rms_dbfs is not None:
        target_rms = 10.0 ** (target_rms_dbfs / 20.0)
        cur_rms = float(np.sqrt(np.mean(np.square(audio))))
        if cur_rms > 1e-9:
            audio = audio * (target_rms / cur_rms)

        cur_peak = float(np.max(np.abs(audio)))
        if cur_peak > target_peak > 0.0:
            # Smooth C-infinity hyperbolic tangent soft-ceiling compression on peaks so target_peak_dbfs is exact
            audio = target_peak * np.tanh(audio / target_peak)
            # Re-scale to exact target_peak
            post_peak = float(np.max(np.abs(audio)))
            if post_peak > 1e-9:
                audio = audio * (target_peak / post_peak)
    else:
        cur_peak = float(np.max(np.abs(audio)))
        if cur_peak > 1e-9:
            audio = audio * (target_peak / cur_peak)

    return audio.astype(np.float32)


@app.default
def generate_reamp_input(
    output_file: Annotated[
        Path,
        Parameter(
            name=["--output", "-o"],
            help="Output path for the re-amping dry excitation WAV file.",
        ),
    ] = DEFAULT_REAMP_INPUT_PATH,
    *,
    duration: Annotated[
        float,
        Parameter(
            name=["--duration", "-d"],
            help="Total excitation duration in seconds.",
        ),
    ] = DEFAULT_REAMP_DURATION_SEC,
    peak_dbfs: Annotated[
        float,
        Parameter(
            name=["--peak-dbfs"],
            help="Target peak amplitude in dBFS (default -3.0 dBFS for Anagram input staging).",
        ),
    ] = DEFAULT_REAMP_PEAK_DBFS,
    rms_dbfs: Annotated[
        float | None,
        Parameter(
            name=["--rms-dbfs"],
            help="Optional broadband RMS target in dBFS (if omitted, scales directly to --peak-dbfs).",
        ),
    ] = None,
) -> Path:
    """Synthesizes and exports a 24-bit 48 kHz dry excitation WAV optimized for hardware amp+cab re-amping."""
    print("=======================================================", flush=True)
    print(" Allomorph Re-Amping Dry Excitation Generator", flush=True)
    print("=======================================================", flush=True)
    print(f"  Output Path:    {output_file}", flush=True)
    print(f"  Sample Rate:    {FS} Hz (24-bit Mono PCM)", flush=True)
    print(f"  Duration:       {duration:.1f}s ({duration / 60.0:.2f} min)", flush=True)
    rms_target_str = f"{rms_dbfs:+.2f} dBFS" if rms_dbfs is not None else "auto (peak-calibrated)"
    print(f"  Target Peak:    {peak_dbfs:+.2f} dBFS | Target RMS: {rms_target_str}", flush=True)
    print("-------------------------------------------------------", flush=True)

    audio = generate_optimal_reamp_dry(
        duration_sec=duration,
        sample_rate=FS,
        peak_dbfs=peak_dbfs,
        target_rms_dbfs=rms_dbfs,
    )

    output_file.parent.mkdir(parents=True, exist_ok=True)
    write_wav_24bit(output_file, audio, sample_rate=FS)

    actual_peak = float(np.max(np.abs(audio)))
    actual_rms = float(np.sqrt(np.mean(np.square(audio, dtype=np.float64))))
    actual_peak_db = 20.0 * math.log10(max(actual_peak, 1e-12))
    actual_rms_db = 20.0 * math.log10(max(actual_rms, 1e-12))
    crest_db = actual_peak_db - actual_rms_db

    print(f"  Generated Samples: {len(audio):,} ({len(audio) / FS:.2f}s)", flush=True)
    print(
        f"  Final Levels:      Peak: {actual_peak_db:+.2f} dBFS | "
        f"RMS: {actual_rms_db:+.2f} dBFS (Crest: {crest_db:.1f} dB)",
        flush=True,
    )
    print(
        f"  File Size:         {output_file.stat().st_size / (1024 * 1024):.2f} MB",
        flush=True,
    )
    print("  Status: Ready for Darkglass Anagram USB Output 9 re-amping.", flush=True)
    print("=======================================================\n", flush=True)
    return output_file


if __name__ == "__main__":
    app()
