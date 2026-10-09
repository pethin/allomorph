"""
Allomorph - Final Audio Stem Diagnostics & Frequency Response Analyzer
Leverages fast logarithmic sine sweep deconvolution to evaluate rendered wet stems
for causal zero-latency alignment, sub-bass DC-blocker transmission, midband balance,
and Farina harmonic distortion (THD, 2nd, and 3rd harmonics).
"""

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from allomorph.circuit.forward import resolve_target_voicing, simulate_instrument_voicing
from allomorph.dsp import (
    deconvolve_log_sweep,
    extract_farina_harmonics,
    synthesize_fast_log_sweep,
    write_wav_24bit,
)


@dataclass
class StemDiagnosticReport:
    """Comprehensive diagnostic metrics for a rendered instrument voicing stem."""

    voice_id: str
    instrument_id: str
    drive_dbfs: float
    is_causal_zero_latency: bool
    onset_sample_index: int
    peak_sample_index: int
    crest_factor_db: float
    rms_dbfs: float
    peak_dbfs: float
    sub_bass_10hz_db: float
    sub_bass_20hz_db: float
    sub_bass_b0_31hz_db: float
    sub_bass_e1_41hz_db: float
    mid_500hz_db: float
    mid_1khz_db: float
    upper_mid_2khz_db: float
    treble_5khz_db: float
    treble_10khz_db: float
    ultra_20khz_db: float
    thd_percent: float
    thd_db: float
    thd2_percent: float
    thd2_db: float
    thd3_percent: float
    thd3_db: float
    output_wav: Path | None = None
    ir_wav: Path | None = None


def debug_voicing_stem(
    voice_id: str,
    instrument: str | None = None,
    drive_dbfs: float = -20.5,
    gate_taps: int | None = 4096,
    export_dir: Path | str | None = None,
    n_samples: int = 16384,
    sr: int = 48000,
) -> StemDiagnosticReport:
    """Renders a fast logarithmic sine sweep through simulate_instrument_voicing() and returns

    a comprehensive diagnostic report evaluating causal latency, frequency response,
    sub-bass transmission, and non-linear harmonic saturation.
    """
    inst, vcfg = resolve_target_voicing(voice_id, instrument=instrument)

    x_sweep = synthesize_fast_log_sweep(
        n_samples=n_samples,
        f_start=10.0,
        f_end=24000.0,
        sr=sr,
        target_dbfs=drive_dbfs,
        fade_len=144,
    )

    y_wet = simulate_instrument_voicing(
        instrument=inst,
        voicing=vcfg,
        input_audio=x_sweep,
        return_audio=True,
        normalize="none",
    )

    f_bins, H_complex, h_time = deconvolve_log_sweep(
        y_wet,
        x_sweep,
        f_start=10.0,
        f_end=24000.0,
        sr=sr,
        gate_taps=gate_taps,
    )

    # Fundamental impulse response onset and peak sample indices
    h_eval = np.abs(h_time[:gate_taps if gate_taps else len(h_time)])
    peak_sample = int(np.argmax(h_eval))
    peak_val = float(h_eval[peak_sample]) if len(h_eval) > 0 else 1.0
    # Onset threshold set at -40 dBc (1% of peak) to reliably detect acoustic onset
    # even on extreme sub-bass low-passed resonances (e.g. 190 Hz dub coils)
    thresh = 0.01 * max(peak_val, 1e-6)
    onset_candidates = np.where(h_eval >= thresh)[0]
    onset_sample = int(onset_candidates[0]) if len(onset_candidates) > 0 else peak_sample

    # Causal zero-latency invariant: the impulse response onset must begin at sample 0..4
    # (no artificial leading zeroes or unaligned dead latency), while multi-pickup spatial arrivals
    # or heavy tone low-pass group delays may position the maximum peak at sample 0..80.
    is_causal = (0 <= onset_sample <= 4) and (0 <= peak_sample <= 80)

    # Signal Levels
    rms = float(np.sqrt(np.mean(y_wet**2)))
    rms_dbfs = 20.0 * math.log10(max(rms, 1e-9))
    peak = float(np.max(np.abs(y_wet)))
    peak_dbfs = 20.0 * math.log10(max(peak, 1e-9))
    crest_factor = peak_dbfs - rms_dbfs

    # Frequency Response Anchors
    mags = np.abs(H_complex)
    mags_db = 20.0 * np.log10(np.maximum(mags, 1e-6))

    def get_db(f_target: float) -> float:
        idx = int(np.argmin(np.abs(f_bins - f_target)))
        return float(np.round(mags_db[idx], 2))

    # Farina Harmonic Distortion (THD)
    thd_info = extract_farina_harmonics(
        h_time,
        n_samples=n_samples,
        f_start=10.0,
        f_end=24000.0,
        sr=sr,
        win_len=512,
    )

    out_wav_path: Path | None = None
    ir_wav_path: Path | None = None
    if export_dir is not None:
        p_dir = Path(export_dir)
        p_dir.mkdir(parents=True, exist_ok=True)
        out_wav_path = p_dir / f"stem_{inst.id}_{voice_id}.wav"
        ir_wav_path = p_dir / f"ir_{inst.id}_{voice_id}.wav"
        write_wav_24bit(out_wav_path, y_wet.astype(np.float32), sr)
        write_wav_24bit(ir_wav_path, (h_time[:4096] / np.max(np.abs(h_time[:4096])) * 0.95).astype(np.float32), sr)

    return StemDiagnosticReport(
        voice_id=voice_id,
        instrument_id=inst.id,
        drive_dbfs=drive_dbfs,
        is_causal_zero_latency=is_causal,
        onset_sample_index=onset_sample,
        peak_sample_index=peak_sample,
        crest_factor_db=round(crest_factor, 2),
        rms_dbfs=round(rms_dbfs, 2),
        peak_dbfs=round(peak_dbfs, 2),
        sub_bass_10hz_db=get_db(10.0),
        sub_bass_20hz_db=get_db(20.0),
        sub_bass_b0_31hz_db=get_db(30.87),
        sub_bass_e1_41hz_db=get_db(41.2),
        mid_500hz_db=get_db(500.0),
        mid_1khz_db=get_db(1000.0),
        upper_mid_2khz_db=get_db(2000.0),
        treble_5khz_db=get_db(5000.0),
        treble_10khz_db=get_db(10000.0),
        ultra_20khz_db=get_db(20000.0),
        thd_percent=thd_info["thd_percent"],
        thd_db=thd_info["thd_db"],
        thd2_percent=thd_info["thd2_percent"],
        thd2_db=thd_info["thd2_db"],
        thd3_percent=thd_info["thd3_percent"],
        thd3_db=thd_info["thd3_db"],
        output_wav=out_wav_path,
        ir_wav=ir_wav_path,
    )


def format_stem_report_table(report: StemDiagnosticReport) -> str:
    """Formats a diagnostic report into an elegant ANSI console table."""
    status_sym = "✅ PASS" if report.is_causal_zero_latency else "❌ FAIL"
    lines = [
        "================================================================================",
        f" ALLOMORPH STEM DIAGNOSTIC REPORT: {report.voice_id} ({report.instrument_id})",
        "================================================================================",
        f"  Excitation Drive Level   : {report.drive_dbfs:+.1f} dBFS",
        f"  Causal Zero-Latency      : {status_sym} (onset sample {report.onset_sample_index}, peak sample {report.peak_sample_index})",
        f"  Stem Peak / RMS Level    : {report.peak_dbfs:+.2f} dBFS peak | {report.rms_dbfs:+.2f} dBFS rms (Crest: {report.crest_factor_db:.1f} dB)",
        "--------------------------------------------------------------------------------",
        " FREQUENCY RESPONSE ANCHORS (Absolute Transmission):",
        f"  Sub-Bass Transmission    : 10 Hz: {report.sub_bass_10hz_db:+.2f} dB | 20 Hz: {report.sub_bass_20hz_db:+.2f} dB",
        f"  Bass Note Registers      : B0 (30.9 Hz): {report.sub_bass_b0_31hz_db:+.2f} dB | E1 (41.2 Hz): {report.sub_bass_e1_41hz_db:+.2f} dB",
        f"  Midrange Registers       : 500 Hz: {report.mid_500hz_db:+.2f} dB | 1 kHz: {report.mid_1khz_db:+.2f} dB | 2 kHz: {report.upper_mid_2khz_db:+.2f} dB",
        f"  Treble & Ultrasonic      : 5 kHz: {report.treble_5khz_db:+.2f} dB | 10 kHz: {report.treble_10khz_db:+.2f} dB | 20 kHz: {report.ultra_20khz_db:+.2f} dB",
        "--------------------------------------------------------------------------------",
        " NON-LINEAR HARMONIC SATURATION (Farina THD Separation):",
        f"  Total Harmonic Distortion: {report.thd_percent:.3f}% ({report.thd_db:+.1f} dBc)",
        f"  2nd Harmonic (Asymmetric): {report.thd2_percent:.3f}% ({report.thd2_db:+.1f} dBc)",
        f"  3rd Harmonic (Core Sat)  : {report.thd3_percent:.3f}% ({report.thd3_db:+.1f} dBc)",
        "================================================================================",
    ]
    if report.output_wav or report.ir_wav:
        lines.append(f" Exported Artifacts: stem -> {report.output_wav}, ir -> {report.ir_wav}")
        lines.append("================================================================================")
    return "\n".join(lines)
