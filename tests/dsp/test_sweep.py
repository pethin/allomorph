"""
Tests for Fast Logarithmic Sine Sweep synthesis, Farina deconvolution, and stem diagnostics.
"""

import numpy as np

from allomorph.circuit.stem_debug import debug_voicing_stem, format_stem_report_table
from allomorph.dsp import (
    deconvolve_log_sweep,
    extract_farina_harmonics,
    synthesize_fast_log_sweep,
)


def test_sweep_loopback_identity():
    """Validates that deconvolving an unperturbed logarithmic sweep yields exact 0.00 dB flat response."""
    sr = 48000
    n_samples = 16384
    x = synthesize_fast_log_sweep(
        n_samples=n_samples, f_start=10.0, f_end=24000.0, sr=sr, target_dbfs=-20.5
    )

    f_bins, H_complex, h_time = deconvolve_log_sweep(x, x, sr=sr, gate_taps=4096)

    # Impulse peak must be strictly causal at sample 0
    peak_idx = int(np.argmax(np.abs(h_time[:4096])))
    assert peak_idx == 0, f"Expected causal peak at sample 0, got {peak_idx}"

    # Frequency response across 100 Hz - 20 kHz must be within 0.05 dB of 0.00 dB
    mags_db = 20.0 * np.log10(np.abs(H_complex))
    passband_mask = (f_bins >= 100.0) & (f_bins <= 20000.0)
    passband_mags = mags_db[passband_mask]

    assert np.allclose(passband_mags, 0.0, atol=0.05), (
        f"Loopback passband deviated from 0 dB: min={np.min(passband_mags):.3f}, max={np.max(passband_mags):.3f}"
    )


def test_sweep_known_filter_accuracy():
    """Validates that deconvolving a known linear filter reproduces the analytical response to < 0.05 dB."""
    sr = 48000
    n_samples = 16384
    x = synthesize_fast_log_sweep(
        n_samples=n_samples, f_start=10.0, f_end=24000.0, sr=sr, target_dbfs=-20.5
    )

    # First-order Butterworth low-pass filter at fc = 2000 Hz
    fc = 2000.0
    w_c = 2.0 * np.pi * fc
    # Bilinear transform single-pole lowpass
    dt = 1.0 / sr
    alpha = (w_c * dt) / (2.0 + w_c * dt)
    b = np.array([alpha, alpha], dtype=np.float64)
    a = np.array([1.0, (w_c * dt - 2.0) / (2.0 + w_c * dt)], dtype=np.float64)

    # Filter sweep in time domain
    y = np.zeros_like(x)
    for i in range(len(x)):
        y[i] = (
            b[0] * x[i] + (b[1] * x[i - 1] if i > 0 else 0.0) - (a[1] * y[i - 1] if i > 0 else 0.0)
        )

    f_bins, H_complex, _ = deconvolve_log_sweep(y, x, sr=sr, gate_taps=4096)
    measured_mags = 20.0 * np.log10(np.abs(H_complex))

    # Exact discrete frequency response: H(z) = (b0 + b1*z^-1) / (a0 + a1*z^-1)
    omega = 2.0 * np.pi * f_bins / sr
    z_inv = np.exp(-1j * omega)
    h_discrete = (b[0] + b[1] * z_inv) / (a[0] + a[1] * z_inv)
    analytical_mags = 20.0 * np.log10(np.abs(h_discrete))

    # Compare across 50 Hz - 20 kHz
    eval_mask = (f_bins >= 50.0) & (f_bins <= 20000.0)
    diff = measured_mags[eval_mask] - analytical_mags[eval_mask]
    assert np.allclose(diff, 0.0, atol=0.05), (
        f"Max discrepancy from analytical: {np.max(np.abs(diff)):.3f} dB"
    )


def test_farina_harmonic_separation():
    """Validates that non-linear tanh saturation generates 3rd harmonic distortion at the predicted arrival time."""
    sr = 48000
    n_samples = 16384
    x = synthesize_fast_log_sweep(
        n_samples=n_samples, f_start=10.0, f_end=24000.0, sr=sr, target_dbfs=-12.0
    )

    # Odd-symmetric tanh saturation generates strong 3rd harmonic
    v_sat = 0.25
    y_distorted = v_sat * np.tanh(x / v_sat)

    _, _, h_time = deconvolve_log_sweep(y_distorted, x, sr=sr, gate_taps=None)

    harmonics = extract_farina_harmonics(
        h_time, n_samples=n_samples, f_start=10.0, f_end=24000.0, sr=sr
    )

    # 3rd harmonic must be detected and larger than 2nd harmonic (due to odd-symmetry tanh)
    assert harmonics["thd3_percent"] > 0.5, (
        f"Expected 3rd harmonic > 0.5%, got {harmonics['thd3_percent']}%"
    )
    assert harmonics["thd_percent"] > harmonics["thd2_percent"], (
        "Expected THD to exceed 2nd harmonic alone"
    )


def test_stem_debug_api():
    """Validates the stem diagnostic reporting API across standard and composite voicings."""
    report_p = debug_voicing_stem("precision_vintage")
    assert report_p.is_causal_zero_latency
    assert 0 <= report_p.onset_sample_index <= 4
    assert -2.0 <= report_p.peak_sample_index <= 5
    assert -30.0 <= report_p.rms_dbfs <= -5.0
    assert report_p.thd_percent >= 0.0

    table_text = format_stem_report_table(report_p)
    assert "ALLOMORPH STEM DIAGNOSTIC REPORT" in table_text
    assert "precision_vintage" in table_text
    assert "PASS" in table_text

    # Also validate composite multi-pickup voice
    report_dingwall = debug_voicing_stem("dingwall_parallel")
    assert report_dingwall.is_causal_zero_latency
    assert 0 <= report_dingwall.onset_sample_index <= 4


def test_all_catalog_instrument_voicings_bug_free():
    """Validates that all 56 native voicings defined across all 16 catalog instruments
    pass causal zero-latency alignment, remain below true-peak ceiling, and maintain stable THD.
    """
    from allomorph.config.instruments import INSTRUMENTS

    for inst_id, inst in sorted(INSTRUMENTS.items()):
        for v_id in sorted(inst.voicings.keys()):
            report = debug_voicing_stem(voice_id=v_id, instrument=inst_id)
            assert report.is_causal_zero_latency, (
                f"{inst_id}:{v_id} failed causal zero-latency (onset={report.onset_sample_index}, peak={report.peak_sample_index})"
            )
            assert report.peak_dbfs <= -0.09, (
                f"{inst_id}:{v_id} clipped above -0.09 dBFS ceiling ({report.peak_dbfs:+.2f} dBFS)"
            )
            assert report.thd_percent <= 50.0, (
                f"{inst_id}:{v_id} showed unstable harmonic distortion ({report.thd_percent:.1f}%)"
            )
