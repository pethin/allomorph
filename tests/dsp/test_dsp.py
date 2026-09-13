"""
Tests for Core Digital Signal Processing utilities in allomorph.dsp.
"""

import math
import tempfile
import wave
from pathlib import Path

import numpy as np

from allomorph.dsp import (
    fft_convolve,
    read_wav,
    read_wav_24bit,
    synthesize_minimum_phase_fir,
    write_wav_24bit,
)


def test_synthesize_minimum_phase_fir():
    # Simple lowpass magnitude curve
    mag_curve = [1.0 if i < 100 else 0.1 for i in range(4096)]
    num_taps = 4096
    fir = synthesize_minimum_phase_fir(mag_curve, num_taps=num_taps)

    assert len(fir) == num_taps

    # Peak normalization check (-0.1 dBFS, i.e., max peak is 0.99)
    max_peak = max(abs(x) for x in fir)
    assert math.isclose(max_peak, 0.99, rel_tol=1e-4)

    # Minimum phase causality: Energy should be concentrated at early taps
    early_energy = sum(x**2 for x in fir[:512])
    late_energy = sum(x**2 for x in fir[2048:])
    assert early_energy > late_energy * 10

    # Tail should taper smoothly towards zero
    assert abs(fir[-1]) < 0.01


def test_write_wav_24bit():
    samples = [0.0, 0.5, -0.5, 0.99, -0.99]
    with tempfile.TemporaryDirectory() as tmpdir:
        wav_path = Path(tmpdir) / "test_out.wav"
        write_wav_24bit(str(wav_path), samples, sample_rate=48000)

        assert wav_path.exists()
        with wave.open(str(wav_path), "rb") as wf:
            assert wf.getnchannels() == 1
            assert wf.getsampwidth() == 3  # 24-bit PCM
            assert wf.getframerate() == 48000
            assert wf.getnframes() == len(samples)


def test_read_wav_24bit_roundtrip():
    samples = np.array([0.0, 0.5, -0.5, 0.85, -0.85], dtype=np.float32)
    with tempfile.TemporaryDirectory() as tmpdir:
        wav_path = Path(tmpdir) / "test_roundtrip.wav"
        write_wav_24bit(str(wav_path), samples, sample_rate=48000)

        audio, sr = read_wav(wav_path)
        assert sr == 48000
        assert len(audio) == len(samples)
        assert np.allclose(audio, samples, atol=1e-4)

        # read_wav_24bit alias
        audio24, sr24 = read_wav_24bit(wav_path)
        assert sr24 == 48000
        assert np.array_equal(audio, audio24)

        # max_samples truncation
        audio_sub, _ = read_wav(wav_path, max_samples=3)
        assert len(audio_sub) == 3
        assert np.allclose(audio_sub, samples[:3], atol=1e-4)


def test_fft_convolve_modes_and_accuracy():
    # Test short and medium lengths
    for n, m in [(12, 5), (5, 12), (100, 32), (1000, 128)]:
        x = np.random.randn(n).astype(np.float64)
        y = np.random.randn(m).astype(np.float64)

        expected_full = np.convolve(x, y, mode="full")
        actual_full = fft_convolve(x, y, mode="full")
        assert np.allclose(actual_full, expected_full, atol=1e-9)

        expected_same = np.convolve(x, y, mode="same")
        actual_same = fft_convolve(x, y, mode="same")
        assert np.allclose(actual_same, expected_same, atol=1e-9)

        expected_causal = expected_full[: max(n, m)]
        actual_causal = fft_convolve(x, y, mode="causal")
        assert np.allclose(actual_causal, expected_causal, atol=1e-9)

    # Test overlap-add path (> 131072 samples)
    n_long = 150000
    m_ir = 256
    x_long = np.random.randn(n_long).astype(np.float32)
    y_ir = np.random.randn(m_ir).astype(np.float32)

    actual_overlap = fft_convolve(x_long, y_ir, mode="same")
    assert len(actual_overlap) == n_long

    actual_causal_long = fft_convolve(x_long, y_ir, mode="causal")
    assert len(actual_causal_long) == n_long

    # Check slice against direct convolve
    slice_len = 1000
    direct_slice = np.convolve(x_long[:slice_len], y_ir, mode="same")
    assert np.allclose(
        actual_overlap[m_ir : slice_len - m_ir], direct_slice[m_ir : slice_len - m_ir], atol=1e-4
    )
    direct_causal_slice = np.convolve(x_long[:slice_len], y_ir, mode="full")[:slice_len]
    assert np.allclose(
        actual_causal_long[: slice_len - m_ir], direct_causal_slice[: slice_len - m_ir], atol=1e-4
    )


def test_causal_preserves_pulse_timing_against_same_shift():
    """Validates that mode='causal' preserves exact impulse timing, while mode='same'

    shifts the signal by (m - 1) // 2 samples (1023 samples for a 2048-tap FIR).
    """
    m = 2048
    ir = np.zeros(m, dtype=np.float64)
    ir[0] = 1.0  # Ideal causal impulse

    n = 20000
    pulse_pos = 10000
    x = np.zeros(n, dtype=np.float64)
    x[pulse_pos] = 1.0

    out_causal = fft_convolve(x, ir, mode="causal")
    out_same = fft_convolve(x, ir, mode="same")

    # In causal mode, the peak is exactly at pulse_pos
    assert np.argmax(out_causal) == pulse_pos

    # In 'same' mode, the peak is shifted backwards by (m - 1) // 2 = 1023 samples
    expected_same_pos = pulse_pos - (m - 1) // 2
    assert np.argmax(out_same) == expected_same_pos
    assert np.argmax(out_causal) - np.argmax(out_same) == 1023


def test_calibrate_nam_v3_latency_detection_and_lookahead():
    """Validates that calibrate_nam_v3_latency detects zero delay, flags lookahead warnings,

    and handles delayed or silent audio appropriately.
    """
    from allomorph.dsp import calibrate_nam_v3_latency

    # 1. Construct synthetic V3 blip audio
    total_len = 580000
    y = np.zeros(total_len, dtype=np.float32)
    # Background noise level 0.001
    rng = np.random.default_rng(123)
    y[492000:498000] = rng.uniform(-0.001, 0.001, 6000).astype(np.float32)
    # Place blips at exactly 504000 and 552000 (amplitude 0.5)
    y[504000] = 0.5
    y[552000] = 0.5

    rec, lookahead_warn, not_det = calibrate_nam_v3_latency(y)
    assert not_det is False
    assert lookahead_warn is False
    assert rec == -1  # delay=0, minus safety_factor=1 -> -1

    # 2. Delayed audio by 1023 samples (e.g. from mode="same" shift)
    y_delayed = np.zeros(total_len, dtype=np.float32)
    y_delayed[504000 + 1023] = 0.5
    y_delayed[552000 + 1023] = 0.5
    rec_d, lookahead_d, not_det_d = calibrate_nam_v3_latency(y_delayed)
    assert not_det_d is False
    assert lookahead_d is False
    assert rec_d == 1022  # delay=1023, minus 1 -> 1022

    # 3. Early audio by 1000 samples (triggers lookahead warning)
    y_early = np.zeros(total_len, dtype=np.float32)
    y_early[504000 - 1000] = 0.5
    y_early[552000 - 1000] = 0.5
    _rec_e, lookahead_e, not_det_e = calibrate_nam_v3_latency(y_early)
    assert not_det_e is False
    assert lookahead_e is True

    # 4. Silent audio -> not detected
    y_silent = np.zeros(total_len, dtype=np.float32)
    _rec_s, _lookahead_s, not_det_s = calibrate_nam_v3_latency(y_silent)
    assert not_det_s is True

    # 5. Too short audio -> not detected
    y_short = np.zeros(1000, dtype=np.float32)
    _rec_sh, _lookahead_sh, not_det_sh = calibrate_nam_v3_latency(y_short)
    assert not_det_sh is True


def test_cinf_smoothstep():
    """Validates that cinf_smoothstep satisfies C^inf boundary and monotonicity properties."""
    from allomorph.dsp import cinf_smoothstep

    # 1. Scalar edge cases
    assert cinf_smoothstep(-1.0) == 0.0
    assert cinf_smoothstep(0.0) == 0.0
    assert cinf_smoothstep(1.0) == 1.0
    assert cinf_smoothstep(2.0) == 1.0
    assert math.isclose(cinf_smoothstep(0.5), 0.5, abs_tol=1e-12)

    # 2. Vectorized evaluation
    t_arr = np.linspace(-0.5, 1.5, 1000)
    out = cinf_smoothstep(t_arr)
    assert len(out) == 1000
    assert np.all(out[t_arr <= 0.0] == 0.0)
    assert np.all(out[t_arr >= 1.0] == 1.0)
    assert np.all(np.diff(out) >= -1e-12), "cinf_smoothstep must be strictly monotonic"

    # 3. Vanishing derivatives at boundaries (t -> 0+ and t -> 1-)
    dt = 1e-4
    d1_0 = (cinf_smoothstep(dt) - cinf_smoothstep(0.0)) / dt
    d1_1 = (cinf_smoothstep(1.0) - cinf_smoothstep(1.0 - dt)) / dt
    assert abs(d1_0) < 1e-4, f"First derivative at t=0 must vanish: {d1_0}"
    assert abs(d1_1) < 1e-4, f"First derivative at t=1 must vanish: {d1_1}"

    # 4. Symmetry: S(1 - t) == 1 - S(t)
    t_mid = np.linspace(0.01, 0.99, 100)
    assert np.allclose(cinf_smoothstep(1.0 - t_mid), 1.0 - cinf_smoothstep(t_mid), atol=1e-12)


