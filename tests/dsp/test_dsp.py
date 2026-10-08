"""
Tests for Core Digital Signal Processing utilities in allomorph.dsp.
"""

import math
import tempfile
import wave
from pathlib import Path
from typing import Any

import numpy as np
import pytest

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


def test_fft_convolve_full_length_exactness():
    """Regression test for Bug 1: Overlap-save convolution must not truncate the ring-out tail."""
    from allomorph.dsp import fft_convolve_multi

    n = 122880
    m = 4096
    np.random.seed(42)
    x = np.random.randn(n).astype(np.float64)
    f = np.random.randn(m).astype(np.float64)
    expected_full_len = n + m - 1

    # 1. Single filter fft_convolve
    res_single = fft_convolve(x, f, mode="full")
    assert len(res_single) == expected_full_len, (
        f"fft_convolve truncated: expected {expected_full_len}, got {len(res_single)}"
    )

    # 2. Multi-filter overlap-save fft_convolve_multi
    res_multi = fft_convolve_multi(x, [f, f], mode="full")
    assert len(res_multi[0]) == expected_full_len, (
        f"fft_convolve_multi truncated: expected {expected_full_len}, got {len(res_multi[0])}"
    )

    # 3. Exactness against numpy reference convolve
    ref_convolve = np.convolve(x, f, mode="full")
    max_err_single = float(np.max(np.abs(res_single - ref_convolve)))
    max_err_multi = float(np.max(np.abs(res_multi[0] - ref_convolve)))
    assert max_err_single < 1e-10, f"Max error single: {max_err_single}"
    assert max_err_multi < 1e-10, f"Max error multi: {max_err_multi}"


def test_write_wav_24bit_fallback_wave(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Verify write_wav_24bit gracefully falls back to stdlib wave when pedalboard is unavailable or fails."""

    # Monkeypatch pedalboard.io.AudioFile to raise RuntimeError
    def _mock_audio_file(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("Mock pedalboard failure")

    monkeypatch.setattr("pedalboard.io.AudioFile", _mock_audio_file)

    test_file = tmp_path / "fallback_24bit.wav"
    samples = np.array([0.0, 0.5, -0.5, 0.9], dtype=np.float32)
    write_wav_24bit(test_file, samples, sample_rate=48000)

    assert test_file.exists()
    audio, sr = read_wav(test_file)
    assert sr == 48000
    assert len(audio) == len(samples)
    assert np.allclose(audio, samples, atol=1e-4)


def test_read_wav_formats_and_multichannel(tmp_path: Path) -> None:
    """Verify read_wav parses 16-bit PCM, 32-bit float, 8-bit PCM, and stereo WAV files."""
    # 1. 16-bit PCM mono
    file_16 = tmp_path / "mono_16bit.wav"
    with wave.open(str(file_16), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(44100)
        data = (np.array([0.0, 0.5, -0.5], dtype=np.float32) * 32767.0).astype("<i2").tobytes()
        wf.writeframes(data)
    audio16, sr16 = read_wav(file_16)
    assert sr16 == 44100
    assert len(audio16) == 3
    assert abs(audio16[1] - 0.5) < 1e-3

    # 2. 32-bit float mono
    file_32 = tmp_path / "mono_32bit.wav"
    with wave.open(str(file_32), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(4)
        wf.setframerate(48000)
        data = np.array([0.0, 0.75, -0.75], dtype=np.float32).tobytes()
        wf.writeframes(data)
    audio32, sr32 = read_wav(file_32)
    assert sr32 == 48000
    assert len(audio32) == 3
    assert abs(audio32[1] - 0.75) < 1e-4

    # 3. 8-bit PCM mono
    file_8 = tmp_path / "mono_8bit.wav"
    with wave.open(str(file_8), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(1)
        wf.setframerate(22050)
        data = np.array([128, 192, 64], dtype=np.uint8).tobytes()
        wf.writeframes(data)
    audio8, sr8 = read_wav(file_8)
    assert sr8 == 22050
    assert len(audio8) == 3
    assert abs(audio8[0]) < 1e-2

    # 4. Stereo 24-bit PCM
    file_stereo = tmp_path / "stereo_24bit.wav"
    with wave.open(str(file_stereo), "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(3)
        wf.setframerate(48000)
        # 2 frames of 2 channels = 4 samples
        samples_l = np.array([0.5, -0.5], dtype=np.float32)
        samples_r = np.array([0.25, -0.25], dtype=np.float32)
        interleaved = np.empty(4, dtype=np.float32)
        interleaved[0::2] = samples_l
        interleaved[1::2] = samples_r
        scaled = np.clip(interleaved * 8388607.0, -8388608.0, 8388607.0).astype(np.int32)
        raw_bytes = scaled.astype("<i4").view(np.uint8).reshape(-1, 4)[:, :3].tobytes()
        wf.writeframes(raw_bytes)
    audio_st, sr_st = read_wav(file_stereo)
    assert sr_st == 48000
    assert audio_st.shape == (2, 2)
    assert np.allclose(audio_st[0], samples_l, atol=1e-3)
    assert np.allclose(audio_st[1], samples_r, atol=1e-3)


def test_apply_biquad():
    """Verify Direct Form II Transposed biquad filtering."""
    from allomorph.dsp import apply_biquad

    # Simple lowpass biquad coefficients
    b = (0.05, 0.10, 0.05)
    a = (1.0, -1.2, 0.4)
    x = np.sin(2.0 * np.pi * 100.0 * np.linspace(0, 0.1, 4800))
    y = apply_biquad(x, b, a)
    assert len(y) == len(x)
    assert np.all(np.isfinite(y))


def test_compute_lufs_branches():
    """Verify ITU-R BS.1770-4 LUFS computation across length and sample rate branches."""
    from allomorph.dsp import compute_lufs

    # 1. Empty audio
    assert compute_lufs(np.array([])) == -100.0

    # 2. Short audio (< 400 ms)
    short_audio = 0.5 * np.sin(2.0 * np.pi * 440.0 * np.linspace(0, 0.2, 9600))
    lufs_short = compute_lufs(short_audio, sample_rate=48000)
    assert -30.0 < lufs_short < 0.0

    # 3. Longer audio with gating (> 400 ms)
    long_audio = np.concatenate(
        [
            np.zeros(4800),
            0.5 * np.sin(2.0 * np.pi * 440.0 * np.linspace(0, 1.0, 48000)),
            np.zeros(4800),
        ]
    )
    lufs_long = compute_lufs(long_audio, sample_rate=48000)
    assert -30.0 < lufs_long < 0.0

    # 4. Non-48 kHz sample rate
    lufs_44k = compute_lufs(short_audio[:8820], sample_rate=44100)
    assert -30.0 < lufs_44k < 0.0

    # 5. Pure digital silence -> gates below -70 LKFS -> returns -100.0
    silence = np.zeros(48000)
    assert compute_lufs(silence, sample_rate=48000) == -100.0


def test_compute_true_peak_branches():
    """Verify true peak calculation across empty, small, and chunked signals."""
    from allomorph.dsp import compute_true_peak, compute_true_peak_dbfs

    # 1. Empty audio
    assert compute_true_peak(np.array([])) == 0.0
    assert compute_true_peak_dbfs(np.array([])) < -200.0

    # 2. Small audio (< 32 samples)
    small = np.array([0.1, -0.4, 0.3], dtype=np.float32)
    assert math.isclose(compute_true_peak(small), 0.4, abs_tol=1e-5)

    # 3. Large signal spanning multiple chunks (> 65536)
    sig = np.sin(2.0 * np.pi * 1000.0 * np.linspace(0, 2.0, 96000))
    tp = compute_true_peak(sig)
    assert math.isclose(tp, 1.0, abs_tol=0.05)
    tp_dbfs = compute_true_peak_dbfs(sig)
    assert abs(tp_dbfs) < 0.5


def test_fft_convolve_multi_modes():
    """Verify fft_convolve_multi causal, same, and short paths."""
    from allomorph.dsp import fft_convolve_multi

    n = 1000
    m = 64
    x = np.random.randn(n)
    h = np.random.randn(m)

    # Causal
    res_c = fft_convolve_multi(x, [h], mode="causal")
    assert len(res_c[0]) == n

    # Same
    res_s = fft_convolve_multi(x, [h], mode="same")
    assert len(res_s[0]) == n
