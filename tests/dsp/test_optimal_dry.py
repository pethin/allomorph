"""
Unit tests for Allomorph optimal synthetic bass dry signal generation.
"""

from pathlib import Path

import numpy as np

from allomorph.dsp import (
    FS,
    OPTIMAL_DRY_PATH,
    ensure_optimal_dry_wav,
    generate_optimal_bass_dry,
    read_wav,
)


def test_generate_optimal_bass_dry_properties():
    """Verify generated synthetic dry audio conforms to standard audio properties."""
    duration = 5.0  # short test duration
    audio = generate_optimal_bass_dry(duration_sec=duration, sample_rate=FS, peak_dbfs=-1.0)

    # 1. Output type and shape
    assert isinstance(audio, np.ndarray)
    assert audio.dtype == np.float32
    assert len(audio) == int(duration * FS)

    # 2. Finite numbers (no NaN, no Inf)
    assert np.all(np.isfinite(audio))

    # 3. Peak ceiling clamping: peak should be close to 10^(-1/20) ~ 0.89125
    peak = float(np.max(np.abs(audio)))
    expected_peak = 10.0 ** (-1.0 / 20.0)
    assert abs(peak - expected_peak) < 0.05
    assert peak <= 0.99  # strictly below full-scale ceiling

    # 4. Zero DC offset: mean must be virtually zero
    mean = float(np.mean(audio))
    assert abs(mean) < 1e-4


def test_generate_optimal_bass_dry_spectral_content():
    """Verify the synthetic dry signal has spectral content across sub-bass, mid, and treble."""
    duration = 10.0
    audio = generate_optimal_bass_dry(duration_sec=duration, sample_rate=FS, peak_dbfs=-1.0)

    # Compute FFT magnitude
    n_fft = len(audio)
    spec = np.abs(np.fft.rfft(audio))
    freqs = np.fft.rfftfreq(n_fft, 1.0 / FS)

    # Sub-bass energy (20 Hz - 60 Hz)
    sub_mask = (freqs >= 20.0) & (freqs <= 60.0)
    sub_energy = float(np.sum(spec[sub_mask] ** 2))
    assert sub_energy > 0.0, "Sub-bass spectrum must contain positive excitation energy"

    # Mid aperture range (500 Hz - 2500 Hz)
    mid_mask = (freqs >= 500.0) & (freqs <= 2500.0)
    mid_energy = float(np.sum(spec[mid_mask] ** 2))
    assert mid_energy > 0.0, "Mid spectrum must contain positive excitation energy"

    # Treble clank range (2500 Hz - 8000 Hz)
    treble_mask = (freqs >= 2500.0) & (freqs <= 8000.0)
    treble_energy = float(np.sum(spec[treble_mask] ** 2))
    assert treble_energy > 0.0, "Treble spectrum must contain positive excitation energy"


def test_ensure_optimal_dry_wav(tmp_path: Path):
    """Verify ensure_optimal_dry_wav synthesizes and writes a valid 24-bit PCM WAV file."""
    test_file = tmp_path / "test_synth_dry.wav"
    assert not test_file.exists()

    result = ensure_optimal_dry_wav(output_path=test_file, duration_sec=2.0)
    assert result == test_file
    assert test_file.exists()

    # Read back and inspect
    audio, sr = read_wav(test_file)
    assert sr == FS
    assert len(audio) == int(2.0 * FS)
    assert np.max(np.abs(audio)) <= 0.9900

    # Ensure calling without overwrite reuses existing file without modifying mtime
    mtime_before = test_file.stat().st_mtime_ns
    result2 = ensure_optimal_dry_wav(output_path=test_file, duration_sec=2.0, overwrite=False)
    assert result2 == test_file
    assert test_file.stat().st_mtime_ns == mtime_before


def test_optimal_bass_dry_zero_artificial_dither_and_silence_bounding():
    """Validates that optimal_bass_dry strictly preserves pure digital silence (0.0),

    contains zero artificial dither/noise floor, and bounds all silence intervals < 1.0s.
    """
    audio = generate_optimal_bass_dry(duration_sec=180.0, sample_rate=FS, peak_dbfs=-1.0)

    # 1. Pure digital silence check: at least 15% of the file consists of bit-exact zeros
    zero_mask = audio == 0.0
    zero_fraction = float(np.mean(zero_mask))
    assert 0.15 <= zero_fraction <= 0.35, f"Expected 15-35% pure digital silence, got {zero_fraction:.2%}"

    # 2. Silence intervals: analyze run lengths of exact zeros
    silence_int = zero_mask.astype(np.int8)
    diffs = np.diff(np.pad(silence_int, (1, 1), "constant"))
    starts = np.where(diffs == 1)[0]
    ends = np.where(diffs == -1)[0]
    runs = (ends - starts) / FS

    # Leading silence <= 0.5s (Tone3000 boundary guard)
    assert runs[0] <= 0.5, f"Leading silence {runs[0]:.2f}s exceeds 0.5s ceiling"
    # Trailing silence <= 0.5s (Tone3000 boundary guard)
    assert runs[-1] <= 0.5, f"Trailing silence {runs[-1]:.2f}s exceeds 0.5s ceiling"
    # No gap anywhere exceeds 1.0s
    assert np.max(runs) < 1.0, f"Max pause {np.max(runs):.2f}s exceeds 1.0s ceiling"


def test_optimal_bass_dry_drop_a_sub_bass_and_determinism():
    """Validates Drop A0 sub-bass energy and bit-exact generation determinism."""
    # Determinism test
    a1 = generate_optimal_bass_dry(duration_sec=5.0, sample_rate=FS, seed=42)
    a2 = generate_optimal_bass_dry(duration_sec=5.0, sample_rate=FS, seed=42)
    assert np.array_equal(a1, a2), "Generation must be 100% bit-exact deterministic"

    # Drop A0 (27.5 Hz) sub-bass energy in 180s track
    audio_full = generate_optimal_bass_dry(duration_sec=180.0, sample_rate=FS)
    n_fft = len(audio_full)
    spec = np.abs(np.fft.rfft(audio_full))
    freqs = np.fft.rfftfreq(n_fft, 1.0 / FS)
    drop_a_mask = (freqs >= 26.0) & (freqs <= 29.0)
    drop_a_energy = float(np.sum(spec[drop_a_mask] ** 2))
    assert drop_a_energy > 0.0, "Must contain measurable Drop A0 (27.5 Hz) sub-bass energy"


def test_optimal_dry_canonical_path():
    """Verify canonical optimal dry path points to audio/canonical/optimal_bass_dry_v2.wav."""
    from allomorph.version import DSP_GENERATION

    assert OPTIMAL_DRY_PATH.name == f"optimal_bass_dry_v{DSP_GENERATION}.wav"
    assert OPTIMAL_DRY_PATH.parent.name == "canonical"


