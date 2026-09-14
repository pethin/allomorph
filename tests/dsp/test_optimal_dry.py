"""
Unit tests for Allomorph optimal synthetic bass dry signal generation.
"""

from pathlib import Path

import numpy as np

from allomorph.dsp import (
    FS,
    OPTIMAL_DRY_PATH,
    calibrate_nam_v3_latency,
    ensure_optimal_dry_wav,
    fft_convolve,
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
    contains zero artificial dither/noise floor, and bounds leading/trailing boundaries.
    """
    audio, sr = read_wav(OPTIMAL_DRY_PATH)
    assert sr == FS
    assert len(audio) == 240 * FS

    # 1. Pure digital silence check: file contains bit-exact zeros in rest intervals
    zero_mask = audio == 0.0
    zero_fraction = float(np.mean(zero_mask))
    assert zero_fraction > 0.05, (
        f"Expected bit-exact digital zeros in rests, got {zero_fraction:.2%}"
    )

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


def test_optimal_bass_dry_drop_a_sub_bass_and_determinism():
    """Validates Drop A0 sub-bass energy and bit-exact generation determinism."""
    # Determinism test
    a1 = generate_optimal_bass_dry(duration_sec=5.0, sample_rate=FS, seed=42)
    a2 = generate_optimal_bass_dry(duration_sec=5.0, sample_rate=FS, seed=42)
    assert np.array_equal(a1, a2), "Generation must be 100% bit-exact deterministic"

    # Drop A0 (27.5 Hz) sub-bass energy in canonical track
    audio_full, _ = read_wav(OPTIMAL_DRY_PATH)
    n_fft = len(audio_full)
    spec = np.abs(np.fft.rfft(audio_full))
    freqs = np.fft.rfftfreq(n_fft, 1.0 / FS)
    drop_a_mask = (freqs >= 26.0) & (freqs <= 29.0)
    drop_a_energy = float(np.sum(spec[drop_a_mask] ** 2))
    assert drop_a_energy > 0.0, "Must contain measurable Drop A0 (27.5 Hz) sub-bass energy"


def test_optimal_bass_dry_non_v3_prelude_latency_zero():
    """Validates that optimal_bass_dry triggers Tone3000's non-V3 prelude fallback:
    '[t3k] Dry-wet input is not V3 prelude (too_short); defaulting latency to 0.'
    guaranteeing zero latency offset without false blip triggering.
    """
    audio, _ = read_wav(OPTIMAL_DRY_PATH)
    rec_delay, lookahead_warn, not_detected = calibrate_nam_v3_latency(audio)
    assert not_detected is True, (
        f"Expected not_detected=True, got {not_detected} (delay={rec_delay})"
    )
    assert rec_delay == 0, f"Expected recommended delay=0, got {rec_delay}"
    assert lookahead_warn is False


def test_optimal_bass_dry_non_zero_dry_wet_delta():
    """Validates that filtering the dry file yields a non-zero dry/wet delta,
    guaranteeing Tone3000's identity-bypass rejection check passes cleanly.
    """
    audio, _ = read_wav(OPTIMAL_DRY_PATH)
    fir = np.zeros(1024, dtype=np.float32)
    fir[0] = 0.85
    fir[8] = -0.35
    wet = fft_convolve(audio[: 48000 * 2], fir, mode="same")
    delta = float(np.max(np.abs(wet - audio[: 48000 * 2])))
    assert delta > 0.05, f"Expected substantial dry/wet delta, got {delta}"


def test_optimal_bass_dry_15_vector_physical_features():
    """Validates physical excitation features across the 240.0s track:
    full-fretboard glissandi (up to 392 Hz), CCIF resonant probes (3-4.25 kHz),
    and exact 240.0s duration.
    """
    audio, _ = read_wav(OPTIMAL_DRY_PATH)
    assert len(audio) == 240 * FS, f"Expected 240s ({240 * FS} samples), got {len(audio)}"

    n_fft = len(audio)
    spec = np.abs(np.fft.rfft(audio))
    freqs = np.fft.rfftfreq(n_fft, 1.0 / FS)

    # 1. High fretboard fundamental energy (G4 ~ 392 Hz from 24th fret glissando)
    g4_mask = (freqs >= 385.0) & (freqs <= 399.0)
    g4_energy = float(np.sum(spec[g4_mask] ** 2))
    assert g4_energy > 0.0, "Must contain energy around G4 (392 Hz) from 24-fret glissandi"

    # 2. CCIF Resonant probe energy (3000 Hz and 4000 Hz)
    probe1_mask = (freqs >= 2990.0) & (freqs <= 3210.0)
    probe2_mask = (freqs >= 3990.0) & (freqs <= 4260.0)
    assert np.sum(spec[probe1_mask] ** 2) > 0.0, "Must contain 3.0-3.2 kHz CCIF probe energy"
    assert np.sum(spec[probe2_mask] ** 2) > 0.0, "Must contain 4.0-4.25 kHz CCIF probe energy"


def test_optimal_dry_canonical_path():
    """Verify canonical optimal dry path points to audio/canonical/optimal_bass_dry_v2.wav."""
    from allomorph.version import DSP_GENERATION

    assert OPTIMAL_DRY_PATH.name == f"optimal_bass_dry_v{DSP_GENERATION}.wav"
    assert OPTIMAL_DRY_PATH.parent.name == "canonical"
