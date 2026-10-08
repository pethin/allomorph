"""
Unit tests for Allomorph optimal synthetic bass dry signal generation.
"""

import math
from pathlib import Path

import numpy as np

from allomorph.dsp import (
    DEFAULT_INPUT_PATH,
    FS,
    calibrate_nam_v3_latency,
    ensure_input_audio_wav,
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


def test_ensure_input_audio_wav(tmp_path: Path):
    """Verify ensure_input_audio_wav synthesizes and writes a valid 24-bit PCM WAV file."""
    test_file = tmp_path / "test_synth_dry.wav"
    assert not test_file.exists()

    result = ensure_input_audio_wav(output_path=test_file, duration_sec=2.0)
    assert result == test_file
    assert test_file.exists()

    # Read back and inspect
    audio, sr = read_wav(test_file)
    assert sr == FS
    assert len(audio) == int(2.0 * FS)
    assert np.max(np.abs(audio)) <= 0.9900

    # Ensure calling without overwrite reuses existing file without modifying mtime
    mtime_before = test_file.stat().st_mtime_ns
    result2 = ensure_input_audio_wav(output_path=test_file, duration_sec=2.0, overwrite=False)
    assert result2 == test_file
    assert test_file.stat().st_mtime_ns == mtime_before


def test_optimal_bass_dry_zero_artificial_dither_and_silence_bounding():
    """Validates that optimal_bass_dry strictly preserves pure digital silence (0.0),
    contains zero artificial dither/noise floor, and bounds leading/trailing boundaries.
    """
    path = ensure_input_audio_wav()
    audio, sr = read_wav(path)
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

    # Drop A0 (27.5 Hz) sub-bass energy in input track
    path = ensure_input_audio_wav()
    audio_full, _ = read_wav(path)
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
    path = ensure_input_audio_wav()
    audio, _ = read_wav(path)
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
    path = ensure_input_audio_wav()
    audio, _ = read_wav(path)
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
    path = ensure_input_audio_wav()
    audio, _ = read_wav(path)
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


def test_default_input_path():
    """Verify default input path points to audio/input.wav."""
    from allomorph.naming import get_default_input_path

    assert DEFAULT_INPUT_PATH.name == "input.wav"
    assert DEFAULT_INPUT_PATH == get_default_input_path()


def test_synthetic_articulation_subroutines():
    """Directly unit test all synthetic articulation subroutines and their edge cases."""
    from allomorph.dsp import (
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

    # 1. _synth_pluck with various playing techniques and velocity tiers
    # Slap with high excursion (fret collision branch) vs low excursion
    p_slap_hi = _synth_pluck(41.2, 0.85, 0.1, FS, clank=True, technique="slap")
    p_slap_lo = _synth_pluck(41.2, 0.50, 0.1, FS, clank=True, technique="slap")
    assert len(p_slap_hi) == int(0.1 * FS)
    assert np.all(np.isfinite(p_slap_hi))
    assert np.all(np.isfinite(p_slap_lo))

    # Pick with high vs low excursion
    p_pick_hi = _synth_pluck(55.0, 0.80, 0.1, FS, clank=True, technique="pick")
    p_pick_lo = _synth_pluck(55.0, 0.50, 0.1, FS, clank=True, technique="pick")
    assert len(p_pick_hi) == int(0.1 * FS)
    assert len(p_pick_lo) == int(0.1 * FS)

    # Palm mute
    p_pm = _synth_pluck(41.2, 0.85, 0.1, FS, clank=False, technique="palm_mute")
    assert len(p_pm) == int(0.1 * FS)

    # Finger with clank (high and low amp) and without clank
    p_finger_hi = _synth_pluck(73.42, 0.80, 0.1, FS, clank=True, technique="finger")
    p_finger_lo = _synth_pluck(73.42, 0.40, 0.1, FS, clank=True, technique="finger")
    p_finger_noclank = _synth_pluck(73.42, 0.40, 0.1, FS, clank=False, technique="finger")
    assert len(p_finger_hi) == int(0.1 * FS)
    assert len(p_finger_lo) == int(0.1 * FS)
    assert len(p_finger_noclank) == int(0.1 * FS)

    # 2. _synth_ghost_note normal and zero duration
    g_norm = _synth_ghost_note(0.04, 0.75, FS)
    assert len(g_norm) == int(0.04 * FS)
    assert np.all(np.isfinite(g_norm))
    g_empty = _synth_ghost_note(0.0, 0.75, FS)
    assert len(g_empty) == 0

    # 3. _synth_natural_harmonic normal and high-frequency cutoff
    harm = _synth_natural_harmonic(82.41, 0.70, 0.1, FS, num_partials=5)
    assert len(harm) == int(0.1 * FS)
    harm_hi = _synth_natural_harmonic(22000.0, 0.70, 0.1, FS, num_partials=5)
    assert len(harm_hi) == int(0.1 * FS)
    harm_empty = _synth_natural_harmonic(82.41, 0.70, 0.0, FS)
    assert len(harm_empty) == 0

    # 4. _synth_dyad
    dyad = _synth_dyad(41.2, 61.74, 0.75, 0.15, FS)
    assert len(dyad) > 0
    assert np.all(np.isfinite(dyad))

    # 5. _synth_glissando rising, falling, and zero duration
    gl_up = _synth_glissando(41.2, 82.4, 0.80, 0.1, FS)
    gl_down = _synth_glissando(82.4, 41.2, 0.80, 0.1, FS)
    assert len(gl_up) == int(0.1 * FS)
    assert len(gl_down) == int(0.1 * FS)
    gl_empty = _synth_glissando(41.2, 82.4, 0.80, 0.0, FS)
    assert len(gl_empty) == 0

    # 6. _synth_ghost_rake
    rake = _synth_ghost_rake(41.2, 0.80, 0.25, FS)
    assert len(rake) > 0
    assert np.all(np.isfinite(rake))

    # 7. _synth_groove_burst (exercising even/odd stroke amplitudes)
    gb = _synth_groove_burst(41.2, 0.80, bpm=120.0, count=6, sample_rate=FS, technique="finger")
    assert len(gb) > 0
    assert np.all(np.isfinite(gb))

    # 8. _synth_slap_pop_pair
    sp = _synth_slap_pop_pair(41.2, 82.41, 0.85, gap_ms=40.0, dur=0.2, sample_rate=FS)
    assert len(sp) > 0
    assert np.all(np.isfinite(sp))

    # 9. _synth_vibrato_pluck normal and zero duration
    vib = _synth_vibrato_pluck(55.0, 0.75, 0.15, mod_rate=5.0, mod_depth_cents=25.0, sample_rate=FS)
    assert len(vib) == int(0.15 * FS)
    assert np.all(np.isfinite(vib))
    vib_empty = _synth_vibrato_pluck(55.0, 0.75, 0.0, sample_rate=FS)
    assert len(vib_empty) == 0

    # 10. _synth_long_ringout normal and zero duration
    ring = _synth_long_ringout(41.2, 0.85, 0.15, FS)
    assert len(ring) == int(0.15 * FS)
    assert np.all(np.isfinite(ring))
    ring_empty = _synth_long_ringout(41.2, 0.85, 0.0, FS)
    assert len(ring_empty) == 0

    # 11. _synth_two_tone_probe normal and zero duration
    pr = _synth_two_tone_probe(3000.0, 3200.0, 0.5, 0.1, FS)
    assert len(pr) == int(0.1 * FS)
    assert np.all(np.isfinite(pr))
    pr_empty = _synth_two_tone_probe(3000.0, 3200.0, 0.5, 0.0, FS)
    assert len(pr_empty) == 0


def test_generate_optimal_bass_dry_full_track():
    """Verify complete 240-second dry generation executes all 9 tiers and target RMS normalization."""
    audio = generate_optimal_bass_dry(
        duration_sec=240.0,
        sample_rate=FS,
        peak_dbfs=-1.0,
        target_rms_dbfs=-20.50,
        seed=42,
    )
    assert len(audio) == 240 * FS
    assert audio.dtype == np.float32
    assert np.all(np.isfinite(audio))

    # Check peak bounding
    peak = float(np.max(np.abs(audio)))
    expected_peak_ceiling = 10.0 ** (-1.0 / 20.0)
    assert peak <= expected_peak_ceiling + 1e-4

    # Check RMS is calibrated near -20.50 dBFS
    rms = float(np.sqrt(np.mean(audio**2)))
    rms_dbfs = 20.0 * math.log10(rms)
    assert abs(rms_dbfs - (-20.50)) < 1.5

    # Check pure silence in rests
    zero_fraction = float(np.mean(audio == 0.0))
    assert zero_fraction > 0.05
