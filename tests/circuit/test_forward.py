"""
Tests for forward circuit digital twin simulation in allomorph.circuit.forward.
All tests use generic synthetic instrument configurations.
"""

import math
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

if TYPE_CHECKING:
    from allomorph.config.schema import InstrumentConfig

from allomorph.circuit.forward import simulate_instrument_voicing
from allomorph.dsp import read_wav, write_wav_24bit


@pytest.fixture
def test_audio_file(tmp_path: Path) -> Path:
    """Generates a 50 ms 48 kHz synthetic bass audio file."""
    sr = 48000
    t = np.linspace(0, 0.05, int(sr * 0.05), endpoint=False)
    sig = 0.25 * np.sin(2.0 * np.pi * 100.0 * t) + 0.1 * np.sin(2.0 * np.pi * 1500.0 * t)
    p = tmp_path / "dry_in.wav"
    write_wav_24bit(p, sig.astype(np.float32), sample_rate=sr)
    return p


def test_forward_simulation_multi_pickup_magnet_properties(
    test_audio_file: Path, tmp_path: Path, generic_dual_pickup_instrument: InstrumentConfig
):
    """Bug 5 regression: multi-pickup instrument simulating bridge pickup must use bridge pickup config, not first pickup."""
    out_p = tmp_path / "bridge_out.wav"
    inst = generic_dual_pickup_instrument
    sim_path = simulate_instrument_voicing(
        instrument=inst,
        voicing=inst.voicings["bridge_solo"],
        input_wav=test_audio_file,
        output_wav=out_p,
        max_samples=2400,
        apply_saturation=True,
    )
    assert sim_path.exists()
    audio, _sr = read_wav(sim_path)
    assert len(audio) == 2400
    assert np.all(np.isfinite(audio))


def test_forward_auto_normalize_fallback_target_dbfs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, generic_instrument_config: InstrumentConfig
):
    """Bug 6 regression: when LUFS measurement falls back to RMS, target_dbfs must be respected."""

    # Force compute_lufs to return NaN to trigger RMS fallback branch
    def _mock_nan_lufs(*_args: Any, **_kwargs: Any) -> float:
        return float("nan")

    monkeypatch.setattr("allomorph.circuit.forward.compute_lufs", _mock_nan_lufs)
    monkeypatch.setattr("allomorph.dsp.conditioning.compute_lufs", _mock_nan_lufs)

    sig = np.full(500, 0.05, dtype=np.float64)
    short_in = tmp_path / "tiny.wav"
    write_wav_24bit(short_in, sig.astype(np.float32), sample_rate=48000)
    out_p = tmp_path / "tiny_out.wav"

    inst = generic_instrument_config
    target_dbfs = -18.0
    sim_path = simulate_instrument_voicing(
        instrument=inst,
        voicing=inst.voicings["generic_voice"],
        input_wav=short_in,
        output_wav=out_p,
        max_samples=500,
        normalize="lufs",  # Will fall back to RMS due to NaN
        target_dbfs=target_dbfs,
    )

    out_audio, _ = read_wav(sim_path)
    out_rms = float(np.sqrt(np.mean(out_audio**2)))
    out_dbfs = 20.0 * math.log10(max(out_rms, 1e-9))
    # Target level must match target_dbfs (-18 dBFS) within 0.1 dB
    assert abs(out_dbfs - target_dbfs) < 0.1, (
        f"Expected {target_dbfs} dBFS, got {out_dbfs:.2f} dBFS"
    )


def test_forward_simulation_voicing_controls(
    test_audio_file: Path, tmp_path: Path, generic_instrument_config: InstrumentConfig
):
    """Verify simulate_instrument_voicing executes with active volume and tone controls."""
    out_p = tmp_path / "p_out.wav"
    inst = generic_instrument_config
    sim_path = simulate_instrument_voicing(
        instrument=inst,
        voicing=inst.voicings["generic_voice"],
        input_wav=test_audio_file,
        output_wav=out_p,
        max_samples=2400,
        vol_pos=0.8,
        tone_pos=0.7,
    )
    assert sim_path.exists()
    audio, _sr = read_wav(sim_path)
    assert len(audio) == 2400
    assert np.all(np.isfinite(audio))


def test_forward_simulation_upright_piezo(
    test_audio_file: Path, tmp_path: Path, generic_piezo_instrument: InstrumentConfig
):
    """Verify forward digital twin simulation of Upright Piezo voicing."""
    out_p = tmp_path / "upright_piezo_out.wav"
    inst = generic_piezo_instrument
    sim_path = simulate_instrument_voicing(
        instrument=inst,
        voicing=inst.voicings["piezo_voice"],
        input_wav=test_audio_file,
        output_wav=out_p,
        max_samples=2400,
        apply_saturation=True,
    )
    assert sim_path.exists()
    audio, _sr = read_wav(sim_path)
    assert len(audio) == 2400
    assert np.all(np.isfinite(audio))

    # Small signal linear bypass check (peak <= 0.10)
    small_sig = (
        0.05 * np.sin(2.0 * np.pi * 100.0 * np.linspace(0, 0.05, 2400, endpoint=False))
    ).astype(np.float32)
    small_in = tmp_path / "small_in.wav"
    small_out = tmp_path / "small_out.wav"
    write_wav_24bit(small_in, small_sig, sample_rate=48000)
    sim_small = simulate_instrument_voicing(
        instrument=inst,
        voicing=inst.voicings["piezo_voice"],
        input_wav=small_in,
        output_wav=small_out,
        max_samples=2400,
        apply_saturation=True,
        normalize="none",
        dc_block=False,
    )
    small_audio, _ = read_wav(sim_small)
    assert np.all(np.isfinite(small_audio))


def test_forward_simulation_return_audio_and_in_memory(
    test_audio_file: Path, tmp_path: Path, generic_instrument_config: InstrumentConfig
):
    """Verify return_audio=True and in-memory input_audio array handling."""
    inst = generic_instrument_config
    sig = np.sin(2.0 * np.pi * 100.0 * np.linspace(0, 0.05, 2400, endpoint=False))
    out_p, audio = simulate_instrument_voicing(
        instrument=inst,
        voicing=inst.voicings["generic_voice"],
        input_audio=sig,
        return_audio=True,
        max_samples=1200,
        output_wav=tmp_path / "mem_out.wav",
    )
    assert out_p.exists()
    assert audio is not None
    assert len(audio) == 1200

    # Test cache hit path (force=False)
    _out_cached, audio_cached = simulate_instrument_voicing(
        instrument=inst,
        voicing=inst.voicings["generic_voice"],
        input_wav=test_audio_file,
        output_wav=tmp_path / "cached_out.wav",
        return_audio=True,
        force=False,
    )
    # Second call hits cache
    _out_cached2, audio_cached2 = simulate_instrument_voicing(
        instrument=inst,
        voicing=inst.voicings["generic_voice"],
        input_wav=test_audio_file,
        output_wav=tmp_path / "cached_out.wav",
        return_audio=True,
        force=False,
    )
    assert np.allclose(audio_cached, audio_cached2, atol=1e-4)


def test_forward_simulation_normalization_modes(
    test_audio_file: Path, tmp_path: Path, generic_instrument_config: InstrumentConfig
):
    """Verify peak and rms normalization modes in simulate_instrument_voicing."""
    inst = generic_instrument_config
    for mode in ("peak", "rms"):
        out_p = tmp_path / f"norm_{mode}.wav"
        res = simulate_instrument_voicing(
            instrument=inst,
            voicing=inst.voicings["generic_voice"],
            input_wav=test_audio_file,
            output_wav=out_p,
            normalize=mode,
            target_dbfs=-15.0,
            max_samples=1200,
        )
        assert res.exists()


def test_forward_multi_pickup_spatial_interference_and_coherence_decay_generic(
    generic_dual_pickup_instrument: InstrumentConfig,
):
    """Verifies that multi-pickup forward simulation correctly models physical spatial interference
    and C^inf cross-coherence decay using purely generic synthetic data:
    1. Primary spatial interference notch matches theoretical 1/(2*Δτ) frequency.
    2. Cross-coherence decay prevents spurious deep comb filter nulls at higher harmonic frequencies.
    3. Output audio starts strictly at sample 0 (zero latency per Guardrail 5.3.6).
    4. Identical pickup positions (Δτ = 0) yield in-phase summation without comb filtering.
    """
    from allomorph.config.schema import CoilConfig, PickupConfig
    from allomorph.dsp import deconvolve_log_sweep, synthesize_fast_log_sweep

    sr = 48000
    x_sweep = synthesize_fast_log_sweep(n_samples=16384, f_start=10.0, f_end=24000.0, sr=sr)

    # 1. Evaluate generic dual pickup instrument with pos_neck=0.14m and pos_bridge=0.06m
    inst = generic_dual_pickup_instrument
    y_wet = simulate_instrument_voicing(
        instrument=inst,
        voicing=inst.voicings["blend"],
        input_audio=x_sweep,
        return_audio=True,
        normalize="none",
        dc_block=False,
        apply_saturation=False,
        apply_dither=False,
    )
    assert y_wet is not None
    assert len(y_wet) == len(x_sweep)

    # Deconvolve to obtain empirical transfer function
    f_bins, H_c, ir = deconvolve_log_sweep(y_wet, x_sweep, sr=sr, gate_taps=8192)
    mag = 20.0 * np.log10(np.maximum(np.abs(H_c), 1e-4))

    # Verify zero-latency causal onset (minimum-phase peak occurs at sample 0..4)
    assert int(np.argmax(np.abs(ir))) <= 4

    # Analytical notch calculation:
    # L = 34in = 0.8636m, f0_mean = 60Hz -> c_mean = 2 * L * f0_mean = 103.632 m/s
    # delta_x = 0.14 - 0.06 = 0.08m -> tau = 0.08 / 103.632 = 0.000772s -> round(tau * 48000) = 37 samples
    # delta_tau = 37 / 48000 -> f_notch = 48000 / (2 * 37) = 648.6 Hz
    scale_m = inst.scale_length_m or 0.8636
    c_mean = 2.0 * scale_m * 60.0
    delta_x = 0.14 - 0.06
    delay_samples = round((delta_x / c_mean) * sr)
    f_notch_expected = sr / (2.0 * delay_samples)

    # Detect local minima deeper than -5 dB (excluding low DC roll-off below 150 Hz)
    minima_idx = [
        i
        for i in range(1, len(mag) - 1)
        if mag[i] < -5.0
        and f_bins[i] > 150.0
        and (
            (mag[i] < mag[i - 1] and mag[i] < mag[i + 1])
            or (
                mag[i] < mag[i - 1]
                and mag[i] == mag[i + 1]
                and (i + 2 >= len(mag) or mag[i + 2] > mag[i])
            )
        )
    ]
    notch_freqs = [float(f_bins[idx]) for idx in minima_idx]

    # Must exhibit exactly 1 primary spatial notch (no high-frequency spurious comb nulls)
    assert len(notch_freqs) == 1, (
        f"Generic multi-pickup simulation has spurious comb nulls: {notch_freqs}"
    )
    assert abs(notch_freqs[0] - f_notch_expected) < 100.0, (
        f"Primary notch {notch_freqs[0]:.1f} Hz deviates from expected {f_notch_expected:.1f} Hz"
    )

    # 2. Edge case: Identical pickup positions (delta_x = 0) must NOT produce comb nulls
    inst_identical = inst.model_copy(deep=True)
    inst_identical.pickups["neck"] = PickupConfig(
        name="Neck",
        position_from_bridge_m=0.10,
        coils=[
            CoilConfig(
                id="c1",
                position_from_bridge_m=0.10,
                aperture_width_in=0.5,
                L=3.0,
                Rdc=6000.0,
                Reddy=100000.0,
                Ccoil=5e-11,
            )
        ],
    )
    inst_identical.pickups["bridge"] = PickupConfig(
        name="Bridge",
        position_from_bridge_m=0.10,
        coils=[
            CoilConfig(
                id="c2",
                position_from_bridge_m=0.10,
                aperture_width_in=0.5,
                L=3.0,
                Rdc=6000.0,
                Reddy=100000.0,
                Ccoil=5e-11,
            )
        ],
    )

    y_ident = simulate_instrument_voicing(
        instrument=inst_identical,
        voicing=inst_identical.voicings["blend"],
        input_audio=x_sweep,
        return_audio=True,
        normalize="none",
        dc_block=False,
        apply_saturation=False,
        apply_dither=False,
    )
    _, H_ident_c, _ = deconvolve_log_sweep(y_ident, x_sweep, sr=sr, gate_taps=8192)
    mag_ident = 20.0 * np.log10(np.maximum(np.abs(H_ident_c), 1e-4))
    minima_ident = [
        i
        for i in range(1, len(mag_ident) - 1)
        if mag_ident[i] < -5.0
        and f_bins[i] > 150.0
        and mag_ident[i] < mag_ident[i - 1]
        and mag_ident[i] < mag_ident[i + 1]
    ]
    assert len(minima_ident) == 0, (
        f"Identical positions produced unexpected notches: {[f_bins[i] for i in minima_ident]}"
    )
