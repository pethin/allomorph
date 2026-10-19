"""Unit tests for `denoiser.noise` (ultrasonic LPF, multi-voice clean corpus, and stochastic 50/60 Hz + SMPS noise banks)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from allomorph.dsp.constants import FS
from allomorph.dsp.io import write_wav_24bit
from denoiser.noise import (
    StochasticNoiseGenerator,
    apply_ultrasonic_lowpass,
    build_clean_bass_corpus,
    build_split_calibration_probes,
    design_ultrasonic_lowpass,
    load_hardware_noise_bank,
    synthesize_mains_hum_bank,
    synthesize_smps_and_hiss_banks,
)


def test_ultrasonic_lowpass_preserves_bass_passband_and_attenuates_smps() -> None:
    """255-tap Blackman ultrasonic LPF at 14 kHz must preserve 20 Hz - 10 kHz at 1:1 zero phase and attenuate 18 kHz SMPS whine by >55 dB."""
    kernel = design_ultrasonic_lowpass(fc=14000.0, fs=float(FS), num_taps=255)
    assert len(kernel) == 255
    assert abs(float(np.sum(kernel)) - 1.0) < 1e-6
    # Symmetric linear-phase Type-I FIR check
    assert np.allclose(kernel, kernel[::-1], atol=1e-7)

    n = FS
    t = np.arange(n, dtype=np.float64) / float(FS)
    bass_sig = (
        0.50 * np.sin(2.0 * np.pi * 82.41 * t)
        + 0.30 * np.sin(2.0 * np.pi * 1200.0 * t)
        + 0.20 * np.sin(2.0 * np.pi * 6000.0 * t)
    ).astype(np.float32)
    smps_18k = (0.25 * np.sin(2.0 * np.pi * 18500.0 * t)).astype(np.float32)

    filtered_bass = apply_ultrasonic_lowpass(bass_sig, fc=14000.0, fs=float(FS), num_taps=255)
    # Interior region (excluding 128-sample filter edge transients) must match to within -70 dB
    interior = slice(256, n - 256)
    err_rms = float(np.sqrt(np.mean(np.square(filtered_bass[interior] - bass_sig[interior]))))
    sig_rms = float(np.sqrt(np.mean(np.square(bass_sig[interior]))))
    assert err_rms / sig_rms < 1e-3

    filtered_smps = apply_ultrasonic_lowpass(smps_18k, fc=14000.0, fs=float(FS), num_taps=255)
    smps_in_rms = float(np.sqrt(np.mean(np.square(smps_18k[interior]))))
    smps_out_rms = float(np.sqrt(np.mean(np.square(filtered_smps[interior]))))
    atten_db = 20.0 * np.log10(max(smps_out_rms, 1e-12) / smps_in_rms)
    assert atten_db < -55.0


def test_synthesize_mains_hum_and_smps_banks_spectral_and_rms_invariants() -> None:
    """Mains hum bank (50, 60, both) and SMPS/hiss banks must have unit RMS and concentrated energy in their target bands."""
    n = FS * 2
    for mains_mode in ("60", "50", "both"):
        hum = synthesize_mains_hum_bank(n, mains=mains_mode, sample_rate=FS, seed=42)
        assert hum.shape == (n,)
        assert hum.dtype == np.float32
        rms = float(np.sqrt(np.mean(np.square(hum, dtype=np.float64))))
        assert abs(rms - 1.0) < 1e-3

    smps, hiss = synthesize_smps_and_hiss_banks(n, sample_rate=FS, seed=99)
    assert smps.shape == (n,)
    assert hiss.shape == (n,)
    assert abs(float(np.sqrt(np.mean(np.square(smps, dtype=np.float64)))) - 1.0) < 1e-3
    assert abs(float(np.sqrt(np.mean(np.square(hiss, dtype=np.float64)))) - 1.0) < 1e-3

    # Verify SMPS bank has >90% of its energy above 2.0 kHz
    spec = np.abs(np.fft.rfft(smps.astype(np.float64))) ** 2
    freqs = np.fft.rfftfreq(n, d=1.0 / float(FS))
    hf_ratio = float(np.sum(spec[freqs >= 2000.0]) / np.sum(spec))
    assert hf_ratio > 0.90


def test_build_clean_bass_corpus_and_split_probes(tmp_path: Path) -> None:
    """build_clean_bass_corpus must combine multi-voice spectral shaping, split probes, and ultrasonic conditioning."""
    probes = build_split_calibration_probes(sample_rate=FS, duration_sec=2.0)
    assert len(probes) == 2 * FS
    # First 70% has active bass probes, final 30% is pure digital silence
    assert float(np.max(np.abs(probes[: int(1.3 * FS)]))) > 0.10
    assert float(np.max(np.abs(probes[int(1.45 * FS) :]))) == 0.0

    n_in = FS * 2
    t = np.arange(n_in, dtype=np.float32) / float(FS)
    dry = (
        0.35 * np.sin(2.0 * np.pi * 110.0 * t) + 0.10 * np.sin(2.0 * np.pi * 19000.0 * t)
    ).astype(np.float32)
    in_wav = tmp_path / "clean_in.wav"
    write_wav_24bit(in_wav, dry, sample_rate=FS)

    corpus, sr = build_clean_bass_corpus(in_wav, ultrasonic_lpf_fc=14000.0)
    assert sr == FS
    assert len(corpus) > 2 * n_in
    # Verify 19 kHz component was stripped from the conditioned clean corpus
    spec = np.abs(np.fft.rfft(corpus[:n_in].astype(np.float64)))
    freqs = np.fft.rfftfreq(n_in, d=1.0 / float(FS))
    f110_mag = float(np.max(spec[(freqs > 100.0) & (freqs < 120.0)]))
    f19k_mag = float(np.max(spec[freqs > 18000.0]))
    assert f19k_mag / f110_mag < 1e-3


def test_stochastic_noise_generator_and_hardware_noise_blending(tmp_path: Path) -> None:
    """StochasticNoiseGenerator must blend synthetic hum/SMPS with optional hardware noise WAV and zero out silence fraction."""
    hw_noise_path = tmp_path / "hw_noise.wav"
    t = np.arange(FS, dtype=np.float32) / float(FS)
    hw_sig = (
        0.01 * np.sin(2.0 * np.pi * 60.0 * t) + 0.005 * np.sin(2.0 * np.pi * 12000.0 * t)
    ).astype(np.float32)
    write_wav_24bit(hw_noise_path, hw_sig, sample_rate=FS)

    loaded_hw = load_hardware_noise_bank(hw_noise_path, target_len=96000, sample_rate=FS)
    assert len(loaded_hw) == 96000
    assert abs(float(np.sqrt(np.mean(np.square(loaded_hw, dtype=np.float64)))) - 1.0) < 1e-3

    device = torch.device("cpu")
    gen = StochasticNoiseGenerator(
        n_samples=96000,
        mains="both",
        profile="balanced",
        device=device,
        dtype=torch.float32,
        sample_rate=FS,
        noise_wav=hw_noise_path,
        seed=42,
    )
    assert gen.hw_noise_bank is not None

    rf_pad = 6346
    ny = 8192
    nx = ny + rf_pad
    y_clean = torch.ones(96000, dtype=torch.float32) * 0.25
    x_clean_padded = torch.cat([torch.zeros(rf_pad), y_clean])
    arange_nx = torch.arange(nx, dtype=torch.int32).unsqueeze(0)
    arange_ny = torch.arange(ny, dtype=torch.int32).unsqueeze(0)
    indices = torch.tensor(
        [[0], [8192], [16384], [24576], [32768], [40960], [49152], [57344]], dtype=torch.int32
    )

    x_noisy, y_target = gen.sample_training_batch(
        x_clean_padded=x_clean_padded,
        y_clean=y_clean,
        clean_indices=indices,
        arange_nx=arange_nx,
        arange_ny=arange_ny,
        rf_pad=rf_pad,
        silence_fraction=0.25,
    )
    assert x_noisy.shape == (8, 1, nx)
    assert y_target.shape == (8, ny)
    # First 2 items (25% of 8) must have y_target == 0.0 (pure silence gate targets) while x_noisy contains idle noise
    assert float(torch.max(torch.abs(y_target[:2])).item()) == 0.0
    assert float(torch.max(torch.abs(x_noisy[:2])).item()) > 0.0
    assert float(torch.mean(y_target[2:]).item()) == pytest.approx(0.25, abs=1e-6)
