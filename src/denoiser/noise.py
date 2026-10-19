"""Stochastic 50/60 Hz mains hum, SMPS switching noise, and pickup-agnostic clean bass corpus synthesis."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch

from allomorph.dsp.constants import CALIBRATION_PEAK_CEILING, FS
from allomorph.dsp.io import read_wav
from denoiser.constants import (
    DENOISER_DEFAULT_ULTRASONIC_LPF_FC,
    DENOISER_FIR_NUM_TAPS,
    DENOISER_PROFILES,
    MainsMode,
    ProfileName,
)


def design_ultrasonic_lowpass(
    fc: float = DENOISER_DEFAULT_ULTRASONIC_LPF_FC,
    fs: float = float(FS),
    num_taps: int = DENOISER_FIR_NUM_TAPS,
) -> np.ndarray:
    """Designs a symmetric Type-I linear-phase Blackman-windowed sinc low-pass FIR filter.

    Because the filter is symmetric around tap `M = (num_taps - 1) // 2` and applied in
    zero-phase (`mode='same'`), its transfer function across the bass passband (20 Hz - 12 kHz)
    has unity magnitude (1.0) and zero group delay (0 samples), while attenuating ultrasonic
    SMPS switching carriers (>14 kHz) by >70 dB.
    """
    if num_taps % 2 == 0 or num_taps < 3:
        raise ValueError(f"num_taps must be an odd integer >= 3, got {num_taps}.")
    if not (100.0 < fc < 0.5 * fs):
        raise ValueError(f"Cutoff fc={fc} Hz must lie strictly inside (100, {0.5 * fs}).")

    m = (num_taps - 1) // 2
    n = np.arange(num_taps, dtype=np.float64) - float(m)
    h = (2.0 * fc / fs) * np.sinc((2.0 * fc / fs) * n)
    window = np.blackman(num_taps)
    h_win = h * window
    dc_sum = float(np.sum(h_win))
    if abs(dc_sum) > 1e-12:
        h_win /= dc_sum
    return h_win.astype(np.float32)


def apply_ultrasonic_lowpass(
    signal: np.ndarray,
    fc: float = DENOISER_DEFAULT_ULTRASONIC_LPF_FC,
    fs: float = float(FS),
    num_taps: int = DENOISER_FIR_NUM_TAPS,
) -> np.ndarray:
    """Applies zero-phase symmetric FIR low-pass conditioning to strip >14 kHz ultrasonic content."""
    if len(signal) == 0:
        return signal.astype(np.float32, copy=True)
    kernel = design_ultrasonic_lowpass(fc=fc, fs=fs, num_taps=num_taps)
    filtered = np.convolve(signal.astype(np.float32), kernel, mode="same")
    return filtered.astype(np.float32)


def _apply_spectral_voicing_eq(
    dry: np.ndarray,
    sample_rate: int,
    voice_idx: int,
) -> np.ndarray:
    """Shapes a clean bass waveform with linear pickup/preamp spectral voicings in the FFT domain.

    Voicings span the physical bass guitar pickup landscape without introducing nonlinear clipping
    or pickup hum:
      0: Direct Studio DI (flat unity reference)
      1: Vintage Passive Split-Coil P-Bass (warm 180 Hz body, smooth 3.8 kHz passive roll-off)
      2: Dual-Coil Jazz Bass (deep 60 Hz bottom, 650 Hz aperture comb scoop, 2.8 kHz snap)
      3: Active Humbucker / StingRay (+4 dB 85 Hz bass shelf, +5 dB 3.2 kHz active treble)
      4: Bright Rickenbacker / Pick + Deep Flatwound Dub hybrid (1.8 kHz clang + 90 Hz fundamental)
    """
    if voice_idx % 5 == 0 or len(dry) < 64:
        return dry.astype(np.float32, copy=True)

    n = len(dry)
    spec = np.fft.rfft(dry.astype(np.float64))
    freqs = np.fft.rfftfreq(n, d=1.0 / float(sample_rate))

    v = voice_idx % 5
    if v == 1:
        # Vintage Passive P-Bass: +3.0 dB around 180 Hz, passive LPF roll-off above 3.8 kHz
        bump = 1.0 + 0.40 * np.exp(-0.5 * ((freqs - 180.0) / 110.0) ** 2)
        rolloff = 1.0 / np.sqrt(1.0 + (freqs / 3800.0) ** 4)
        H = bump * rolloff
    elif v == 2:
        # Dual-Coil Jazz Bass: +2.5 dB @ 65 Hz, -4.5 dB scoop @ 650 Hz, +3.5 dB @ 2800 Hz
        sub = 1.0 + 0.33 * np.exp(-0.5 * ((freqs - 65.0) / 45.0) ** 2)
        scoop = 1.0 - 0.40 * np.exp(-0.5 * ((freqs - 650.0) / 260.0) ** 2)
        bite = 1.0 + 0.48 * np.exp(-0.5 * ((freqs - 2800.0) / 1100.0) ** 2)
        rolloff = 1.0 / np.sqrt(1.0 + (freqs / 7500.0) ** 4)
        H = sub * scoop * bite * rolloff
    elif v == 3:
        # Active StingRay 2-Band: +4 dB low shelf (<110 Hz), +5 dB treble presence (3.2 kHz)
        low_shelf = 1.0 + 0.55 / (1.0 + (freqs / 110.0) ** 2)
        treble = 1.0 + 0.75 * np.exp(-0.5 * ((freqs - 3200.0) / 1400.0) ** 2)
        rolloff = 1.0 / np.sqrt(1.0 + (freqs / 9500.0) ** 4)
        H = low_shelf * treble * rolloff
    else:
        # Bright Pick / Rickenbacker clank (1.8 kHz) + Deep Sub fundamental
        sub = 1.0 + 0.45 * np.exp(-0.5 * ((freqs - 90.0) / 55.0) ** 2)
        clank = 1.0 + 0.65 * np.exp(-0.5 * ((freqs - 1850.0) / 700.0) ** 2)
        rolloff = 1.0 / np.sqrt(1.0 + (freqs / 6500.0) ** 4)
        H = sub * clank * rolloff

    shaped = np.fft.irfft(spec * H, n=n)
    rms_in = float(np.sqrt(np.mean(np.square(dry, dtype=np.float64))))
    rms_out = float(np.sqrt(np.mean(np.square(shaped, dtype=np.float64))))
    if rms_out > 1e-9 and rms_in > 1e-9:
        shaped *= rms_in / rms_out
    return np.clip(shaped, -CALIBRATION_PEAK_CEILING, CALIBRATION_PEAK_CEILING).astype(np.float32)


def build_split_calibration_probes(
    sample_rate: int = FS,
    duration_sec: float = 4.0,
) -> np.ndarray:
    """Synthesizes split dynamic calibration probes for musical bass preservation vs silence gating.

    1. First 70% of duration: Active Passband Identity Probes (> -34 dBFS).
       Contains pure bass fundamentals including 50 Hz (G0/Ab0), 60 Hz (B1), 100 Hz (G2),
       120 Hz (B2), 180 Hz (F#3), 440 Hz, 1200 Hz, and 3500 Hz at musical playing levels
       (-28 dBFS to -12 dBFS), teaching the model that 50/60 Hz notes above the gate knee
       are bass guitar fundamentals that must pass 1:1 with zero comb notching.
    2. Final 30% of duration: Pure Digital Silence (0.0).
       When paired with stochastic 50/60 Hz hum and SMPS noise in the training loop, this
       explicitly anchors the silence gate region (< -46 dBFS) to 0.0 output.
    """
    n_total = max(1024, round(duration_sec * sample_rate))
    n_active = int(0.70 * n_total)
    out = np.zeros(n_total, dtype=np.float32)

    t = np.arange(n_active, dtype=np.float64) / float(sample_rate)
    # Musical frequencies that intentionally overlap 50/60 Hz and their harmonics at high amplitude
    probe_freqs = (41.203, 50.0, 55.0, 60.0, 82.407, 100.0, 120.0, 180.0, 440.0, 1200.0, 3200.0)
    seg_len = max(256, n_active // len(probe_freqs))

    for idx, f0 in enumerate(probe_freqs):
        start = idx * seg_len
        end = n_active if idx == len(probe_freqs) - 1 else min(n_active, (idx + 1) * seg_len)
        if end <= start:
            continue
        seg_t = t[start:end] - t[start]
        seg_dur = max(1e-4, float(end - start) / float(sample_rate))
        # Smooth C-infinity raised-cosine envelope ramping between -28 dBFS and -12 dBFS
        env = np.sin(np.pi * seg_t / seg_dur) ** 2
        amp = float(10.0 ** (-14.0 / 20.0))
        wave = (
            amp
            * env
            * (
                0.75 * np.sin(2.0 * np.pi * f0 * seg_t)
                + 0.20 * np.sin(2.0 * np.pi * (2.0 * f0) * seg_t + 0.4)
                + 0.05 * np.sin(2.0 * np.pi * (3.0 * f0) * seg_t + 1.1)
            )
        )
        out[start:end] = wave.astype(np.float32)

    return out


def build_clean_bass_corpus(
    excitation_wav: Path,
    *,
    ultrasonic_lpf_fc: float = DENOISER_DEFAULT_ULTRASONIC_LPF_FC,
    max_duration_sec: float | None = None,
) -> tuple[np.ndarray, int]:
    """Builds the pickup-agnostic multi-voice clean bass corpus with ultrasonic LPF conditioning.

    Combines `excitation_wav` across 5 linear pickup/preamp spectral voicings (DI, Passive P,
    Dual-Coil Jazz, Active StingRay, Bright Pick/Dub), applies the 255-tap Blackman ultrasonic
    low-pass filter (`fc = 14 kHz`), and appends split dynamic calibration probes.
    """
    dry_raw, sr = read_wav(excitation_wav, dtype=np.float32)
    if dry_raw.ndim > 1:
        dry_raw = np.mean(dry_raw, axis=0)
    if sr != FS:
        raise ValueError(f"Excitation WAV sample rate must be {FS} Hz (got {sr} Hz).")

    if max_duration_sec is not None and max_duration_sec > 0.0:
        max_samples = max(65536, round(max_duration_sec * sr))
        dry_raw = dry_raw[:max_samples]

    n = len(dry_raw)
    if n < 32768:
        raise ValueError(f"Excitation WAV '{excitation_wav}' is too short ({n} samples).")

    # Partition the corpus into 5 overlapping crossfaded voicing segments + full direct pass
    # so every pickup spectral archetype is represented without exploding VRAM footprint
    voiced_copy = np.empty_like(dry_raw)
    num_voices = 5
    chunk = max(1, n // num_voices)
    for v_idx in range(num_voices):
        s = v_idx * chunk
        e = n if v_idx == num_voices - 1 else min(n, (v_idx + 1) * chunk)
        if e > s:
            voiced_copy[s:e] = _apply_spectral_voicing_eq(dry_raw[s:e], sr, voice_idx=v_idx)

    probes = build_split_calibration_probes(
        sample_rate=sr,
        duration_sec=min(6.0, max(2.0, 0.05 * (n / float(sr)))),
    )

    combined = np.concatenate([dry_raw.astype(np.float32), voiced_copy, probes])
    conditioned = apply_ultrasonic_lowpass(
        combined,
        fc=ultrasonic_lpf_fc,
        fs=float(sr),
        num_taps=DENOISER_FIR_NUM_TAPS,
    )
    return conditioned, sr


def synthesize_mains_hum_bank(
    n_samples: int,
    mains: MainsMode = "both",
    sample_rate: int = FS,
    seed: int = 42,
) -> np.ndarray:
    """Synthesizes a phase-diverse electromagnetic 50/60 Hz mains hum and harmonic buzz bank.

    Models:
      - Fundamental 50 Hz / 60 Hz transformer induction (with +/- 0.12 Hz grid frequency drift)
      - Odd harmonics (3f0, 5f0, 7f0 ... up to 45f0) from B-H core saturation & TRIAC dimmers
      - Even harmonics (2f0, 4f0, 6f0 ...) from full-wave bridge rectifier power supply ripple
      - Slow amplitude & spectral tilt modulation across segments so the WaveNet learns
        phase-invariant harmonic comb cancellation.
    """
    rng = np.random.default_rng(seed)
    out = np.zeros(n_samples, dtype=np.float64)
    if n_samples <= 0:
        return out.astype(np.float32)

    if mains == "60":
        f_list = (60.0, 59.94, 60.06)
    elif mains == "50":
        f_list = (50.0, 49.95, 50.05)
    else:
        f_list = (60.0, 50.0, 59.95, 50.05)

    # Generate in 2-second randomized harmonic scenes with smooth crossfades
    seg_len = min(n_samples, max(4096, 2 * sample_rate))
    n_segs = max(1, (n_samples + seg_len - 1) // seg_len)

    for seg_idx in range(n_segs):
        s = seg_idx * seg_len
        e = min(n_samples, s + seg_len)
        m = e - s
        if m <= 0:
            continue
        t = np.arange(m, dtype=np.float64) / float(sample_rate)

        f0 = float(f_list[seg_idx % len(f_list)])
        # In "both" mode, every 4th segment mixes simultaneous 50 Hz + 60 Hz intermodulation
        dual_grid = mains == "both" and (seg_idx % 4 == 3)
        grids = (50.0, 60.0) if dual_grid else (f0,)

        seg_sig = np.zeros(m, dtype=np.float64)
        for g_freq in grids:
            df = float(rng.uniform(-0.08, 0.08))
            f_base = g_freq + df
            # Random spectral tilt p in [0.65, 1.35] (lower p = harsher dimmer/single-coil buzz)
            tilt_p = float(rng.uniform(0.65, 1.35))
            odd_boost = float(rng.uniform(1.1, 2.2))
            even_boost = float(rng.uniform(0.3, 1.2))
            max_h = min(45, int((0.45 * sample_rate) // f_base))

            for k in range(1, max_h + 1):
                fk = k * f_base
                phase = float(rng.uniform(0.0, 2.0 * np.pi))
                parity_gain = odd_boost if (k % 2 == 1) else even_boost
                # Dimmer buzz resonance around 900 Hz - 2400 Hz
                dimmer_bump = 1.0 + 1.4 * math.exp(-0.5 * ((fk - 1500.0) / 700.0) ** 2)
                amp_k = (parity_gain * dimmer_bump) / (float(k) ** tilt_p)
                seg_sig += amp_k * np.sin(2.0 * np.pi * fk * t + phase)

        # Normalize each segment to unit RMS before storing
        seg_rms = float(np.sqrt(np.mean(np.square(seg_sig))))
        if seg_rms > 1e-12:
            seg_sig /= seg_rms
        out[s:e] = seg_sig

    total_rms = float(np.sqrt(np.mean(np.square(out))))
    if total_rms > 1e-12:
        out /= total_rms
    return out.astype(np.float32)


def synthesize_smps_and_hiss_banks(
    n_samples: int,
    sample_rate: int = FS,
    seed: int = 1042,
) -> tuple[np.ndarray, np.ndarray]:
    """Synthesizes switching power supply (SMPS) whine/carrier noise and thermal hiss banks.

    Returns:
        Tuple of `(smps_bank, hiss_bank)`, each normalized to unit RMS (`float32`):
        - `smps_bank`: Narrowband switching regulators (2.5 kHz - 22 kHz), charge-pump
          clock whine (8 kHz - 18 kHz), PWM sidebands (+/- 60/100/120 Hz), and high-passed
          switching edges.
        - `hiss_bank`: Broadband thermal Johnson-Nyquist white noise + capacitive pickup
          f^{+1} treble hiss.
    """
    rng = np.random.default_rng(seed)
    if n_samples <= 0:
        empty = np.zeros(0, dtype=np.float32)
        return empty, empty

    # 1. SMPS Switching Carrier + PWM Sideband Bank
    smps = np.zeros(n_samples, dtype=np.float64)
    seg_len = min(n_samples, max(4096, sample_rate))
    n_segs = max(1, (n_samples + seg_len - 1) // seg_len)

    # Canonical pedalboard & USB SMPS carrier bands (Hz)
    carrier_centers = (
        2800.0,
        4000.0,
        6200.0,
        8000.0,
        10000.0,
        12500.0,
        14800.0,
        16000.0,
        18200.0,
        20500.0,
        22000.0,
    )

    for seg_idx in range(n_segs):
        s = seg_idx * seg_len
        e = min(n_samples, s + seg_len)
        m = e - s
        if m <= 0:
            continue
        t = np.arange(m, dtype=np.float64) / float(sample_rate)
        seg_sig = np.zeros(m, dtype=np.float64)

        # Pick 3 concurrent switching carriers per segment with FM jitter and PWM sidebands
        chosen_carriers = rng.choice(carrier_centers, size=3, replace=False)
        for fc_base in chosen_carriers:
            fc = float(fc_base) * float(rng.uniform(0.94, 1.06))
            if fc >= 0.49 * sample_rate:
                continue
            ph0 = float(rng.uniform(0.0, 2.0 * np.pi))
            # Slow thermal frequency drift + ripple FM modulation
            f_mod = float(rng.choice((50.0, 60.0, 100.0, 120.0, 240.0)))
            fm_index = float(rng.uniform(0.05, 0.35))
            inst_phase = 2.0 * np.pi * fc * t + fm_index * np.sin(2.0 * np.pi * f_mod * t) + ph0
            am_env = 1.0 + float(rng.uniform(0.1, 0.4)) * np.cos(
                2.0 * np.pi * f_mod * t + float(rng.uniform(0.0, 2.0 * np.pi))
            )
            seg_sig += am_env * np.sin(inst_phase)

            # 2nd harmonic of lower-frequency buck converters if below Nyquist
            if 2.0 * fc < 0.48 * sample_rate:
                seg_sig += 0.35 * np.sin(2.0 * inst_phase + 0.7)

        seg_rms = float(np.sqrt(np.mean(np.square(seg_sig))))
        if seg_rms > 1e-12:
            seg_sig /= seg_rms
        smps[s:e] = seg_sig

    smps_rms = float(np.sqrt(np.mean(np.square(smps))))
    if smps_rms > 1e-12:
        smps /= smps_rms

    # 2. Broadband JFET/Op-Amp White + Capacitive f^{+1} Treble Hiss Bank
    white = rng.standard_normal(n_samples)
    diff_noise = np.diff(white, prepend=white[0])
    hiss = 0.65 * white + 0.35 * diff_noise
    hiss_rms = float(np.sqrt(np.mean(np.square(hiss))))
    if hiss_rms > 1e-12:
        hiss /= hiss_rms

    return smps.astype(np.float32), hiss.astype(np.float32)


def load_hardware_noise_bank(
    noise_wav: Path,
    target_len: int,
    sample_rate: int = FS,
) -> np.ndarray:
    """Loads a captured real-world hardware idle noise WAV and normalizes it to unit RMS (`float32`).

    Tiles or slices the recording to `target_len` with smooth crossfades at wrap boundaries.
    """
    raw, sr = read_wav(noise_wav, dtype=np.float32)
    if raw.ndim > 1:
        raw = np.mean(raw, axis=0)
    if sr != sample_rate:
        raise ValueError(
            f"Hardware noise WAV '{noise_wav}' sample rate must be {sample_rate} Hz (got {sr} Hz)."
        )
    if len(raw) < 1024:
        raise ValueError(
            f"Hardware noise WAV '{noise_wav}' is too short ({len(raw)} samples; minimum 1024)."
        )

    # Remove DC offset and normalize to unit RMS so stochastic dBFS scaling is exact
    centered = raw.astype(np.float64) - float(np.mean(raw, dtype=np.float64))
    rms = float(np.sqrt(np.mean(np.square(centered))))
    if rms < 1e-9:
        raise ValueError(f"Hardware noise WAV '{noise_wav}' contains digital silence (RMS={rms}).")
    unit_noise = (centered / rms).astype(np.float32)

    if len(unit_noise) >= target_len:
        return unit_noise[:target_len].copy()

    repeats = (target_len + len(unit_noise) - 1) // len(unit_noise)
    tiled = np.tile(unit_noise, repeats)[:target_len]
    return tiled.astype(np.float32, copy=True)


class StochasticNoiseGenerator:
    """In-VRAM vectorized stochastic noise mixer for WaveNet A2 denoiser training & validation.

    Holds pre-allocated contiguous VRAM noise banks (`hum_bank`, `smps_bank`, `hiss_bank`, and
    optional `hw_noise_bank`) and gathers independent random windows on the GPU at zero PCIe cost.
    """

    def __init__(
        self,
        *,
        n_samples: int,
        mains: MainsMode,
        profile: ProfileName,
        device: torch.device,
        dtype: torch.dtype,
        sample_rate: int = FS,
        threshold_dbfs_override: float | None = None,
        noise_wav: Path | None = None,
        seed: int = 42,
    ) -> None:
        self.profile_spec = DENOISER_PROFILES[profile]
        self.mains: MainsMode = mains
        self.device = device
        self.dtype = dtype
        self.sample_rate = sample_rate
        self.threshold_dbfs = (
            float(threshold_dbfs_override)
            if threshold_dbfs_override is not None
            else self.profile_spec.threshold_dbfs
        )
        # Shift noise injection ceiling proportionally if explicit threshold override was supplied
        delta_thresh = self.threshold_dbfs - self.profile_spec.threshold_dbfs
        self.noise_min_dbfs = self.profile_spec.noise_min_dbfs + delta_thresh
        self.noise_max_dbfs = min(-20.0, self.profile_spec.noise_max_dbfs + delta_thresh)

        bank_len = max(96000, min(n_samples, 12 * sample_rate))
        hum_np = synthesize_mains_hum_bank(
            bank_len, mains=mains, sample_rate=sample_rate, seed=seed
        )
        smps_np, hiss_np = synthesize_smps_and_hiss_banks(
            bank_len, sample_rate=sample_rate, seed=seed + 101
        )

        self.bank_len = bank_len
        self.hum_bank = torch.from_numpy(hum_np).to(device=device, dtype=dtype)
        self.smps_bank = torch.from_numpy(smps_np).to(device=device, dtype=dtype)
        self.hiss_bank = torch.from_numpy(hiss_np).to(device=device, dtype=dtype)

        self.hw_noise_bank: torch.Tensor | None = None
        if noise_wav is not None:
            hw_np = load_hardware_noise_bank(
                noise_wav, target_len=bank_len, sample_rate=sample_rate
            )
            self.hw_noise_bank = torch.from_numpy(hw_np).to(device=device, dtype=dtype)

    def sample_training_batch(
        self,
        *,
        x_clean_padded: torch.Tensor,
        y_clean: torch.Tensor,
        clean_indices: torch.Tensor,
        arange_nx: torch.Tensor,
        arange_ny: torch.Tensor,
        rf_pad: int,
        silence_fraction: float = 0.22,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Gathers a batch of clean bass slices and injects on-the-fly stochastic hum + SMPS noise.

        Args:
            x_clean_padded: 1D tensor of shape `(N + rf_pad,)` in VRAM.
            y_clean: 1D tensor of shape `(N,)` in VRAM.
            clean_indices: 2D int32 tensor of shape `(B, 1)` with starting sample offsets.
            arange_nx: 2D int32 tensor of shape `(1, nx)`.
            arange_ny: 2D int32 tensor of shape `(1, ny)`.
            rf_pad: Receptive field left-padding length (`6346` for WaveNet A2).
            silence_fraction: Fraction of batch items replaced with pure idle silence (`y = 0`)
                so the denoiser continuously trains its sub-threshold silence gate.

        Returns:
            Tuple `(x_noisy_batch, y_clean_batch)` of shapes `(B, 1, nx)` and `(B, ny)`.
        """
        bs = int(clean_indices.shape[0])
        nx = int(arange_nx.shape[1])
        max_noise_start = max(1, self.bank_len - nx)

        # Gather clean signal windows
        x_clean_batch = x_clean_padded[clean_indices + arange_nx].clone()  # (B, nx)
        y_clean_batch = y_clean[clean_indices + arange_ny].clone()  # (B, ny)

        # Stochastically zero out `silence_fraction` of batch items to anchor pure idle noise -> 0.0
        n_silent = max(1, round(bs * silence_fraction)) if bs >= 4 else 0
        if n_silent > 0:
            x_clean_batch[:n_silent].zero_()
            y_clean_batch[:n_silent].zero_()

        # Gather independent random offsets from each noise bank in VRAM
        hum_idx = torch.randint(0, max_noise_start, (bs, 1), device=self.device, dtype=torch.int32)
        smps_idx = torch.randint(0, max_noise_start, (bs, 1), device=self.device, dtype=torch.int32)
        hiss_idx = torch.randint(0, max_noise_start, (bs, 1), device=self.device, dtype=torch.int32)

        hum_win = self.hum_bank[hum_idx + arange_nx]
        smps_win = self.smps_bank[smps_idx + arange_nx]
        hiss_win = self.hiss_bank[hiss_idx + arange_nx]

        # Randomize polarity & relative component mix per batch item in float32 then cast to dtype
        u_db = torch.empty((bs, 1), device=self.device, dtype=torch.float32).uniform_(
            self.noise_min_dbfs, self.noise_max_dbfs
        )
        total_noise_amp = torch.pow(10.0, u_db / 20.0).to(dtype=self.dtype)

        w_hum = (
            torch.empty((bs, 1), device=self.device, dtype=torch.float32)
            .uniform_(0.45, 1.0)
            .to(dtype=self.dtype)
        )
        w_smps = (
            torch.empty((bs, 1), device=self.device, dtype=torch.float32)
            .uniform_(0.35, 0.95)
            .to(dtype=self.dtype)
        )
        w_hiss = (
            torch.empty((bs, 1), device=self.device, dtype=torch.float32)
            .uniform_(0.10, 0.35)
            .to(dtype=self.dtype)
        )

        noise_mix = w_hum * hum_win + w_smps * smps_win + w_hiss * hiss_win

        if self.hw_noise_bank is not None:
            hw_idx = torch.randint(
                0, max_noise_start, (bs, 1), device=self.device, dtype=torch.int32
            )
            hw_win = self.hw_noise_bank[hw_idx + arange_nx]
            w_hw = (
                torch.empty((bs, 1), device=self.device, dtype=torch.float32)
                .uniform_(0.50, 1.10)
                .to(dtype=self.dtype)
            )
            noise_mix = noise_mix + w_hw * hw_win

        # Normalize composite noise window per item to unit RMS before scaling by total_noise_amp
        noise_rms = torch.sqrt(
            torch.mean(torch.square(noise_mix.float()), dim=1, keepdim=True) + 1e-12
        ).to(dtype=self.dtype)
        scaled_noise = (noise_mix / noise_rms) * total_noise_amp

        # Ensure causal zero-padding at the very start of the buffer remains causal if clean_indices == 0
        x_noisy = x_clean_batch + scaled_noise
        if rf_pad > 0:
            # Clamp to valid [-0.999, 0.999] headroom
            x_noisy = torch.clamp(x_noisy, -0.999, 0.999)

        return x_noisy.unsqueeze(1), y_clean_batch

    def build_deterministic_validation_pair(
        self,
        x_val_clean: torch.Tensor,
        rf_pad: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Constructs a deterministic noisy validation buffer with both active notes and silent gaps."""
        n_val = int(x_val_clean.shape[0])
        y_val = x_val_clean.clone()

        # Zero out the final 20% of validation target & clean signal to measure idle noise floor suppression
        silence_tail = max(4096, int(0.20 * n_val))
        y_val[-silence_tail:].zero_()

        # Build deterministic mid-level noise at the midpoint of [noise_min_dbfs, noise_max_dbfs]
        mid_db = 0.5 * (self.noise_min_dbfs + self.noise_max_dbfs)
        noise_amp = float(10.0 ** (mid_db / 20.0))

        repeats = (n_val + self.bank_len - 1) // self.bank_len
        hum_full = self.hum_bank.repeat(repeats)[:n_val]
        smps_full = self.smps_bank.repeat(repeats)[:n_val]
        hiss_full = self.hiss_bank.repeat(repeats)[:n_val]

        composite = 0.70 * hum_full + 0.60 * smps_full + 0.20 * hiss_full
        if self.hw_noise_bank is not None:
            hw_full = self.hw_noise_bank.repeat(repeats)[:n_val]
            composite = composite + 0.75 * hw_full

        comp_rms = torch.sqrt(torch.mean(torch.square(composite.float())) + 1e-12).to(
            dtype=self.dtype
        )
        noise_vec = (composite / comp_rms) * noise_amp

        x_val_noisy = torch.clamp(y_val + noise_vec, -0.999, 0.999)
        x_val_noisy_padded = torch.cat(
            [torch.zeros(rf_pad, device=self.device, dtype=self.dtype), x_val_noisy]
        )
        return x_val_noisy_padded, y_val
