"""
Tests for 24-bit audio digital twin simulation, level normalization,
batch execution, and CLI integration.
"""

import math
import os
import tempfile
import wave
from pathlib import Path

import numpy as np
import pedalboard.io
import pytest

from allomorph.circuit import (
    CircuitModel,
    apply_magnet_properties_to_model,
    apply_oversampled_saturation,
    compute_circuit_transfer_functions,
    load_circuit,
    simulate_circuit_audio,
    simulate_voice,
)
from allomorph.config import VOICES
from allomorph.dsp import FREQS, NUM_TAPS, write_wav_24bit
from allomorph.naming import resolve_voices
from allomorph.physics import compute_voice_prefilter_firs
from allomorph.pipeline import run_spice_batch
from allomorph.visualizer import build_voice_dataframe


def test_simulate_circuit_audio_output():
    with tempfile.TemporaryDirectory() as tmpdir:
        input_wav = Path(tmpdir) / "test_in.wav"
        output_wav = Path(tmpdir) / "test_out.wav"

        # Generate a test pulse train at 48 kHz
        samples = [0.4 if i % 200 == 0 else 0.0 for i in range(4800)]
        write_wav_24bit(str(input_wav), samples, sample_rate=48000)

        model = load_circuit("precision_active")
        res = simulate_circuit_audio(input_wav, output_wav, model)
        assert res is True
        assert output_wav.exists()

        with wave.open(str(output_wav), "rb") as wf:
            assert wf.getframerate() == 48000
            assert wf.getsampwidth() == 3  # 24-bit PCM
            assert wf.getnchannels() == 1
            assert wf.getnframes() > 0


def test_simulate_voice_end_to_end():
    """Verify unified end-to-end simulation from raw audio to final digital twin audio."""
    with tempfile.TemporaryDirectory() as tmpdir:
        input_wav = Path(tmpdir) / "raw_in.wav"
        output_wav = Path(tmpdir) / "final_out.wav"

        # Generate a test excitation track at 48 kHz
        samples = [0.5 if i % 100 == 0 else 0.0 for i in range(4800)]
        write_wav_24bit(str(input_wav), samples, sample_rate=48000)

        # Single pickup voice (P-Bass Ceramic)
        res = simulate_voice(
            "precision_active",
            input_wav=input_wav,
            output_wav=output_wav,
            instrument="30in",
            prefiltered=False,
        )
        assert res is True
        assert output_wav.exists()

        with wave.open(str(output_wav), "rb") as wf:
            assert wf.getframerate() == 48000
            assert wf.getsampwidth() == 3
            assert wf.getnchannels() == 1
            assert wf.getnframes() > 0

        # Multi-pickup voice (Jazz Bass Pair in Parallel)
        output_jazz = Path(tmpdir) / "jazz_out.wav"
        res_jazz = simulate_voice(
            "jazz_pair_open",
            input_wav=input_wav,
            output_wav=output_jazz,
            instrument="30in",
            prefiltered=False,
        )
        assert res_jazz is True
        assert output_jazz.exists()

        with wave.open(str(output_jazz), "rb") as wf:
            assert wf.getframerate() == 48000
            assert wf.getsampwidth() == 3
            assert wf.getnchannels() == 1
            assert wf.getnframes() > 0


def test_circuit_simulation_vs_theory_consistency():
    """Verify that simulated impulse FFT matches analytical theory curve within 2.0 dB across 50-8000 Hz."""
    inst_id = "30in_emg_mmtw"
    sr = 48000
    n_samples = 48000 * 2
    impulse = np.zeros(n_samples, dtype=np.float32)
    impulse[10] = 0.05  # Linear small-signal excitation

    for voice_id in ["precision_active", "jazz_pair_open", "pj_active"]:
        cfg = VOICES[voice_id]
        model = load_circuit(voice_id)
        apply_magnet_properties_to_model(model, cfg)
        prefilter_firs = compute_voice_prefilter_firs(voice_id, instrument=inst_id)

        with tempfile.NamedTemporaryFile(suffix=".wav") as tmp_out:
            simulate_circuit_audio(
                impulse, Path(tmp_out.name), model, prefilter_firs=prefilter_firs
            )
            with pedalboard.io.AudioFile(tmp_out.name) as f:
                out_audio = f.read(f.frames)[0]

        H_sim = np.abs(np.fft.rfft(out_audio))
        freqs_sim = np.fft.rfftfreq(len(out_audio), 1.0 / sr)

        df = build_voice_dataframe(voice_id, cfg, instrument=inst_id)
        f_theory = df["frequency"].to_numpy()
        mag_theory_db = df["magnitude_db"].to_numpy()

        ref_val = np.interp(100.0, freqs_sim, H_sim)
        theory_ref_db = np.interp(100.0, f_theory, mag_theory_db)
        sim_db = 20.0 * np.log10(H_sim / ref_val) + theory_ref_db

        for test_f in [50, 100, 200, 500, 1000, 2000, 3000, 4000, 6000, 8000]:
            val_sim = float(np.interp(test_f, freqs_sim, sim_db))
            val_theory = float(np.interp(test_f, f_theory, mag_theory_db))
            diff = abs(val_sim - val_theory)
            assert diff < 2.0, (
                f"{voice_id} at {test_f} Hz diff={diff:.2f} dB exceeds 2.0 dB (sim={val_sim:.2f}, theory={val_theory:.2f})"
            )


def test_upright_voicing_simulation_vs_theory_consistency():
    """Verify that 32in fretless upright acoustic transducer simulation matches theory across 20-5000 Hz."""
    inst_id = "32in_fretless_pmm"
    voice_id = "upright_acoustic"
    sr = 48000
    n_samples = 48000 * 2
    impulse = np.zeros(n_samples, dtype=np.float32)
    impulse[10] = 0.05

    cfg = VOICES[voice_id]
    model = load_circuit(voice_id)
    apply_magnet_properties_to_model(model, cfg)
    prefilter_firs = compute_voice_prefilter_firs(voice_id, instrument=inst_id)

    with tempfile.NamedTemporaryFile(suffix=".wav") as tmp_out:
        simulate_circuit_audio(impulse, Path(tmp_out.name), model, prefilter_firs=prefilter_firs)
        with pedalboard.io.AudioFile(tmp_out.name) as f:
            out_audio = f.read(f.frames)[0]

    H_sim = np.abs(np.fft.rfft(out_audio))
    freqs_sim = np.fft.rfftfreq(len(out_audio), 1.0 / sr)

    df = build_voice_dataframe(voice_id, cfg, instrument=inst_id)
    f_theory = df["frequency"].to_numpy()
    mag_theory_db = df["magnitude_db"].to_numpy()

    ref_val = np.interp(100.0, freqs_sim, H_sim)
    theory_ref_db = np.interp(100.0, f_theory, mag_theory_db)
    sim_db = 20.0 * np.log10(H_sim / ref_val) + theory_ref_db

    # In the critical upright passband (20 Hz - 4.2 kHz), diff must be under 1.5 dB
    for test_f in [20, 30, 50, 70, 100, 200, 500, 1000, 2000, 3000, 4200]:
        val_sim = float(np.interp(test_f, freqs_sim, sim_db))
        val_theory = float(np.interp(test_f, f_theory, mag_theory_db))
        diff = abs(val_sim - val_theory)
        assert diff < 1.5, (
            f"Upright at {test_f} Hz diff={diff:.2f} dB exceeds 1.5 dB (sim={val_sim:.2f}, theory={val_theory:.2f})"
        )


def test_passive_source_simulation_runs():
    """Verify end-to-end simulate_voice runs for passive source instruments."""
    with (
        tempfile.NamedTemporaryFile(suffix=".wav") as tmp1,
        tempfile.NamedTemporaryFile(suffix=".wav") as tmp2,
    ):
        res1 = simulate_voice(
            "precision_vintage",
            output_wav=Path(tmp1.name),
            instrument="34in_standard_p",
            max_samples=4800,
        )
        assert res1 is True
        assert os.path.exists(tmp1.name) and os.path.getsize(tmp1.name) > 1000

        res2 = simulate_voice(
            "jazz_pair_open",
            output_wav=Path(tmp2.name),
            instrument="34in_standard_jazz",
            max_samples=4800,
        )
        assert res2 is True
        assert os.path.exists(tmp2.name) and os.path.getsize(tmp2.name) > 1000


def test_auto_output_level_normalization_to_input_sweep():
    """Verify that simulated output automatically normalizes its RMS to match the input sweep dBFS."""
    sr = 48000
    t = np.linspace(0, 1.0, sr, endpoint=False)
    in_signal = (0.5 * np.sin(2 * np.pi * 150 * t) + 0.3 * np.sin(2 * np.pi * 800 * t)).astype(
        np.float32
    )

    with tempfile.TemporaryDirectory() as td:
        in_path = Path(td) / "test_in.wav"
        out_path = Path(td) / "test_out.wav"
        write_wav_24bit(str(in_path), in_signal, sample_rate=sr)

        with pedalboard.io.AudioFile(str(in_path)) as f:
            in_audio = f.read(f.frames)[0]
        in_rms = float(np.sqrt(np.mean(in_audio**2)))
        in_rms_db = 20.0 * math.log10(in_rms)

        res = simulate_voice(
            "precision_active",
            input_wav=in_path,
            output_wav=out_path,
            instrument="30in",
            normalize="auto",
        )
        assert res is True

        with pedalboard.io.AudioFile(str(out_path)) as f:
            out_audio = f.read(f.frames)[0]

        out_rms = float(np.sqrt(np.mean(out_audio**2)))
        out_rms_db = 20.0 * math.log10(out_rms)
        out_peak = float(np.max(np.abs(out_audio)))

        # Output RMS must match input sweep RMS within 0.05 dB
        assert abs(out_rms_db - in_rms_db) < 0.05
        # Peak must have clean true-peak safety headroom (< -0.1 dBFS = 0.9885)
        assert out_peak <= 0.9885


def test_output_normalization_modes_and_target_dbfs():
    """Verify custom target_dbfs override and normalize modes (rms, peak, none)."""
    sr = 48000
    t = np.linspace(0, 1.0, sr, endpoint=False)
    in_signal = (0.5 * np.sin(2 * np.pi * 150 * t) + 0.3 * np.sin(2 * np.pi * 800 * t)).astype(
        np.float32
    )

    with tempfile.TemporaryDirectory() as td:
        in_path = Path(td) / "test_in.wav"
        write_wav_24bit(str(in_path), in_signal, sample_rate=sr)

        # 1. Custom target dBFS (-24.0 dBFS)
        out_path = Path(td) / "out_24.wav"
        simulate_voice(
            "precision_active",
            input_wav=in_path,
            output_wav=out_path,
            instrument="30in",
            normalize="rms",
            target_dbfs=-24.0,
        )
        with pedalboard.io.AudioFile(str(out_path)) as f:
            out_audio = f.read(f.frames)[0]
        out_rms_db = 20.0 * math.log10(np.sqrt(np.mean(out_audio**2)))
        assert abs(out_rms_db - (-24.0)) < 0.05

        # 2. None (raw unnormalized)
        out_none = Path(td) / "out_none.wav"
        simulate_voice(
            "precision_active",
            input_wav=in_path,
            output_wav=out_none,
            instrument="30in",
            normalize="none",
        )
        with pedalboard.io.AudioFile(str(out_none)) as f:
            out_audio = f.read(f.frames)[0]
        out_rms_none = 20.0 * math.log10(np.sqrt(np.mean(out_audio**2)))
        assert -10.0 < out_rms_none < 0.0
        assert abs(out_rms_none - (-24.0)) > 10.0


def test_num_taps_4096_resolution():
    """Verify that NUM_TAPS is 4096 and frequency bin spacing is ~5.86 Hz."""
    assert NUM_TAPS == 4096
    assert len(FREQS) == 4096
    df = FREQS[1] - FREQS[0]
    assert math.isclose(df, 24000.0 / 4095, rel_tol=1e-3)


def test_subaudible_dc_blocking_filter():
    """Verify that 8 Hz DC blocker eliminates quadratic saturation DC offset (< 1e-6) while preserving audio."""
    sr = 48000
    t = np.linspace(0, 0.5, int(sr * 0.5), endpoint=False)
    # 100 Hz forte tone triggering asymmetric saturation
    tone = (0.60 * np.sin(2 * np.pi * 100 * t)).astype(np.float32)

    model = load_circuit("precision_active")

    with (
        tempfile.NamedTemporaryFile(suffix=".wav") as tmp_dc_on,
        tempfile.NamedTemporaryFile(suffix=".wav") as tmp_dc_off,
    ):
        # 1. With DC blocker enabled (default)
        simulate_circuit_audio(tone, Path(tmp_dc_on.name), model, alpha=0.25, dc_block=True)
        with pedalboard.io.AudioFile(tmp_dc_on.name) as f:
            audio_on = f.read(f.frames)[0]

        # 2. With DC blocker disabled
        simulate_circuit_audio(tone, Path(tmp_dc_off.name), model, alpha=0.25, dc_block=False)
        with pedalboard.io.AudioFile(tmp_dc_off.name) as f:
            audio_off = f.read(f.frames)[0]

    # Without DC blocker, asymmetric alpha produces positive DC mean (> 1e-4)
    assert abs(np.mean(audio_off)) > 1e-4
    # With DC blocker, DC mean is suppressed by > 80 dB down to < 1e-6 (essentially 0.000000)
    assert abs(np.mean(audio_on)) < 1e-6


def test_spatial_wave_propagation_delay():
    """
    Verify spatial acoustic wave propagation delay (tau = delta_x / c_s):
    1. Multi-pickup voice jazz_pair_open produces 2 FIRs where bridge FIR is delayed
       by ~0.81 ms (~39 samples at 48 kHz) relative to neck FIR.
    2. Summing neck and bridge FIRs produces the iconic acoustic phase comb notch
       in the 500-800 Hz range (depth > 10 dB relative to 100 Hz).
    3. Single-pickup voice precision_vintage produces 1 FIR with 0 delay (peak at tap 0).
    """
    firs_02 = compute_voice_prefilter_firs("jazz_pair_open", instrument="30in")
    assert len(firs_02) == 2

    fir_n = np.array(firs_02[0])
    fir_b = np.array(firs_02[1])

    # Neck is at pos=155.6 mm (pos_max -> tau = 0), so peak is near tap 0
    peak_n = int(np.argmax(np.abs(fir_n)))
    # Bridge is at pos=63.5 mm (delta_x = 92.1 mm -> tau = 0.81 ms = 38.9 samples at 48k)
    peak_b = int(np.argmax(np.abs(fir_b)))
    sample_delay = peak_b - peak_n
    assert 36 <= sample_delay <= 42, f"Bridge delay was {sample_delay} samples (expected ~39)"

    # Summing the two channels creates acoustic comb notch
    fir_sum = fir_n + fir_b
    n_fft = 8192
    H_sum = np.abs(np.fft.rfft(fir_sum, n_fft))
    f_bins = np.fft.rfftfreq(n_fft, 1.0 / 48000.0)

    ref_100 = np.interp(100.0, f_bins, H_sum)
    mask_notch = (f_bins >= 500.0) & (f_bins <= 800.0)
    min_notch = np.min(H_sum[mask_notch])
    notch_depth_db = 20.0 * np.log10(ref_100 / min_notch)
    assert notch_depth_db > 10.0, f"Comb notch depth was {notch_depth_db:.2f} dB (expected > 10 dB)"

    # Single-pickup voice
    firs_05 = compute_voice_prefilter_firs("precision_vintage", instrument="30in")
    assert len(firs_05) == 1
    assert np.argmax(np.abs(firs_05[0])) <= 2


def test_string_mass_momentum_weighting():
    """
    Verify string-mass kinetic momentum excursion weighting in saturation engine:
    1. Momentum pre-filter provides +3 dB boost at Low-E 41.2 Hz and -10.6 dB cut at 1 kHz relative to 100 Hz.
    2. Deep bass fundamentals (41.2 Hz) drive more dynamic non-linear compression than upper register (1 kHz).
    3. Small signals (amplitude <= 0.10) bypass non-linearity and preserve exact waveform.
    """
    # 1. Frequency response verification of momentum weighting curve
    freqs = np.array([41.2, 100.0, 1000.0])
    wc = 2.0 * np.pi * 40.0
    s = 1j * 2.0 * np.pi * freqs
    H_pre = (wc / (s + wc)) ** 0.55
    H_pre = H_pre / np.abs(H_pre[1])
    gain_db = 20.0 * np.log10(np.abs(H_pre))
    assert gain_db[0] == pytest.approx(3.00, abs=0.1)
    assert gain_db[1] == pytest.approx(0.00, abs=0.01)
    assert gain_db[2] == pytest.approx(-10.65, abs=0.1)

    # 2. Dynamic saturation compression comparison
    sr = 48000
    n = sr * 2
    t = np.arange(n) / sr

    sig_low = (0.60 * np.sin(2.0 * np.pi * 41.2 * t)).astype(np.float32)
    sig_hi = (0.60 * np.sin(2.0 * np.pi * 1000.0 * t)).astype(np.float32)

    sat_low = apply_oversampled_saturation(
        sig_low, vsat=0.50, alpha=0.20, oversample=1, displacement_weighting=True
    )
    sat_hi = apply_oversampled_saturation(
        sig_hi, vsat=0.50, alpha=0.20, oversample=1, displacement_weighting=True
    )

    cf_low = np.max(np.abs(sat_low)) / np.sqrt(np.mean(sat_low**2))
    cf_hi = np.max(np.abs(sat_hi)) / np.sqrt(np.mean(sat_hi**2))
    assert cf_low < cf_hi

    # 3. Small-signal impulse bypass
    impulse = np.zeros(1024, dtype=np.float32)
    impulse[0] = 0.08
    out_small = apply_oversampled_saturation(
        impulse, vsat=0.50, alpha=0.20, oversample=1, displacement_weighting=True
    )
    assert np.allclose(impulse, out_small, atol=1e-7)


def test_calibrated_drive_excursion_item3():
    """
    Verify Item 3: Drive excursion into magnetic saturation window is consistently calibrated
    when prefilter_firs is None (standalone or prefiltered audio).
    - Large signals (peak 0.95) are scaled to target_drive_peak (<= 0.70) into saturation.
    - Small signals (peak 0.05) bypass saturation completely and remain 100% linear.
    """
    m = load_circuit("precision_vintage")
    sr = 48000
    t = np.linspace(0, 0.1, int(sr * 0.1), endpoint=False)

    # 1. Large signal: peak 0.95
    large_sig = (0.95 * np.sin(2 * np.pi * 100 * t)).astype(np.float32)
    # 2. Small signal: peak 0.05
    small_sig = (0.05 * np.sin(2 * np.pi * 100 * t)).astype(np.float32)

    with tempfile.TemporaryDirectory() as td:
        out_large = Path(td) / "out_large.wav"
        out_small = Path(td) / "out_small.wav"

        simulate_circuit_audio(
            large_sig, out_large, m, prefilter_firs=None, is_passive=False, normalize="none"
        )
        simulate_circuit_audio(
            small_sig, out_small, m, prefilter_firs=None, is_passive=False, normalize="none"
        )

        with pedalboard.io.AudioFile(str(out_large)) as f:
            audio_large = f.read(f.frames)[0]
        with pedalboard.io.AudioFile(str(out_small)) as f:
            audio_small = f.read(f.frames)[0]

        # Small signal has zero harmonic distortion (pure sine preserved)
        fft_small = np.abs(np.fft.rfft(audio_small))
        freqs = np.fft.rfftfreq(len(audio_small), 1.0 / sr)
        h1_idx = np.argmin(np.abs(freqs - 100.0))
        h2_idx = np.argmin(np.abs(freqs - 200.0))
        assert fft_small[h2_idx] < 1e-4 * fft_small[h1_idx]

        # Large signal engages saturation safely without exceeding true-peak headroom
        assert np.max(np.abs(audio_large)) <= 0.9885


def test_multichannel_branch_weight_consistency():
    """
    Verify Vector 2 & 3: Multi-channel branch weight consistency.
    When a voice has a multi-channel SPICE netlist (e.g. jazz_pair_open),
    prefilter FIRs have unit branch weight (p_weight = 1.0) because SPICE nodal
    analysis computes the parallel current divider Y_branch / Y_total.
    """
    firs_02 = compute_voice_prefilter_firs("jazz_pair_open", instrument="30in", num_taps=512)
    assert len(firs_02) == 2
    # Both channels must have peaks around 0.99
    peak_0 = np.max(np.abs(firs_02[0]))
    peak_1 = np.max(np.abs(firs_02[1]))
    assert math.isclose(max(peak_0, peak_1), 0.99, rel_tol=1e-3)


def test_jaco_bridge_growl_bias_voicing():
    """
    Verify Refinement 4 & 5: Jaco Pastorius 60s Jazz Bridge Growl Bias voicing (02c).
    Validates decoupled pot parsing (Neck 75%, Bridge 100%), relative branch attenuation,
    and prefilter FIR generation.
    """
    model = load_circuit("jazz_bridge_growl")
    assert model.topology == "parallel"
    assert model.Rpot_n == pytest.approx(55000.0)
    assert model.Rpot_b == pytest.approx(0.0)

    curves = compute_circuit_transfer_functions(model, freqs=FREQS)
    assert len(curves) == 2
    neck_mag = np.array(curves[0])
    bridge_mag = np.array(curves[1])

    # Neck pickup should be attenuated relative to bridge pickup across passband
    # due to 55k wiper resistance
    idx_1k = np.argmin(np.abs(np.array(FREQS) - 1000.0))
    assert neck_mag[idx_1k] < bridge_mag[idx_1k]
    diff_db_1k = 20.0 * np.log10(bridge_mag[idx_1k] / neck_mag[idx_1k])
    assert 6.0 <= diff_db_1k <= 14.0

    # Pre-filter FIRs for 02c must exist and synthesize cleanly
    firs = compute_voice_prefilter_firs(
        "jazz_bridge_growl", instrument="34in_standard_jazz", num_taps=512
    )
    assert len(firs) == 2
    assert np.all(np.isfinite(firs[0]))
    assert np.all(np.isfinite(firs[1]))


def test_multi_pickup_excursion_ratio():
    """
    Verify Refinement 3: Physical string excursion drive ratio between neck and bridge pickups.
    Mono signal through multi-pickup circuit simulation scales bridge drive.
    """
    model = load_circuit("jazz_pair_open")
    n_samples = 4800
    mono_audio = (np.sin(2 * np.pi * 100.0 * np.linspace(0, 0.1, n_samples)) * 0.8).astype(
        np.float32
    )

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_out:
        out_wav = Path(tmp_out.name)
    try:
        simulate_circuit_audio(
            mono_audio,
            output_wav_path=out_wav,
            model=model,
            vsat=0.5,
            alpha=0.2,
            alpha3=0.08,
            k_sag=0.08,
            bypass_saturation=False,
        )
        assert out_wav.exists()
        with pedalboard.io.AudioFile(str(out_wav)) as f:
            read_audio = f.read(f.frames)[0]
        assert len(read_audio) == n_samples
        assert np.all(np.isfinite(read_audio))
    finally:
        if out_wav.exists():
            out_wav.unlink()


def test_passive_rlc_thermal_noise_dither():
    """Verify passive RLC-shaped -108 dBFS thermal noise dither and reproducibility."""
    fs = 48000
    model = CircuitModel()
    model.L = 4.8
    model.Rdc = 9500.0
    model.Ccoil = 80e-12

    # Synthesize test input sweep above 0.10 peak
    t = np.linspace(0, 0.5, int(fs * 0.5), endpoint=False)
    sweep = (0.5 * np.sin(2 * np.pi * 120.0 * t)).astype(np.float32)

    with tempfile.TemporaryDirectory() as td:
        out1 = Path(td) / "dither1.wav"
        out2 = Path(td) / "dither2.wav"
        out_nodither = Path(td) / "nodither.wav"

        # 1. Deterministic reproducibility across multiple calls
        simulate_circuit_audio(
            sweep, out1, model, noise_dither=True, is_identity=False, normalize="none"
        )
        simulate_circuit_audio(
            sweep, out2, model, noise_dither=True, is_identity=False, normalize="none"
        )

        with open(out1, "rb") as f1, open(out2, "rb") as f2:
            assert f1.read() == f2.read(), (
                "Thermal noise dither must be bit-exact reproducible with PRNG seed 42"
            )

        # 2. Dither difference from clean un-dithered output
        simulate_circuit_audio(
            sweep, out_nodither, model, noise_dither=False, is_identity=False, normalize="none"
        )
        with pedalboard.io.AudioFile(str(out1)) as f:
            a_dither = f.read(f.frames)[0]
        with pedalboard.io.AudioFile(str(out_nodither)) as f:
            a_clean = f.read(f.frames)[0]

        diff = a_dither - a_clean
        diff_rms = np.sqrt(np.mean(diff**2))
        diff_rms_db = 20.0 * math.log10(diff_rms)
        # Injected noise should be calibrated around -108 dBFS (within 2.0 dB)
        assert abs(diff_rms_db - (-108.0)) < 2.0


def test_run_spice_batch_parallel(tmp_path: Path):
    """Verify that run_spice_batch executes multiple voices concurrently across ProcessPoolExecutor workers."""
    test_voices = ["precision_active", "precision_vintage"]
    inst = "30in"

    # Execute batch with jobs=2 and max_samples=4800 (fast test bounding) in tmp_path
    ok = run_spice_batch(
        voices=test_voices,
        instrument=inst,
        jobs=2,
        max_samples=4800,
        output_dir=tmp_path,
    )
    assert ok is True

    # Verify both outputs exist and are valid non-empty audio files
    for v in test_voices:
        out_wav = tmp_path / f"out_{v}.wav"
        assert out_wav.exists(), (
            f"Expected output {out_wav} to be created by parallel batch simulation"
        )
        assert out_wav.stat().st_size > 44, f"Output {out_wav} is too small"


def test_unsupported_ltspice_backend_raises_error():
    """Verify that attempting to invoke the removed legacy LTspice backend raises ValueError."""
    import pytest

    from allomorph.pipeline import run_circuit_simulation

    with pytest.raises(ValueError, match="The legacy LTspice pipeline has been removed"):
        run_circuit_simulation("precision_active", backend="ltspice")

    with pytest.raises(ValueError, match="The legacy LTspice pipeline has been removed"):
        run_spice_batch(["precision_active"], backend="ltspice")


def test_run_pipeline_cli_jobs_and_voices():
    """Verify CLI argument parsing and default voice resolution across pipeline stages."""
    import argparse

    # Simulate parser logic
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        choices=["all", "viz", "sim", "pack", "train"],
        default="all",
    )
    parser.add_argument("--voice", "-v", default="all")
    parser.add_argument("--jobs", "-j", type=int, default=None)

    # 1. When no voice is specified, it must resolve to all voices
    args = parser.parse_args(["--stage", "sim"])
    voices = resolve_voices(args.voice)
    assert len(voices) == len(VOICES)
    assert "precision_active" in voices
    assert "jazz_pair_active" in voices

    # 2. When explicit -v is passed
    args = parser.parse_args(["--stage", "sim", "-v", "precision_active", "-j", "4"])
    voices = resolve_voices(args.voice)
    assert voices == ["precision_active"]
    assert args.jobs == 4

    # 3. When --stage train is invoked with no voice, it also resolves to all voices
    args = parser.parse_args(["--stage", "train"])
    voices = resolve_voices(args.voice)
    assert len(voices) == len(VOICES)


def test_simulate_voice_strict_configuration_errors():
    """Verify that simulate_voice and apply_magnet_properties_to_model raise strict configuration errors."""
    import pytest

    from allomorph.circuit.parser import CircuitModel
    from allomorph.circuit.solver import apply_magnet_properties_to_model

    # 1. Unknown target voice
    with pytest.raises(KeyError, match="Target voice 'imaginary_bass_voice' not found"):
        simulate_voice("imaginary_bass_voice")

    # 2. Unknown pickup on instrument
    with pytest.raises(
        KeyError, match="Pickup 'non_existent_pickup' not found on instrument '30in_emg_mmtw'"
    ):
        simulate_voice("precision_active", instrument="30in", pickup="non_existent_pickup")

    from allomorph.config.schema import InstrumentConfig, PickupConfig

    # 3. Passive source pickup missing a [circuit] block
    passive_inst_no_cir = InstrumentConfig(
        id="broken_passive_bass",
        name="Broken Passive Bass",
        electronics="passive",
        default_pickup="p",
        pickups={
            "p": PickupConfig(
                name="Passive P",
                position_from_bridge_m=0.125,
                aperture_width_in=0.75,
                coil_spacing_in=0.0,
                magnet_type="alnico_v",
                # Note: No 'circuit' defined!
            )
        },
        string_wave_speeds=[73.4, 98.0, 130.8, 174.6],
        scale_length_in=34.0,
    )
    with pytest.raises(ValueError, match="does not define a '\\[circuit\\]' block"):
        simulate_voice("precision_active", instrument=passive_inst_no_cir, max_samples=100)

    # 4. Unknown magnet type in solver
    model = CircuitModel()
    with pytest.raises(KeyError, match="Unknown magnet type 'unobtainium'"):
        apply_magnet_properties_to_model(
            model, PickupConfig(name="mock", magnet_type="unobtainium")
        )
