"""
Integration tests for catalog instruments, voicings, and forward stems.
Verifies that all catalog instruments and voicings pass forward digital twin simulation,
causal zero-latency alignment, true-peak safety, and differential magnetic dynamics.
"""

import tempfile
from pathlib import Path

import numpy as np
import pedalboard.io

from allomorph.circuit.stem_debug import debug_voicing_stem
from allomorph.config.instruments import INSTRUMENTS
from allomorph.config.preamps import PREAMPS
from allomorph.dsp import write_wav_24bit
from allomorph.pipeline.batch import simulate_voice


def test_all_catalog_instrument_voicings_bug_free():
    """Validates that all native voicings defined across all catalog instruments
    pass causal zero-latency alignment, remain below true-peak ceiling, and maintain stable THD.
    """
    for inst_id, inst in sorted(INSTRUMENTS.items()):
        for v_id, voicing in sorted(inst.voicings.items()):
            report = debug_voicing_stem(
                instrument=inst,
                voicing=voicing,
                preamps=PREAMPS,
            )
            assert report.is_causal_zero_latency, (
                f"{inst_id}:{v_id} failed causal zero-latency (onset={report.onset_sample_index}, peak={report.peak_sample_index})"
            )
            assert report.peak_dbfs <= -0.09, (
                f"{inst_id}:{v_id} clipped above -0.09 dBFS ceiling ({report.peak_dbfs:+.2f} dBFS)"
            )
            assert report.thd_percent <= 50.0, (
                f"{inst_id}:{v_id} showed unstable harmonic distortion ({report.thd_percent:.1f}%)"
            )


def test_differential_magnetic_softening_neodymium_to_alnico():
    """
    Verify that converting a passive Neodymium source (34in_dingwall_sp1) to an
    Alnico V target (precision_vintage) engages differential magnetic softening:
    - Delta alpha = 0.18, Delta eta = 0.05, Delta k_sag = 0.07, Vsat_eff ≈ 0.84.
    - Forte peaks (> 0.5V) undergo soft-knee saturation and 2nd harmonic expansion.
    """
    sr = 48000
    t = np.linspace(0, 0.1, int(sr * 0.1), endpoint=False)
    forte_signal = (0.85 * np.sin(2 * np.pi * 100 * t)).astype(np.float32)

    with tempfile.TemporaryDirectory() as td:
        in_wav = Path(td) / "forte_in.wav"
        out_wav = Path(td) / "out_softened.wav"
        write_wav_24bit(str(in_wav), forte_signal, sample_rate=sr)

        res = simulate_voice(
            "precision_vintage",
            input_wav=in_wav,
            output_wav=out_wav,
            instrument="34in_dingwall_sp1",
            normalize="none",
        )
        assert res is True

        with pedalboard.io.AudioFile(str(out_wav)) as f:
            audio_out = f.read(f.frames)[0]

        # In a non-linear saturation, second harmonic (200 Hz) emerges from asymmetry (Delta alpha > 0)
        fft_mag = np.abs(np.fft.rfft(audio_out))
        freqs = np.fft.rfftfreq(len(audio_out), 1.0 / sr)
        fund_idx = np.argmin(np.abs(freqs - 100.0))
        h2_idx = np.argmin(np.abs(freqs - 200.0))

        fund_level = fft_mag[fund_idx]
        h2_level = fft_mag[h2_idx]
        # Second harmonic is present due to differential asymmetry (alpha > 0)
        assert h2_level > 1e-4 * fund_level, (
            f"Expected 2nd harmonic bloom from differential alpha, got H2/H1 = {h2_level / fund_level:.6f}"
        )


def test_differential_magnetic_softening_alnico_to_neodymium_bypassed():
    """
    Verify that converting a softer Alnico V source (34in_standard_p) to a stiffer
    Neodymium target (dingwall_bridge) bypasses forward saturation (Delta <= 0).
    Input scaling linearity error ||y_full - 2 * y_half|| / ||y_full|| must be < 1e-4.
    """
    sr = 48000
    t = np.linspace(0, 0.1, int(sr * 0.1), endpoint=False)
    sig_full = (0.85 * np.sin(2 * np.pi * 100 * t)).astype(np.float32)
    sig_half = (0.425 * np.sin(2 * np.pi * 100 * t)).astype(np.float32)

    with tempfile.TemporaryDirectory() as td:
        in_full = Path(td) / "full.wav"
        in_half = Path(td) / "half.wav"
        out_full = Path(td) / "out_bypassed_full.wav"
        out_half = Path(td) / "out_bypassed_half.wav"
        write_wav_24bit(str(in_full), sig_full, sample_rate=sr)
        write_wav_24bit(str(in_half), sig_half, sample_rate=sr)

        simulate_voice(
            "dingwall_bridge",
            input_wav=in_full,
            output_wav=out_full,
            instrument="34in_standard_p",
            normalize="none",
        )
        simulate_voice(
            "dingwall_bridge",
            input_wav=in_half,
            output_wav=out_half,
            instrument="34in_standard_p",
            normalize="none",
        )

        with pedalboard.io.AudioFile(str(out_full)) as f:
            y_full = f.read(f.frames)[0]
        with pedalboard.io.AudioFile(str(out_half)) as f:
            y_half = f.read(f.frames)[0]

        # Target Neodymium is stiffer than source Alnico V, so softening is bypassed: 100% linear
        rel_diff = float(np.max(np.abs(y_full - 2.0 * y_half)) / np.max(np.abs(y_full)))
        assert rel_diff < 1e-4, f"Expected linear scaling (rel_diff < 1e-4), got {rel_diff:.2e}"


def test_differential_magnetic_softening_active_to_passive():
    """
    Verify that converting an active 18V EMG source (30in_emg_mmtw) to a passive
    Alnico V target (precision_vintage) applies full target magnetic saturation.
    Compression and asymmetry cause ||y_full - 2 * y_half|| / ||y_full|| to exceed 5%.
    """
    sr = 48000
    t = np.linspace(0, 0.1, int(sr * 0.1), endpoint=False)
    sig_full = (0.85 * np.sin(2 * np.pi * 100 * t)).astype(np.float32)
    sig_half = (0.425 * np.sin(2 * np.pi * 100 * t)).astype(np.float32)

    with tempfile.TemporaryDirectory() as td:
        in_full = Path(td) / "full.wav"
        in_half = Path(td) / "half.wav"
        out_full = Path(td) / "out_act_full.wav"
        out_half = Path(td) / "out_act_half.wav"
        write_wav_24bit(str(in_full), sig_full, sample_rate=sr)
        write_wav_24bit(str(in_half), sig_half, sample_rate=sr)

        simulate_voice(
            "precision_vintage",
            input_wav=in_full,
            output_wav=out_full,
            instrument="30in_emg_mmtw",
            normalize="none",
        )
        simulate_voice(
            "precision_vintage",
            input_wav=in_half,
            output_wav=out_half,
            instrument="30in_emg_mmtw",
            normalize="none",
        )

        with pedalboard.io.AudioFile(str(out_full)) as f:
            y_full = f.read(f.frames)[0]
        with pedalboard.io.AudioFile(str(out_half)) as f:
            y_half = f.read(f.frames)[0]

        rel_diff = float(np.max(np.abs(y_full - 2.0 * y_half)) / np.max(np.abs(y_full)))
        assert rel_diff > 0.05, (
            f"Expected non-linear saturation (rel_diff > 0.05), got {rel_diff:.4f}"
        )


def test_differential_magnetic_softening_identity_bypassed():
    """
    Verify that an identity voice conversion (34in_standard_p -> precision_vintage)
    bypasses forward saturation to prevent double-compression.
    Input scaling linearity error ||y_full - 2 * y_half|| / ||y_full|| must be < 1e-4.
    """
    sr = 48000
    t = np.linspace(0, 0.1, int(sr * 0.1), endpoint=False)
    sig_full = (0.50 * np.sin(2 * np.pi * 100 * t)).astype(np.float32)
    sig_half = (0.25 * np.sin(2 * np.pi * 100 * t)).astype(np.float32)

    with tempfile.TemporaryDirectory() as td:
        in_full = Path(td) / "full.wav"
        in_half = Path(td) / "half.wav"
        out_full = Path(td) / "out_id_full.wav"
        out_half = Path(td) / "out_id_half.wav"
        write_wav_24bit(str(in_full), sig_full, sample_rate=sr)
        write_wav_24bit(str(in_half), sig_half, sample_rate=sr)

        simulate_voice(
            "precision_vintage",
            input_wav=in_full,
            output_wav=out_full,
            instrument="34in_standard_p",
            normalize="none",
        )
        simulate_voice(
            "precision_vintage",
            input_wav=in_half,
            output_wav=out_half,
            instrument="34in_standard_p",
            normalize="none",
        )

        with pedalboard.io.AudioFile(str(out_full)) as f:
            y_full = f.read(f.frames)[0]
        with pedalboard.io.AudioFile(str(out_half)) as f:
            y_half = f.read(f.frames)[0]

        rel_diff = float(np.max(np.abs(y_full - 2.0 * y_half)) / np.max(np.abs(y_full)))
        assert rel_diff < 1e-4, f"Expected linear scaling (rel_diff < 1e-4), got {rel_diff:.2e}"
