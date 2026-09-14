"""
Tests for non-linear magnetic feel, Dahl hysteresis, Lenz velocity drag,
soft-knee core saturation, and Numba accelerated kernels.
"""

import math
import tempfile
from pathlib import Path

import numpy as np
import pedalboard.io
import pytest

from allomorph.circuit import (
    MAGNET_PROPERTIES,
    CircuitModel,
    _dahl_core,
    _lenz_envelope_core,
    _lenz_velocity_drag_core,
    _slew_limit_core,
    apply_dahl_hysteresis,
    apply_elliptical_orbit_projection,
    apply_magnet_properties_to_model,
    apply_oversampled_saturation,
    compute_core_impedance,
    load_circuit,
    simulate_circuit_audio,
    simulate_voice,
)
from allomorph.config.schema import PickupConfig
from allomorph.dsp import write_wav_24bit


def test_passive_saturation_bypassed():
    """Verify that forward tanh saturation is bypassed when is_passive is True."""
    n_samples = 4800
    # High amplitude input (0.80) exceeding vsat (0.45)
    in_heavy = np.full((1, n_samples), 0.80, dtype=np.float32)

    m = load_circuit("precision_vintage")

    with (
        tempfile.NamedTemporaryFile(suffix=".wav") as tmp_act,
        tempfile.NamedTemporaryFile(suffix=".wav") as tmp_pas,
    ):
        # Active simulation: applies tanh
        simulate_circuit_audio(in_heavy, Path(tmp_act.name), m, is_passive=False)
        # Passive simulation: bypasses tanh
        simulate_circuit_audio(in_heavy, Path(tmp_pas.name), m, is_passive=True)

        with pedalboard.io.AudioFile(tmp_act.name) as f:
            audio_act = f.read(f.frames)[0]
        with pedalboard.io.AudioFile(tmp_pas.name) as f:
            audio_pas = f.read(f.frames)[0]

    # Passive output should maintain linear proportional gain (higher uncompressed peak)
    assert np.max(np.abs(audio_pas)) > np.max(np.abs(audio_act))


def test_anti_aliased_oversampling_suppression():
    """Verify that 2x oversampled saturation suppresses folded aliasing by >60 dB."""
    sr = 48000
    t = np.linspace(0, 0.1, int(sr * 0.1), endpoint=False)
    tone = 0.5 * np.sin(2 * np.pi * 15000 * t).astype(np.float32)

    out_1x = apply_oversampled_saturation(
        tone, vsat=0.45, alpha=0.20, oversample=1, displacement_weighting=False, magnet_drag=False
    )
    out_2x = apply_oversampled_saturation(
        tone, vsat=0.45, alpha=0.20, oversample=2, displacement_weighting=False, magnet_drag=False
    )

    f_arr = np.fft.rfftfreq(len(tone), 1.0 / sr)
    fft_1x = np.abs(np.fft.rfft(out_1x))
    fft_2x = np.abs(np.fft.rfft(out_2x))

    alias_3k_1x = fft_1x[np.argmin(np.abs(f_arr - 3000))]
    alias_3k_2x = fft_2x[np.argmin(np.abs(f_arr - 3000))]
    fund_1x = fft_1x[np.argmin(np.abs(f_arr - 15000))]
    fund_2x = fft_2x[np.argmin(np.abs(f_arr - 15000))]

    db_1x = 20 * np.log10(alias_3k_1x / fund_1x)
    db_2x = 20 * np.log10(max(alias_3k_2x, 1e-9) / fund_2x)

    suppression = db_1x - db_2x
    assert suppression > 60.0, f"Aliasing suppression {suppression:.1f} dB did not exceed 60 dB"


def test_displacement_domain_imd_reduction():
    """Verify that displacement weighting suppresses treble intermodulation distortion by >6 dB."""
    sr = 48000
    t = np.linspace(0, 0.2, int(sr * 0.2), endpoint=False)
    sig = 0.6 * np.sin(2 * np.pi * 50 * t) + 0.1 * np.sin(2 * np.pi * 2500 * t)
    sig = sig.astype(np.float32)

    out_flat = apply_oversampled_saturation(
        sig, vsat=0.45, alpha=0.20, oversample=1, displacement_weighting=False, magnet_drag=False
    )
    out_disp = apply_oversampled_saturation(
        sig, vsat=0.45, alpha=0.20, oversample=1, displacement_weighting=True, magnet_drag=False
    )

    f_arr = np.fft.rfftfreq(len(sig), 1.0 / sr)
    fft_flat = np.abs(np.fft.rfft(out_flat))
    fft_disp = np.abs(np.fft.rfft(out_disp))

    idx_2500 = np.argmin(np.abs(f_arr - 2500))
    idx_2450 = np.argmin(np.abs(f_arr - 2450))

    imd_flat_db = 20 * np.log10(fft_flat[idx_2450] / fft_flat[idx_2500])
    imd_disp_db = 20 * np.log10(fft_disp[idx_2450] / fft_disp[idx_2500])

    imd_reduction = imd_flat_db - imd_disp_db
    assert imd_reduction > 6.0, f"IMD reduction {imd_reduction:.1f} dB did not exceed 6 dB"


def test_magnet_specific_saturation_voicing():
    """Verify that Ceramic (alpha=0.12) generates less even-harmonic energy than Alnico V (alpha=0.26), and Piezo (alpha=0) is purely odd."""
    sr = 48000
    t = np.linspace(0, 0.2, int(sr * 0.2), endpoint=False)
    fund_f = 200.0
    tone = (0.50 * np.sin(2 * np.pi * fund_f * t)).astype(np.float32)

    # 1. Alnico V (alpha=0.26)
    out_alnico = apply_oversampled_saturation(
        tone, vsat=0.45, alpha=0.26, oversample=2, displacement_weighting=False, magnet_drag=False
    )
    # 2. Ceramic (alpha=0.12)
    out_ceramic = apply_oversampled_saturation(
        tone, vsat=0.45, alpha=0.12, oversample=2, displacement_weighting=False, magnet_drag=False
    )
    # 3. Piezo (alpha=0.00)
    out_piezo = apply_oversampled_saturation(
        tone, vsat=0.45, alpha=0.00, oversample=2, displacement_weighting=False, magnet_drag=False
    )

    f_arr = np.fft.rfftfreq(len(tone), 1.0 / sr)
    fft_alnico = np.abs(np.fft.rfft(out_alnico))
    fft_ceramic = np.abs(np.fft.rfft(out_ceramic))
    fft_piezo = np.abs(np.fft.rfft(out_piezo))

    h2_idx = np.argmin(np.abs(f_arr - 400.0))  # 2nd harmonic (400 Hz)
    fund_idx = np.argmin(np.abs(f_arr - 200.0))  # Fundamental (200 Hz)

    h2_alnico_db = 20 * np.log10(fft_alnico[h2_idx] / fft_alnico[fund_idx])
    h2_ceramic_db = 20 * np.log10(fft_ceramic[h2_idx] / fft_ceramic[fund_idx])
    h2_piezo_db = 20 * np.log10(max(fft_piezo[h2_idx], 1e-9) / fft_piezo[fund_idx])

    # Alnico V must generate more 2nd harmonic bloom than Ceramic (> 5 dB difference)
    assert h2_alnico_db > h2_ceramic_db + 5.0
    # Piezo must have negligible 2nd harmonic (< -60 dB relative to fundamental)
    assert h2_piezo_db < -60.0


def test_fractional_core_eddy_diffusion():
    """
    Verify Foster 2-stage ladder core eddy diffusion:
    1. Alnico V has ~8% inductance drop and authentic damping at 10 kHz.
    2. Ceramic has <= 2% inductance drop due to non-conductive ferrite.
    3. Core impedance introduces real resistive losses Re(Z_L) > 0 at audio frequencies.
    4. Disabling eddy diffusion preserves constant ideal inductance s * L.
    """
    L0 = 4.8  # H
    # 1. Alnico V
    model_alnico = CircuitModel()
    model_alnico.L = L0
    apply_magnet_properties_to_model(
        model_alnico, PickupConfig(name="mock", magnet_type="alnico_v")
    )
    assert model_alnico.L_core == pytest.approx(0.08 * L0)
    assert model_alnico.R_core > 0.0

    # Evaluate at 10 kHz
    w_hi = 2.0 * math.pi * 10000.0
    s_hi = 1j * w_hi
    Z_hi_alnico = compute_core_impedance(
        s_hi, model_alnico.L, model_alnico.L_core, model_alnico.R_core
    )

    # Inductance at high frequency should be dropped by ~7-8%
    L_eff_hi = Z_hi_alnico.imag / w_hi
    assert L_eff_hi < L0
    drop_pct = (L0 - L_eff_hi) / L0 * 100.0
    assert 6.0 <= drop_pct <= 8.5
    # Real part must represent eddy loss resistance (Re(Z) > 0)
    assert Z_hi_alnico.real > 500.0

    # 2. Ceramic (insulating ferrite core)
    model_ceramic = CircuitModel()
    model_ceramic.L = L0
    apply_magnet_properties_to_model(
        model_ceramic, PickupConfig(name="mock", magnet_type="ceramic")
    )
    assert model_ceramic.L_core == pytest.approx(0.02 * L0)
    Z_hi_ceramic = compute_core_impedance(
        s_hi, model_ceramic.L, model_ceramic.L_core, model_ceramic.R_core
    )
    L_eff_ceramic = Z_hi_ceramic.imag / w_hi
    drop_ceramic = (L0 - L_eff_ceramic) / L0 * 100.0
    assert drop_ceramic <= 2.2

    # 3. Disabled eddy diffusion
    model_disabled = CircuitModel()
    model_disabled.L = L0
    apply_magnet_properties_to_model(
        model_disabled, PickupConfig(name="mock", magnet_type="alnico_v"), eddy_diffusion=False
    )
    assert model_disabled.L_core == 0.0
    assert model_disabled.R_core == 0.0
    Z_hi_disabled = compute_core_impedance(
        s_hi, model_disabled.L, model_disabled.L_core, model_disabled.R_core
    )
    assert Z_hi_disabled.imag / w_hi == pytest.approx(L0)
    assert Z_hi_disabled.real == 0.0


def test_dahl_magnetic_hysteresis():
    """
    Verify Dahl magnetic hysteresis friction modeling:
    1. Produces phase lag on sinusoidal signals without generating DC bias.
    2. Alnico V (eta=0.06) exhibits hysteresis lag; Piezo (eta=0.0) has 0 phase lag.
    3. Small signals (<= 0.10 peak) bypass saturation/hysteresis.
    """
    sr = 96000
    t = np.linspace(0, 0.1, int(sr * 0.1), endpoint=False)
    freq = 100.0
    x = 0.60 * np.sin(2 * np.pi * freq * t)

    # 1. Alnico V hysteresis
    y_alnico = apply_dahl_hysteresis(x, eta=0.06)
    # Zero crossing phase lag: y should cross 0 slightly later than x
    idx_cross = int(sr * 0.005)
    # Just after crossing where x is negative:
    assert x[idx_cross + 1] < 0.0
    # y should lag behind x (more positive)
    assert y_alnico[idx_cross + 1] > x[idx_cross + 1]

    # DC offset must be minimal (< 1e-3)
    assert abs(np.mean(y_alnico)) < 1e-3

    # 2. Piezo (eta=0.0) must be exact identity
    y_piezo = apply_dahl_hysteresis(x, eta=0.0)
    assert np.array_equal(x, y_piezo)

    # 3. Small-signal test impulse bypass in apply_oversampled_saturation
    impulse = np.zeros(2048, dtype=np.float32)
    impulse[0] = 0.05
    out_impulse = apply_oversampled_saturation(impulse, vsat=0.5, alpha=0.26, eta_hyst=0.06)
    assert np.array_equal(impulse, out_impulse)


def test_higher_order_dipole_expansion_and_sag():
    """
    Verify higher-order magnetic dipole expansion (alpha3) and Lenz-law flux sag (k_sag):
    1. MAGNET_PROPERTIES defines alpha3 and k_sag across all magnet types.
    2. Cubic expansion generates 3rd harmonic (3f) excitation.
    3. Flux sag reduces peak excursion envelope on forte attacks, proportional to k_sag.
    4. Small signals (<= 0.10) preserve mathematical linearity.
    """
    # 1. Magnet properties dictionary check
    for mag_type in [
        "alnico_v",
        "alnico_ii",
        "alnico_iii",
        "ceramic",
        "hybrid",
        "neodymium",
        "piezo",
        "ideal",
    ]:
        assert mag_type in MAGNET_PROPERTIES
        props = MAGNET_PROPERTIES[mag_type]
        assert hasattr(props, "alpha3")
        assert hasattr(props, "k_sag")
        assert 0.0 <= props.alpha3 <= 0.20
        assert 0.0 <= props.k_sag <= 0.20

    # Alnico II has highest sag and proximity stiffening; Piezo and Ideal have 0.0
    assert MAGNET_PROPERTIES["alnico_ii"].alpha3 > MAGNET_PROPERTIES["ceramic"].alpha3
    assert MAGNET_PROPERTIES["alnico_ii"].k_sag > MAGNET_PROPERTIES["ceramic"].k_sag
    assert MAGNET_PROPERTIES["piezo"].alpha3 == 0.0
    assert MAGNET_PROPERTIES["piezo"].k_sag == 0.0
    assert MAGNET_PROPERTIES["ideal"].alpha3 == 0.0
    assert MAGNET_PROPERTIES["ideal"].k_sag == 0.0

    # 2. Cubic expansion generating 3rd harmonic
    sr = 48000
    n = 8192
    t = np.arange(n) / sr
    f0 = 150.0
    sig = 0.50 * np.sin(2.0 * np.pi * f0 * t)

    # Pure quadratic (alpha=0.20, alpha3=0.0)
    out_quad = apply_oversampled_saturation(
        sig, vsat=0.6, alpha=0.20, alpha3=0.0, k_sag=0.0, displacement_weighting=False, oversample=1
    )
    # Cubic proximity stiffening (alpha=0.20, alpha3=0.10)
    out_cubic = apply_oversampled_saturation(
        sig,
        vsat=0.6,
        alpha=0.20,
        alpha3=0.10,
        k_sag=0.0,
        displacement_weighting=False,
        oversample=1,
    )

    diff_sig = out_cubic - out_quad
    fft_diff = np.abs(np.fft.rfft(diff_sig))
    freqs = np.fft.rfftfreq(n, 1.0 / sr)

    idx_h3 = int(np.argmin(np.abs(freqs - 3.0 * f0)))
    h3_diff = fft_diff[idx_h3]

    # Cubic proximity stiffening must produce significant 3rd harmonic modulation and increased excursion
    assert h3_diff > 2.0
    assert np.max(out_cubic) > np.max(out_quad)

    # 3. Dynamic Lenz-law core flux sag
    # Large burst signal exceeding vsat
    burst = 0.80 * np.sin(2.0 * np.pi * f0 * t)
    out_nosag = apply_oversampled_saturation(
        burst, vsat=0.5, alpha=0.0, alpha3=0.0, k_sag=0.0, magnet_drag=True, oversample=1
    )
    out_sag = apply_oversampled_saturation(
        burst, vsat=0.5, alpha=0.0, alpha3=0.0, k_sag=0.12, magnet_drag=True, oversample=1
    )

    # RMS of sagged burst must be lower due to dynamic Lenz braking
    rms_nosag = np.sqrt(np.mean(out_nosag**2))
    rms_sag = np.sqrt(np.mean(out_sag**2))
    assert rms_sag < rms_nosag

    # 4. Small-signal linearity
    small_sig = 0.05 * np.sin(2.0 * np.pi * f0 * t)
    out_small = apply_oversampled_saturation(
        small_sig, vsat=0.5, alpha=0.20, alpha3=0.10, k_sag=0.12, oversample=1
    )
    assert np.allclose(small_sig, out_small, atol=1e-6)


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


def test_asymmetric_lenz_flux_sag_attack_release():
    """
    Verify Refinement 1: Asymmetric Lenz flux sag envelope follower.
    Fast attack (<= 8 ms) on sudden transient burst, gradual release (>= 35 ms) on decay.
    """
    fs = 48000.0
    tau_att = 0.006  # 6 ms
    tau_rel = 0.045  # 45 ms
    alpha_att = 1.0 - math.exp(-1.0 / (fs * tau_att))
    alpha_rel = 1.0 - math.exp(-1.0 / (fs * tau_rel))

    # Test Step Attack: sudden burst from 0.0 to 1.0
    n_samples = int(fs * 0.1)  # 100 ms
    step_input = np.ones(n_samples, dtype=np.float64)
    env_attack = _lenz_envelope_core(step_input, alpha_att, alpha_rel)

    # In 1 tau_att (6 ms = 288 samples), envelope should reach ~63.2%
    idx_6ms = int(fs * 0.006)
    assert 0.60 <= env_attack[idx_6ms] <= 0.66
    # In 8 ms (384 samples), envelope should exceed 70% (fast attack <= 8 ms)
    idx_8ms = int(fs * 0.008)
    assert env_attack[idx_8ms] >= 0.70

    # Test Decay Release: held at 1.0 for 100 ms, then sudden drop to 0.0 for 100 ms
    pulse_input = np.zeros(int(fs * 0.2), dtype=np.float64)
    pulse_input[:n_samples] = 1.0
    env_release = _lenz_envelope_core(pulse_input, alpha_att, alpha_rel)
    # At n_samples (100 ms), env is ~1.0. In 1 tau_rel (45 ms), it should decay to ~36.8% of ~1.0
    idx_45ms_after = n_samples + int(fs * 0.045)
    assert 0.33 <= env_release[idx_45ms_after] <= 0.40
    # In 35 ms after drop, it should still retain significant energy (> 40%), proving release >= 35 ms
    idx_35ms_after = n_samples + int(fs * 0.035)
    assert env_release[idx_35ms_after] >= 0.40

    # Verify apply_oversampled_saturation incorporates Lenz sag
    x = np.sin(2 * np.pi * 100 * np.linspace(0, 0.2, int(fs * 0.2))) * 0.8
    y_sag = apply_oversampled_saturation(x, vsat=0.4, k_sag=0.15, magnet_drag=True)
    y_nosag = apply_oversampled_saturation(x, vsat=0.4, k_sag=0.0, magnet_drag=False)
    assert np.max(np.abs(y_sag)) < np.max(np.abs(y_nosag))


def test_asymmetric_dahl_proximity_pinning():
    """
    Verify Refinement: Asymmetric Dahl magnetic domain-wall pinning.
    Approach excursion (x > 0 towards pole) experiences higher pinning coupling than departure (x < 0).
    Verifies zero DC bias drift on cyclic zero-mean AC excitation.
    """
    n = 1000
    pos_step = np.full(n, 0.6, dtype=np.float64)
    neg_step = np.full(n, -0.6, dtype=np.float64)
    pos_step[0] = 0.0
    neg_step[0] = 0.0

    out_pos = _dahl_core(pos_step, eta=0.10, r=0.06)
    out_neg = _dahl_core(neg_step, eta=0.10, r=0.06)

    z_pos_1 = (out_pos[1] - (1.0 - 0.10) * pos_step[1]) / 0.10
    z_neg_1 = abs((out_neg[1] - (1.0 - 0.10) * neg_step[1]) / 0.10)
    assert z_pos_1 > z_neg_1

    # Verify tiny DC drift on cyclic audio (< 0.005, blocked downstream by 8 Hz DC blocker)
    t = np.linspace(0, 0.5, 24000)
    sine = np.sin(2 * np.pi * 100.0 * t) * 0.7
    out_sine = apply_dahl_hysteresis(sine, eta=0.06, r=0.06)
    assert abs(np.mean(out_sine)) < 0.005


def test_frequency_selective_lenz_velocity_drag():
    """
    Verify Refinement: Velocity-proportional / frequency-selective Lenz drag.
    High-frequency transient clank is damped more heavily than low-frequency fundamental.
    """
    fs = 48000
    t = np.linspace(0, 0.2, int(fs * 0.2))
    # Dual-tone signal: 60 Hz bass fundamental + 3000 Hz transient clank
    sig_low = np.sin(2 * np.pi * 60.0 * t) * 0.5
    sig_high = np.sin(2 * np.pi * 3000.0 * t) * 0.5
    sig_dual = (sig_low + sig_high).astype(np.float32)

    # Saturate with Lenz drag engaged
    out_sag = apply_oversampled_saturation(sig_dual, vsat=0.4, k_sag=0.20, magnet_drag=True)
    out_nosag = apply_oversampled_saturation(sig_dual, vsat=0.4, k_sag=0.0, magnet_drag=False)

    f_bins = np.fft.rfftfreq(len(t), 1.0 / fs)
    idx_60 = np.argmin(np.abs(f_bins - 60.0))
    idx_3k = np.argmin(np.abs(f_bins - 3000.0))

    fft_sag = np.abs(np.fft.rfft(out_sag))
    fft_nosag = np.abs(np.fft.rfft(out_nosag))

    low_ratio = fft_sag[idx_60] / fft_nosag[idx_60]
    high_ratio = fft_sag[idx_3k] / fft_nosag[idx_3k]

    # High frequency must be damped more heavily than low frequency
    assert high_ratio < low_ratio
    delta_db_diff = 20.0 * np.log10(low_ratio / high_ratio)
    assert delta_db_diff >= 0.40

    # Small-signal test vector (<= 0.10) must be 100% linear bypass
    small_sig = (sig_dual * 0.05).astype(np.float32)
    out_small = apply_oversampled_saturation(small_sig, vsat=0.4, k_sag=0.20)
    assert np.allclose(small_sig, out_small, atol=1e-6)


def test_dynamic_eddy_de_qing():
    """Verify dynamic eddy-current transient core de-Qing physics."""
    fs = 48000
    n = 2400
    # High-frequency transient clank on forte attack
    x = np.sin(2 * np.pi * 3500.0 * np.linspace(0, n / fs, n)) * 0.8
    env = np.full(n, 0.8)  # Forte excursion above vsat=0.45
    alpha_c = 1.0 - math.exp(-2.0 * math.pi * 750.0 / fs)

    # Low eddy (ceramic / modern active) vs high eddy (vintage Alnico V)
    out_no_eddy = _lenz_velocity_drag_core(
        x, env, vsat=0.45, k_sag=0.08, alpha_c=alpha_c, k_eddy=0.0
    )
    out_with_eddy = _lenz_velocity_drag_core(
        x, env, vsat=0.45, k_sag=0.08, alpha_c=alpha_c, k_eddy=0.16
    )

    rms_no = np.sqrt(np.mean(out_no_eddy**2))
    rms_with = np.sqrt(np.mean(out_with_eddy**2))

    # Eddy current de-Qing must add transient damping to the peak attack
    assert rms_with < rms_no
    damping_db = 20.0 * np.log10(rms_no / rms_with)
    assert 0.4 <= damping_db <= 2.5


def test_elliptical_orbit_projection():
    """Verify elliptical string orbit quadrature second-harmonic (2f0) projection."""
    fs = 48000
    n = 24000
    t = np.linspace(0, n / fs, n, endpoint=False)
    sine = (0.5 * np.sin(2 * np.pi * 100.0 * t)).astype(np.float64)

    # 1. kappa_orbit == 0 must return input untouched
    out_zero = apply_elliptical_orbit_projection(sine, vsat=0.5, kappa_orbit=0.0)
    assert np.allclose(out_zero, sine, atol=1e-12)

    # 2. Realistic Alnico V orbit projection (kappa = 0.06)
    out_orbit = apply_elliptical_orbit_projection(sine, vsat=0.5, kappa_orbit=0.06)

    # DC offset must be zero
    assert abs(np.mean(out_orbit)) < 1e-6

    # 2f0 harmonic (200 Hz) must emerge in the spectrum
    fft_in = np.abs(np.fft.rfft(sine))
    fft_out = np.abs(np.fft.rfft(out_orbit))
    f_bins = np.fft.rfftfreq(n, 1.0 / fs)
    idx_100 = np.argmin(np.abs(f_bins - 100.0))
    idx_200 = np.argmin(np.abs(f_bins - 200.0))

    # In input sine, 200 Hz has near-zero energy
    assert fft_in[idx_200] / fft_in[idx_100] < 1e-4
    # In orbit projected output, authentic 2f0 bloom is present at approx -40 to -50 dB
    h2_ratio = fft_out[idx_200] / fft_out[idx_100]
    h2_db = 20.0 * np.log10(h2_ratio)
    assert -52.0 <= h2_db <= -36.0


def test_dynamic_core_inductance_wobble():
    """Verify dynamic core inductance curvature beta_curv produces transient modulation on forte signals."""
    sr = 48000
    t = np.linspace(0, 0.2, int(sr * 0.2), endpoint=False)
    # Forte signal exceeding vsat (0.75 peak with vsat=0.5)
    sig_forte = (0.75 * np.sin(2.0 * np.pi * 200.0 * t)).astype(np.float32)

    out_no_curv = apply_oversampled_saturation(
        sig_forte,
        vsat=0.5,
        alpha=0.0,
        alpha3=0.0,
        k_sag=0.0,
        k_eddy=0.0,
        beta_curv=0.0,
        slew_limit=False,
        oversample=1,
        displacement_weighting=False,
    )
    out_with_curv = apply_oversampled_saturation(
        sig_forte,
        vsat=0.5,
        alpha=0.0,
        alpha3=0.0,
        k_sag=0.0,
        k_eddy=0.0,
        beta_curv=0.035,  # Alnico V
        slew_limit=False,
        oversample=1,
        displacement_weighting=False,
    )

    diff = out_with_curv - out_no_curv
    assert np.max(np.abs(diff)) > 1e-5, (
        "Inductance curvature must modulate signal on forte excursions"
    )

    # Small-signal test (<= 0.10): must bypass completely and remain bit-exact
    sig_small = (0.05 * np.sin(2.0 * np.pi * 200.0 * t)).astype(np.float32)
    out_small_no = apply_oversampled_saturation(
        sig_small,
        vsat=0.5,
        beta_curv=0.0,
        slew_limit=False,
    )
    out_small_curv = apply_oversampled_saturation(
        sig_small,
        vsat=0.5,
        beta_curv=0.035,
        slew_limit=False,
    )
    assert np.allclose(out_small_curv, out_small_no), "Small-signal must preserve exact linearity"


def test_transient_magnetic_slew_limiting():
    """Verify domain-wall slew-rate limiting transparently handles continuous audio while limiting extreme clank."""
    sr = 48000
    vsat = 0.5
    f_slew = 16000.0
    max_delta = 2.0 * np.pi * f_slew * vsat / float(sr)

    # 1. Smooth 1 kHz sine wave: slew rate = 2*pi*1000*0.5/48000 = 0.065 < max_delta (1.047)
    t = np.linspace(0, 0.05, int(sr * 0.05), endpoint=False)
    sine = (0.5 * np.sin(2.0 * np.pi * 1000.0 * t)).astype(np.float64)
    slewed_sine = _slew_limit_core(sine, max_delta)
    # Should be virtually identical to clean sine
    assert np.allclose(slewed_sine, sine, atol=1e-3), (
        "Smooth musical audio must pass through slew-limiter untouched"
    )

    # 2. Extreme step discontinuity (e.g. fret clank transient with instantaneous jump from 0.0 to 1.0)
    step = np.zeros(100, dtype=np.float64)
    step[50:] = 1.0
    slewed_step = _slew_limit_core(step, max_delta)
    diffs = np.abs(np.diff(slewed_step))
    # Slew limiter must smoothly constrain delta to <= max_delta * tanh(jump/max_delta)
    assert np.all(diffs <= max_delta + 1e-6), "Slew limiter must strictly bound maximum step delta"


def test_nonlinear_magnetic_string_pull_dynamics():
    """Verify nonlinear magnetic string pull produces attack pitch sag and damping on forte strikes, preserving linearity on small signals."""
    sr = 48000
    vsat = 0.4
    k_pull = 0.04
    k_sag = 0.05

    # 1. Small signal (amplitude 0.05 << vsat): must be completely linear and identical regardless of k_pull
    t = np.linspace(0, 0.05, int(sr * 0.05), endpoint=False)
    small_x = (0.05 * np.sin(2.0 * np.pi * 440.0 * t)).astype(np.float64)
    small_env = np.full_like(small_x, 0.05)

    out_small_nopull = _lenz_velocity_drag_core(
        small_x, small_env, vsat, k_sag=k_sag, alpha_c=0.1, k_eddy=0.0, beta_curv=0.0, k_pull=0.0
    )
    out_small_pull = _lenz_velocity_drag_core(
        small_x, small_env, vsat, k_sag=k_sag, alpha_c=0.1, k_eddy=0.0, beta_curv=0.0, k_pull=k_pull
    )
    assert np.allclose(out_small_nopull, out_small_pull, atol=1e-6), (
        "Small signals must not experience magnetic string pull"
    )

    # 2. Forte signal (amplitude 0.9 >> vsat): magnetic string pull must damp upper harmonics and induce pitch sag
    forte_x = (
        0.9 * np.sin(2.0 * np.pi * 440.0 * t) + 0.3 * np.sin(2.0 * np.pi * 1320.0 * t)
    ).astype(np.float64)
    forte_env = np.full_like(forte_x, 0.9)

    out_forte_nopull = _lenz_velocity_drag_core(
        forte_x, forte_env, vsat, k_sag=k_sag, alpha_c=0.1, k_eddy=0.0, beta_curv=0.0, k_pull=0.0
    )
    out_forte_pull = _lenz_velocity_drag_core(
        forte_x, forte_env, vsat, k_sag=k_sag, alpha_c=0.1, k_eddy=0.0, beta_curv=0.0, k_pull=k_pull
    )

    # Output with pull must differ on forte attack, demonstrating physical interaction
    diff = out_forte_pull - out_forte_nopull
    assert np.max(np.abs(diff)) > 1e-4, (
        "Nonlinear magnetic pull must modulate signal on forte excursions"
    )


def test_excursion_dependent_touch_spectral_tilt():
    """Verify displacement-domain touch spectral tilt dynamically expands high-end brightness on forte strikes."""
    sr = 48000
    vsat = 0.4
    tau_touch = 0.18

    # 1. Forte transient signal (amplitude 0.85 >> vsat)
    t = np.linspace(0, 0.1, int(sr * 0.1), endpoint=False)
    forte_in = (0.85 * np.sin(2.0 * np.pi * 200.0 * t)).astype(np.float32)

    out_notilt = apply_oversampled_saturation(forte_in, vsat=vsat, tau_touch=0.0)
    out_tilt = apply_oversampled_saturation(forte_in, vsat=vsat, tau_touch=tau_touch)

    # High frequencies (> 1 kHz) should have higher energy with touch spectral tilt engaged
    fft_notilt = np.abs(np.fft.rfft(out_notilt))
    fft_tilt = np.abs(np.fft.rfft(out_tilt))
    freqs = np.fft.rfftfreq(len(forte_in), 1.0 / sr)
    hf_mask = (freqs > 1000.0) & (freqs < 8000.0)

    hf_energy_notilt = np.sum(fft_notilt[hf_mask] ** 2)
    hf_energy_tilt = np.sum(fft_tilt[hf_mask] ** 2)
    assert hf_energy_tilt > hf_energy_notilt, (
        "Touch spectral tilt must increase harmonic excitation on forte strikes"
    )

    # 2. Quiet signal (amplitude 0.05 << vsat): linear small-signal bypass
    quiet_in = (0.05 * np.sin(2.0 * np.pi * 200.0 * t)).astype(np.float32)
    out_quiet_notilt = apply_oversampled_saturation(quiet_in, vsat=vsat, tau_touch=0.0)
    out_quiet_tilt = apply_oversampled_saturation(quiet_in, vsat=vsat, tau_touch=tau_touch)
    assert np.allclose(out_quiet_notilt, out_quiet_tilt, atol=1e-5), (
        "Small signals must be identical"
    )


def test_conformal_geometric_clearance_asymmetry():
    """Verify conformal geometric clearance asymmetry creates proximity growl on positive excursions and linear bypass on small signals."""
    sr = 48000
    vsat = 0.45
    kappa_geom = 0.22

    t = np.linspace(0, 0.1, int(sr * 0.1), endpoint=False)

    # 1. Forte signal: positive excursions diverge as string approaches pole piece
    forte_in = (0.75 * np.sin(2.0 * np.pi * 120.0 * t)).astype(np.float32)
    out_base = apply_oversampled_saturation(forte_in, vsat=vsat, kappa_geom=0.0)
    out_geom = apply_oversampled_saturation(forte_in, vsat=vsat, kappa_geom=kappa_geom)

    # Positive peak should be pulled higher (proximity field divergence)
    pos_max_base = np.max(out_base)
    pos_max_geom = np.max(out_geom)
    assert pos_max_geom > pos_max_base, (
        "Positive excursion must increase due to pole proximity divergence"
    )

    # Must be bounded, no NaNs
    assert not np.any(np.isnan(out_geom))
    assert np.max(np.abs(out_geom)) < 2.0

    # 2. Small signal (<= 0.10): exact linear bypass
    quiet_in = (0.05 * np.sin(2.0 * np.pi * 120.0 * t)).astype(np.float32)
    out_q_base = apply_oversampled_saturation(quiet_in, vsat=vsat, kappa_geom=0.0)
    out_q_geom = apply_oversampled_saturation(quiet_in, vsat=vsat, kappa_geom=kappa_geom)
    assert np.allclose(out_q_base, out_q_geom, atol=1e-5), (
        "Small signal must linearly bypass geometric clearance"
    )


def test_dynamic_steinmetz_ac_core_loss():
    """Verify Steinmetz AC loss damps high-frequency flux transients during hard attack plucks."""
    vsat = 0.40
    k_stein = 0.035
    n = 1024
    # Fast forte transient with rapid high-frequency displacement changes (high dB/dt)
    t = np.linspace(0, 0.02, n, endpoint=False)
    x_transient = (0.80 * np.sin(2.0 * np.pi * 800.0 * t)).astype(np.float64)
    env = np.full(n, 0.80, dtype=np.float64)

    out_nostein = _lenz_velocity_drag_core(
        x_transient,
        env,
        vsat,
        k_sag=0.05,
        alpha_c=0.1,
        k_eddy=0.0,
        beta_curv=0.0,
        k_pull=0.0,
        k_stein=0.0,
    )
    out_stein = _lenz_velocity_drag_core(
        x_transient,
        env,
        vsat,
        k_sag=0.05,
        alpha_c=0.1,
        k_eddy=0.0,
        beta_curv=0.0,
        k_pull=0.0,
        k_stein=k_stein,
    )

    # Steinmetz loss adds high-frequency damping on rapid flux changes
    rms_nostein = np.sqrt(np.mean(out_nostein**2))
    rms_stein = np.sqrt(np.mean(out_stein**2))
    assert rms_stein < rms_nostein, "Steinmetz loss must damp high dB/dt transient flux spikes"
    assert not np.any(np.isnan(out_stein))


def test_register_dependent_string_pull():
    """Verify register-dependent weighting: low register fundamental experiences stronger pull damping than high register."""
    sr = 48000
    vsat = 0.4
    k_pull = 0.05
    k_sag = 0.02
    t = np.linspace(0, 0.05, int(sr * 0.05), endpoint=False)

    # 1. Low register note (40 Hz fundamental) + upper harmonic clank
    low_note = (
        0.80 * np.sin(2.0 * np.pi * 40.0 * t) + 0.30 * np.sin(2.0 * np.pi * 1200.0 * t)
    ).astype(np.float64)
    env_low = np.full_like(low_note, 0.80)

    out_low_nopull = _lenz_velocity_drag_core(
        low_note, env_low, vsat, k_sag=k_sag, alpha_c=0.1, k_pull=0.0
    )
    out_low_pull = _lenz_velocity_drag_core(
        low_note, env_low, vsat, k_sag=k_sag, alpha_c=0.1, k_pull=k_pull
    )
    diff_low = np.max(np.abs(out_low_pull - out_low_nopull))

    # 2. High register note (400 Hz fundamental) + upper harmonic clank
    high_note = (
        0.80 * np.sin(2.0 * np.pi * 400.0 * t) + 0.30 * np.sin(2.0 * np.pi * 1200.0 * t)
    ).astype(np.float64)
    env_high = np.full_like(high_note, 0.80)

    out_high_nopull = _lenz_velocity_drag_core(
        high_note, env_high, vsat, k_sag=k_sag, alpha_c=0.1, k_pull=0.0
    )
    out_high_pull = _lenz_velocity_drag_core(
        high_note, env_high, vsat, k_sag=k_sag, alpha_c=0.1, k_pull=k_pull
    )
    diff_high = np.max(np.abs(out_high_pull - out_high_nopull))

    # Low register has higher |x_low| so w_reg is higher, resulting in greater pull modulation
    assert diff_low > diff_high, (
        f"Low register pull diff ({diff_low:.4f}) must exceed high register ({diff_high:.4f})"
    )


def test_dynamic_reluctance_inductance_modulation():
    """Verify dynamic reluctance inductance modulation (lambda_L) produces dynamic phase lag on forte attacks."""
    sr = 48000
    t = np.linspace(0, 0.05, int(sr * 0.05), endpoint=False)

    # 1. Small signal linearity (peak <= 0.10): must be bit-exact linear bypass
    small_sig = (0.05 * np.sin(2.0 * np.pi * 200.0 * t)).astype(np.float32)
    out_active = apply_oversampled_saturation(small_sig, vsat=0.5, lambda_L=0.0)
    out_alnico = apply_oversampled_saturation(small_sig, vsat=0.5, lambda_L=0.05)
    np.testing.assert_array_equal(out_active, small_sig)
    np.testing.assert_array_equal(out_alnico, small_sig)

    # 2. Large signal transient (peak = 0.70 > vsat = 0.50): lambda_L modulates transient clank
    forte_sig = (
        0.70 * np.sin(2.0 * np.pi * 100.0 * t) + 0.30 * np.sin(2.0 * np.pi * 2500.0 * t)
    ).astype(np.float32)
    out_nolin = apply_oversampled_saturation(forte_sig, vsat=0.5, lambda_L=0.0)
    out_mod = apply_oversampled_saturation(forte_sig, vsat=0.5, lambda_L=0.05)
    diff = float(np.max(np.abs(out_mod - out_nolin)))
    assert diff > 1e-4, "Reluctance inductance modulation must engage on forte excursions"


def test_electromechanical_back_emf_braking():
    """Verify electromechanical back-EMF string braking (k_emf) compresses sharp transient peaks on passive pickups."""
    sr = 48000
    t = np.linspace(0, 0.05, int(sr * 0.05), endpoint=False)

    # 1. Small signal: bit-exact linear identity
    small_sig = (0.08 * np.sin(2.0 * np.pi * 150.0 * t)).astype(np.float32)
    out_no_emf = apply_oversampled_saturation(small_sig, vsat=0.5, k_emf=0.0)
    out_emf = apply_oversampled_saturation(small_sig, vsat=0.5, k_emf=0.04)
    np.testing.assert_array_equal(out_no_emf, small_sig)
    np.testing.assert_array_equal(out_emf, small_sig)

    # 2. Large transient spike: back-EMF decelerates string velocity, adding dynamic drag
    forte_spike = (
        0.75 * np.sin(2.0 * np.pi * 80.0 * t) + 0.35 * np.sin(2.0 * np.pi * 3000.0 * t)
    ).astype(np.float32)
    out_no_emf = apply_oversampled_saturation(forte_spike, vsat=0.5, k_emf=0.0)
    out_emf = apply_oversampled_saturation(forte_spike, vsat=0.5, k_emf=0.04)
    diff = float(np.max(np.abs(out_emf - out_no_emf)))
    assert diff > 1e-4, "Back-EMF braking must dynamically engage on forte excursions"
    rms_no_emf = np.sqrt(np.mean(out_no_emf**2))
    rms_emf = np.sqrt(np.mean(out_emf**2))
    assert rms_emf <= rms_no_emf, "Back-EMF damping must reduce or maintain total energy"
