"""
Allomorph - Circuit Simulation & Audio Processing Engine
Applies aperture pre-filtering, dynamic magnetic saturation, Foster 2-stage
core eddy diffusion, and differential RLC transfer functions to 24-bit audio buffers.
"""

import functools
import math
import wave
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from allomorph.circuit.audio import (
    apply_prefilter_to_audio,
    find_default_input_audio,
)
from allomorph.circuit.parser import (
    MAGNET_PROPERTIES,
    CircuitModel,
    load_circuit,
)
from allomorph.circuit.saturation import (
    apply_oversampled_saturation,
)
from allomorph.circuit.schema import (
    HarnessControls,
    MagnetPropertiesConfig,
    SaturationConfig,
    SimulationConfig,
)
from allomorph.circuit.solver import (
    apply_magnet_properties_to_model,
    compute_circuit_transfer_functions,
    compute_differential_circuit_transfer_functions,
)
from allomorph.config.instruments import get_source_pickup, load_instrument
from allomorph.config.scales import REPO_ROOT
from allomorph.config.schema import InstrumentConfig, VoiceConfig
from allomorph.config.strings import get_instrument_string
from allomorph.config.voices import VOICES
from allomorph.dsp import (
    FREQS,
    NUM_TAPS,
    fft_convolve,
    synthesize_minimum_phase_fir,
)
from allomorph.physics import (
    compute_voice_prefilter_firs,
    is_voice_matching_source,
)

AUDIO_DIR = REPO_ROOT / "audio"
MODELS_DIR = REPO_ROOT / "models"
MODELS_FRONTENDS_DIR = MODELS_DIR / "frontends"
from allomorph.version import DSP_GENERATION

CANONICAL_SWEEP_PATH = AUDIO_DIR / "canonical" / f"canonical_sweep_v{DSP_GENERATION}.wav"
FRONTENDS_DIR = AUDIO_DIR / "frontends"
TARGETS_DIR = AUDIO_DIR / "targets"
INTERMEDIATE_TARGET_PEAK_DBFS = -1.5
INTERMEDIATE_TARGET_RMS_DBFS = -16.5
CALIBRATION_PEAK_CEILING = 0.9900  # -0.087 dBFS (matching optimal_bass_dry.wav calibration peak)


@functools.lru_cache(maxsize=4)
def _get_white_noise_vector(n: int) -> np.ndarray:
    """Generates and caches deterministic Gaussian white noise for Johnson-Nyquist thermal dither."""
    rng = np.random.RandomState(42)
    return rng.normal(0.0, 1.0, n).astype(np.float64)


def simulate_circuit_audio(
    input_audio: str | Path | np.ndarray,
    output_wav_path: str | Path,
    model: CircuitModel,
    prefilter_firs: Sequence[Any] | None = None,
    circuit_curves: Sequence[Any] | None = None,
    is_passive: bool = False,
    bypass_saturation: bool | None = None,
    normalize: str = "auto",
    target_dbfs: float | None = None,
    oversample: int = 2,
    displacement_weighting: bool = True,
    magnet_drag: bool = True,
    alpha: float = 0.20,
    alphas: Sequence[float] | None = None,
    alpha3: float = 0.08,
    alpha3s: Sequence[float] | None = None,
    eta_hyst: float = 0.06,
    eta_hysts: Sequence[float] | None = None,
    k_sag: float = 0.08,
    k_sags: Sequence[float] | None = None,
    k_eddy: float = 0.0,
    k_eddys: Sequence[float] | None = None,
    kappa_orbit: float = 0.0,
    kappa_orbits: Sequence[float] | None = None,
    beta_curv: float = 0.0,
    beta_curvs: Sequence[float] | None = None,
    k_pull: float = 0.0,
    k_pulls: Sequence[float] | None = None,
    tau_touch: float = 0.0,
    tau_touches: Sequence[float] | None = None,
    kappa_geom: float = 0.0,
    kappa_geoms: Sequence[float] | None = None,
    k_stein: float = 0.0,
    k_steins: Sequence[float] | None = None,
    k_emf: float = 0.0,
    k_emfs: Sequence[float] | None = None,
    lambda_L: float = 0.0,
    lambda_Ls: Sequence[float] | None = None,
    vol_pos: float | None = None,
    tone_pos: float | None = None,
    blend_pos: float | None = None,
    pot_taper: str | None = None,
    slew_limit: bool = True,
    f_slew: float = 16000.0,
    is_identity: bool = False,
    noise_dither: bool = True,
    vsat: float | None = None,
    vsats: Sequence[float] | None = None,
    dc_block: bool = True,
    max_samples: int | None = None,
    skip_identity: bool = False,
    saturation_config: SaturationConfig | None = None,
    harness_controls: HarnessControls | None = None,
):
    """
    Executes native Virtual Analog circuit simulation on audio.
    If prefilter_firs is provided, convolves input audio through acoustic aperture and
    scale-tension FIRs in memory first.
    For active instruments and differential softening conversions, applies anti-aliased
    oversampled soft-knee saturation with displacement-domain weighting, dynamic Lenz flux sag,
    dipole cubic proximity expansion, magnet-specific alpha asymmetry, and Dahl magnetic hysteresis.
    For matching passive source instruments or stiffer targets, bypasses forward saturation (to prevent
    double-compression) and applies regularized differential SPICE transfer functions (H_target / H_source).
    Applies sub-audible DC-blocking high-pass filtering (8.0 Hz) to eliminate DC offset before
    feeding downstream high-gain overdrive stages.
    Automatically normalizes output level based on the input sweep's dBFS (or explicit target_dbfs).
    Writes canonical 24-bit 48 kHz mono audio.
    """
    if harness_controls is not None:
        vol_pos = harness_controls.vol_pos
        tone_pos = harness_controls.tone_pos
        blend_pos = harness_controls.blend_pos
        pot_taper = harness_controls.pot_taper

    if saturation_config is not None:
        vsat = saturation_config.vsat if vsat is None else vsat
        alpha = saturation_config.alpha
        alpha3 = saturation_config.alpha3
        eta_hyst = saturation_config.eta_hyst
        k_sag = saturation_config.k_sag
        k_eddy = saturation_config.k_eddy
        kappa_orbit = saturation_config.kappa_orbit
        beta_curv = saturation_config.beta_curv
        k_pull = saturation_config.k_pull
        tau_touch = saturation_config.tau_touch
        kappa_geom = saturation_config.kappa_geom
        k_stein = saturation_config.k_stein
        k_emf = saturation_config.k_emf
        lambda_L = saturation_config.lambda_L
        slew_limit = saturation_config.slew_limit
        f_slew = saturation_config.f_slew
        oversample = saturation_config.oversample
        displacement_weighting = saturation_config.displacement_weighting
        magnet_drag = saturation_config.magnet_drag

    import pedalboard
    from pedalboard.io import AudioFile

    if isinstance(input_audio, (str, Path)):
        with AudioFile(str(input_audio)) as f:
            num_frames = min(f.frames, max_samples) if max_samples else f.frames
            audio = f.read(num_frames)
            sr = int(f.samplerate)
    elif isinstance(input_audio, np.ndarray):
        audio = (
            input_audio[:, :max_samples]
            if input_audio.ndim > 1
            else input_audio[:max_samples]
            if max_samples
            else input_audio
        )
        sr = 48000
    else:
        raise TypeError(f"Unsupported input_audio type: {type(input_audio)}")

    if bypass_saturation is None:
        bypass_saturation = is_passive

    # Capture input sweep baseline levels before filtering
    in_mono = audio[0] if audio.ndim > 1 else audio
    in_peak = float(np.max(np.abs(in_mono)))
    in_rms = float(np.sqrt(np.mean(in_mono**2)))
    in_rms_db = 20.0 * math.log10(max(in_rms, 1e-9))

    can_fuse_stages = bool(bypass_saturation and prefilter_firs is not None)

    if can_fuse_stages:
        pass
    elif prefilter_firs is not None:
        audio = apply_prefilter_to_audio(audio, sr, prefilter_firs)
    elif not bypass_saturation and in_peak > 0.10:
        target_drive_peak = min(in_peak * 0.687, 0.70)
        audio = (audio / max(in_peak, 1e-9)) * target_drive_peak

    if (
        vol_pos is not None
        or tone_pos is not None
        or blend_pos is not None
        or pot_taper is not None
    ):
        model.apply_pot_positions(
            vol_pos=vol_pos,
            tone_pos=tone_pos,
            blend_pos=blend_pos,
            pot_taper=pot_taper,
        )

    if circuit_curves is not None:
        mag_curves = circuit_curves
    else:
        mag_curves = compute_circuit_transfer_functions(model, freqs=FREQS)

    channel_outputs = []

    for ch_idx, mag_curve in enumerate(mag_curves):
        if audio.ndim > 1:
            in_ch = audio[ch_idx] if audio.shape[0] > ch_idx else audio[0]
        else:
            if ch_idx == 1 and model.topology in ["parallel", "series"]:
                in_ch = audio * 0.75
            else:
                in_ch = audio

        if can_fuse_stages and prefilter_firs is not None:
            p_fir = prefilter_firs[ch_idx] if ch_idx < len(prefilter_firs) else prefilter_firs[0]
            c_fir = np.array(
                synthesize_minimum_phase_fir(mag_curve, num_taps=NUM_TAPS, normalize=False),
                dtype=np.float32,
            )
            # Convolve p_fir and c_fir into fused compound impulse response
            n_fir_fft = 1 << (len(p_fir) + len(c_fir) - 1).bit_length()
            fused_fir = np.fft.irfft(
                np.fft.rfft(p_fir, n_fir_fft) * np.fft.rfft(c_fir, n_fir_fft),
                n_fir_fft,
            )[: (len(p_fir) + len(c_fir) - 1)].astype(np.float32)

            # High-speed FFT convolution of input with compound FIR
            out_ch = fft_convolve(in_ch, fused_fir, mode="causal")[: len(in_ch)].astype(
                np.float32
            )
            channel_outputs.append(out_ch)
            continue

        # Dynamic magnetic saturation: bypassed when linear or already physically saturated
        if bypass_saturation:
            in_dyn = in_ch.copy().astype(np.float32)
        else:
            if vsats is not None and len(vsats) > ch_idx:
                ch_vsat = vsats[ch_idx]
            elif vsat is not None:
                ch_vsat = vsat
            elif model.topology in ["parallel", "series"]:
                ch_vsat = model.vsat_n if ch_idx == 0 else model.vsat_b
            else:
                ch_vsat = model.vsat

            ch_alpha = (
                alphas[ch_idx]
                if (isinstance(alphas, (list, tuple)) and len(alphas) > ch_idx)
                else alpha
            )
            ch_alpha3 = (
                alpha3s[ch_idx]
                if (isinstance(alpha3s, (list, tuple)) and len(alpha3s) > ch_idx)
                else alpha3
            )
            ch_eta = (
                eta_hysts[ch_idx]
                if (isinstance(eta_hysts, (list, tuple)) and len(eta_hysts) > ch_idx)
                else eta_hyst
            )
            ch_sag = (
                k_sags[ch_idx]
                if (isinstance(k_sags, (list, tuple)) and len(k_sags) > ch_idx)
                else k_sag
            )
            ch_eddy = (
                k_eddys[ch_idx]
                if (isinstance(k_eddys, (list, tuple)) and len(k_eddys) > ch_idx)
                else k_eddy
            )
            ch_orbit = (
                kappa_orbits[ch_idx]
                if (isinstance(kappa_orbits, (list, tuple)) and len(kappa_orbits) > ch_idx)
                else kappa_orbit
            )
            ch_beta = (
                beta_curvs[ch_idx]
                if (isinstance(beta_curvs, (list, tuple)) and len(beta_curvs) > ch_idx)
                else beta_curv
            )
            ch_pull = (
                k_pulls[ch_idx]
                if (isinstance(k_pulls, (list, tuple)) and len(k_pulls) > ch_idx)
                else k_pull
            )
            ch_touch = (
                tau_touches[ch_idx]
                if (isinstance(tau_touches, (list, tuple)) and len(tau_touches) > ch_idx)
                else tau_touch
            )
            ch_geom = (
                kappa_geoms[ch_idx]
                if (isinstance(kappa_geoms, (list, tuple)) and len(kappa_geoms) > ch_idx)
                else kappa_geom
            )
            ch_stein = (
                k_steins[ch_idx]
                if (isinstance(k_steins, (list, tuple)) and len(k_steins) > ch_idx)
                else k_stein
            )
            ch_emf = (
                k_emfs[ch_idx]
                if (isinstance(k_emfs, (list, tuple)) and len(k_emfs) > ch_idx)
                else k_emf
            )
            ch_lambda = (
                lambda_Ls[ch_idx]
                if (isinstance(lambda_Ls, (list, tuple)) and len(lambda_Ls) > ch_idx)
                else lambda_L
            )

            if (
                ch_vsat >= 10.0
                and ch_alpha <= 0.001
                and ch_alpha3 <= 0.001
                and ch_eta <= 0.001
                and ch_sag <= 0.001
                and ch_eddy <= 0.001
                and ch_orbit <= 0.001
                and ch_beta <= 0.001
                and ch_pull <= 0.001
                and ch_touch <= 0.001
                and ch_geom <= 0.001
                and ch_stein <= 0.001
                and ch_emf <= 0.001
                and ch_lambda <= 0.001
            ):
                in_dyn = in_ch.copy().astype(np.float32)
            else:
                in_dyn = apply_oversampled_saturation(
                    in_ch,
                    vsat=ch_vsat,
                    alpha=ch_alpha,
                    alpha3=ch_alpha3,
                    eta_hyst=ch_eta,
                    k_sag=ch_sag,
                    k_eddy=ch_eddy,
                    kappa_orbit=ch_orbit,
                    beta_curv=ch_beta,
                    k_pull=ch_pull,
                    tau_touch=ch_touch,
                    kappa_geom=ch_geom,
                    k_stein=ch_stein,
                    k_emf=ch_emf,
                    lambda_L=ch_lambda,
                    slew_limit=slew_limit,
                    f_slew=f_slew,
                    oversample=oversample,
                    displacement_weighting=displacement_weighting,
                    magnet_drag=magnet_drag,
                )

        # Synthesize minimum-phase causal impulse response
        fir = np.array(
            synthesize_minimum_phase_fir(mag_curve, num_taps=NUM_TAPS, normalize=False),
            dtype=np.float32,
        )

        # High-speed FFT block convolution
        out_ch = fft_convolve(in_dyn, fir, mode="causal")[: len(in_ch)].astype(np.float32)
        channel_outputs.append(out_ch)

    # Sum all pickup contributions
    if len(channel_outputs) > 1 and prefilter_firs is not None and len(prefilter_firs) > 1:
        peaks = [int(np.argmax(np.abs(fir))) for fir in prefilter_firs]
        delta_samples = max(peaks) - min(peaks) if len(peaks) > 1 else 0
        if delta_samples > 0:
            # Multi-string wave dispersion cross-coherence decay (Guardrail 5.1.2)
            # Evaluated analytically in frequency domain to yield a causal, minimum-phase correction filter.
            # Strictly avoids non-causal circular FFT phase division or audio-dependent noise smearing.
            N_spec = 4096
            f_bins_spec = np.fft.rfftfreq(N_spec, 1.0 / sr)
            H_chs = []
            for i in range(len(channel_outputs)):
                fir_p = np.array(prefilter_firs[i], dtype=np.float32)
                m_curve = mag_curves[i] if i < len(mag_curves) else mag_curves[0]
                fir_c = np.array(
                    synthesize_minimum_phase_fir(m_curve, num_taps=NUM_TAPS, normalize=False),
                    dtype=np.float32,
                )
                H_p = np.fft.rfft(fir_p, N_spec)
                H_c = np.fft.rfft(fir_c, N_spec)
                H_chs.append(H_p * H_c)

            P_coh = np.abs(np.sum(H_chs, axis=0)) ** 2
            P_incoh = np.sum(np.abs(H_chs) ** 2, axis=0)

            delta_tau = delta_samples / float(sr)
            f_notch = 1.0 / (2.0 * delta_tau)
            f_mid = 1.35 * f_notch
            f_sigma = max(0.35 * f_notch, 1.0)
            gamma = 0.5 * (1.0 - np.tanh((f_bins_spec - f_mid) / f_sigma))
            M_blend = np.sqrt(gamma * P_coh + (1.0 - gamma) * P_incoh)

            H_coh = np.sum(H_chs, axis=0)
            mag_coh = np.abs(H_coh)
            H_spatial = M_blend / np.maximum(mag_coh, 1e-4)

            fir_spatial = np.array(
                synthesize_minimum_phase_fir(H_spatial, num_taps=1024, normalize=False),
                dtype=np.float32,
            )
            raw_sum = np.sum(channel_outputs, axis=0)
            out_total = fft_convolve(raw_sum, fir_spatial, mode="causal")[
                : len(channel_outputs[0])
            ].astype(np.float32)
        else:
            out_total = np.sum(channel_outputs, axis=0)
    else:
        out_total = np.sum(channel_outputs, axis=0)

    # Sub-Audible DC-Blocking High-Pass Filter (fc ≈ 8.0 Hz)
    if dc_block and in_peak > 0.10 and (np.min(in_mono) < 0.0):
        hp = pedalboard.HighpassFilter(cutoff_frequency_hz=8.0)
        out_total = hp(out_total[np.newaxis, :], sr)[0]
        out_total = out_total - float(np.mean(out_total))

    # Passive RLC-Shaped Johnson-Nyquist Thermal Noise Dither (-108 dBFS)
    if noise_dither and in_peak > 0.10 and (not is_identity):
        n_sig_d = len(out_total)
        white_noise = _get_white_noise_vector(n_sig_d)
        avg_mag = (
            np.mean(mag_curves, axis=0) if isinstance(mag_curves, (list, tuple)) else mag_curves
        )
        n_dither_taps = 512
        dither_fir = np.array(
            synthesize_minimum_phase_fir(avg_mag, num_taps=n_dither_taps, normalize=True),
            dtype=np.float64,
        )
        colored_noise = fft_convolve(white_noise, dither_fir, mode="causal")[:n_sig_d]
        colored_rms = max(float(np.sqrt(np.mean(colored_noise**2))), 1e-9)
        target_dither_rms = 10.0 ** (-108.0 / 20.0)
        dither = (colored_noise / colored_rms) * target_dither_rms
        out_total = out_total + dither.astype(np.float32)

    raw_peak = float(np.max(np.abs(out_total)))
    raw_rms = float(np.sqrt(np.mean(out_total**2)))
    raw_rms_db = 20.0 * math.log10(max(raw_rms, 1e-9))

    should_normalize = (
        normalize in ["auto", "rms", "peak"]
        and in_peak > 0.10
        and in_rms > 0.005
        and (np.min(in_mono) < 0.0)
    )

    if should_normalize:
        norm_mode = "rms" if normalize == "auto" else normalize
        if norm_mode == "rms":
            target_rms = 10.0 ** (target_dbfs / 20.0) if target_dbfs is not None else in_rms
            if raw_rms > 1e-9:
                scale = target_rms / raw_rms
                out_total = out_total * scale
        elif norm_mode == "peak":
            target_peak = (
                10.0 ** (target_dbfs / 20.0) if target_dbfs is not None else min(in_peak, 0.988)
            )
            if raw_peak > 1e-9:
                scale = target_peak / raw_peak
                out_total = out_total * scale

    # True-Peak Safety Headroom: strictly prevent 0 dBFS full-scale clipping by bounding
    # to the calibration sweep peak ceiling (0.9900 / -0.09 dBFS)
    max_val = float(np.max(np.abs(out_total)))
    if max_val > CALIBRATION_PEAK_CEILING:
        out_total = out_total * (CALIBRATION_PEAK_CEILING / max_val)

    final_peak = float(np.max(np.abs(out_total)))
    final_rms = float(np.sqrt(np.mean(out_total**2)))
    final_peak_db = 20.0 * math.log10(max(final_peak, 1e-9))
    final_rms_db = 20.0 * math.log10(max(final_rms, 1e-9))

    if should_normalize:
        gain_applied_db = 20.0 * math.log10(max(final_rms / max(raw_rms, 1e-9), 1e-9))
        tgt_desc = (
            f"{target_dbfs:.1f} dBFS"
            if target_dbfs is not None
            else f"{in_rms_db:.1f} dBFS (Input Sweep)"
        )
        print(
            f"     Level Normalized: RMS {raw_rms_db:.1f} -> {final_rms_db:.1f} dBFS ({gain_applied_db:+.1f} dB, target: {tgt_desc}) | Peak: {final_peak_db:.1f} dBFS"
        )

    output_wav_path = Path(output_wav_path)
    if skip_identity and (
        is_identity
        or (
            len(out_total) == len(in_mono)
            and np.max(np.abs(in_mono)) > 1e-4
            and np.allclose(out_total, in_mono, atol=1e-6, rtol=1e-4)
        )
    ):
        if output_wav_path.exists():
            output_wav_path.unlink(missing_ok=True)
        print(
            f"     [Identity Bypass] Output is bit-for-bit identical to input. Skipping file: {output_wav_path.name}"
        )
        return False

    output_wav_path.parent.mkdir(parents=True, exist_ok=True)

    # Write 24-bit 48 kHz mono PCM WAV
    with wave.open(str(output_wav_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(3)  # 24-bit
        wf.setframerate(sr)
        int24_max = 8388607.0
        scaled = np.clip(out_total * int24_max, -8388608.0, 8388607.0).astype(np.int32)
        raw_bytes = scaled.astype("<i4").view(np.uint8).reshape(-1, 4)[:, :3].tobytes()
        wf.writeframes(raw_bytes)

    return True


def simulate_voice(
    voice_id: str,
    input_wav: str | Path | None = None,
    output_wav: str | Path | None = None,
    instrument: InstrumentConfig | str | Path | None = "30in",
    pickup: str | None = None,
    tier: str | None = None,
    prefiltered: bool = False,
    cir_path: str | Path | None = None,
    normalize: str = "auto",
    target_dbfs: float | None = None,
    oversample: int = 2,
    displacement_weighting: bool = True,
    magnet_drag: bool = True,
    alpha: float | None = None,
    alpha3: float | None = None,
    eta_hyst: float | None = None,
    k_sag: float | None = None,
    k_eddy: float | None = None,
    kappa_orbit: float | None = None,
    beta_curv: float | None = None,
    k_pull: float | None = None,
    tau_touch: float | None = None,
    kappa_geom: float | None = None,
    k_stein: float | None = None,
    k_emf: float | None = None,
    lambda_L: float | None = None,
    vol_pos: float | None = None,
    tone_pos: float | None = None,
    blend_pos: float | None = None,
    pot_taper: str | None = None,
    cable_pf: float | None = None,
    slew_limit: bool = True,
    f_slew: float = 16000.0,
    noise_dither: bool = True,
    eddy_diffusion: bool = True,
    dc_block: bool = True,
    max_samples: int | None = None,
    skip_identity: bool = False,
    config: SimulationConfig | None = None,
):
    """
    Simulates a target voice digital twin using the native Virtual Analog engine.
    By default, applies acoustic aperture pre-filtering and circuit simulation
    end-to-end in memory from raw calibration audio.
    Outputs are saved by default to audio/{instrument_id}/out_{voice_id}.wav.
    """
    if config is not None:
        input_wav = config.input_wav if input_wav is None else input_wav
        output_wav = config.output_wav if output_wav is None else output_wav
        instrument = (
            config.instrument
            if instrument == "30in" and config.instrument is not None
            else instrument
        )
        pickup = config.pickup if pickup is None else pickup
        tier = config.tier if tier is None else tier
        prefiltered = config.prefiltered if prefiltered is False else prefiltered
        cir_path = config.cir_path if cir_path is None else cir_path
        normalize = config.normalize if normalize == "auto" else normalize
        target_dbfs = config.target_dbfs if target_dbfs is None else target_dbfs
        oversample = config.oversample
        displacement_weighting = config.displacement_weighting
        magnet_drag = config.magnet_drag
        alpha = config.alpha if alpha is None else alpha
        alpha3 = config.alpha3 if alpha3 is None else alpha3
        eta_hyst = config.eta_hyst if eta_hyst is None else eta_hyst
        k_sag = config.k_sag if k_sag is None else k_sag
        k_eddy = config.k_eddy if k_eddy is None else k_eddy
        kappa_orbit = config.kappa_orbit if kappa_orbit is None else kappa_orbit
        beta_curv = config.beta_curv if beta_curv is None else beta_curv
        k_pull = config.k_pull if k_pull is None else k_pull
        tau_touch = config.tau_touch if tau_touch is None else tau_touch
        kappa_geom = config.kappa_geom if kappa_geom is None else kappa_geom
        k_stein = config.k_stein if k_stein is None else k_stein
        k_emf = config.k_emf if k_emf is None else k_emf
        lambda_L = config.lambda_L if lambda_L is None else lambda_L
        vol_pos = config.vol_pos if vol_pos is None else vol_pos
        tone_pos = config.tone_pos if tone_pos is None else tone_pos
        blend_pos = config.blend_pos if blend_pos is None else blend_pos
        pot_taper = config.pot_taper if pot_taper is None else pot_taper
        cable_pf = config.cable_pf if cable_pf is None else cable_pf
        slew_limit = config.slew_limit
        f_slew = config.f_slew
        noise_dither = config.noise_dither
        eddy_diffusion = config.eddy_diffusion
        dc_block = config.dc_block
        max_samples = config.max_samples if max_samples is None else max_samples
        skip_identity = config.skip_identity
    if cir_path:
        model = load_circuit(cir_path)
        vcfg = VOICES.get(voice_id) or VoiceConfig(
            id=voice_id,
            name=voice_id,
            topology="single",
            description="",
            fr=3000.0,
            Q=1.5,
            circuit=model,
        )
    else:
        if voice_id not in VOICES:
            raise KeyError(
                f"Target voice '{voice_id}' not found in voice catalog. "
                f"Available voices: {list(VOICES.keys())}"
            )
        vcfg = VOICES[voice_id]
        model = load_circuit(vcfg.circuit)

    inst_cfg = (
        instrument
        if isinstance(instrument, InstrumentConfig)
        else load_instrument("30in" if instrument is None else instrument)
    )
    inst_id = inst_cfg.id
    inst_audio_dir = AUDIO_DIR / inst_id

    if not input_wav or not Path(input_wav).exists():
        found = find_default_input_audio()
        if found:
            in_wav_path = found
        elif (inst_audio_dir / f"aperture_{voice_id}.wav").exists():
            in_wav_path = inst_audio_dir / f"aperture_{voice_id}.wav"
            prefiltered = True
        else:
            raise FileNotFoundError(
                f"Input audio '{input_wav}' not found, and no standard calibration audio (optimal_bass_dry.wav) was detected."
            )
    else:
        in_wav_path = Path(input_wav)
        if in_wav_path.name.startswith("aperture_") and not prefiltered:
            prefiltered = True
            print(
                f"  [Auto-detected pre-filtered aperture input: {in_wav_path.name} -> setting prefiltered=True]"
            )

    out_wav: Path = inst_audio_dir / f"out_{voice_id}.wav" if not output_wav else Path(output_wav)

    apply_magnet_properties_to_model(model, vcfg, eddy_diffusion=eddy_diffusion)
    if (
        vol_pos is not None
        or tone_pos is not None
        or blend_pos is not None
        or pot_taper is not None
    ):
        model.apply_pot_positions(
            vol_pos=vol_pos,
            tone_pos=tone_pos,
            blend_pos=blend_pos,
            pot_taper=pot_taper,
        )
    if cable_pf is not None:
        model.Ccable = cable_pf * 1e-12 if cable_pf > 1e-6 else cable_pf

    # Dynamic bridge compliance scaling based on source string pluck excursion
    if "upright_bridge_transducer" in voice_id:
        src_string = get_instrument_string(inst_cfg)
        excursion = float(src_string.pluck_excursion_factor or 1.0)
        if excursion > 0:
            model.vsat = round(model.vsat / excursion, 3)

    # Resolve magnet-specific saturation profile and Dahl hysteresis coupling
    mag_type_global = vcfg.magnet_type or "alnico_v"
    global_props = MAGNET_PROPERTIES.get(mag_type_global, MAGNET_PROPERTIES["alnico_v"])

    voice_alpha = (
        alpha if alpha is not None else (vcfg.alpha if vcfg.alpha != 0.0 else global_props.alpha)
    )
    voice_alpha3 = (
        alpha3
        if alpha3 is not None
        else (vcfg.alpha3 if vcfg.alpha3 is not None else global_props.alpha3)
    )
    voice_eta = (
        eta_hyst
        if eta_hyst is not None
        else (vcfg.eta_hyst if vcfg.eta_hyst is not None else global_props.eta_hyst)
    )
    voice_sag = (
        k_sag
        if k_sag is not None
        else (vcfg.k_sag if vcfg.k_sag is not None else global_props.k_sag)
    )
    voice_eddy = (
        k_eddy
        if k_eddy is not None
        else (vcfg.k_eddy if vcfg.k_eddy is not None else global_props.k_eddy)
    )
    voice_orbit = (
        kappa_orbit
        if kappa_orbit is not None
        else (vcfg.kappa_orbit if vcfg.kappa_orbit is not None else global_props.kappa_orbit)
    )
    voice_beta = (
        beta_curv
        if beta_curv is not None
        else (vcfg.beta_curv if vcfg.beta_curv is not None else global_props.beta_curv)
    )
    voice_pull = (
        k_pull
        if k_pull is not None
        else (vcfg.k_pull if vcfg.k_pull is not None else global_props.k_pull)
    )
    voice_touch = (
        tau_touch
        if tau_touch is not None
        else (vcfg.tau_touch if vcfg.tau_touch is not None else global_props.tau_touch)
    )
    voice_geom = (
        kappa_geom
        if kappa_geom is not None
        else (vcfg.kappa_geom if vcfg.kappa_geom is not None else global_props.kappa_geom)
    )
    voice_stein = (
        k_stein
        if k_stein is not None
        else (vcfg.k_stein if vcfg.k_stein is not None else global_props.k_stein)
    )
    voice_emf = (
        k_emf
        if k_emf is not None
        else (vcfg.k_emf if vcfg.k_emf is not None else global_props.k_emf)
    )
    voice_lambda = (
        lambda_L
        if lambda_L is not None
        else (vcfg.lambda_L if vcfg.lambda_L is not None else global_props.lambda_L)
    )

    pickups_cfg = vcfg.pickups or []
    if len(pickups_cfg) > 1:
        voice_alphas = []
        voice_alpha3s = []
        voice_eta_hysts = []
        voice_k_sags = []
        voice_k_eddys = []
        voice_kappa_orbits = []
        voice_beta_curvs = []
        voice_k_pulls = []
        voice_tau_touches = []
        voice_kappa_geoms = []
        voice_k_steins = []
        voice_k_emfs = []
        voice_lambda_Ls = []
        for p in pickups_cfg:
            p_mag = p.magnet_type or mag_type_global
            p_props = MAGNET_PROPERTIES.get(p_mag, MAGNET_PROPERTIES["alnico_v"])
            voice_alphas.append(float(p.alpha) if p.alpha is not None else p_props.alpha)
            voice_alpha3s.append(p_props.alpha3)
            voice_eta_hysts.append(p_props.eta_hyst)
            voice_k_sags.append(p_props.k_sag)
            voice_k_eddys.append(p_props.k_eddy)
            voice_kappa_orbits.append(p_props.kappa_orbit)
            voice_beta_curvs.append(p_props.beta_curv)
            voice_k_pulls.append(p_props.k_pull)
            voice_tau_touches.append(p_props.tau_touch)
            voice_kappa_geoms.append(p_props.kappa_geom)
            voice_k_steins.append(p_props.k_stein)
            voice_k_emfs.append(p_props.k_emf)
            voice_lambda_Ls.append(p_props.lambda_L)
    else:
        voice_alphas = None
        voice_alpha3s = None
        voice_eta_hysts = None
        voice_k_sags = None
        voice_k_eddys = None
        voice_kappa_orbits = None
        voice_beta_curvs = None
        voice_k_pulls = None
        voice_tau_touches = None
        voice_kappa_geoms = None
        voice_k_steins = None
        voice_k_emfs = None
        voice_lambda_Ls = None

    is_passive = inst_cfg.electronics == "passive"
    is_spatial_match = is_voice_matching_source(inst_cfg, voice_id, vcfg)
    if pickup and pickup != "auto":
        pickups = inst_cfg.pickups
        if pickup not in pickups:
            raise KeyError(
                f"Pickup '{pickup}' not found on instrument '{inst_id}'. "
                f"Available pickups: {list(pickups.keys())}"
            )
        p_raw = pickups[pickup]
        src_pickup = p_raw.model_copy()
        src_pickup.id = pickup
    else:
        src_pickup = get_source_pickup(inst_cfg, voice_id)

    diff_curves = None
    if model.no_eq:
        diff_curves = [np.ones(len(FREQS), dtype=np.float64).tolist()]
    elif src_pickup.circuit is not None:
        src_model = load_circuit(src_pickup.circuit)
        apply_magnet_properties_to_model(src_model, src_pickup, eddy_diffusion=eddy_diffusion)
        diff_curves = compute_differential_circuit_transfer_functions(model, src_model, freqs=FREQS)
    elif is_passive:
        raise ValueError(
            f"Passive instrument '{inst_id}' pickup '{src_pickup.id or 'unknown'}' "
            f"does not define a '[circuit]' block. Passive source pickups require an explicit "
            f"circuit model for differential deconvolution."
        )

    has_source_circuit = diff_curves is not None
    is_circuit_match = bool(
        diff_curves is not None
        and len(diff_curves) > 0
        and np.allclose(diff_curves[0], 1.0, rtol=1e-3)
    )
    is_identity = is_spatial_match and (is_circuit_match if has_source_circuit else True)

    if skip_identity and is_identity:
        out_p = Path(out_wav)
        if out_p.exists():
            out_p.unlink(missing_ok=True)
        print(
            f"  [Identity Bypass] Target voice '{voice_id}' is bit-for-bit identical to source instrument '{inst_id}' "
            f"({src_pickup.name}). Skipping output file: {out_p.name}"
        )
        return False

    # Differential magnetic softening parameters
    if not is_passive:
        src_props = MAGNET_PROPERTIES["active"]
    else:
        src_mag = src_pickup.magnet_type
        if not src_mag and src_pickup.components:
            for c in src_pickup.components:
                if c.pickup and c.pickup in inst_cfg.pickups:
                    c_p = inst_cfg.pickups[c.pickup]
                    if c_p.magnet_type:
                        src_mag = c_p.magnet_type
                        break
        if not src_mag:
            raise KeyError(
                f"Passive pickup '{src_pickup.id or 'unknown'}' on instrument '{inst_id}' "
                f"does not specify 'magnet_type'. Available magnet types: {list(MAGNET_PROPERTIES.keys())}"
            )
        if src_mag not in MAGNET_PROPERTIES:
            raise KeyError(
                f"Unknown magnet type '{src_mag}' on pickup '{src_pickup.id or 'unknown'}'. "
                f"Available magnet types: {list(MAGNET_PROPERTIES.keys())}"
            )
        src_props = MAGNET_PROPERTIES[src_mag]

    src_alpha = src_props.alpha
    src_alpha3 = src_props.alpha3
    src_eta = src_props.eta_hyst
    src_sag = src_props.k_sag
    src_eddy = src_props.k_eddy
    src_orbit = src_props.kappa_orbit
    src_beta = src_props.beta_curv
    src_pull = src_props.k_pull
    src_touch = src_props.tau_touch
    src_geom = src_props.kappa_geom
    src_stein = src_props.k_stein
    src_emf = src_props.k_emf
    src_lambda = src_props.lambda_L
    src_vsat = src_props.vsat

    voice_props = MagnetPropertiesConfig(
        alpha=voice_alpha,
        alpha3=voice_alpha3,
        eta_hyst=voice_eta,
        k_sag=voice_sag,
        k_eddy=voice_eddy,
        kappa_orbit=voice_orbit,
        beta_curv=voice_beta,
        k_pull=voice_pull,
        tau_touch=voice_touch,
        kappa_geom=voice_geom,
        k_stein=voice_stein,
        k_emf=voice_emf,
        lambda_L=voice_lambda,
    )
    diff_props = voice_props.diff(src_props)

    tgt_vsat = model.vsat
    diff_alpha = diff_props.alpha
    diff_alpha3 = diff_props.alpha3
    diff_eta = diff_props.eta_hyst
    diff_sag = diff_props.k_sag
    diff_eddy = diff_props.k_eddy
    diff_orbit = diff_props.kappa_orbit
    diff_beta = diff_props.beta_curv
    diff_pull = diff_props.k_pull
    diff_touch = diff_props.tau_touch
    diff_geom = diff_props.kappa_geom
    diff_stein = diff_props.k_stein
    diff_emf = diff_props.k_emf
    diff_lambda = diff_props.lambda_L
    src_vsat = src_props.vsat

    if not is_passive:
        eff_vsat = tgt_vsat
    elif tgt_vsat < src_vsat:
        denom = 1.0 - min(0.85, tgt_vsat / src_vsat) + 0.15
        eff_vsat = tgt_vsat / denom
    else:
        eff_vsat = 10.0

    if voice_alphas and len(voice_alphas) > 1:
        eff_alphas = []
        eff_alpha3s = []
        eff_eta_hysts = []
        eff_k_sags = []
        eff_k_eddys = []
        eff_kappa_orbits = []
        eff_beta_curvs = []
        eff_k_pulls = []
        eff_tau_touches = []
        eff_kappa_geoms = []
        eff_k_steins = []
        eff_k_emfs = []
        eff_lambda_Ls = []
        eff_vsats = []
        for i in range(len(voice_alphas)):
            ch_a = max(voice_alphas[i] - src_alpha, 0.0)
            ch_a3 = max(voice_alpha3s[i] - src_alpha3, 0.0) if voice_alpha3s else diff_alpha3
            ch_eta = max(voice_eta_hysts[i] - src_eta, 0.0) if voice_eta_hysts else diff_eta
            ch_sag = max(voice_k_sags[i] - src_sag, 0.0) if voice_k_sags else diff_sag
            ch_eddy = max(voice_k_eddys[i] - src_eddy, 0.0) if voice_k_eddys else diff_eddy
            ch_orbit = (
                max(voice_kappa_orbits[i] - src_orbit, 0.0) if voice_kappa_orbits else diff_orbit
            )
            ch_beta = max(voice_beta_curvs[i] - src_beta, 0.0) if voice_beta_curvs else diff_beta
            ch_pull = max(voice_k_pulls[i] - src_pull, 0.0) if voice_k_pulls else diff_pull
            ch_touch = (
                max(voice_tau_touches[i] - src_touch, 0.0) if voice_tau_touches else diff_touch
            )
            ch_geom = max(voice_kappa_geoms[i] - src_geom, 0.0) if voice_kappa_geoms else diff_geom
            ch_stein = max(voice_k_steins[i] - src_stein, 0.0) if voice_k_steins else diff_stein
            ch_emf = max(voice_k_emfs[i] - src_emf, 0.0) if voice_k_emfs else diff_emf
            ch_lambda = (
                max(voice_lambda_Ls[i] - src_lambda, 0.0) if voice_lambda_Ls else diff_lambda
            )
            eff_alphas.append(ch_a)
            eff_alpha3s.append(ch_a3)
            eff_eta_hysts.append(ch_eta)
            eff_k_sags.append(ch_sag)
            eff_k_eddys.append(ch_eddy)
            eff_kappa_orbits.append(ch_orbit)
            eff_beta_curvs.append(ch_beta)
            eff_k_pulls.append(ch_pull)
            eff_tau_touches.append(ch_touch)
            eff_kappa_geoms.append(ch_geom)
            eff_k_steins.append(ch_stein)
            eff_k_emfs.append(ch_emf)
            eff_lambda_Ls.append(ch_lambda)

            ch_tgt_vsat = model.vsat_n if i == 0 else model.vsat_b
            if not is_passive:
                eff_vsats.append(ch_tgt_vsat)
            elif ch_tgt_vsat < src_vsat:
                denom = 1.0 - min(0.85, ch_tgt_vsat / src_vsat) + 0.15
                eff_vsats.append(ch_tgt_vsat / denom)
            else:
                eff_vsats.append(10.0)
        check_alpha = max(eff_alphas)
        check_eta = max(eff_eta_hysts)
        check_sag = max(eff_k_sags)
        check_vsat = min(eff_vsats)
    else:
        eff_alphas = None
        eff_alpha3s = None
        eff_eta_hysts = None
        eff_k_sags = None
        eff_k_eddys = None
        eff_kappa_orbits = None
        eff_beta_curvs = None
        eff_k_pulls = None
        eff_tau_touches = None
        eff_kappa_geoms = None
        eff_k_steins = None
        eff_k_emfs = None
        eff_lambda_Ls = None
        eff_vsats = None
        check_alpha = diff_alpha
        check_eta = diff_eta
        check_sag = diff_sag
        check_vsat = eff_vsat

    is_target_more_saturated = (
        (check_alpha > 0.02)
        or (check_eta > 0.01)
        or (check_sag > 0.01)
        or (diff_eddy > 0.01)
        or (diff_orbit > 0.01)
        or (diff_beta > 0.005)
        or (diff_pull > 0.005)
        or (diff_touch > 0.005)
        or (diff_geom > 0.01)
        or (diff_stein > 0.005)
        or (diff_emf > 0.005)
        or (diff_lambda > 0.005)
        or (check_vsat < src_vsat - 0.03)
    )

    if tier == "clean":
        diff_alpha = 0.0
        diff_alpha3 = 0.0
        diff_eta = 0.0
        diff_sag = 0.0
        diff_eddy = 0.0
        diff_orbit = 0.0
        diff_beta = 0.0
        diff_pull = 0.0
        diff_touch = 0.0
        diff_geom = 0.0
        diff_stein = 0.0
        diff_emf = 0.0
        diff_lambda = 0.0
        eff_vsat = 10.0
        should_soften = False
        bypass_saturation = True
    elif tier in ["standard", "std"]:
        diff_alpha = voice_alpha
        diff_alpha3 = voice_alpha3
        diff_eta = voice_eta
        diff_sag = voice_sag
        diff_eddy = voice_eddy
        diff_orbit = voice_orbit
        diff_beta = voice_beta
        diff_pull = voice_pull
        diff_touch = voice_touch
        diff_geom = voice_geom
        diff_stein = voice_stein
        diff_emf = voice_emf
        diff_lambda = voice_lambda
        eff_vsat = tgt_vsat
        should_soften = not is_identity
        bypass_saturation = not should_soften
    elif tier == "hotrod":
        diff_alpha = min(1.0, voice_alpha * 1.75)
        diff_alpha3 = min(0.5, voice_alpha3 * 1.75)
        diff_eta = min(0.3, voice_eta * 1.5)
        diff_sag = min(0.5, voice_sag * 1.5)
        diff_eddy = min(0.5, voice_eddy * 1.5)
        diff_orbit = min(0.3, voice_orbit * 1.5)
        diff_beta = min(0.2, voice_beta * 1.5)
        diff_pull = min(0.2, voice_pull * 1.5)
        diff_touch = min(0.2, voice_touch * 1.5)
        diff_geom = min(0.5, voice_geom * 1.5)
        diff_stein = min(0.2, voice_stein * 1.5)
        diff_emf = min(0.2, voice_emf * 1.5)
        diff_lambda = min(0.2, voice_lambda * 1.5)
        eff_vsat = max(0.20, tgt_vsat / 1.35)
        should_soften = not is_identity
        bypass_saturation = not should_soften
    else:
        # tier in ["dynamic", "dyn"] or tier is None
        if is_identity:
            should_soften = False
        elif not is_passive:
            should_soften = True
        else:
            should_soften = is_target_more_saturated
        bypass_saturation = not should_soften

    prefilter_firs = None
    if not prefiltered:
        prefilter_firs = compute_voice_prefilter_firs(
            voice_id, instrument=inst_cfg, src_pickup_key=src_pickup.id
        )
        if has_source_circuit:
            stage_desc = f"Acoustic Aperture + Differential Circuit Simulation ({'Passive' if is_passive else 'Active'} Source)"
        else:
            stage_desc = "Acoustic Aperture + Circuit Simulation"
    else:
        stage_desc = "Circuit Simulation (Pre-filtered Input)"

    samples_desc = f", Samples: {max_samples}" if max_samples is not None else ""
    cir_label = Path(cir_path).name if cir_path else f"{voice_id}.toml"
    print(
        f"  -> Simulating Native VA ({stage_desc}{samples_desc}): {cir_label} (Topology: {model.topology}, Source: {inst_id}, Soften: {'Yes' if should_soften else 'No'}, Alpha: {diff_alpha:.2f}, Alpha3: {diff_alpha3:.2f}, Eta: {diff_eta:.2f}, Sag: {diff_sag:.2f}, Eddy: {diff_eddy:.2f}, Orbit: {diff_orbit:.2f}, Beta: {diff_beta:.3f}, Pull: {diff_pull:.3f}, Touch: {diff_touch:.3f}, Geom: {diff_geom:.2f}, Stein: {diff_stein:.3f}, EMF: {diff_emf:.2f}, Lambda: {diff_lambda:.2f}, Vsat: {eff_vsat:.2f})..."
    )
    ok = simulate_circuit_audio(
        in_wav_path,
        out_wav,
        model,
        prefilter_firs=prefilter_firs,
        circuit_curves=diff_curves,
        bypass_saturation=bypass_saturation,
        is_passive=bypass_saturation,
        is_identity=is_identity,
        normalize=normalize,
        target_dbfs=target_dbfs,
        oversample=oversample,
        displacement_weighting=displacement_weighting,
        magnet_drag=magnet_drag,
        alpha=diff_alpha,
        alphas=eff_alphas,
        alpha3=diff_alpha3,
        alpha3s=eff_alpha3s,
        eta_hyst=diff_eta,
        eta_hysts=eff_eta_hysts,
        k_sag=diff_sag,
        k_sags=eff_k_sags,
        k_eddy=diff_eddy,
        k_eddys=eff_k_eddys,
        kappa_orbit=diff_orbit,
        kappa_orbits=eff_kappa_orbits,
        beta_curv=diff_beta,
        beta_curvs=eff_beta_curvs,
        k_pull=diff_pull,
        k_pulls=eff_k_pulls,
        tau_touch=diff_touch,
        tau_touches=eff_tau_touches,
        kappa_geom=diff_geom,
        kappa_geoms=eff_kappa_geoms,
        k_stein=diff_stein,
        k_steins=eff_k_steins,
        k_emf=diff_emf,
        k_emfs=eff_k_emfs,
        lambda_L=diff_lambda,
        lambda_Ls=eff_lambda_Ls,
        vol_pos=vol_pos,
        tone_pos=tone_pos,
        blend_pos=blend_pos,
        pot_taper=pot_taper,
        slew_limit=slew_limit,
        f_slew=f_slew,
        noise_dither=noise_dither,
        vsat=eff_vsat,
        vsats=eff_vsats,
        dc_block=dc_block,
        max_samples=max_samples,
        skip_identity=skip_identity,
    )
    if not ok:
        return False
    print(f"     Exported: {out_wav}")
    return True


def _simulate_voice_task(task_args: tuple[str, SimulationConfig]) -> bool:
    v, sim_cfg = task_args
    return simulate_voice(v, config=sim_cfg)
