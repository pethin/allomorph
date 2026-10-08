"""
Allomorph - Direct Unified Forward Simulation Engine (DSP Gen 4 / v0.4.0)
Convolves dry string excitation audio with authentic instrument physical aperture,
loaded RLC circuit, active preamps, and string mechanics into 24-bit PCM wet stems.
"""

import functools
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pedalboard

from allomorph.circuit.audio import find_default_input_audio
from allomorph.circuit.parser import MAGNET_PROPERTIES, CircuitModel, load_circuit
from allomorph.circuit.saturation import apply_oversampled_saturation
from allomorph.circuit.solver import (
    apply_magnet_properties_to_model,
    compute_active_preamp_eq,
    compute_active_preamp_transfer,
    compute_circuit_transfer_functions,
)
from allomorph.config.geometry import compute_effective_position, resolve_pickup_coils
from allomorph.config.instruments import INSTRUMENTS, load_instrument
from allomorph.config.scales import REPO_ROOT, resolve_scale_range
from allomorph.config.schema import InstrumentConfig, VoicingConfig
from allomorph.config.strings import STRINGS, get_instrument_string
from allomorph.dsp import (
    FREQS,
    NUM_TAPS,
    compute_lufs,
    compute_true_peak,
    compute_true_peak_dbfs,
    fft_convolve,
    read_wav,
    synthesize_minimum_phase_fir,
    write_wav_24bit,
)
from allomorph.physics import MEAN_BASS_F0
from allomorph.physics.aperture import (
    compute_displacement_proximity_shelf,
    compute_pickup_isolation_leveling,
    compute_saddle_boundary_coupling,
    numpy_pickup_acoustic_response,
)
from allomorph.physics.strings import (
    compute_differential_longitudinal_transfer,
    compute_differential_string_transfer,
)
from allomorph.version import (
    DSP_GENERATION,
    compute_file_sha256,
    is_wet_stem_valid,
    resolve_tri_part_version,
    write_manifest,
)

AUDIO_DIR = REPO_ROOT / "audio"
WET_AUDIO_DIR = AUDIO_DIR / "wet"
CALIBRATION_PEAK_CEILING = 0.9900  # -0.087 dBFS (matching optimal_bass_dry calibration ceiling)


@functools.lru_cache(maxsize=4)
def _get_white_noise_vector(n: int) -> np.ndarray:
    """Generates and caches deterministic Gaussian white noise for Johnson-Nyquist thermal dither."""
    rng = np.random.RandomState(42)
    return rng.normal(0.0, 1.0, n).astype(np.float64)


def _stft(x: np.ndarray, n_fft: int, hop_length: int) -> np.ndarray:
    x_pad = np.pad(x, (n_fft // 2, n_fft // 2), mode="constant")
    n_frames = 1 + (len(x_pad) - n_fft) // hop_length
    frames = np.lib.stride_tricks.as_strided(
        x_pad, shape=(n_frames, n_fft), strides=(x_pad.strides[0] * hop_length, x_pad.strides[0])
    )
    window = np.hanning(n_fft)
    return np.fft.rfft(frames * window, axis=1)


def _istft(X: np.ndarray, n_fft: int, hop_length: int, orig_len: int) -> np.ndarray:
    n_frames = X.shape[0]
    x_out = np.zeros(n_frames * hop_length + n_fft, dtype=np.float64)
    window_sq = np.zeros_like(x_out)
    win = np.hanning(n_fft)

    frames = np.fft.irfft(X, n=n_fft, axis=1)
    for i in range(n_frames):
        start = i * hop_length
        x_out[start : start + n_fft] += frames[i] * win
        window_sq[start : start + n_fft] += win**2

    mask = window_sq > 1e-10
    x_out[mask] /= window_sq[mask]

    return x_out[n_fft // 2 : n_fft // 2 + orig_len]


def resolve_target_voicing(
    voicing: VoicingConfig | str,
    instrument: InstrumentConfig | str | Path | None = None,
) -> tuple[InstrumentConfig, VoicingConfig]:
    """Resolves an instrument and voicing from either a native voicing ID or a universal voice slug."""
    if isinstance(voicing, VoicingConfig):
        inst = (
            load_instrument(instrument)
            if instrument is not None and not isinstance(instrument, InstrumentConfig)
            else (instrument if isinstance(instrument, InstrumentConfig) else None)
        )
        if inst is None:
            raise ValueError("Must provide instrument when voicing is a VoicingConfig")
        return inst, voicing

    clean_voicing = str(voicing).lower().replace(" ", "_").replace("∕", "_").replace("/", "_")

    # If instrument was provided, check its native voicings first
    if instrument is not None:
        inst = (
            load_instrument(instrument)
            if not isinstance(instrument, InstrumentConfig)
            else instrument
        )
        if voicing in inst.voicings:
            return inst, inst.voicings[voicing]
        for v in inst.voicings.values():
            slug = (
                v.tone_name.lower().replace(" ", "_").replace("∕", "_").replace("/", "_")
                if v.tone_name
                else (v.id or "")
            )
            if clean_voicing in (v.id, slug):
                return inst, v

        # Check if voicing matches a declared pickup on the instrument
        if clean_voicing in inst.pickups or str(voicing) in inst.pickups:
            p_key = clean_voicing if clean_voicing in inst.pickups else str(voicing)
            p_cfg = inst.pickups[p_key]

            # 1. Search for a native nominal voicing on that pickup
            best_match: VoicingConfig | None = None
            for v in inst.voicings.values():
                if v.pickup == p_key:
                    if v.id == p_key:
                        best_match = v
                        break
                    if (
                        v.vol_pos == 1.0
                        and v.tone_pos == 1.0
                        and (v.gain_db == 0.0 or v.gain_db is None)
                        and len(v.preamp_bands) == 0
                        and v.hpf is None
                        and v.circuit is None
                    ):
                        best_match = v
                        break
                    if best_match is None:
                        best_match = v

            if best_match is not None:
                return inst, best_match

            # 2. If no identity voicing, construct a clean nominal VoicingConfig
            sensor = (
                "bridge_force"
                if ("piezo" in getattr(p_cfg, "type", "") or p_key.endswith("piezo"))
                else ("direct" if getattr(p_cfg, "type", "") == "direct" else "magnetic")
            )
            str_preset = (
                getattr(getattr(inst, "strings", None), "preset", None)
                or "roundwound_nickel_standard"
            )
            p_preamp = getattr(p_cfg.circuit, "preamp", None) if p_cfg.circuit else None
            if p_preamp in ("none", ""):
                p_preamp = None
            dyn_v = VoicingConfig(
                id=p_key,
                name=p_cfg.name or p_key,
                pickup=p_key,
                vol_pos=1.0,
                tone_pos=1.0,
                sensor_type=sensor,
                string_preset_override=str_preset,
                preamp_preset=p_preamp,
            )
            return inst, dyn_v

    # Search STANDARD_CATALOG_TARGETS across all instruments
    from allomorph.config.instruments import STANDARD_CATALOG_TARGETS

    for iid, vid in STANDARD_CATALOG_TARGETS:
        t_inst = load_instrument(iid)
        if vid in t_inst.voicings:
            t_v = t_inst.voicings[vid]
            slug = (
                t_v.tone_name.lower().replace(" ", "_").replace("∕", "_").replace("/", "_")
                if t_v.tone_name
                else vid
            )
            if clean_voicing in (vid, slug, (t_v.id or "").lower()):
                return t_inst, t_v

    # Fallback error reporting
    if instrument is not None:
        inst = (
            load_instrument(instrument)
            if not isinstance(instrument, InstrumentConfig)
            else instrument
        )
        raise KeyError(
            f"Voicing or pickup '{voicing}' not found on instrument '{inst.id}'. "
            f"Available voicings: {list(inst.voicings.keys())}, pickups: {list(inst.pickups.keys())}"
        )

    raise KeyError(
        f"Target voicing '{voicing}' could not be resolved from any instrument in catalog."
    )


def simulate_instrument_voicing(
    instrument: InstrumentConfig | str | Path | None = None,
    voicing: VoicingConfig | str = "default_voicing",
    input_wav: Path | str | None = None,
    output_wav: Path | str | None = None,
    max_samples: int | None = None,
    num_taps: int = NUM_TAPS,
    apply_dither: bool = True,
    apply_saturation: bool = True,
    vol_pos: float | None = None,
    tone_pos: float | None = None,
    blend_pos: float | None = None,
    cable_pf: float | None = None,
    dc_block: bool = True,
    normalize: str = "auto",
    target_dbfs: float | None = None,
    force: bool = False,
) -> Path:
    """
    Simulates a physical instrument voicing digital twin directly from dry string excitation.
    Convolves:
      1. Physical sensor/aperture acoustics H_ac(f) with multi-pickup branch coupling
      2. Loaded RLC circuit transfer H_elec(f) with volume/tone wiper tapers
      3. Scale-length tension snap H_tension(f) and longitudinal clank resonance H_long(f)
      4. String damping and tension compliance differential mechanics H_string(f)
      5. Onboard active preamp EQ contour H_preamp(f)
      6. Oversampled non-linear magnetic saturation across all 16 physical parameters
      7. Sub-audible 8 Hz DC blocking and passive RLC-colored Johnson noise dither
    Synthesizes a causal minimum-phase FIR starting at sample 0 (zero latency).
    Exports 24-bit 48 kHz PCM audio with calibrated LUFS volume matching and 4x true-peak ceiling protection.
    """
    inst, voicing_cfg = resolve_target_voicing(voicing, instrument=instrument)
    voicing_id = (
        voicing_cfg.tone_name.lower().replace(" ", "_").replace("∕", "_").replace("/", "_")
        if voicing_cfg.tone_name
        else (voicing_cfg.id or "default_voicing")
    )

    # Resolve input audio
    if not input_wav or not Path(input_wav).exists():
        found = find_default_input_audio()
        if not found:
            raise FileNotFoundError(
                f"Input audio '{input_wav}' not found, and no standard excitation audio (input.wav) was detected."
            )
        in_path = Path(found)
    else:
        in_path = Path(input_wav)

    # Resolve output path
    if output_wav is not None:
        out_path = Path(output_wav)
    elif isinstance(voicing, str) and (
        voicing in inst.pickups or str(voicing).lower() in inst.pickups
    ):
        p_key = voicing if voicing in inst.pickups else str(voicing).lower()
        out_path = WET_AUDIO_DIR / inst.id / f"{p_key}.wav"
    else:
        out_path = WET_AUDIO_DIR / inst.id / f"{voicing_id}.wav"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Cryptographic Bit-Provenance Cache Check
    inst_ver = getattr(inst, "version", 1)
    voice_ver = getattr(voicing_cfg, "version", 1)
    v_tag = resolve_tri_part_version(DSP_GENERATION, inst_ver, voice_ver)
    if not force and is_wet_stem_valid(out_path, base_dry_path=in_path, expected_version=v_tag):
        print(f"[Forward Sim] Cached {inst.id}:{voicing_id} ({v_tag}) -> {out_path.name}")
        return out_path

    # Lookup physical pickup
    pickup_key = voicing_cfg.pickup
    if pickup_key not in inst.pickups:
        raise KeyError(
            f"Voicing '{voicing_id}' references unknown pickup '{pickup_key}' on instrument '{inst.id}'. "
            f"Available pickups: {list(inst.pickups.keys())}"
        )
    pickup_cfg = inst.pickups[pickup_key]

    if (
        inst.electronics == "passive"
        and pickup_cfg.circuit is None
        and voicing_cfg.sensor_type == "magnetic"
    ):
        raise ValueError(
            f"Passive instrument '{inst.id}' pickup '{pickup_key}' "
            f"does not define a '[circuit]' block. Passive source pickups require an explicit "
            f"circuit model for forward circuit simulation."
        )

    f = np.asarray(FREQS, dtype=np.float64)
    s = 2j * np.pi * f
    scale_range = resolve_scale_range(inst)
    scale_m = (scale_range[0] + scale_range[1]) / 2.0
    scale_in = inst.scale_length_in or (scale_m / 0.0254)

    # Audio Loading
    raw_audio, sr = read_wav(in_path, max_samples=max_samples, dtype=np.float64)
    input_mono = raw_audio[0] if raw_audio.ndim > 1 else raw_audio
    n_samples = len(input_mono)

    # 1. Sensor Aperture Acoustics & Loaded RLC Circuit Transfer
    circ_model = None
    curves = None
    if pickup_cfg.circuit is not None:
        circ_model = load_circuit(pickup_cfg.circuit)
        eff_vol = vol_pos if vol_pos is not None else voicing_cfg.vol_pos
        eff_tone = tone_pos if tone_pos is not None else voicing_cfg.tone_pos
        eff_blend = blend_pos if blend_pos is not None else voicing_cfg.blend_pos
        circ_model.apply_pot_positions(
            vol_pos=eff_vol,
            tone_pos=eff_tone,
            blend_pos=eff_blend,
        )
        if cable_pf is not None:
            circ_model.Ccable = float(cable_pf) * 1e-12
        if voicing_cfg.tone_cap_f is not None:
            circ_model.Ctone = float(voicing_cfg.tone_cap_f)
        apply_magnet_properties_to_model(circ_model, pickup_cfg, eddy_diffusion=True)
        curves = compute_circuit_transfer_functions(circ_model, freqs=f, return_numpy=True)

    # 2. Active Preamp EQ Contour H_preamp(f)
    circ_preamp = getattr(circ_model, "preamp", None) if circ_model is not None else None

    if voicing_cfg.preamp_bands:
        h_pre_raw = compute_active_preamp_transfer(voicing_cfg.preamp_bands, s)
        h_preamp = np.abs(h_pre_raw).astype(np.float64)
    elif voicing_cfg.preamp_preset:
        h_pre_raw = compute_active_preamp_eq(voicing_cfg.preamp_preset, s)
        h_preamp = np.abs(h_pre_raw).astype(np.float64)
    elif circ_preamp and circ_preamp not in ("none", ""):
        h_pre_raw = compute_active_preamp_eq(circ_preamp, s)
        h_preamp = np.abs(h_pre_raw).astype(np.float64)
    else:
        h_preamp = np.ones_like(f, dtype=np.float64)

    # 3. Scale-Length Tension Dynamics H_tension(f)
    r_L = scale_in / 34.0
    if voicing_cfg.sensor_type == "direct":
        h_tension = np.ones_like(f, dtype=np.float64)
    elif voicing_cfg.sensor_type == "bridge_force":
        g_snap = r_L**1.5
        h_tension = np.sqrt((1.0 + g_snap**2 * (f / 2800.0) ** 2) / (1.0 + (f / 2800.0) ** 2))
    else:
        g_excursion = 1.0 / r_L
        g_snap = r_L**1.5
        h_tension = np.sqrt(
            (g_excursion**2 + (f / 100.0) ** 2) / (1.0 + (f / 100.0) ** 2)
        ) * np.sqrt((1.0 + g_snap**2 * (f / 2800.0) ** 2) / (1.0 + (f / 2800.0) ** 2))

    # 4. Steel Core Longitudinal Clank Resonance H_long(f) & String Damping H_string(f)
    inst_string = get_instrument_string(inst)
    ref_string = STRINGS["roundwound_nickel_standard"]
    target_string = (
        STRINGS[voicing_cfg.string_preset_override]
        if voicing_cfg.string_preset_override and voicing_cfg.string_preset_override in STRINGS
        else inst_string
    )
    if voicing_cfg.sensor_type not in ("bridge_force", "direct") and target_string != ref_string:
        h_str_raw = compute_differential_string_transfer(f, ref_string, target_string)
        h_string = np.asarray(h_str_raw, dtype=np.float64)
        h_long = compute_differential_longitudinal_transfer(
            f, ref_string, target_string, scale_length_inches=scale_in
        )
    else:
        h_string = np.ones_like(f, dtype=np.float64)
        h_long = np.ones_like(f, dtype=np.float64)

    is_composite_sim = False
    composite_audio = np.zeros(n_samples, dtype=np.float64)

    if voicing_cfg.sensor_type == "direct":
        if curves is not None:
            h_base = np.asarray(curves[0], dtype=np.float64)
        else:
            h_base = np.ones_like(f, dtype=np.float64)
    elif voicing_cfg.sensor_type == "bridge_force":
        f_damp = 3800.0
        h_ac = 1.0 / np.sqrt(1.0 + (f / f_damp) ** 4)
        f_sub = 10.0
        h_sub = np.sqrt(f**2 / (f_sub**2 + f**2))
        h_ac = h_ac * h_sub
        if curves is not None:
            h_elec = np.asarray(curves[0], dtype=np.float64)
            h_base = h_ac * h_elec
        else:
            h_base = h_ac
    else:
        # Magnetic sensor: determine single pickup vs multi-pickup composite
        is_composite = (
            pickup_cfg.type == "composite"
            or bool(pickup_cfg.components)
            or (curves is not None and len(curves) > 1)
        )

        single_positions = [
            float(p.position_from_bridge_m)
            for p in inst.pickups.values()
            if not p.components and p.position_from_bridge_m is not None
        ]
        # Self-referential leveling: multi-pickup basses level bridge pickups relative to their own forward-most pickup.
        # Single-pickup basses receive ref_pos = None and evaluate to exact 1.0000 (0.00 dB).
        ref_pos = max(single_positions) if len(single_positions) > 1 else None

        if is_composite and pickup_cfg.components:
            is_composite_sim = True
            N = 8192
            f_bins = np.fft.rfftfreq(N, 1.0 / 48000.0)
            c_mean = 2.0 * scale_m * MEAN_BASS_F0

            # Resolve branch components
            branch_sub_pickups = []
            for comp in pickup_cfg.components:
                if comp.pickup and comp.pickup in inst.pickups:
                    branch_sub_pickups.append(
                        (inst.pickups[comp.pickup], float(comp.weight), float(comp.polarity))
                    )

            branch_coils_list = []
            branch_positions = []
            for sp, w_comp, pol_comp in branch_sub_pickups:
                b_coils = resolve_pickup_coils(sp, inst)
                branch_coils_list.append((b_coils, w_comp, pol_comp))
                branch_positions.append(compute_effective_position(b_coils))

            pos_max = max(branch_positions) if branch_positions else 0.0
            H_channels = []
            peaks = []
            branch_audios = []

            for i, (b_coils, w_comp, pol_comp) in enumerate(branch_coils_list):
                c_curve_raw = (
                    curves[i]
                    if (curves is not None and i < len(curves))
                    else (curves[0] if curves is not None else np.ones_like(FREQS))
                )
                c_curve = np.interp(f_bins, FREQS, np.asarray(c_curve_raw, dtype=np.float64))

                weight_fac = 1.0 if (curves is not None and len(curves) > 1) else w_comp
                ac_raw = numpy_pickup_acoustic_response(
                    f_bins, b_coils, scale_length_m=scale_range
                ) * (weight_fac * pol_comp)

                b_pos = branch_positions[i]
                h_pos = compute_displacement_proximity_shelf(f_bins, b_pos, scale_m=scale_m)
                k_iso = compute_pickup_isolation_leveling(b_pos, scale_m=scale_m, ref_pos_m=ref_pos)
                ac = ac_raw * h_pos * k_iso

                min_pos = min((c.position_from_bridge_m for c in b_coils), default=0.10)
                if min_pos < 0.075:
                    h_saddle = compute_saddle_boundary_coupling(f_bins, min_pos)
                    ac = ac * np.asarray(h_saddle, dtype=np.float64)

                fir_ac = synthesize_minimum_phase_fir(ac, num_taps=2048, normalize=False)
                tau_i = (pos_max - b_pos) / c_mean if len(branch_coils_list) > 1 else 0.0
                delay_samples = round(tau_i * 48000.0)
                if 0 < delay_samples < 2048:
                    fir_ac = [0.0] * delay_samples + fir_ac[: 2048 - delay_samples]
                peaks.append(int(np.argmax(np.abs(fir_ac))))

                fir_circ = synthesize_minimum_phase_fir(c_curve, num_taps=2048, normalize=False)
                H_ac_fft = np.fft.rfft(fir_ac, N)
                H_circ_fft = np.fft.rfft(fir_circ, N)
                H_channels.append(H_ac_fft * H_circ_fft)

                fir_branch = fft_convolve(
                    np.asarray(fir_ac, dtype=np.float64),
                    np.asarray(fir_circ, dtype=np.float64),
                    mode="causal",
                )
                b_audio = fft_convolve(input_mono, fir_branch, mode="causal")[:n_samples]

                # Branch-wise magnetic saturation
                if apply_saturation:
                    sp = branch_sub_pickups[i][0]
                    sp_mag = sp.magnet_type or (
                        "active" if getattr(inst, "electronics", "") == "active" else "alnico_v"
                    )
                    if sp_mag not in MAGNET_PROPERTIES:
                        raise KeyError(f"Unrecognized magnet type: '{sp_mag}'")
                    sp_props = MAGNET_PROPERTIES[sp_mag]
                    sp_vsat = (
                        float(voicing_cfg.vsat)
                        if voicing_cfg.vsat is not None
                        else (
                            float(circ_model.vsat)
                            if (circ_model is not None and circ_model.vsat is not None)
                            else float(sp_props.vsat)
                        )
                    )
                    sp_alpha = (
                        float(voicing_cfg.alpha)
                        if voicing_cfg.alpha is not None
                        else (float(sp.alpha) if sp.alpha is not None else float(sp_props.alpha))
                    )
                    drive_db = float(getattr(voicing_cfg, "gain_db", 0.0) or 0.0)
                    drive_in = b_audio if drive_db == 0.0 else b_audio * (10.0 ** (drive_db / 20.0))
                    b_audio = apply_oversampled_saturation(
                        drive_in.astype(np.float32),
                        vsat=sp_vsat,
                        alpha=sp_alpha,
                        alpha3=float(sp_props.alpha3),
                        eta_hyst=float(sp_props.eta_hyst),
                        k_sag=float(sp_props.k_sag),
                        k_eddy=float(sp_props.k_eddy),
                        kappa_orbit=float(sp_props.kappa_orbit),
                        beta_curv=float(sp_props.beta_curv),
                        k_pull=float(sp_props.k_pull),
                        tau_touch=float(sp_props.tau_touch),
                        kappa_geom=float(sp_props.kappa_geom),
                        k_stein=float(sp_props.k_stein),
                        k_emf=float(sp_props.k_emf),
                        lambda_L=float(sp_props.lambda_L),
                        kappa_ap=float(sp_props.kappa_ap),
                        slew_limit=True,
                        f_slew=16000.0,
                        oversample=2,
                        displacement_weighting=True,
                        magnet_drag=True,
                    ).astype(np.float64)

                branch_audios.append(b_audio)

            raw_sum = np.sum(branch_audios, axis=0)
            H_channels_arr = np.array(H_channels)
            delta_samples = max(peaks) - min(peaks) if len(peaks) > 1 else 0

            if len(H_channels_arr) > 1 and delta_samples > 0:
                P_coh_raw = np.abs(np.sum(H_channels_arr, axis=0)) ** 2
                P_incoh = np.sum(np.abs(H_channels_arr) ** 2, axis=0)

                eps_quad = 0.18
                P_coh_reg = P_coh_raw + (eps_quad**2) * P_incoh

                H_dc = np.abs(H_channels_arr[:, 0])
                total_w = np.sum(H_dc)
                dc_incoh = np.sum(H_dc**2)
                dc_norm = (
                    math.sqrt(total_w**2 + (eps_quad**2) * dc_incoh) / total_w
                    if total_w > 0
                    else 1.0
                )

                delta_tau = delta_samples / 48000.0
                f_notch = 1.0 / (2.0 * delta_tau)
                f_mid = 1.35 * f_notch
                f_sigma = max(0.35 * f_notch, 1.0)
                gamma = 0.5 * (1.0 - np.tanh((f_bins - f_mid) / f_sigma))

                M_blend = np.sqrt(gamma * P_coh_reg + (1.0 - gamma) * P_incoh) / dc_norm

                n_fft = 4096
                hop_length = 1024

                # STFT Magnitude Forcing (Perfect physical realization of spatial incoherence without FIR ringing)
                X_branches = [_stft(b, n_fft, hop_length) for b in branch_audios]
                X_coh = np.sum(X_branches, axis=0)
                P_incoh_stft = np.sum([np.abs(X) ** 2 for X in X_branches], axis=0)

                f_stft = np.fft.rfftfreq(n_fft, d=1.0 / 48000.0)
                gamma_stft = 0.5 * (1.0 - np.tanh((f_stft - f_mid) / f_sigma))
                gamma_stft = gamma_stft[np.newaxis, :]

                M_target = (
                    np.sqrt(
                        gamma_stft * (np.abs(X_coh) ** 2 + (eps_quad**2) * P_incoh_stft)
                        + (1.0 - gamma_stft) * P_incoh_stft
                    )
                    / dc_norm
                )

                phase_ref = X_branches[0].copy()
                mask = np.abs(X_coh) < 1e-6
                phase_ref[mask] = X_branches[0][mask] if len(X_branches) > 0 else 0.0

                X_out = M_target * np.exp(1j * np.angle(phase_ref))
                composite_audio = _istft(X_out, n_fft, hop_length, n_samples)

                mag_spectrum = M_blend
            elif len(H_channels_arr) > 1:
                composite_audio = raw_sum
                mag_spectrum = np.abs(np.sum(H_channels_arr, axis=0))
            else:
                composite_audio = raw_sum
                mag_spectrum = np.abs(H_channels_arr[0])

            h_base = np.interp(f, f_bins, mag_spectrum)
        else:
            # Single pickup or unified coil aperture
            coils = resolve_pickup_coils(pickup_cfg, inst)
            eff_pos = compute_effective_position(coils)
            h_ac_raw = numpy_pickup_acoustic_response(f, coils, scale_length_m=scale_range)
            h_ac = np.asarray(h_ac_raw, dtype=np.float64)

            # Spatial bridge proximity displacement excursion relative to calibration baseline
            h_pos = compute_displacement_proximity_shelf(f, eff_pos, scale_m=scale_m)
            k_iso = compute_pickup_isolation_leveling(eff_pos, scale_m=scale_m, ref_pos_m=ref_pos)
            h_ac = h_ac * h_pos * k_iso

            # Saddle boundary coupling for close bridge pickups
            min_pos = min((c.position_from_bridge_m for c in coils), default=0.10)
            if min_pos < 0.075:
                h_saddle = compute_saddle_boundary_coupling(f, min_pos)
                h_ac = h_ac * np.asarray(h_saddle, dtype=np.float64)

            if curves is not None:
                h_elec = np.asarray(curves[0], dtype=np.float64)
            else:
                h_elec = np.ones_like(f, dtype=np.float64)

            h_base = np.abs(h_ac) * np.abs(h_elec)

    # 5. Composite Magnitude Transfer Function & Convolutions
    H_downstream = np.maximum(
        h_preamp * h_tension * h_long * h_string,
        1e-6,
    )
    h_total = np.maximum(
        h_base * H_downstream,
        1e-6,
    )

    if is_composite_sim:
        if not np.allclose(H_downstream, 1.0, atol=1e-4):
            fir_down = synthesize_minimum_phase_fir(
                H_downstream, num_taps=num_taps, normalize=False
            )
            filtered = fft_convolve(
                composite_audio, np.asarray(fir_down, dtype=np.float64), mode="causal"
            )[:n_samples]
        else:
            filtered = composite_audio
    else:
        fir_base = synthesize_minimum_phase_fir(h_base, num_taps=num_taps, normalize=False)
        fir_base_np = np.asarray(fir_base, dtype=np.float64)
        filtered = fft_convolve(input_mono, fir_base_np, mode="causal")[:n_samples]

        # Single-pickup Oversampled Magnetic Saturation & Core Dynamics (All 16 Parameters)
        has_direct_dynamics = (
            voicing_cfg.sensor_type == "direct"
            and circ_model is not None
            and not getattr(circ_model, "no_eq", False)
        )
        if apply_saturation and (voicing_cfg.sensor_type == "magnetic" or has_direct_dynamics):
            sp = pickup_cfg if pickup_cfg is not None else next(iter(inst.pickups.values()))
            sp_mag = sp.magnet_type or (
                "active" if getattr(inst, "electronics", "") == "active" else "alnico_v"
            )
            if sp_mag not in MAGNET_PROPERTIES:
                raise KeyError(f"Unrecognized magnet type: '{sp_mag}'")
            sp_props = MAGNET_PROPERTIES[sp_mag]
            sp_vsat = (
                float(voicing_cfg.vsat)
                if voicing_cfg.vsat is not None
                else (
                    float(circ_model.vsat)
                    if (circ_model is not None and circ_model.vsat is not None)
                    else float(sp_props.vsat)
                )
            )
            sp_alpha = (
                float(voicing_cfg.alpha)
                if voicing_cfg.alpha is not None
                else (float(sp.alpha) if sp.alpha is not None else float(sp_props.alpha))
            )
            drive_db = float(getattr(voicing_cfg, "gain_db", 0.0) or 0.0)
            drive_in = filtered if drive_db == 0.0 else filtered * (10.0 ** (drive_db / 20.0))
            filtered = apply_oversampled_saturation(
                drive_in.astype(np.float32),
                vsat=sp_vsat,
                alpha=sp_alpha,
                alpha3=float(sp_props.alpha3),
                eta_hyst=float(sp_props.eta_hyst),
                k_sag=float(sp_props.k_sag),
                k_eddy=float(sp_props.k_eddy),
                kappa_orbit=float(sp_props.kappa_orbit),
                beta_curv=float(sp_props.beta_curv),
                k_pull=float(sp_props.k_pull),
                tau_touch=float(sp_props.tau_touch),
                kappa_geom=float(sp_props.kappa_geom),
                k_stein=float(sp_props.k_stein),
                k_emf=float(sp_props.k_emf),
                lambda_L=float(sp_props.lambda_L),
                kappa_ap=float(sp_props.kappa_ap),
                slew_limit=True,
                f_slew=16000.0,
                oversample=2,
                displacement_weighting=True,
                magnet_drag=True,
            ).astype(np.float64)

        if not np.allclose(H_downstream, 1.0, atol=1e-4):
            fir_down = synthesize_minimum_phase_fir(
                H_downstream, num_taps=num_taps, normalize=False
            )
            filtered = fft_convolve(
                filtered, np.asarray(fir_down, dtype=np.float64), mode="causal"
            )[:n_samples]

    # 9. Sub-Audible DC-Blocking Filter (8 Hz)
    if dc_block:
        hp = pedalboard.HighpassFilter(cutoff_frequency_hz=8.0)
        filtered = hp(filtered.astype(np.float32)[np.newaxis, :], sr)[0].astype(np.float64)
        filtered = filtered - float(np.mean(filtered))

    # 10. Johnson-Nyquist Passive RLC-Colored Dither (-108 dBFS)
    if apply_dither:
        white_noise = _get_white_noise_vector(n_samples)
        n_dither_taps = 512
        dither_fir = synthesize_minimum_phase_fir(h_total, num_taps=n_dither_taps, normalize=True)
        colored_noise = fft_convolve(
            white_noise, np.asarray(dither_fir, dtype=np.float64), mode="causal"
        )[:n_samples]
        colored_rms = max(float(np.sqrt(np.mean(colored_noise**2))), 1e-9)
        target_dither_rms = 10.0 ** (-108.0 / 20.0)
        dither = (colored_noise / colored_rms) * target_dither_rms
        filtered = filtered + dither

    # 11. Calibrated Level Normalization & Headroom Ceiling
    if normalize in ("auto", "lufs"):
        in_lufs = compute_lufs(input_mono, sample_rate=sr)
        cur_lufs = compute_lufs(filtered, sample_rate=sr)
        if not (
            math.isinf(cur_lufs)
            or math.isnan(cur_lufs)
            or math.isinf(in_lufs)
            or math.isnan(in_lufs)
        ):
            target_lufs = float(target_dbfs) if target_dbfs is not None else in_lufs
            gain_db = target_lufs - cur_lufs
            filtered = filtered * (10.0 ** (gain_db / 20.0))
        else:
            in_rms = float(np.sqrt(np.mean(input_mono**2)))
            out_rms = float(np.sqrt(np.mean(filtered**2)))
            target_rms = 10.0 ** (float(target_dbfs) / 20.0) if target_dbfs is not None else in_rms
            if out_rms > 1e-9:
                filtered = filtered * (target_rms / out_rms)
    elif normalize == "rms":
        target_rms = (
            10.0 ** (target_dbfs / 20.0)
            if target_dbfs is not None
            else float(np.sqrt(np.mean(input_mono**2)))
        )
        cur_rms = float(np.sqrt(np.mean(filtered**2)))
        if cur_rms > 1e-9:
            filtered = filtered * (target_rms / cur_rms)
    elif normalize == "peak":
        target_peak = (
            10.0 ** (target_dbfs / 20.0)
            if target_dbfs is not None
            else float(np.max(np.abs(input_mono)))
        )
        cur_peak = float(np.max(np.abs(filtered)))
        if cur_peak > 1e-9:
            filtered = filtered * (target_peak / cur_peak)

    if normalize not in ("none", "raw"):
        tp = compute_true_peak(filtered)
        if tp > CALIBRATION_PEAK_CEILING:
            filtered = filtered * (CALIBRATION_PEAK_CEILING / tp)

    # 12. 24-bit PCM Export
    write_wav_24bit(out_path, filtered.astype(np.float32), sample_rate=sr)

    final_tp_db = compute_true_peak_dbfs(filtered)
    final_tp_lin = compute_true_peak(filtered)
    final_rms_db = 20.0 * math.log10(max(float(np.sqrt(np.mean(filtered**2))), 1e-9))
    final_lufs = compute_lufs(filtered, sample_rate=sr)
    lufs_str = (
        f"{final_lufs:.2f} LUFS"
        if not (math.isinf(final_lufs) or math.isnan(final_lufs))
        else "-inf LUFS"
    )
    print(
        f"[Forward Sim] {inst.id}:{voicing_id} -> {out_path.name}: "
        f"True Peak = {final_tp_db:.2f} dBFS (lin={final_tp_lin:.4f}), "
        f"RMS = {final_rms_db:.2f} dBFS, Loudness = {lufs_str}"
    )

    # 13. Manifest Provenance Tracking (base dry SHA256)
    try:
        base_dry_sha = compute_file_sha256(in_path)
        inst_ver = getattr(inst, "version", 1)
        voice_ver = getattr(voicing_cfg, "version", 1)
        v_tag = resolve_tri_part_version(
            DSP_GENERATION,
            inst_ver,
            voice_ver,
        )
        write_manifest(
            output_dir=out_path.parent,
            stage="voicing",
            files=[out_path],
            version_tag=v_tag,
            base_dry_sha256=base_dry_sha,
            base_dry_file=in_path.name,
            instrument_version=inst_ver,
            voicing_version=voice_ver,
        )
    except OSError, ValueError, RuntimeError:
        pass

    return out_path


def simulate_all_instrument_voicings(
    instrument: InstrumentConfig | str | Path | None = None,
    input_wav: Path | str | None = None,
    max_samples: int | None = None,
    jobs: int | None = None,
    force: bool = False,
) -> list[Path]:
    """
    Simulates all native voicings for a specified instrument (or all instruments in catalog).
    Outputs files to audio/wet/<inst_id>/<voicing_id>.wav.
    """
    if instrument is not None:
        target_insts = [
            load_instrument(instrument)
            if not isinstance(instrument, InstrumentConfig)
            else instrument
        ]
    else:
        target_insts = list(INSTRUMENTS.values())

    exported_paths: list[Path] = []
    tasks: list[tuple[InstrumentConfig, VoicingConfig | str]] = []
    for inst in target_insts:
        for voicing in inst.voicings.values():
            tasks.append((inst, voicing))
        for p_key in inst.pickups:
            tasks.append((inst, p_key))

    def _sim(item: tuple[InstrumentConfig, VoicingConfig | str]) -> Path:
        i, v = item
        return simulate_instrument_voicing(
            instrument=i,
            voicing=v,
            input_wav=input_wav,
            max_samples=max_samples,
            force=force,
        )

    eff_jobs = jobs if jobs is not None and jobs > 0 else 1
    if eff_jobs > 1 and len(tasks) > 1:
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=eff_jobs) as executor:
            exported_paths = list(executor.map(_sim, tasks))
    else:
        exported_paths = [_sim(t) for t in tasks]

    return exported_paths


def simulate_circuit_audio(
    input_audio: str | Path | np.ndarray,
    output_wav_path: str | Path,
    model: CircuitModel,
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
    kappa_ap: float = 0.0,
    kappa_aps: Sequence[float] | None = None,
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
    saturation_config: Any = None,
    harness_controls: Any = None,
    **kwargs: Any,
) -> bool:
    """
    Executes circuit simulation on audio through a CircuitModel.
    Convolves with the circuit's transfer function, applies non-linear saturation,
    sub-audible DC-blocking, thermal dither, and level matching.
    """
    if harness_controls is not None:
        vol_pos = getattr(harness_controls, "vol_pos", vol_pos)
        tone_pos = getattr(harness_controls, "tone_pos", tone_pos)
        blend_pos = getattr(harness_controls, "blend_pos", blend_pos)
        pot_taper = getattr(harness_controls, "pot_taper", pot_taper)

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

    # 1. Audio Loading
    if isinstance(input_audio, np.ndarray):
        if input_audio.ndim == 2:
            audio_data = (
                input_audio[0] if input_audio.shape[0] < input_audio.shape[1] else input_audio[:, 0]
            )
        else:
            audio_data = input_audio
        sr = 48000
    else:
        audio_data, sr = read_wav(input_audio)

    if max_samples is not None and len(audio_data) > max_samples:
        audio_data = audio_data[:max_samples]

    audio_mono = np.asarray(audio_data, dtype=np.float64)
    should_bypass = bypass_saturation is True or is_passive
    in_peak = float(np.max(np.abs(audio_mono)))
    if not should_bypass and in_peak > 0.10:
        target_drive_peak = min(in_peak * 0.687, 0.70)
        audio_mono = (audio_mono / max(in_peak, 1e-9)) * target_drive_peak

    # 2. Circuit Transfer Function
    f = np.asarray(FREQS, dtype=np.float64)
    if circuit_curves is not None:
        curves = circuit_curves
    else:
        curves = compute_circuit_transfer_functions(model, freqs=f, return_numpy=True)
    h_elec = np.asarray(curves[0], dtype=np.float64)

    # 3. FIR Synthesis & Linear Stage Fusion
    circ_fir = synthesize_minimum_phase_fir(h_elec, num_taps=NUM_TAPS, normalize=False)
    filtered = fft_convolve(audio_mono, circ_fir, mode="causal")[: len(audio_mono)]

    # 4. Non-Linear Saturation
    should_bypass = bypass_saturation is True or is_passive
    if not should_bypass:
        if saturation_config is not None:
            vsat = getattr(saturation_config, "vsat", vsat)
            alpha = getattr(saturation_config, "alpha", alpha)
            alpha3 = getattr(saturation_config, "alpha3", alpha3)
            eta_hyst = getattr(saturation_config, "eta_hyst", eta_hyst)
            k_sag = getattr(saturation_config, "k_sag", k_sag)
            k_eddy = getattr(saturation_config, "k_eddy", k_eddy)
            kappa_orbit = getattr(saturation_config, "kappa_orbit", kappa_orbit)
            beta_curv = getattr(saturation_config, "beta_curv", beta_curv)
            k_pull = getattr(saturation_config, "k_pull", k_pull)
            tau_touch = getattr(saturation_config, "tau_touch", tau_touch)
            kappa_geom = getattr(saturation_config, "kappa_geom", kappa_geom)
            k_stein = getattr(saturation_config, "k_stein", k_stein)
            k_emf = getattr(saturation_config, "k_emf", k_emf)
            lambda_L = getattr(saturation_config, "lambda_L", lambda_L)
            kappa_ap = getattr(saturation_config, "kappa_ap", kappa_ap)
        vsat_val = vsat if vsat is not None else 0.45
        filtered = apply_oversampled_saturation(
            filtered.astype(np.float32),
            vsat=vsat_val,
            alpha=alpha,
            alpha3=alpha3,
            eta_hyst=eta_hyst,
            k_sag=k_sag,
            k_eddy=k_eddy,
            kappa_orbit=kappa_orbit,
            beta_curv=beta_curv,
            k_pull=k_pull,
            tau_touch=tau_touch,
            kappa_geom=kappa_geom,
            k_stein=k_stein,
            k_emf=k_emf,
            lambda_L=lambda_L,
            kappa_ap=kappa_ap,
            slew_limit=slew_limit,
            f_slew=f_slew,
            oversample=oversample,
            displacement_weighting=displacement_weighting,
            magnet_drag=magnet_drag,
        ).astype(np.float64)

    # 5. DC Blocking
    in_peak = float(np.max(np.abs(audio_mono)))
    if dc_block and in_peak > 0.10:
        hp = pedalboard.HighpassFilter(cutoff_frequency_hz=8.0)
        filtered = hp(filtered.astype(np.float32)[np.newaxis, :], sr)[0].astype(np.float64)
        filtered = filtered - float(np.mean(filtered))

    # 6. Thermal Dither
    if noise_dither and float(np.max(np.abs(audio_mono))) > 0.10:
        n_samples = len(filtered)
        white_noise = _get_white_noise_vector(n_samples)
        dither_fir = synthesize_minimum_phase_fir(h_elec, num_taps=512, normalize=True)
        colored = fft_convolve(
            white_noise, np.asarray(dither_fir, dtype=np.float64), mode="causal"
        )[:n_samples]
        c_rms = max(float(np.sqrt(np.mean(colored**2))), 1e-9)
        filtered = filtered + (colored / c_rms) * (10.0 ** (-108.0 / 20.0))

    # 7. Level Normalization
    in_rms = float(np.sqrt(np.mean(audio_mono**2)))
    should_normalize = (
        normalize in ("auto", "rms", "peak")
        and in_peak > 0.10
        and in_rms > 0.005
        and (np.min(audio_mono) < 0.0)
    )
    if normalize == "rms" and target_dbfs is not None:
        target_rms = 10.0 ** (target_dbfs / 20.0)
        cur_rms = float(np.sqrt(np.mean(filtered**2)))
        if cur_rms > 1e-9:
            filtered = filtered * (target_rms / cur_rms)
    elif normalize == "peak" and target_dbfs is not None:
        target_peak = 10.0 ** (target_dbfs / 20.0)
        cur_peak = float(np.max(np.abs(filtered)))
        if cur_peak > 1e-9:
            filtered = filtered * (target_peak / cur_peak)
    elif should_normalize:
        out_rms = float(np.sqrt(np.mean(filtered**2)))
        if in_rms > 1e-9 and out_rms > 1e-9:
            filtered = filtered * (in_rms / out_rms)

    if normalize not in ("none", "raw"):
        peak = float(np.max(np.abs(filtered)))
        if peak > CALIBRATION_PEAK_CEILING:
            filtered = filtered * (CALIBRATION_PEAK_CEILING / peak)

    # 8. WAV Export
    out_p = Path(output_wav_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    write_wav_24bit(out_p, filtered.astype(np.float32), sample_rate=sr)
    return True


def simulate_voice(
    voice: str,
    output_wav: Path | str | None = None,
    input_wav: Path | str | None = None,
    instrument: InstrumentConfig | str | Path | None = None,
    normalize: str = "auto",
    target_dbfs: float | None = None,
    max_samples: int | None = None,
    config: Any = None,
    pickup: str | None = None,
    force: bool = False,
    **kwargs: Any,
) -> bool:
    """
    Simulates an instrument voicing digital twin.
    Dispatches directly to simulate_instrument_voicing.
    """
    if config is not None:
        inst_target = config.instrument or instrument
        out_target = config.output_wav or output_wav
        in_target = config.input_wav or input_wav
        max_s = config.max_samples or max_samples
        skip_id = getattr(config, "skip_identity", False)
        norm = getattr(config, "normalize", normalize)
        tgt_db = getattr(config, "target_dbfs", target_dbfs)
        force_val = getattr(config, "force", force)
    else:
        inst_target = instrument
        out_target = output_wav
        in_target = input_wav
        max_s = max_samples
        skip_id = kwargs.get("skip_identity", False)
        norm = normalize
        tgt_db = target_dbfs
        force_val = force

    if inst_target is not None:
        inst_obj = (
            load_instrument(inst_target)
            if not isinstance(inst_target, InstrumentConfig)
            else inst_target
        )
        if pickup and pickup != "auto" and pickup not in inst_obj.pickups:
            raise KeyError(
                f"Pickup '{pickup}' not found on instrument '{inst_obj.id}'. "
                f"Available pickups: {list(inst_obj.pickups.keys())}"
            )
        if inst_obj.electronics == "passive":
            p_key = pickup or getattr(inst_obj, "default_pickup", None) or "p"
            if p_key in inst_obj.pickups and inst_obj.pickups[p_key].circuit is None:
                raise ValueError(
                    f"Passive instrument '{inst_obj.id}' pickup '{p_key}' "
                    f"does not define a '[circuit]' block. Passive source pickups require an explicit "
                    f"circuit model for forward circuit simulation."
                )

        from allomorph.physics import is_voice_matching_source

        is_id = is_voice_matching_source(inst_obj, voice)
        if skip_id and is_id:
            if out_target and Path(out_target).exists():
                Path(out_target).unlink()
            return False

        apply_sat = kwargs.get("apply_saturation", True)
        p_key = (
            pickup
            or getattr(inst_obj, "default_pickup", None)
            or (next(iter(inst_obj.pickups.keys())) if inst_obj.pickups else None)
        )
        src_p = inst_obj.pickups.get(p_key) if p_key else None
        src_mag = (src_p.magnet_type if src_p else None) or (
            "active" if inst_obj.electronics == "active" else "alnico_v"
        )
        src_props = MAGNET_PROPERTIES.get(src_mag, MAGNET_PROPERTIES["alnico_v"])

        try:
            t_inst, v_cfg = resolve_target_voicing(voice, instrument=inst_obj)
            tgt_p = t_inst.pickups.get(v_cfg.pickup)
            tgt_mag = (tgt_p.magnet_type if tgt_p else None) or "alnico_v"
            tgt_props = MAGNET_PROPERTIES.get(tgt_mag, MAGNET_PROPERTIES["alnico_v"])
            if is_id or (
                float(tgt_props.alpha) <= float(src_props.alpha)
                and float(tgt_props.vsat) >= float(src_props.vsat)
            ):
                apply_sat = False
        except KeyError:
            pass
    else:
        apply_sat = kwargs.get("apply_saturation", True)

    try:
        out = simulate_instrument_voicing(
            instrument=inst_target,
            voicing=voice,
            input_wav=in_target,
            output_wav=out_target,
            max_samples=max_s,
            apply_saturation=apply_sat,
            normalize=norm,
            target_dbfs=tgt_db,
            force=force_val,
        )
        return out.exists()
    except KeyError:
        raise KeyError(f"Target voice '{voice}' not found")
