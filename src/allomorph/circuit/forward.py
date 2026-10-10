"""
Allomorph - Direct Unified Forward Simulation Engine (DSP Gen 6)
Convolves dry string excitation audio with authentic instrument physical aperture,
loaded RLC circuit, active preamps, and string mechanics into 24-bit PCM wet stems.
"""

import functools
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Literal, overload

import numpy as np
import pedalboard

from allomorph.circuit.audio import find_default_input_audio
from allomorph.circuit.parser import MAGNET_PROPERTIES
from allomorph.circuit.saturation import (
    apply_active_pickup_dynamics,
    apply_oversampled_saturation,
)
from allomorph.circuit.solver import solve_mna_harness
from allomorph.config.geometry import compute_effective_position, resolve_pickup_coils
from allomorph.config.scales import REPO_ROOT, resolve_scale_range
from allomorph.config.schema import (
    InstrumentConfig,
    PreampConfig,
    StringPresetConfig,
    VoicingConfig,
)
from allomorph.config.strings import get_instrument_string
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
    compute_forward_string_transfer,
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


@overload
def simulate_instrument_voicing(
    instrument: InstrumentConfig,
    voicing: VoicingConfig,
    input_wav: Path | str | None = ...,
    input_audio: np.ndarray | None = ...,
    output_wav: Path | str | None = ...,
    return_audio: Literal[False] = False,
    max_samples: int | None = ...,
    num_taps: int = ...,
    apply_dither: bool = ...,
    apply_saturation: bool = ...,
    vol_pos: float | None = ...,
    tone_pos: float | None = ...,
    blend_pos: float | None = ...,
    cable_pf: float | None = ...,
    dc_block: bool = ...,
    normalize: str = ...,
    target_dbfs: float | None = ...,
    force: bool = ...,
    preamps: Mapping[str, PreampConfig] | None = ...,
    strings: Mapping[str, StringPresetConfig] | None = ...,
) -> Path: ...


@overload
def simulate_instrument_voicing(
    instrument: InstrumentConfig,
    voicing: VoicingConfig,
    input_wav: Path | str | None = ...,
    input_audio: np.ndarray | None = ...,
    output_wav: None = None,
    return_audio: Literal[True] = ...,
    max_samples: int | None = ...,
    num_taps: int = ...,
    apply_dither: bool = ...,
    apply_saturation: bool = ...,
    vol_pos: float | None = ...,
    tone_pos: float | None = ...,
    blend_pos: float | None = ...,
    cable_pf: float | None = ...,
    dc_block: bool = ...,
    normalize: str = ...,
    target_dbfs: float | None = ...,
    force: bool = ...,
    preamps: Mapping[str, PreampConfig] | None = ...,
    strings: Mapping[str, StringPresetConfig] | None = ...,
) -> np.ndarray: ...


@overload
def simulate_instrument_voicing(
    instrument: InstrumentConfig,
    voicing: VoicingConfig,
    input_wav: Path | str | None = ...,
    input_audio: np.ndarray | None = ...,
    output_wav: Path | str = ...,
    return_audio: Literal[True] = ...,
    max_samples: int | None = ...,
    num_taps: int = ...,
    apply_dither: bool = ...,
    apply_saturation: bool = ...,
    vol_pos: float | None = ...,
    tone_pos: float | None = ...,
    blend_pos: float | None = ...,
    cable_pf: float | None = ...,
    dc_block: bool = ...,
    normalize: str = ...,
    target_dbfs: float | None = ...,
    force: bool = ...,
    preamps: Mapping[str, PreampConfig] | None = ...,
    strings: Mapping[str, StringPresetConfig] | None = ...,
) -> tuple[Path, np.ndarray]: ...


def simulate_instrument_voicing(
    instrument: InstrumentConfig,
    voicing: VoicingConfig,
    input_wav: Path | str | None = None,
    input_audio: np.ndarray | None = None,
    output_wav: Path | str | None = None,
    return_audio: bool = False,
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
    preamps: Mapping[str, PreampConfig] | None = None,
    strings: Mapping[str, StringPresetConfig] | None = None,
) -> Path | tuple[Path, np.ndarray] | np.ndarray:
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
    inst = instrument
    voicing_cfg = voicing
    voicing_id = (
        voicing_cfg.tone_name.lower().replace(" ", "_").replace("∕", "_").replace("/", "_")
        if voicing_cfg.tone_name
        else (voicing_cfg.id or "default_voicing")
    )

    # Resolve input audio
    in_path: Path | None = None
    if input_audio is not None:
        raw_audio = np.asarray(input_audio, dtype=np.float64)
        if max_samples is not None and len(raw_audio) > max_samples:
            raw_audio = raw_audio[:max_samples]
        sr = 48000
    elif not input_wav or not Path(input_wav).exists():
        found = find_default_input_audio()
        if not found:
            raise FileNotFoundError(
                f"Input audio '{input_wav}' not found, and no standard excitation audio (input.wav) was detected."
            )
        in_path = Path(found)
        raw_audio, sr = read_wav(in_path, max_samples=max_samples, dtype=np.float64)
    else:
        in_path = Path(input_wav)
        raw_audio, sr = read_wav(in_path, max_samples=max_samples, dtype=np.float64)

    # Resolve output path
    out_path: Path | None = None
    if output_wav is not None:
        out_path = Path(output_wav)
        out_path.parent.mkdir(parents=True, exist_ok=True)
    elif not return_audio:
        out_path = WET_AUDIO_DIR / inst.id / f"{voicing_id}.wav"
        out_path.parent.mkdir(parents=True, exist_ok=True)

    # Cryptographic Bit-Provenance Cache Check
    if in_path is not None and out_path is not None and not force:
        inst_ver = getattr(inst, "version", 1)
        voice_ver = getattr(voicing_cfg, "version", 1)
        v_tag = resolve_tri_part_version(DSP_GENERATION, inst_ver, voice_ver)
        if is_wet_stem_valid(out_path, base_dry_path=in_path, expected_version=v_tag):
            print(f"[Forward Sim] Cached {inst.id}:{voicing_id} ({v_tag}) -> {out_path.name}")
            if return_audio:
                audio_cached, _ = read_wav(out_path)
                return out_path, audio_cached
            return out_path

    if preamps is None:
        from allomorph.config.preamps import PREAMPS

        preamps = PREAMPS

    # Resolve active harness and MNA
    harness_id = voicing_cfg.harness
    used_mna = bool(getattr(inst, "harnesses", None) and harness_id in inst.harnesses)

    from allomorph.config.voices import (
        resolve_voicing_active_coils,
        resolve_voicing_active_pickups,
    )

    active_pickups = resolve_voicing_active_pickups(inst, voicing_cfg)
    pickup_key = active_pickups[0] if active_pickups else next(iter(inst.pickups.keys()))
    pickup_cfg = inst.pickups[pickup_key]

    if not used_mna:
        harnesses_list: list[str] = (
            list(inst.harnesses.keys()) if hasattr(inst, "harnesses") and inst.harnesses else []
        )
        raise KeyError(
            f"Voicing '{voicing_cfg.name}' on instrument '{inst.id}' references harness '{harness_id}', "
            f"which was not found in inst.harnesses: {harnesses_list}"
        )

    f = np.asarray(FREQS, dtype=np.float64)
    s = 2j * np.pi * f
    scale_range = resolve_scale_range(inst)
    scale_m = (scale_range[0] + scale_range[1]) / 2.0
    scale_in = scale_m / 0.0254

    input_mono = raw_audio[0] if raw_audio.ndim > 1 else raw_audio
    n_samples = len(input_mono)

    curves_dict = solve_mna_harness(
        inst, inst.harnesses[harness_id], voicing_cfg, freqs=f, preamps=preamps
    )

    def _lookup_curve(*keys: str) -> np.ndarray:
        for k in keys:
            if k and k in curves_dict and curves_dict[k] is not None:
                return curves_dict[k]
        return next(iter(curves_dict.values()))

    if len(active_pickups) >= 2:
        curves = [_lookup_curve(p_id) for p_id in active_pickups]
    else:
        p_coils = resolve_pickup_coils(pickup_cfg, inst)
        active_c_ids = resolve_voicing_active_coils(inst, voicing_cfg, pickup_key)
        if active_c_ids:
            p_coils = [c for c in p_coils if c.id in active_c_ids or not c.id]
        if len(p_coils) > 1:
            curves = [
                _lookup_curve(
                    f"{pickup_cfg.id}.{c.id}",
                    c.id or "",
                    f"{pickup_key}.{c.id}",
                    pickup_key,
                )
                for c in p_coils
            ]
        else:
            curves = [_lookup_curve(pickup_key, pickup_cfg.id or "")]
    h_preamp = np.ones_like(f, dtype=np.float64)

    # 3. Viscoelastic String Wrap Damping H_wrap(f)
    inst_string = get_instrument_string(inst, strings=strings)
    target_string = (
        strings[voicing_cfg.string_preset_override]
        if (
            voicing_cfg.string_preset_override
            and strings
            and voicing_cfg.string_preset_override in strings
        )
        else inst_string
    )
    if voicing_cfg.sensor_type == "direct":
        h_wrap = np.ones_like(f, dtype=np.float64)
    else:
        h_wrap = compute_forward_string_transfer(
            f, target_string, scale_length_inches=scale_in, sensor_type=voicing_cfg.sensor_type
        )

    is_composite_sim = False
    composite_audio = np.zeros(n_samples, dtype=np.float64)
    h_ac: np.ndarray = np.ones_like(f, dtype=np.float64)
    h_elec: np.ndarray = np.ones_like(f, dtype=np.float64)

    if voicing_cfg.sensor_type == "direct":
        if curves is not None:
            h_base = np.asarray(curves[0], dtype=np.float64)
        else:
            h_base = np.ones_like(f, dtype=np.float64)
    elif voicing_cfg.sensor_type == "bridge_force":
        # 1. Double bass maple bridge rocking resonance (Woodhouse/Guettler: 800 Hz, Q ~ 1.8, +2.5 dB)
        f_rock = 800.0
        q_rock = 1.8
        g_rock = 10.0 ** (2.5 / 20.0)
        w = 2.0 * np.pi * f
        w0 = 2.0 * np.pi * f_rock
        s = 1j * w
        h_rock = np.abs(
            (s**2 + (g_rock * w0 / q_rock) * s + w0**2) / (s**2 + (w0 / q_rock) * s + w0**2)
        )

        # 2. Bridge structure wood mass inertial roll-off above 3.2 kHz
        f_damp = 3200.0
        h_mass = 1.0 / np.sqrt(1.0 + (f / f_damp) ** 4)

        h_ac = h_rock * h_mass
        if curves is not None:
            h_elec = np.asarray(curves[0], dtype=np.float64)
            h_base = h_ac * h_elec
        else:
            h_base = h_ac
    else:
        # Magnetic sensor: determine single pickup vs multi-pickup composite
        p_components = getattr(pickup_cfg, "components", None) or []
        is_composite = (
            len(active_pickups) >= 2
            or pickup_cfg.type == "composite"
            or bool(p_components)
            or (curves is not None and len(curves) > 1)
        )

        single_positions = [
            float(p.position_from_bridge_m)
            for p in inst.pickups.values()
            if not getattr(p, "components", None) and p.position_from_bridge_m is not None
        ]
        # Self-referential leveling: multi-pickup basses level bridge pickups relative to their own forward-most pickup.
        # Single-pickup basses receive ref_pos = None and evaluate to exact 1.0000 (0.00 dB).
        ref_pos = max(single_positions) if len(single_positions) > 1 else None

        if is_composite and (p_components or len(active_pickups) >= 2):
            is_composite_sim = True
            N = 8192
            f_bins = np.fft.rfftfreq(N, 1.0 / 48000.0)
            c_mean = 2.0 * scale_m * MEAN_BASS_F0

            # Resolve branch components
            branch_sub_pickups = []
            if p_components:
                for comp in p_components:
                    if comp.pickup and comp.pickup in inst.pickups:
                        branch_sub_pickups.append(
                            (inst.pickups[comp.pickup], float(comp.weight), float(comp.polarity))
                        )
            else:
                for p_id in active_pickups:
                    branch_sub_pickups.append((inst.pickups[p_id], 1.0, 1.0))

            branch_coils_list = []
            branch_positions = []
            for sp, w_comp, pol_comp in branch_sub_pickups:
                b_coils = resolve_pickup_coils(sp, inst)
                branch_coils_list.append((b_coils, w_comp, pol_comp))
                branch_positions.append(compute_effective_position(b_coils))

            pos_max = max(branch_positions) if branch_positions else 0.0
            H_channels = []
            H_channels_delayed = []
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
                fir_circ = synthesize_minimum_phase_fir(c_curve, num_taps=2048, normalize=False)
                H_ac_fft = np.fft.rfft(fir_ac, N)
                H_circ_fft = np.fft.rfft(fir_circ, N)
                H_channels.append(H_ac_fft * H_circ_fft)

                tau_i = (pos_max - b_pos) / c_mean if len(branch_coils_list) > 1 else 0.0
                delay_samples = round(tau_i * 48000.0)
                fir_ac_del = list(fir_ac)
                if 0 < delay_samples < 2048:
                    fir_ac_del = [0.0] * delay_samples + fir_ac_del[: 2048 - delay_samples]
                H_ac_del_fft = np.fft.rfft(fir_ac_del, N)
                H_channels_delayed.append(H_ac_del_fft * H_circ_fft)
                peaks.append(delay_samples)

                # Block 1: Transducer Stage for this branch (zero latency, starts strictly at sample 0)
                v_ac = fft_convolve(
                    input_mono, np.asarray(fir_ac, dtype=np.float64), mode="causal"
                )[:n_samples]

                if apply_saturation:
                    sp = branch_sub_pickups[i][0]
                    sp_mag = sp.magnet_type or (
                        "active" if getattr(inst, "electronics", "") == "active" else "alnico_v"
                    )
                    if sp_mag not in MAGNET_PROPERTIES:
                        raise KeyError(f"Unrecognized magnet type: '{sp_mag}'")
                    sp_props = MAGNET_PROPERTIES[sp_mag]
                    sp_vsat = float(sp.vsat) if sp.vsat is not None else float(sp_props.vsat)
                    sp_alpha = float(sp.alpha) if sp.alpha is not None else float(sp_props.alpha)
                    drive_db = float(getattr(voicing_cfg, "gain_db", 0.0) or 0.0)
                    drive_in = v_ac if drive_db == 0.0 else v_ac * (10.0 ** (drive_db / 20.0))
                    emf = apply_oversampled_saturation(
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
                else:
                    emf = v_ac

                # Block 2: Coil Output Stage for this branch
                v_coil = fft_convolve(emf, np.asarray(fir_circ, dtype=np.float64), mode="causal")[
                    :n_samples
                ]

                # Check if sub-pickup defines its own active circuit
                sub_p = branch_sub_pickups[i][0]
                sub_active_var = sub_p.active_variant
                if sub_active_var is not None and apply_saturation:
                    sub_vsat = float(sub_p.vsat) if sub_p.vsat is not None else 2.40
                    b_audio = apply_active_pickup_dynamics(
                        v_coil.astype(np.float32),
                        vsat=sub_vsat,
                        k_level=0.0,
                        tau_att=0.003,
                        tau_rel=0.045,
                        sr=sr,
                    ).astype(np.float64)
                else:
                    b_audio = v_coil

                branch_audios.append(b_audio)

            raw_sum = np.sum(branch_audios, axis=0)
            H_del_arr = np.array(H_channels_delayed)
            delta_samples = max(peaks) - min(peaks) if len(peaks) > 1 else 0

            if len(H_del_arr) > 1 and delta_samples > 0:
                P_coh_raw = np.abs(np.sum(H_del_arr, axis=0)) ** 2
                P_incoh = np.sum(np.abs(H_del_arr) ** 2, axis=0)

                eps_quad = 0.18
                P_coh_reg = P_coh_raw + (eps_quad**2) * P_incoh

                H_dc = np.abs(H_del_arr[:, 0])
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

                # Universal C^inf Cross-Coherence Decay Spatial Filter
                # Shapes the in-phase sum of saturated pickup branches into the theoretical
                # blended spatial aperture response, seamlessly transitioning from low-frequency
                # coherent interference (mid notch) to high-frequency incoherent power summation
                # without injecting non-invertible comb nulls into the time domain.
                H_nodelay_arr = np.array(H_channels)
                S_in_phase = np.abs(np.sum(H_nodelay_arr, axis=0))
                H_spatial_ratio = M_blend / np.maximum(S_in_phase, 1e-6)
                fir_spatial = synthesize_minimum_phase_fir(
                    H_spatial_ratio, num_taps=num_taps, normalize=False
                )
                composite_audio = fft_convolve(
                    raw_sum, np.asarray(fir_spatial, dtype=np.float64), mode="causal"
                )[:n_samples]

                mag_spectrum = M_blend
            elif len(H_channels) > 1:
                composite_audio = raw_sum
                mag_spectrum = np.abs(np.sum(np.array(H_channels), axis=0))
            else:
                composite_audio = raw_sum
                mag_spectrum = np.abs(H_channels[0])

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
        h_preamp * h_wrap,
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
        # Block 1: Transducer Stage (Spatial Macro-Aperture -> Induced Open-Circuit EMF)
        fir_ac = synthesize_minimum_phase_fir(np.abs(h_ac), num_taps=num_taps, normalize=False)
        fir_ac_np = np.asarray(fir_ac, dtype=np.float64)
        v_ac = fft_convolve(input_mono, fir_ac_np, mode="causal")[:n_samples]

        # Magnetic core saturation produces open-circuit coil EMF
        has_direct_dynamics = voicing_cfg.sensor_type == "direct" and (
            pickup_cfg is not None and pickup_cfg.vsat is not None
        )
        if apply_saturation and (voicing_cfg.sensor_type == "magnetic" or has_direct_dynamics):
            sp = pickup_cfg if pickup_cfg is not None else next(iter(inst.pickups.values()))
            sp_mag = sp.magnet_type or (
                "active" if getattr(inst, "electronics", "") == "active" else "alnico_v"
            )
            if sp_mag not in MAGNET_PROPERTIES:
                raise KeyError(f"Unrecognized magnet type: '{sp_mag}'")
            sp_props = MAGNET_PROPERTIES[sp_mag]
            mag_vsat = float(sp.vsat) if sp.vsat is not None else float(sp_props.vsat)
            sp_alpha = float(sp.alpha) if sp.alpha is not None else float(sp_props.alpha)
            drive_db = float(getattr(voicing_cfg, "gain_db", 0.0) or 0.0)
            drive_in = v_ac if drive_db == 0.0 else v_ac * (10.0 ** (drive_db / 20.0))
            emf = apply_oversampled_saturation(
                drive_in.astype(np.float32),
                vsat=mag_vsat,
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
        elif apply_saturation and voicing_cfg.sensor_type == "bridge_force":
            p_vsat = (
                float(pickup_cfg.vsat)
                if (pickup_cfg is not None and pickup_cfg.vsat is not None)
                else 0.42
            )
            drive_db = float(getattr(voicing_cfg, "gain_db", 0.0) or 0.0)
            drive_in = v_ac if drive_db == 0.0 else v_ac * (10.0 ** (drive_db / 20.0))
            max_in = float(np.max(np.abs(drive_in))) if len(drive_in) > 0 else 0.0
            if max_in > 0.10 and p_vsat > 0.0:
                alpha_p = (
                    float(pickup_cfg.alpha)
                    if (pickup_cfg is not None and pickup_cfg.alpha is not None)
                    else 0.15
                )
                v_piezo = drive_in * (1.0 + alpha_p * np.tanh(drive_in / p_vsat))
                emf = (p_vsat * np.tanh(v_piezo / p_vsat)).astype(np.float64)
            else:
                emf = drive_in
        else:
            emf = v_ac

        # Block 2: Pickup Output Stage (Coil RLC & Active Pickup Dynamics)
        fir_elec = synthesize_minimum_phase_fir(np.abs(h_elec), num_taps=num_taps, normalize=False)
        fir_elec_np = np.asarray(fir_elec, dtype=np.float64)
        v_coil = fft_convolve(emf, fir_elec_np, mode="causal")[:n_samples]

        # Active pickup internal op-amp dynamics
        active_var = pickup_cfg.active_variant if pickup_cfg is not None else None
        if active_var is not None and apply_saturation:
            v_sat = (
                float(pickup_cfg.vsat)
                if (pickup_cfg is not None and pickup_cfg.vsat is not None)
                else 2.40
            )
            filtered = apply_active_pickup_dynamics(
                v_coil.astype(np.float32),
                vsat=v_sat,
                k_level=0.0,
                tau_att=0.003,
                tau_rel=0.045,
                sr=sr,
            ).astype(np.float64)
        else:
            filtered = v_coil

        # Block 3: Downstream Electronics & Cable Stage
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
    if out_path is not None:
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
        if in_path is not None:
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

    if return_audio:
        if out_path is not None:
            return out_path, filtered
        return filtered

    assert out_path is not None
    return out_path
