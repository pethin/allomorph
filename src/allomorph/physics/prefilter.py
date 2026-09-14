"""
Allomorph - Minimum-Phase FIR Filter Synthesis & Pre-filtering Engine
Synthesizes causal minimum-phase FIR pre-filters modeling compound spatial acoustics,
scale-length wave-speed conversions, and transducer deconvolution.
"""

from pathlib import Path

import numpy as np

from allomorph.circuit.solver import smooth_soft_knee_db
from allomorph.config.geometry import (
    compute_effective_position,
    resolve_pickup_coils,
    resolve_voice_pickups,
)
from allomorph.config.instruments import get_source_pickup, load_instrument
from allomorph.config.scales import SCALES
from allomorph.config.schema import InstrumentConfig, PickupComponentConfig
from allomorph.config.strings import get_instrument_string
from allomorph.config.voices import VOICES
from allomorph.dsp import FREQS, NUM_TAPS, cinf_smoothstep, synthesize_minimum_phase_fir
from allomorph.physics.aperture import (
    compute_body_microphonic_coupling,
    compute_saddle_boundary_coupling,
    is_voice_matching_source,
    numpy_pickup_acoustic_response,
    numpy_pickup_macro_aperture,
    soft_clamp_displacement_ratio,
)
from allomorph.physics.strings import (
    MEAN_BASS_F0,
    compute_differential_longitudinal_transfer,
    compute_differential_string_transfer,
    get_voice_string,
    resolve_scale_range,
)


def compute_voice_prefilter_firs(
    voice_id: str,
    instrument: InstrumentConfig | str | Path = "30in",
    src_scale: InstrumentConfig | str | float | tuple[float, float] | list[float] | None = None,
    num_taps: int = NUM_TAPS,
    src_pickup_key: str | None = None,
    normalize: bool = False,
) -> list[list[float]]:
    """
    Computes acoustic pre-filter FIRs for each pickup in a target voice configuration using NumPy.
    For single-pickup voices, returns a list with 1 FIR: [fir].
    For multi-pickup voices (e.g. Jazz pair, P/J, P/MM), returns a list of FIRs:
    [fir_pickup_0, fir_pickup_1, ...], enabling independent channel excitation in SPICE.
    """
    cfg = VOICES[voice_id]
    if cfg.preserve_aperture:
        impulse = [1.0] + [0.0] * (num_taps - 1)
        return [impulse]

    target_scale_key = cfg.scale
    if target_scale_key not in SCALES:
        raise KeyError(
            f"Target scale '{target_scale_key}' not found in SCALES configuration. "
            f"Available scales: {list(SCALES.keys())}"
        )
    tgt = SCALES[target_scale_key]

    inst_selector = src_scale if src_scale is not None else instrument
    if isinstance(inst_selector, InstrumentConfig):
        inst = inst_selector
    elif isinstance(inst_selector, (str, Path)):
        inst = load_instrument(inst_selector)
    else:
        inst = load_instrument(instrument)

    src_scale_range = resolve_scale_range(inst)
    tgt_scale_range = resolve_scale_range(tgt if target_scale_key in SCALES else target_scale_key)
    src_scale_m = (src_scale_range[0] + src_scale_range[1]) / 2.0
    tgt_scale_m = (tgt_scale_range[0] + tgt_scale_range[1]) / 2.0
    src_scale_in = src_scale_m / 0.0254
    tgt_scale_in = tgt_scale_m / 0.0254

    if src_pickup_key and src_pickup_key != "auto":
        pickups = inst.pickups
        if src_pickup_key not in pickups:
            raise KeyError(
                f"Pickup '{src_pickup_key}' not found on instrument '{inst.id}'. "
                f"Available pickups: {list(pickups.keys())}"
            )
        p_raw = pickups[src_pickup_key]
        src_pickup = p_raw.model_copy(deep=True)
        src_pickup.id = src_pickup_key
    else:
        src_pickup = get_source_pickup(inst, voice_id)
    src_coils = resolve_pickup_coils(src_pickup, inst)
    src_pos_eff = compute_effective_position(src_coils)

    freqs = np.asarray(FREQS, dtype=np.float64)

    h_src_acoustic = numpy_pickup_acoustic_response(
        freqs, src_coils, scale_length_m=src_scale_range
    )

    src_string = get_instrument_string(inst)
    tgt_string = get_voice_string(cfg)

    sensor_type = cfg.sensor_type
    pickups = resolve_voice_pickups(cfg)
    is_identity = (sensor_type not in ["bridge_force", "direct"]) and is_voice_matching_source(
        inst, voice_id, cfg
    )

    # Scale-Length Tension Snap Filter
    if is_identity:
        h_tension = np.ones_like(freqs)
    else:
        delta_scale = tgt_scale_in - src_scale_in
        if delta_scale <= 0.0:
            h_tension = np.ones_like(freqs)
        else:
            snap_db = 3.5 * np.tanh((1.8 * delta_scale) / (4.0 * 3.5))
            g_snap = 10.0 ** (snap_db / 20.0)
            h_tension = np.sqrt(
                (1.0 + g_snap**2 * (freqs / 2800.0) ** 2) / (1.0 + (freqs / 2800.0) ** 2)
            )

    has_multichannel_circuit = bool(len(pickups) > 1)

    src_components: list[PickupComponentConfig] = (
        src_pickup.components if src_pickup.type == "composite" and src_pickup.components else []
    )
    use_branch_matching = len(src_components) == len(pickups) and len(pickups) > 1

    positions = [compute_effective_position(p.coils) for p in pickups]
    pos_max = max(positions) if positions else 0.0
    c_mean = 2.0 * tgt_scale_m * MEAN_BASS_F0

    tau_diffs: list[float] = []
    if is_identity:
        tau_diffs = [0.0] * len(pickups)
    elif use_branch_matching:
        src_positions = []
        for c in src_components:
            if not c.pickup or c.pickup not in inst.pickups:
                raise KeyError(
                    f"Component pickup '{c.pickup}' not found in instrument '{inst.id}'. "
                    f"Available pickups: {list(inst.pickups.keys())}"
                )
            src_positions.append(
                compute_effective_position(resolve_pickup_coils(inst.pickups[c.pickup], inst))
            )
        src_pos_max = max(src_positions) if src_positions else 0.0
        src_c_mean = 2.0 * src_scale_m * MEAN_BASS_F0
        for i in range(len(pickups)):
            tau_src_i = (
                (src_pos_max - src_positions[i]) / src_c_mean if i < len(src_positions) else 0.0
            )
            tau_tgt_i = (pos_max - positions[i]) / c_mean
            tau_diffs.append(tau_tgt_i - tau_src_i)
    elif len(pickups) > 1:
        for i in range(len(pickups)):
            tau_diffs.append((pos_max - positions[i]) / c_mean)
    else:
        tau_diffs = [0.0] * len(pickups)

    # Vector Causal Normalization: subtract global minimum to guarantee 100% causality (tau_i >= 0)
    # while preserving exact relative inter-pickup phase differentials without clamping
    min_tau = min(tau_diffs) if tau_diffs else 0.0
    tau_causal = [t - min_tau for t in tau_diffs]

    raw_firs = []
    for i, p in enumerate(pickups):
        p_coils = p.coils
        tgt_pos_eff = positions[i]

        if use_branch_matching:
            comp_sub_id = src_components[i].pickup
            if not comp_sub_id or comp_sub_id not in inst.pickups:
                raise KeyError(
                    f"Component pickup '{comp_sub_id}' not found in instrument '{inst.id}'. "
                    f"Available pickups: {list(inst.pickups.keys())}"
                )
            comp_sub_p = inst.pickups[comp_sub_id]
            b_src_coils = resolve_pickup_coils(comp_sub_p, inst)
            b_src_pos_eff = compute_effective_position(b_src_coils)
            b_src_acoustic = numpy_pickup_acoustic_response(
                freqs, b_src_coils, scale_length_m=src_scale_range
            )
        else:
            b_src_coils = src_coils
            b_src_pos_eff = src_pos_eff
            b_src_acoustic = h_src_acoustic

        if sensor_type == "bridge_force":
            eps = 0.08
            h_decomb_raw = b_src_acoustic / (b_src_acoustic**2 + eps)
            mid_mask = (freqs >= 100.0) & (freqs <= 1000.0)
            h_decomb_raw = h_decomb_raw / np.median(h_decomb_raw[mid_mask])

            c_mean_src = 2.0 * src_scale_m * MEAN_BASS_F0
            pos_eff = max(b_src_pos_eff, 0.035)
            f_peak_src = c_mean_src / pos_eff
            f_taper_start = min(f_peak_src, 2500.0)
            f_taper_end = min(1.8 * f_taper_start, 4500.0)
            t = (freqs - f_taper_start) / (f_taper_end - f_taper_start)
            w = 1.0 - cinf_smoothstep(t)
            h_decomb = w * h_decomb_raw + (1.0 - w) * 1.0

            is_flatwound = "flat" in (src_string.type or "")
            f_damp = 4200.0 if is_flatwound else 3800.0
            h_damp = 1.0 / np.sqrt(1.0 + (freqs / f_damp) ** 4)

            f_sub = 10.0
            h_sub = np.sqrt(freqs**2 / (f_sub**2 + freqs**2))

            h_acoustic_transfer = h_decomb * h_damp * h_sub
            h_pos = np.ones_like(freqs)
        elif is_identity:
            h_acoustic_transfer = np.ones_like(freqs)
            h_pos = np.ones_like(freqs)
        else:
            if sensor_type == "direct":
                h_tgt_acoustic = np.ones_like(freqs)
            else:
                h_tgt_acoustic = numpy_pickup_acoustic_response(
                    freqs, p_coils, scale_length_m=tgt_scale_range
                )
            h_src_macro = numpy_pickup_macro_aperture(
                freqs, b_src_coils, scale_length_m=src_scale_range
            )
            eps = 0.01
            h_quotient = (h_tgt_acoustic * h_src_macro) / (h_src_macro**2 + eps**2)
            q_db = 20.0 * np.log10(np.maximum(h_quotient, 1e-6))
            g_max_db = 12.0 if sensor_type == "direct" else 8.0
            g_min_db = -14.0
            sigma = 0.5 * (1.0 + np.tanh(0.5 * q_db))
            f_pos = smooth_soft_knee_db(q_db, thresh=6.0, ceiling=g_max_db, alpha=2.0)
            f_neg = -smooth_soft_knee_db(-q_db, thresh=10.0, ceiling=abs(g_min_db), alpha=2.0)
            q_soft_db = sigma * f_pos + (1.0 - sigma) * f_neg
            h_acoustic_transfer = 10.0 ** (q_soft_db / 20.0)

            if sensor_type == "direct":
                h_pos = np.ones_like(freqs)
            else:
                eta_tgt = tgt_pos_eff / tgt_scale_m
                eta_src = b_src_pos_eff / src_scale_m
                delta_g = 20.0 * np.log10(max(eta_tgt / max(eta_src, 1e-4), 1e-6))
                delta_g_soft = soft_clamp_displacement_ratio(delta_g)
                g_0 = 10.0 ** (delta_g_soft / 20.0)
                h_pos = np.sqrt((g_0**2 + (freqs / 220.0) ** 2) / (1.0 + (freqs / 220.0) ** 2))

        p_weight = 1.0 if has_multichannel_circuit else p.weight
        p_pol = p.polarity
        scale_fac = abs(p_weight * p_pol)

        if (
            sensor_type != "bridge_force"
            and cfg.target_string
            and cfg.target_string != "roundwound_nickel_standard"
        ):
            h_str_diff = compute_differential_string_transfer(freqs, src_string, tgt_string)
            h_long_diff = compute_differential_longitudinal_transfer(
                freqs, src_string, tgt_string, scale_length_inches=src_scale_in
            )
        else:
            h_str_diff = np.ones_like(freqs)
            h_long_diff = np.ones_like(freqs)

        h_scale_tension = np.ones_like(freqs) if is_identity else h_tension
        h_body = (
            np.ones_like(freqs)
            if is_identity
            else compute_body_microphonic_coupling(freqs, src_pickup, cfg, inst=inst)
        )

        if is_identity or sensor_type == "bridge_force":
            h_saddle_diff = np.ones_like(freqs)
        else:
            h_saddle_tgt = compute_saddle_boundary_coupling(freqs, tgt_pos_eff, tgt_scale_m)
            h_saddle_src = compute_saddle_boundary_coupling(freqs, b_src_pos_eff, src_scale_m)
            r_saddle_db = 20.0 * np.log10(
                np.maximum(h_saddle_tgt / np.maximum(h_saddle_src, 1e-6), 1e-6)
            )
            h_saddle_diff = 10.0 ** (r_saddle_db / 20.0)

        prefilter_curve = (
            scale_fac
            * h_acoustic_transfer
            * h_pos
            * h_scale_tension
            * h_str_diff
            * h_long_diff
            * h_body
            * h_saddle_diff
        )
        fir_raw = synthesize_minimum_phase_fir(prefilter_curve, num_taps=num_taps, normalize=False)

        tau_i = tau_causal[i]
        if tau_i > 0.0:
            delay_samples = round(tau_i * 48000.0)
            if 0 < delay_samples < num_taps:
                fir_raw = [0.0] * delay_samples + fir_raw[: num_taps - delay_samples]

        raw_firs.append(fir_raw)

    global_peak = max(max(abs(x) for x in fir) for fir in raw_firs)
    if (normalize and global_peak > 0) or global_peak > 0.99:
        return [[(x / global_peak) * 0.99 for x in fir] for fir in raw_firs]
    return raw_firs
