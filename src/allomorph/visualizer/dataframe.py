"""
Allomorph Visualizer - Frequency Response Modeling and DataFrame Generation
Calculates magnitude frequency responses for target voicings, frontend deconvolutions,
and composite signal flow stages using Polars and NumPy.
"""

from collections.abc import Sequence
from typing import Any

import numpy as np
import polars as pl

from allomorph.circuit import (
    apply_magnet_properties_to_model,
    compute_circuit_transfer_functions,
    compute_differential_circuit_transfer_functions,
    load_circuit,
    smooth_soft_knee_db,
)
from allomorph.config.geometry import (
    compute_effective_position,
    resolve_voice_pickups,
)
from allomorph.config.instruments import (
    get_source_pickup,
    load_instrument,
)
from allomorph.config.scales import SCALES, resolve_scale_range
from allomorph.config.schema import InstrumentConfig, VoiceConfig
from allomorph.config.strings import STRINGS, get_voice_string
from allomorph.config.voices import VOICES
from allomorph.dsp import (
    FREQS,
    synthesize_minimum_phase_fir,
)
from allomorph.physics import (
    MEAN_BASS_F0,
    UNIVERSAL_DATUM_POS_M,
    compute_differential_longitudinal_transfer,
    compute_differential_string_transfer,
    compute_displacement_proximity_shelf,
    compute_pickup_isolation_leveling,
    compute_saddle_boundary_coupling,
    compute_voice_prefilter_firs,
    is_voice_matching_source,
    numpy_pickup_acoustic_response,
)

NUM_POINTS = 600
F_MIN = 20.0
F_MAX = 20000.0

log_freqs = [F_MIN * (F_MAX / F_MIN) ** (i / (NUM_POINTS - 1)) for i in range(NUM_POINTS)]

_OUTPUT_VOICE_DF_CACHE: dict[str, pl.DataFrame] = {}
_DIFF_VOICE_DF_CACHE: dict[tuple[str, str, str], pl.DataFrame] = {}


def build_voice_dataframe(
    voice_id: str,
    cfg: VoiceConfig,
    instrument: InstrumentConfig | str = "30in",
    src_scale: InstrumentConfig | str | None = None,
    mode: str = "difference",
    include_mode_col: bool = False,
    src_pickup_key: str | None = None,
) -> pl.DataFrame:
    """
    Calculates magnitude frequency response in dB for a voice using NumPy vector math and Polars.
    mode="output": Absolute acoustic aperture + loaded SPICE circuit frequency response of the target voice.
    mode="difference": Regularized differential transfer function (H_target / H_source) applied to the source instrument.
    """
    if mode == "output" and cfg == VOICES.get(voice_id) and voice_id in _OUTPUT_VOICE_DF_CACHE:
        base_df = _OUTPUT_VOICE_DF_CACHE[voice_id]
        return (
            base_df.with_columns(pl.lit("Output Voice").alias("mode"))
            if include_mode_col
            else base_df
        )

    inst_selector = src_scale if src_scale is not None else instrument
    inst = (
        inst_selector
        if isinstance(inst_selector, InstrumentConfig)
        else load_instrument(inst_selector)
    )

    cache_key_diff = (inst.id, voice_id, src_pickup_key or "")
    if (
        mode == "difference"
        and cfg == VOICES.get(voice_id)
        and cache_key_diff in _DIFF_VOICE_DF_CACHE
    ):
        base_df = _DIFF_VOICE_DF_CACHE[cache_key_diff]
        return (
            base_df.with_columns(pl.lit("Input/Output Difference").alias("mode"))
            if include_mode_col
            else base_df
        )

    tgt_scale = cfg.scale
    _ = SCALES[tgt_scale]

    freqs = np.asarray(log_freqs, dtype=np.float64)
    pickups = resolve_voice_pickups(cfg)
    tgt_circuit = cfg.circuit

    sensor_type = cfg.sensor_type
    tgt_string = get_voice_string(cfg)
    is_passive = inst.electronics == "passive"
    if src_pickup_key:
        is_spatial_match = (
            (mode != "output")
            and (inst.pickup_mapping.get(voice_id, inst.default_pickup) == src_pickup_key)
            and is_voice_matching_source(inst, voice_id, cfg)
        )
    else:
        is_spatial_match = (mode != "output") and is_voice_matching_source(inst, voice_id, cfg)

    is_pure_di = voice_id == "studio_direct" or (
        tgt_circuit is not None and getattr(load_circuit(tgt_circuit), "no_eq", False)
    )
    if is_pure_di and mode == "output":
        data = {
            "frequency": log_freqs,
            "magnitude_db": [0.0] * len(log_freqs),
            "voice_id": voice_id,
            "voice_name": cfg.name,
            "topology": cfg.topology,
            "description": cfg.description,
        }
        df = pl.DataFrame(data)
        if cfg == VOICES.get(voice_id):
            _OUTPUT_VOICE_DF_CACHE[voice_id] = df
        if include_mode_col:
            return df.with_columns(pl.lit("Output Voice").alias("mode"))
        return df

    if mode == "output":
        # 1. Output Voice: Target acoustic aperture + loaded SPICE circuit + string mechanics
        model = load_circuit(tgt_circuit)
        apply_magnet_properties_to_model(model, cfg)
        circuit_curves = compute_circuit_transfer_functions(model, freqs=FREQS)

        if sensor_type == "bridge_force":
            f_lin = freqs
            is_flatwound = "flat" in (tgt_string.type or "")
            f_damp = 4200.0 if is_flatwound else 3800.0
            h_damp = 1.0 / np.sqrt(1.0 + (f_lin / f_damp) ** 4)
            f_sub = 10.0
            h_sub = np.sqrt(f_lin**2 / (f_sub**2 + f_lin**2))
            c_curve = circuit_curves[0] if circuit_curves else [1.0] * len(FREQS)
            branch = np.interp(freqs, FREQS, np.asarray(c_curve, dtype=np.float64)) * h_damp * h_sub
            h_tgt_total = branch
        elif sensor_type == "direct":
            c_curve = circuit_curves[0] if circuit_curves else [1.0] * len(FREQS)
            h_tgt_total = np.interp(freqs, FREQS, np.asarray(c_curve, dtype=np.float64))
        else:
            tgt_scale_range = resolve_scale_range(tgt_scale)
            tgt_scale_m = (tgt_scale_range[0] + tgt_scale_range[1]) / 2.0
            positions = [compute_effective_position(p.coils) for p in pickups]
            pos_max = max(positions) if positions else 0.0
            ref_pos = cfg.ref_pos_m or max(pos_max, UNIVERSAL_DATUM_POS_M)
            c_mean = 2.0 * tgt_scale_m * MEAN_BASS_F0

            N = 8192
            f_bins = np.fft.rfftfreq(N, 1.0 / 48000.0)
            H_channels = []
            peaks = []

            for i, p in enumerate(pickups):
                c_curve_raw = circuit_curves[i] if i < len(circuit_curves) else [1.0] * len(FREQS)
                c_curve = np.asarray(c_curve_raw, dtype=np.float64)
                p_weight = p.weight
                p_pol = p.polarity
                weight_fac = 1.0 if len(circuit_curves) > 1 else p_weight

                ac_raw = numpy_pickup_acoustic_response(
                    f_bins, p.coils, scale_length_m=tgt_scale_range
                ) * (weight_fac * p_pol)

                p_pos = compute_effective_position(p.coils)
                h_pos = compute_displacement_proximity_shelf(f_bins, p_pos, scale_m=tgt_scale_m)
                k_iso = compute_pickup_isolation_leveling(
                    p_pos, scale_m=tgt_scale_m, ref_pos_m=ref_pos
                )
                ac = ac_raw * h_pos * k_iso

                min_pos = min((c.position_from_bridge_m for c in p.coils), default=0.10)
                if min_pos < 0.075:
                    h_saddle = compute_saddle_boundary_coupling(f_bins, min_pos)
                    ac = ac * np.asarray(h_saddle, dtype=np.float64)

                fir_ac = synthesize_minimum_phase_fir(ac, num_taps=2048, normalize=False)
                tau_i = (pos_max - positions[i]) / c_mean if len(pickups) > 1 else 0.0
                delay_samples = round(tau_i * 48000.0)
                if 0 < delay_samples < 2048:
                    fir_ac = [0.0] * delay_samples + fir_ac[: 2048 - delay_samples]
                peaks.append(int(np.argmax(np.abs(fir_ac))))

                fir_circ = synthesize_minimum_phase_fir(c_curve, num_taps=2048, normalize=False)
                H_channels.append(np.fft.rfft(fir_ac, N) * np.fft.rfft(fir_circ, N))

            H_channels = np.array(H_channels)
            delta_samples = max(peaks) - min(peaks) if len(peaks) > 1 else 0

            if len(H_channels) > 1 and delta_samples > 0:
                P_coherent = np.abs(np.sum(H_channels, axis=0)) ** 2
                P_incoherent = np.sum(np.abs(H_channels) ** 2, axis=0)
                delta_tau = delta_samples / 48000.0
                f_notch = 1.0 / (2.0 * delta_tau)
                f_mid = 1.35 * f_notch
                f_sigma = max(0.35 * f_notch, 1.0)
                gamma = 0.5 * (1.0 - np.tanh((f_bins - f_mid) / f_sigma))
                mag_spectrum = np.sqrt(gamma * P_coherent + (1.0 - gamma) * P_incoherent)
            elif len(H_channels) > 1:
                mag_spectrum = np.abs(np.sum(H_channels, axis=0))
            else:
                mag_spectrum = np.abs(H_channels[0])

            h_tgt_total = np.interp(freqs, f_bins, mag_spectrum)

        # Scale-Length Tension Dynamics for target instrument
        tgt_scale_in = (
            41.25
            if tgt_scale == "upright"
            else (
                37.0
                if tgt_scale in ["multiscale", "37in"]
                else (35.0 if tgt_scale == "multiscale_super" else 34.0)
            )
        )
        r_L = tgt_scale_in / 34.0
        if sensor_type == "bridge_force":
            g_snap = r_L**1.5
            h_tension = np.sqrt(
                (1.0 + g_snap**2 * (freqs / 2800.0) ** 2) / (1.0 + (freqs / 2800.0) ** 2)
            )
        else:
            g_excursion = 1.0 / r_L
            g_snap = r_L**1.5
            h_tension = np.sqrt(
                (g_excursion**2 + (freqs / 100.0) ** 2) / (1.0 + (freqs / 100.0) ** 2)
            ) * np.sqrt((1.0 + g_snap**2 * (freqs / 2800.0) ** 2) / (1.0 + (freqs / 2800.0) ** 2))

        # String voicing for target instrument (relative to standard nickel roundwound)
        if (
            sensor_type != "bridge_force"
            and cfg.target_string
            and cfg.target_string != "roundwound_nickel_standard"
        ):
            std_str = STRINGS["roundwound_nickel_standard"]
            scale_in = tgt_scale_in
            h_str = compute_differential_string_transfer(freqs, std_str, tgt_string)
            h_long = compute_differential_longitudinal_transfer(
                freqs, std_str, tgt_string, scale_length_inches=scale_in
            )
        else:
            h_str = np.ones_like(freqs)
            h_long = np.ones_like(freqs)

        mag_raw = h_tgt_total * h_tension * h_str * h_long

    else:
        # 2. Input/Output Difference: H_diff = H_target / H_source
        if src_pickup_key and src_pickup_key in inst.pickups:
            p_raw = inst.pickups[src_pickup_key]
            src_pickup = p_raw.model_copy(deep=True)
            src_pickup.id = src_pickup_key
        else:
            src_pickup = get_source_pickup(inst, voice_id)
        src_circuit = src_pickup.circuit

        if not src_circuit and is_passive:
            raise ValueError(
                f"Passive instrument '{inst.id}' pickup '{src_pickup.id or 'unknown'}' "
                f"does not define a '[circuit]' block. Passive source pickups require an explicit "
                f"circuit model for differential deconvolution."
            )

        if src_circuit:
            model = load_circuit(tgt_circuit)
            apply_magnet_properties_to_model(model, cfg)
            src_model = load_circuit(src_circuit)
            apply_magnet_properties_to_model(src_model, src_pickup)
            circuit_curves = compute_differential_circuit_transfer_functions(
                model, src_model, freqs=FREQS
            )
        else:
            model = load_circuit(tgt_circuit)
            apply_magnet_properties_to_model(model, cfg)
            circuit_curves = compute_circuit_transfer_functions(model, freqs=FREQS)

        # Multi-rate FFT evaluation matching native circuit simulator synthesis exactly
        prefilter_firs = compute_voice_prefilter_firs(
            voice_id, instrument=inst, num_taps=2048, src_pickup_key=src_pickup_key
        )
        N = 8192
        f_bins = np.fft.rfftfreq(N, 1.0 / 48000.0)
        H_channels = []
        for i in range(len(prefilter_firs)):
            pf = np.array(prefilter_firs[i], dtype=np.float32)
            c_curve = circuit_curves[i] if i < len(circuit_curves) else circuit_curves[0]
            cf = np.array(
                synthesize_minimum_phase_fir(c_curve, num_taps=2048, normalize=False),
                dtype=np.float32,
            )
            H_channels.append(np.fft.rfft(pf, N) * np.fft.rfft(cf, N))

        H_channels = np.array(H_channels)
        peaks = [int(np.argmax(np.abs(fir))) for fir in prefilter_firs]
        delta_samples = max(peaks) - min(peaks) if len(peaks) > 1 else 0
        has_spatial_delay = len(prefilter_firs) > 1 and delta_samples > 0

        if has_spatial_delay:
            # Acoustic inter-pickup spatial coherence decay:
            P_coherent = np.abs(np.sum(H_channels, axis=0)) ** 2
            P_incoherent = np.sum(np.abs(H_channels) ** 2, axis=0)
            delta_tau = delta_samples / 48000.0
            f_notch = 1.0 / (2.0 * delta_tau)
            f_mid = 1.35 * f_notch
            f_sigma = max(0.35 * f_notch, 1.0)
            gamma = 0.5 * (1.0 - np.tanh((f_bins - f_mid) / f_sigma))
            mag_spectrum = np.sqrt(gamma * P_coherent + (1.0 - gamma) * P_incoherent)
        elif len(H_channels) > 1:
            mag_spectrum = np.abs(np.sum(H_channels, axis=0))
        else:
            mag_spectrum = np.abs(H_channels[0])

        mag_raw = np.interp(freqs, f_bins, mag_spectrum)

    is_circuit_match = bool(
        circuit_curves
        and len(circuit_curves) > 0
        and np.allclose(circuit_curves[0], 1.0, rtol=1e-3)
    )
    is_full_identity = is_spatial_match and (is_circuit_match if mode == "difference" else True)
    gain_offset = 0.0 if is_full_identity else cfg.gain_db

    if mode == "difference":
        # Differential transfer function evaluated in absolute gain units
        # Identity match (source == target) evaluates to bit-exact 0.00 dB across all bins
        if is_full_identity:
            mag_norm = np.ones_like(freqs)
        else:
            mag_norm = mag_raw
    else:
        # Output voice magnitude: preserve absolute physical excursion relative to datum
        hpf_val = cfg.hpf
        if hpf_val is not None and float(hpf_val) >= 80.0:
            ref_idx = np.argmin(np.abs(freqs - 1000.0))
            ref_val = mag_raw[ref_idx]
            mag_norm = mag_raw / ref_val if ref_val > 0 else mag_raw
        elif cfg.sensor_type == "bridge_force":
            ref_idx = np.argmin(np.abs(freqs - 100.0))
            ref_val = mag_raw[ref_idx]
            mag_norm = mag_raw / ref_val if ref_val > 0 else mag_raw
        else:
            mag_norm = mag_raw

    mag_db = 20.0 * np.log10(np.clip(mag_norm, 1e-5, 20.0)) + gain_offset

    data = {
        "frequency": log_freqs,
        "magnitude_db": mag_db.tolist(),
        "voice_id": voice_id,
        "voice_name": cfg.name,
        "topology": cfg.topology,
        "description": cfg.description,
    }
    df = pl.DataFrame(data)
    if mode == "output" and cfg == VOICES.get(voice_id):
        _OUTPUT_VOICE_DF_CACHE[voice_id] = df
    elif mode == "difference" and cfg == VOICES.get(voice_id):
        _DIFF_VOICE_DF_CACHE[cache_key_diff] = df

    if include_mode_col:
        return df.with_columns(
            pl.lit("Output Voice" if mode == "output" else "Input/Output Difference").alias("mode")
        )
    return df


_TARGET_DFS_CACHE: dict[int, dict[str, tuple[str, np.ndarray]]] = {}


def get_cached_target_dfs(step: int = 1) -> dict[str, tuple[str, np.ndarray]]:
    """Caches precomputed target voice responses downsampled by step."""
    if step in _TARGET_DFS_CACHE:
        return _TARGET_DFS_CACHE[step]

    target_dfs: dict[str, tuple[str, np.ndarray]] = {}
    for vid, cfg in sorted(VOICES.items()):
        vname = cfg.name
        vdf = build_voice_dataframe(vid, cfg, mode="output")
        mag_full = np.asarray(vdf["magnitude_db"], dtype=np.float64)
        target_dfs[vid] = (vname, mag_full[::step] if step > 1 else mag_full)

    _TARGET_DFS_CACHE[step] = target_dfs
    return target_dfs


VOICE_FAMILIES: dict[str, str] = {
    "precision_vintage": "Precision",
    "precision_mids": "Precision",
    "precision_warm": "Precision",
    "precision_active": "Precision",
    "precision_dub": "Precision",
    "jazz_pair_open": "Jazz",
    "jazz_pair_mids": "Jazz",
    "jazz_pair_active": "Jazz",
    "jazz_bridge_growl": "Jazz",
    "jazz_bridge_open": "Jazz",
    "jazz_neck_warm": "Jazz",
    "stingray_parallel": "StingRay",
    "stingray_series": "StingRay",
    "stingray_active": "StingRay",
    "dingwall_bridge": "Dingwall",
    "dingwall_middle": "Dingwall",
    "dingwall_parallel": "Dingwall",
    "rickenbacker_clank": "Rickenbacker",
    "rickenbacker_open": "Rickenbacker",
    "pj_passive": "PJ",
    "pj_active": "PJ",
    "p_mm_parallel": "P∕MM",
    "p_mm_series": "P∕MM",
    "mudbucker_deep": "Mudbucker",
    "upright_acoustic": "Upright",
    "studio_direct": "Studio",
    "studio_active": "Studio",
    "studio_passive": "Studio",
}


def build_voicings_comparison_data(step: int = 3) -> dict[str, Any]:
    """
    Builds the compact data structure containing frequency responses and metadata
    for all 28 target voicings in VOICES.
    Used by voicings.html to display the 3-line graph:
      - Line 1: Source Voicing (H_src)
      - Line 2: Target Voicing (H_tgt)
      - Line 3: Difference (H_diff = H_tgt - H_src)
    When Source == Target, the difference line evaluates to exact 0.00 dB.
    """
    freqs = np.asarray(log_freqs[::step], dtype=np.float64)
    f_pts = np.round(freqs, 1).tolist()

    target_dfs = get_cached_target_dfs(step=step)

    voices_dict: dict[str, dict[str, Any]] = {}
    families_set: set[str] = set()

    for vid, (vname, db_tgt) in sorted(target_dfs.items()):
        vcfg = VOICES[vid]
        family = VOICE_FAMILIES.get(vid, "Specialty")
        families_set.add(family)
        rms_db = compute_curve_rms_db(db_tgt)
        voices_dict[vid] = {
            "id": vid,
            "name": vname,
            "tone_name": vcfg.tone_name or vname,
            "family": family,
            "topology": vcfg.topology,
            "sensor_type": vcfg.sensor_type,
            "description": vcfg.description,
            "fr": float(vcfg.fr),
            "q": float(vcfg.Q),
            "alpha": float(vcfg.alpha or 0.0),
            "vsat": float(vcfg.vsat or 1.0),
            "magnitude_db": np.round(db_tgt, 2).tolist(),
            "rms_db": round(rms_db, 2),
        }

    family_order = [
        "Precision",
        "Jazz",
        "StingRay",
        "P∕MM",
        "Dingwall",
        "Rickenbacker",
        "PJ",
        "Mudbucker",
        "Upright",
        "Studio",
    ]
    sorted_families = [f for f in family_order if f in families_set] + sorted(
        families_set - set(family_order)
    )

    return {
        "frequencies": f_pts,
        "voices": voices_dict,
        "families": sorted_families,
        "default_source": "precision_vintage",
        "default_target": "jazz_bridge_growl",
    }


def compute_curve_rms_db(mag_db: np.ndarray | Sequence[float] | pl.Series) -> float:
    """Computes the equivalent broadband RMS gain in dB for a magnitude frequency response."""
    arr = np.asarray(mag_db, dtype=np.float64)
    return float(20.0 * np.log10(np.sqrt(np.mean((10.0 ** (arr / 20.0)) ** 2))))


def compute_regularized_differential_db(
    db_tgt: np.ndarray,
    db_src: np.ndarray,
    freqs: np.ndarray | None = None,
    max_boost_db: float = 7.0,
    snr_db: float = 35.0,
    f_c_hz: float = 1200.0,
) -> np.ndarray:
    """
    Computes regularized differential gain in decibels modeling the empirical
    transfer function of a Neural Amp Modeler (NAM) trained on normalized audio:
      H_nam(f) = (h_tgt,norm * h_src,norm * S_dry(f)) / (|h_src,norm|^2 * S_dry(f) + eps)
    where:
      - S_dry(f) = 1 / (1 + (f / f_c)^2) is the dry bass string excitation power spectrum
      - eps = 10^(-SNR/10) * max(P_in) is the regularization floor corresponding to NAM training ESR
      - smooth_soft_knee_db smoothly bounds any resonance peaks at max_boost_db (+7.0 dB)

    This accurately mirrors trained neural models:
      - Passband (< 3.0 kHz): Perfect linear EQ matching with < 0.09 dB error.
      - Treble / Stopband (> 5 kHz): Smooth, natural physical roll-off governed by
        input signal energy, completely eliminating unconstrained ultrasonic boost.
      - Identity pairs (h_tgt == h_src): Bit-exact 0.00 dB.
    """
    if np.allclose(db_tgt, db_src, atol=1e-5):
        return np.zeros_like(db_tgt, dtype=np.float64)

    n_pts = len(db_tgt)
    if freqs is None:
        f_arr = np.asarray(
            [F_MIN * (F_MAX / F_MIN) ** (i / (n_pts - 1)) for i in range(n_pts)],
            dtype=np.float64,
        )
    else:
        f_arr = np.asarray(freqs, dtype=np.float64)

    h_src = 10.0 ** (np.asarray(db_src, dtype=np.float64) / 20.0)
    h_tgt = 10.0 ** (np.asarray(db_tgt, dtype=np.float64) / 20.0)

    s_dry = 1.0 / (1.0 + (f_arr / f_c_hz) ** 2)
    p_in = (h_src**2) * s_dry
    eps = (10.0 ** (-snr_db / 10.0)) * float(np.max(p_in))

    h_nam = (h_tgt * h_src * s_dry) / (p_in + eps)
    db_nam = 20.0 * np.log10(np.maximum(h_nam, 1e-3))

    knee_width = min(2.5, max_boost_db / 2.0)
    thresh = max_boost_db - knee_width
    h_db_soft = smooth_soft_knee_db(db_nam, thresh=thresh, ceiling=max_boost_db, alpha=2.0)

    return np.asarray(h_db_soft, dtype=np.float64)


def build_voicings_comparison_dataframe(
    source_id: str = "precision_vintage",
    target_id: str = "jazz_bridge_growl",
    step: int = 1,
) -> pl.DataFrame:
    """
    Constructs a 3-line Polars DataFrame comparing a Source Voicing and Target Voicing:
      1. Source Voicing (H_src)
      2. Target Voicing (H_tgt)
      3. Normalized Difference (Norm. Diff = H_tgt,norm - H_src,norm)
    When source_id == target_id, the differential line evaluates to exact 0.00 dB.
    """
    if source_id not in VOICES:
        raise KeyError(f"Source voice '{source_id}' not found in VOICES: {list(VOICES.keys())}")
    if target_id not in VOICES:
        raise KeyError(f"Target voice '{target_id}' not found in VOICES: {list(VOICES.keys())}")

    freqs = np.asarray(log_freqs[::step], dtype=np.float64)
    n_pts = len(freqs)
    f_pts = np.round(freqs, 1).tolist()

    target_dfs = get_cached_target_dfs(step=step)

    src_name, db_src = target_dfs[source_id]
    tgt_name, db_tgt = target_dfs[target_id]

    if source_id == target_id:
        db_diff = np.zeros(n_pts, dtype=np.float64)
    else:
        src_rms = compute_curve_rms_db(db_src)
        tgt_rms = compute_curve_rms_db(db_tgt)
        db_src_norm = db_src - src_rms
        db_tgt_norm = db_tgt - tgt_rms
        db_diff = np.round(compute_regularized_differential_db(db_tgt_norm, db_src_norm, freqs), 2)

    freq_col = f_pts * 3
    mag_col = np.round(db_src, 2).tolist() + np.round(db_tgt, 2).tolist() + db_diff.tolist()
    line_type_col = (
        ["1. Source Voicing"] * n_pts
        + ["2. Target Voicing"] * n_pts
        + ["3. Normalized Difference (Norm. Diff)"] * n_pts
    )
    vid_col = [source_id] * n_pts + [target_id] * n_pts + [f"{source_id}_to_{target_id}"] * n_pts
    vname_col = (
        [src_name] * n_pts + [tgt_name] * n_pts + [f"Norm. Diff: {tgt_name} - {src_name}"] * n_pts
    )

    return pl.DataFrame(
        {
            "frequency": freq_col,
            "magnitude_db": mag_col,
            "line_type": line_type_col,
            "voice_id": vid_col,
            "voice_name": vname_col,
        }
    )


def compute_fir_csd(
    fir: Sequence[float] | np.ndarray,
    freqs: Sequence[float] | np.ndarray,
    num_slices: int = 24,
    max_time_ms: float = 3.0,
    sr: int = 48000,
    n_fft: int = 2048,
    n_rise: int = 8,
) -> tuple[list[float], list[list[float]]]:
    """
    Computes Cumulative Spectral Decay (CSD) waterfall slices for an impulse response.
    Slices the FIR across time from t=0 to max_time_ms with a smooth half-Hann onset taper
    to prevent truncation spectral splatter.
    Returns:
      (time_ms, csd_matrix) where csd_matrix has shape [num_slices, len(freqs)] with dB values.
    """
    fir_arr = np.asarray(fir, dtype=np.float64)
    total_samples = len(fir_arr)
    max_sample = round((max_time_ms / 1000.0) * sr)
    max_sample = min(max_sample, total_samples - 1)

    time_indices = np.linspace(0, max_sample, num_slices, dtype=int)
    time_ms = [round(float(idx / sr * 1000.0), 2) for idx in time_indices]

    rise = 0.5 * (1.0 - np.cos(np.pi * np.arange(n_rise) / n_rise))
    rfft_freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)
    f_eval = np.asarray(freqs, dtype=np.float64)

    csd_matrix: list[list[float]] = []
    for t_start in time_indices:
        gated = fir_arr.copy()
        if t_start > 0:
            gated[:t_start] = 0.0
            r_end = min(t_start + n_rise, total_samples)
            gated[t_start:r_end] *= rise[: (r_end - t_start)]

        spec = np.abs(np.fft.rfft(gated, n_fft))
        interp_spec = np.interp(f_eval, rfft_freqs, spec)
        db = 20.0 * np.log10(np.maximum(interp_spec, 1e-3))
        csd_matrix.append([round(float(v), 1) for v in db])

    return time_ms, csd_matrix


_VOICING_3D_CACHE: dict[tuple[int, int, float], dict[str, Any]] = {}


def build_voicing_ir_diff_3d_data(
    num_freqs: int = 50,
    num_slices: int = 24,
    max_time_ms: float = 3.0,
) -> dict[str, Any]:
    """
    Builds the 3D data structure for voicing IR difference visualization across all voicing pairs.
    For each (source, target) pair:
      - Evaluates the difference frequency response H_diff = H_tgt - H_src in dB
      - Synthesizes the minimum-phase difference impulse response h_diff(t) using a linear frequency grid
      - Computes the Cumulative Spectral Decay (CSD) waterfall matrix and FIR waveform
    """
    cache_key = (num_freqs, num_slices, float(max_time_ms))
    if cache_key in _VOICING_3D_CACHE:
        return _VOICING_3D_CACHE[cache_key]

    f_eval = [round(float(f), 1) for f in np.geomspace(20.0, 20000.0, num_freqs)]
    f_lin = np.linspace(0.0, 24000.0, 2049)

    # Compute time_ms using dummy impulse
    dummy_impulse = np.zeros(2048, dtype=np.float64)
    dummy_impulse[0] = 1.0
    time_ms, _ = compute_fir_csd(
        dummy_impulse, f_eval, num_slices=num_slices, max_time_ms=max_time_ms
    )

    # 1. Precompute magnitude response for each voice (both 50-pt display and linear synthesis grid)
    voice_mags_50: dict[str, np.ndarray] = {}
    voice_mags_lin: dict[str, np.ndarray] = {}
    voices_meta: dict[str, dict[str, Any]] = {}
    f_orig = np.asarray(log_freqs, dtype=np.float64)

    for vid, cfg in sorted(VOICES.items()):
        vdf = build_voice_dataframe(vid, cfg, mode="output")
        mag_full = vdf["magnitude_db"].to_numpy()
        mag_50 = np.interp(f_eval, f_orig, mag_full)
        mag_lin = np.interp(f_lin, f_orig, mag_full)
        voice_mags_50[vid] = mag_50
        voice_mags_lin[vid] = mag_lin
        voices_meta[vid] = {
            "id": vid,
            "name": cfg.name,
            "tone_name": cfg.tone_name or cfg.name,
            "family": VOICE_FAMILIES.get(vid, "Specialty"),
            "magnitude_db": [round(float(v), 1) for v in mag_50],
        }

    # 2. Build difference IR matrix
    responses: dict[str, dict[str, dict[str, Any]]] = {}
    n_rise = max(min(round((max_time_ms / 1000.0 * 48000) / num_slices), 8), 4)

    for s_vid in sorted(VOICES.keys()):
        responses[s_vid] = {}
        for t_vid in sorted(VOICES.keys()):
            db_diff_50 = voice_mags_50[t_vid] - voice_mags_50[s_vid]
            if s_vid == t_vid:
                # Identity pair: unit impulse at t=0, -60 dB floor elsewhere
                csd_matrix = [
                    [0.0 if m == 0 else -60.0 for _ in range(num_freqs)] for m in range(num_slices)
                ]
                fir_head = [1.0] + [0.0] * 127
            else:
                db_diff_lin = voice_mags_lin[t_vid] - voice_mags_lin[s_vid]
                mag_lin_grid = 10.0 ** (db_diff_lin / 20.0)
                fir = np.array(
                    synthesize_minimum_phase_fir(mag_lin_grid, num_taps=1024, normalize=False),
                    dtype=np.float64,
                )
                _, csd_matrix = compute_fir_csd(
                    fir,
                    f_eval,
                    num_slices=num_slices,
                    max_time_ms=max_time_ms,
                    n_rise=n_rise,
                )
                fir_head = [round(float(x), 4) for x in fir[:128]]

            responses[s_vid][t_vid] = {
                "magnitude_db": [round(float(x), 1) for x in db_diff_50],
                "csd_matrix": csd_matrix,
                "fir_waveform": fir_head,
            }

    data: dict[str, Any] = {
        "frequencies": f_eval,
        "time_ms": time_ms,
        "voices": voices_meta,
        "responses": responses,
        "default_source": "precision_vintage",
        "default_target": "jazz_bridge_growl",
    }

    _VOICING_3D_CACHE[cache_key] = data
    return data
