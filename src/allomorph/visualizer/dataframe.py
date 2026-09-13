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
    compute_frontend_transfer_function,
    load_circuit,
)
from allomorph.config.geometry import (
    compute_effective_position,
    resolve_voice_pickups,
)
from allomorph.config.instruments import (
    get_source_pickup,
    load_all_instruments,
    load_instrument,
)
from allomorph.config.scales import SCALES, resolve_scale_range
from allomorph.config.schema import CoilConfig, InstrumentConfig, VoiceConfig
from allomorph.config.strings import STRINGS, get_voice_string
from allomorph.config.voices import VOICES
from allomorph.dsp import (
    FREQS,
    synthesize_minimum_phase_fir,
)
from allomorph.physics import (
    MEAN_BASS_F0,
    compute_differential_longitudinal_transfer,
    compute_differential_string_transfer,
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

    if sensor_type == "direct" and mode == "output":
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
        # 1. Output Voice: Target acoustic aperture + loaded SPICE circuit + string + body bloom
        model = load_circuit(tgt_circuit)
        apply_magnet_properties_to_model(model, cfg)
        circuit_curves = compute_circuit_transfer_functions(model, freqs=FREQS)

        if sensor_type == "bridge_force":
            f_lin = freqs
            is_flatwound = "flat" in (tgt_string.type or "")
            f_damp = 4200.0 if is_flatwound else 3600.0
            h_damp = 1.0 / np.sqrt((1.0 - (f_lin / f_damp) ** 2) ** 2 + 2.0 * (f_lin / f_damp) ** 2)
            g_sub = 0.15
            h_sub = np.sqrt((g_sub**2 * 32.0**2 + f_lin**2) / (32.0**2 + f_lin**2))
            h_tilt_raw = np.sqrt((1.0 + (f_lin / 250.0) ** 2) / (1.0 + (f_lin / 70.0) ** 2))
            h_tilt = h_tilt_raw / np.max(h_tilt_raw)
            c_curve = circuit_curves[0] if circuit_curves else [1.0] * len(FREQS)
            branch = (
                np.interp(freqs, FREQS, np.asarray(c_curve, dtype=np.float64))
                * h_damp
                * h_sub
                * h_tilt
            )
            h_tgt_total = branch
        elif sensor_type == "direct":
            c_curve = circuit_curves[0] if circuit_curves else [1.0] * len(FREQS)
            h_tgt_total = np.interp(freqs, FREQS, np.asarray(c_curve, dtype=np.float64))
        else:
            tgt_scale_range = resolve_scale_range(tgt_scale)
            tgt_scale_m = (tgt_scale_range[0] + tgt_scale_range[1]) / 2.0
            positions = [compute_effective_position(p.coils) for p in pickups]
            pos_max = max(positions) if positions else 0.0
            c_mean = 2.0 * tgt_scale_m * MEAN_BASS_F0

            N = 8192
            f_bins = np.fft.rfftfreq(N, 1.0 / 48000.0)
            H_channels = []
            peaks = []

            for i, p in enumerate(pickups):
                c_curve = circuit_curves[i] if i < len(circuit_curves) else [1.0] * len(FREQS)
                p_weight = p.weight
                p_pol = p.polarity
                weight_fac = 1.0 if len(circuit_curves) > 1 else p_weight

                ac = numpy_pickup_acoustic_response(
                    f_bins, p.coils, scale_length_m=tgt_scale_range
                ) * (weight_fac * p_pol)
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

        # Tension / Bloom for target instrument
        if tgt_scale == "upright":
            delta_bloom = float(tgt_string.bloom_db or 2.8)
            g_bloom = 10.0 ** (max(delta_bloom, 0.5) / 20.0)
            h_tension = np.sqrt((g_bloom**2 + (freqs / 100.0) ** 2) / (1.0 + (freqs / 100.0) ** 2))
        else:
            tgt_scale_in = (
                37.0
                if tgt_scale in ["multiscale", "37in"]
                else (35.0 if tgt_scale == "multiscale_super" else 34.0)
            )
            delta_scale = tgt_scale_in - 34.0
            if delta_scale <= 0.0:
                h_tension = np.ones_like(freqs)
            else:
                snap_db = 3.5 * np.tanh((1.8 * delta_scale) / (4.0 * 3.5))
                g_snap = 10.0 ** (snap_db / 20.0)
                h_tension = np.sqrt(
                    (1.0 + g_snap**2 * (freqs / 2800.0) ** 2) / (1.0 + (freqs / 2800.0) ** 2)
                )

        # String voicing for target instrument (relative to standard nickel roundwound)
        if (
            sensor_type != "bridge_force"
            and cfg.target_string
            and cfg.target_string != "roundwound_nickel_standard"
        ):
            std_str = STRINGS["roundwound_nickel_standard"]
            scale_in = (
                37.0
                if tgt_scale in ["multiscale", "37in"]
                else (35.0 if tgt_scale == "multiscale_super" else 34.0)
            )
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
            cf = np.array(
                synthesize_minimum_phase_fir(circuit_curves[i], num_taps=2048, normalize=False),
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

    hpf_val = cfg.hpf
    if hpf_val is not None and float(hpf_val) >= 80.0:
        ref_idx = np.argmin(np.abs(freqs - 1000.0))
    elif cfg.sensor_type == "bridge_force":
        ref_idx = np.argmin(np.abs(freqs - 100.0))
    else:
        ref_idx = 0

    ref_val = mag_raw[ref_idx]
    mag_norm = mag_raw / ref_val if ref_val > 0 else mag_raw
    is_circuit_match = bool(
        circuit_curves
        and len(circuit_curves) > 0
        and np.allclose(circuit_curves[0], 1.0, rtol=1e-3)
    )
    is_full_identity = is_spatial_match and (is_circuit_match if mode == "difference" else True)
    gain_offset = 0.0 if is_full_identity else cfg.gain_db
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


def compute_canonical_acoustic_response(freqs: np.ndarray) -> np.ndarray:
    """
    Evaluates the Canonical Intermediate baseline acoustic aperture response.
    Scale length: 34.0", Single 0.75" magnetic slit @ 93.5mm datum from bridge.
    """
    can_coils = [
        CoilConfig(
            strings=["all"], position_from_bridge_m=0.0935, aperture_width_in=0.75, weight=1.0
        )
    ]
    h_can_ac = numpy_pickup_acoustic_response(freqs, can_coils, scale_length_m=(0.8636, 0.8636))
    return h_can_ac / max(h_can_ac[0], 1e-9)


def compute_canonical_intermediate_response(freqs: np.ndarray) -> np.ndarray:
    """
    Evaluates the Canonical Intermediate total baseline response (acoustic aperture + wideband passive reference circuit).
    Scale length: 34.0", Single 0.75" magnetic slit @ 93.5mm datum from bridge.
    Circuit: Wideband passive reference pickup (4.8 kHz, Q=0.75).
    """
    h_can_ac_norm = compute_canonical_acoustic_response(freqs)
    can_voice = VOICES.get("00_canonical_intermediate")
    if can_voice and can_voice.circuit:
        can_model = load_circuit(can_voice.circuit)
        curves = compute_circuit_transfer_functions(can_model, freqs=freqs, return_numpy=True)
        h_can_elec = curves[0]
        h_can_elec_norm = h_can_elec / max(h_can_elec[0], 1e-9)
        return h_can_ac_norm * h_can_elec_norm

    return h_can_ac_norm


_TARGET_DFS_CACHE: dict[int, dict[str, tuple[str, np.ndarray | None]]] = {}


def get_cached_target_dfs(step: int = 1) -> dict[str, tuple[str, np.ndarray | None]]:
    """Caches precomputed target voice responses downsampled by step."""
    if step in _TARGET_DFS_CACHE:
        return _TARGET_DFS_CACHE[step]

    target_dfs: dict[str, tuple[str, np.ndarray | None]] = {}
    for vid, cfg in sorted(VOICES.items()):
        if vid == "00_canonical_intermediate":
            continue
        vname = cfg.name
        if cfg.sensor_type == "direct":
            target_dfs[vid] = (vname, None)
        else:
            vdf = build_voice_dataframe(vid, cfg, mode="output")
            mag_full = np.asarray(vdf["magnitude_db"], dtype=np.float64)
            target_dfs[vid] = (vname, mag_full[::step] if step > 1 else mag_full)

    _TARGET_DFS_CACHE[step] = target_dfs
    return target_dfs


def build_universal_targets_dataframe() -> pl.DataFrame:
    """
    Calculates magnitude frequency responses for all 23 Universal Target Voicings relative to
    the Canonical Intermediate baseline: H_backend = H_target / H_canonical.
    """
    freqs = np.asarray(log_freqs, dtype=np.float64)
    h_can_norm = compute_canonical_intermediate_response(freqs)
    db_can = 20.0 * np.log10(np.clip(h_can_norm, 1e-5, 20.0))

    target_dfs = get_cached_target_dfs(step=1)

    freq_col: list[float] = []
    mag_col: list[float] = []
    vid_col: list[str] = []
    vname_col: list[str] = []
    topo_col: list[str] = []
    fr_col: list[float] = []
    q_col: list[float] = []
    desc_col: list[str] = []

    for vid, cfg in sorted(VOICES.items()):
        if vid == "00_canonical_intermediate":
            continue
        vname, db_tgt = target_dfs[vid]
        if db_tgt is None:
            db_backend = np.zeros_like(freqs)
        else:
            db_backend = db_tgt - db_can

        topo = cfg.topology
        fr = float(cfg.fr)
        q = float(cfg.Q)
        desc = cfg.description

        freq_col.extend(log_freqs)
        mag_col.extend(db_backend.tolist())
        vid_col.extend([vid] * NUM_POINTS)
        vname_col.extend([vname] * NUM_POINTS)
        topo_col.extend([topo] * NUM_POINTS)
        fr_col.extend([fr] * NUM_POINTS)
        q_col.extend([q] * NUM_POINTS)
        desc_col.extend([desc] * NUM_POINTS)

    return pl.DataFrame(
        {
            "frequency": freq_col,
            "magnitude_db": mag_col,
            "voice_id": vid_col,
            "voice_name": vname_col,
            "topology": topo_col,
            "fr": fr_col,
            "Q": q_col,
            "description": desc_col,
        }
    )


def build_frontend_deconvolutions_dataframe() -> pl.DataFrame:
    """
    Builds the master dataframe for frontend_deconvolutions.html containing Block 1
    deconvolution curves (H_front = H_can / H_src) for all source instruments and pickups
    to the 0.00 dB Canonical Intermediate baseline.
    """
    freqs = np.asarray(log_freqs, dtype=np.float64)

    can_voice = VOICES.get("00_canonical_intermediate")
    can_circuit = can_voice.circuit if can_voice is not None else None
    can_model = load_circuit(can_circuit) if can_circuit else None

    all_insts = load_all_instruments()

    mag_arrays: list[np.ndarray] = []
    iid_col: list[str] = []
    iname_col: list[str] = []
    pkey_col: list[str] = []
    pname_col: list[str] = []
    label_col: list[str] = []
    scale_col: list[float] = []
    pos_col: list[float] = []

    for inst_id, inst in sorted(all_insts.items()):
        if inst_id == "canonical_intermediate":
            continue
        inst_name = inst.name
        scale_in = float(inst.scale_length_in or 34.0)
        pickups = inst.pickups

        for p_key, p_cfg in sorted(pickups.items()):
            p_name = p_cfg.name
            pos_m = p_cfg.position_from_bridge_m or 0.0
            pos_mm = float(pos_m * 1000.0) if pos_m else 0.0

            h_front = compute_frontend_transfer_function(
                inst, p_key, freqs=freqs, can_model=can_model
            )
            db_front = 20.0 * np.log10(np.maximum(h_front, 1e-6))
            label = f"{inst_name} - {p_name}"

            mag_arrays.append(db_front)
            iid_col.extend([inst_id] * NUM_POINTS)
            iname_col.extend([inst_name] * NUM_POINTS)
            pkey_col.extend([p_key] * NUM_POINTS)
            pname_col.extend([p_name] * NUM_POINTS)
            label_col.extend([label] * NUM_POINTS)
            scale_col.extend([scale_in] * NUM_POINTS)
            pos_col.extend([pos_mm] * NUM_POINTS)

    n_pickups = len(mag_arrays)
    all_freqs = np.tile(log_freqs, n_pickups)
    all_mags = np.concatenate(mag_arrays) if mag_arrays else np.array([], dtype=np.float64)

    return pl.DataFrame(
        {
            "frequency": all_freqs,
            "magnitude_db": all_mags,
            "instrument_id": iid_col,
            "instrument_name": iname_col,
            "pickup_key": pkey_col,
            "pickup_name": pname_col,
            "label": label_col,
            "scale_in": scale_col,
            "position_mm": pos_col,
        }
    )


def build_instrument_frontend_dataframe(inst: InstrumentConfig) -> pl.DataFrame:
    """
    Calculates magnitude frequency responses for all pickup switch positions of a source instrument:
    H_frontend = H_canonical / H_source.
    """
    freqs = np.asarray(log_freqs, dtype=np.float64)

    inst_id = inst.id
    inst_name = inst.name
    scale_in = float(inst.scale_length_in or 34.0)
    pickups = inst.pickups

    can_voice = VOICES.get("00_canonical_intermediate")
    can_circuit = can_voice.circuit if can_voice is not None else None
    can_model = load_circuit(can_circuit) if can_circuit else None

    mag_arrays: list[np.ndarray] = []
    iid_col: list[str] = []
    iname_col: list[str] = []
    pkey_col: list[str] = []
    pname_col: list[str] = []
    scale_col: list[float] = []
    pos_col: list[float] = []

    for p_key, p_cfg in sorted(pickups.items()):
        p_name = p_cfg.name
        pos_m = p_cfg.position_from_bridge_m or 0.0
        pos_mm = float(pos_m * 1000.0) if pos_m else 0.0

        h_front = compute_frontend_transfer_function(inst, p_key, freqs=freqs, can_model=can_model)
        db_front = 20.0 * np.log10(np.maximum(h_front, 1e-6))

        mag_arrays.append(db_front)
        iid_col.extend([inst_id] * NUM_POINTS)
        iname_col.extend([inst_name] * NUM_POINTS)
        pkey_col.extend([p_key] * NUM_POINTS)
        pname_col.extend([p_name] * NUM_POINTS)
        scale_col.extend([scale_in] * NUM_POINTS)
        pos_col.extend([pos_mm] * NUM_POINTS)

    n_pickups = len(mag_arrays)
    all_freqs = np.tile(log_freqs, n_pickups)
    all_mags = np.concatenate(mag_arrays) if mag_arrays else np.array([], dtype=np.float64)

    return pl.DataFrame(
        {
            "frequency": all_freqs,
            "magnitude_db": all_mags,
            "instrument_id": iid_col,
            "instrument_name": iname_col,
            "pickup_key": pkey_col,
            "pickup_name": pname_col,
            "scale_in": scale_col,
            "position_mm": pos_col,
        }
    )


def build_composite_instrument_dataframe(
    inst: InstrumentConfig,
    step: int = 3,
) -> pl.DataFrame:
    """
    Calculates the 5-stage physical signal flow progression for a source instrument
    relative to the standardized Canonical Intermediate datum (34" @ 93.5mm datum, wideband passive reference circuit):
      1. Source Bass Input: Physical response of the source pickup relative to Canonical Intermediate.
      2. Block 1 Deconvolution: 2048-tap FIR deconvolution filter (H_front = H_can / H_src) neutralizing source pickup.
      3. Canonical Intermediate (0 dB): Neutral baseline reference datum (Stage 1 + Stage 2 = 0.00 dB).
      4. Block 2 Target Voicing: Universal target model transfer function (H_back = H_tgt / H_can).
      5. Target Voice Output: Authentic acoustic target voice produced after Block 2.
    Illustrates: Source Bass Input + Block 1 Deconvolution = Canonical Intermediate (0 dB) -> Block 2 Target Voicing -> Target Voice Output.
    """
    freqs = np.asarray(log_freqs[::step], dtype=np.float64)
    n_pts = len(freqs)
    f_pts = np.round(freqs, 1).tolist()
    h_can_norm = compute_canonical_intermediate_response(freqs)
    db_can = 20.0 * np.log10(np.clip(h_can_norm, 1e-5, 20.0))

    pickups = inst.pickups

    target_dfs = get_cached_target_dfs(step=step)

    can_voice = VOICES.get("00_canonical_intermediate")
    can_circuit = can_voice.circuit if can_voice is not None else None
    can_model = load_circuit(can_circuit) if can_circuit else None

    freq_col: list[float] = []
    mag_col: list[float] = []
    stage_col: list[str] = []
    vname_col: list[str] = []
    pname_col: list[str] = []

    # Stages 1, 2, 3: Per-pickup curves (deduplicated across target voices)
    for _p_key, p_cfg in sorted(pickups.items()):
        p_name = p_cfg.name
        h_front = compute_frontend_transfer_function(inst, _p_key, freqs=freqs, can_model=can_model)
        db_front = np.round(20.0 * np.log10(np.maximum(h_front, 1e-6)), 2)
        # Source Bass Input entering Block 1 (relative to Canonical Intermediate baseline)
        db_src = -db_front
        db_ci = [0.0] * n_pts

        # Stage 1: Source Bass Input
        freq_col.extend(f_pts)
        mag_col.extend(db_src.tolist())
        stage_col.extend(["1. Source Bass Input"] * n_pts)
        vname_col.extend([""] * n_pts)
        pname_col.extend([p_name] * n_pts)

        # Stage 2: Block 1 Deconvolution
        freq_col.extend(f_pts)
        mag_col.extend(db_front.tolist())
        stage_col.extend(["2. Block 1 Deconvolution"] * n_pts)
        vname_col.extend([""] * n_pts)
        pname_col.extend([p_name] * n_pts)

        # Stage 3: Canonical Intermediate (0 dB)
        freq_col.extend(f_pts)
        mag_col.extend(db_ci)
        stage_col.extend(["3. Canonical Intermediate (0 dB)"] * n_pts)
        vname_col.extend([""] * n_pts)
        pname_col.extend([p_name] * n_pts)

    # Stage 4: Block 2 Target Voicing (deduplicated across pickups)
    for vid, (vname, db_tgt) in sorted(target_dfs.items()):
        if db_tgt is None:
            db_back = [0.0] * n_pts
        else:
            db_back = np.round(db_tgt - db_can, 2).tolist()

        freq_col.extend(f_pts)
        mag_col.extend(db_back)
        stage_col.extend(["4. Block 2 Target Voicing"] * n_pts)
        vname_col.extend([vname] * n_pts)
        pname_col.extend([""] * n_pts)

    # Stage 5: Target Voice Output (per pickup and target voice)
    for _p_key, p_cfg in sorted(pickups.items()):
        p_name = p_cfg.name
        for vid, (vname, db_tgt) in sorted(target_dfs.items()):
            if db_tgt is None:
                db_out = [0.0] * n_pts
            else:
                db_out = np.round(db_tgt - db_can, 2).tolist()

            freq_col.extend(f_pts)
            mag_col.extend(db_out)
            stage_col.extend(["5. Target Voice Output"] * n_pts)
            vname_col.extend([vname] * n_pts)
            pname_col.extend([p_name] * n_pts)

    return pl.DataFrame(
        {
            "frequency": freq_col,
            "magnitude_db": mag_col,
            "stage": stage_col,
            "voice_name": vname_col,
            "pickup_name": pname_col,
        }
    )


VOICE_FAMILIES: dict[str, str] = {
    "01_modern_jazz_active": "Jazz",
    "02_jazz_bass_pair": "Jazz",
    "02b_jazz_bass_pair_22nf": "Jazz",
    "02c_jazz_bridge_growl_bias": "Jazz",
    "03_jazz_bridge_60s": "Jazz",
    "04_modern_p_ceramic": "Precision",
    "05_vintage_62_p_alnico": "Precision",
    "05b_vintage_62_p_22nf": "Precision",
    "05c_vintage_62_p_47nf": "Precision",
    "05d_vintage_50s_p_100nf": "Precision",
    "07_modern_pj_active": "PJ",
    "08_vintage_pj_passive": "PJ",
    "09_stingray_mm_parallel": "StingRay",
    "09b_stingray_mm_series": "StingRay",
    "10_rickenbacker_bridge_hpf": "Rickenbacker",
    "11_modern_pmm_active": "P∕MM",
    "11b_pmm_hybrid_series": "P∕MM",
    "12_mudbucker_ultra_series": "Mudbucker",
    "13_dingwall_multiscale_bridge": "Dingwall",
    "14_upright_bridge_transducer": "Upright",
    "15_neutral_character": "Character",
    "15b_active_character": "Character",
    "15c_passive_character": "Character",
}


def build_baked_responses_data(step: int = 3) -> dict[str, Any]:
    """
    Builds the compact matrix data structure containing frequency response curves for all source
    instruments and target voicings in the 1-block monolithic baked configuration (H_diff = H_tgt / H_src).
    Uses the decoupled two-stage vectorization: dB_baked = dB_front + dB_back.
    Identity pairs (where source instrument matches target voice) evaluate to exact 0.00 dB.
    """
    freqs = np.asarray(log_freqs[::step], dtype=np.float64)
    n_pts = len(freqs)
    f_pts = np.round(freqs, 1).tolist()

    can_voice = VOICES.get("00_canonical_intermediate")
    can_circuit = can_voice.circuit if can_voice is not None else None
    can_model = load_circuit(can_circuit) if can_circuit else None

    h_can_norm = compute_canonical_intermediate_response(freqs)
    db_can = 20.0 * np.log10(np.clip(h_can_norm, 1e-5, 20.0))

    target_dfs = get_cached_target_dfs(step=step)
    all_insts = load_all_instruments()

    insts = {
        iid: icfg for iid, icfg in sorted(all_insts.items()) if iid != "canonical_intermediate"
    }

    # 1. Precompute frontend deconvolution per instrument & pickup
    inst_pickup_front: dict[str, dict[str, np.ndarray]] = {}
    instruments_meta: dict[str, dict[str, Any]] = {}

    for iid, inst in insts.items():
        inst_pickup_front[iid] = {}
        for pkey in inst.pickups:
            h_front = compute_frontend_transfer_function(
                inst, pkey, freqs=freqs, can_model=can_model
            )
            db_front = np.round(20.0 * np.log10(np.maximum(h_front, 1e-6)), 2)
            inst_pickup_front[iid][pkey] = db_front

        scale_in = float(inst.scale_length_in or 34.0)
        scale_m = float(inst.scale_length_m or (scale_in * 0.0254))
        def_pkey = inst.default_pickup or next(iter(inst.pickups.keys()))
        instruments_meta[iid] = {
            "id": iid,
            "name": inst.name,
            "scale_in": scale_in,
            "scale_m": round(scale_m, 4),
            "default_pickup": def_pkey,
            "pickups_count": len(inst.pickups),
        }

    # 2. Target voices metadata
    voices_meta: dict[str, dict[str, Any]] = {}
    for vid, (vname, _) in sorted(target_dfs.items()):
        vcfg = VOICES[vid]
        family = VOICE_FAMILIES.get(vid, "Specialty")
        voices_meta[vid] = {
            "id": vid,
            "name": vname,
            "tone_name": vcfg.tone_name or vname,
            "family": family,
            "topology": vcfg.topology,
            "sensor_type": vcfg.sensor_type,
            "description": vcfg.description,
            "alpha": float(vcfg.alpha or 0.0),
            "vsat": float(vcfg.vsat or 1.0),
        }

    # 3. Monolithic baked responses matrix
    responses: dict[str, dict[str, dict[str, Any]]] = {}
    for iid, inst in insts.items():
        responses[iid] = {}
        for vid, (vname, db_tgt) in sorted(target_dfs.items()):
            vcfg = VOICES[vid]
            p = get_source_pickup(inst, vid)
            pkey = p.id or inst.default_pickup or next(iter(inst.pickups.keys()))
            db_front = inst_pickup_front[iid][pkey]

            if db_tgt is None:
                db_back = np.zeros(n_pts)
            else:
                db_back = np.round(db_tgt - db_can, 2)

            if vid == "15_neutral_character" or (
                vcfg.preserve_aperture and getattr(vcfg.circuit, "no_eq", False)
            ):
                db_baked = [0.0] * n_pts
            elif vcfg.preserve_aperture:
                tgt_circuit = vcfg.circuit
                model = load_circuit(tgt_circuit)
                apply_magnet_properties_to_model(model, vcfg)
                if p.circuit:
                    src_model = load_circuit(p.circuit)
                    apply_magnet_properties_to_model(src_model, p)
                    curves = compute_differential_circuit_transfer_functions(
                        model, src_model, freqs=freqs
                    )
                else:
                    curves = compute_circuit_transfer_functions(model, freqs=freqs)
                h_diff = np.asarray(curves[0], dtype=np.float64)
                h_diff = h_diff / max(h_diff[0], 1e-6)
                db_baked = np.round(20.0 * np.log10(np.maximum(h_diff, 1e-6)), 2).tolist()
            elif is_voice_matching_source(inst, vid, vcfg):
                if p.circuit and vcfg.circuit:
                    src_m = load_circuit(p.circuit)
                    apply_magnet_properties_to_model(src_m, p)
                    tgt_m = load_circuit(vcfg.circuit)
                    apply_magnet_properties_to_model(tgt_m, vcfg)
                    diff_c = compute_differential_circuit_transfer_functions(
                        tgt_m, src_m, freqs=freqs
                    )
                    diff_arr = np.asarray(diff_c[0], dtype=np.float64)
                    if np.allclose(diff_arr, 1.0, rtol=1e-3):
                        db_baked = [0.0] * n_pts
                    else:
                        db_baked = np.round(20.0 * np.log10(np.maximum(diff_arr, 1e-6)), 2).tolist()
                elif not p.circuit and not vcfg.circuit:
                    db_baked = [0.0] * n_pts
                else:
                    db_baked = np.round(db_front + db_back, 2).tolist()
            else:
                db_baked = np.round(db_front + db_back, 2).tolist()

            responses[iid][vid] = {
                "pickup_key": pkey,
                "pickup_name": p.name or pkey,
                "magnitude_db": db_baked,
            }

    return {
        "frequencies": f_pts,
        "instruments": instruments_meta,
        "voices": voices_meta,
        "responses": responses,
    }


def build_baked_responses_dataframe(
    instrument_ids: Sequence[str] | None = None,
    voice_ids: Sequence[str] | None = None,
    step: int = 3,
) -> pl.DataFrame:
    """
    Builds a tabular Polars DataFrame containing monolithic baked curves (H_diff = H_tgt / H_src)
    for selected source instruments and target voicings.
    """
    data = build_baked_responses_data(step=step)
    f_pts: list[float] = data["frequencies"]
    n_pts = len(f_pts)

    filter_insts = set(instrument_ids) if instrument_ids else None
    filter_voices = set(voice_ids) if voice_ids else None

    freq_col: list[float] = []
    mag_col: list[float] = []
    iid_col: list[str] = []
    iname_col: list[str] = []
    vid_col: list[str] = []
    vname_col: list[str] = []
    pkey_col: list[str] = []
    pname_col: list[str] = []
    label_col: list[str] = []
    topo_col: list[str] = []
    desc_col: list[str] = []

    for iid, inst_info in sorted(data["instruments"].items()):
        if filter_insts and iid not in filter_insts:
            continue
        iname = inst_info["name"]

        for vid, v_info in sorted(data["voices"].items()):
            if filter_voices and vid not in filter_voices:
                continue
            vname = v_info["name"]
            resp = data["responses"][iid][vid]
            mags: list[float] = resp["magnitude_db"]
            pkey = resp["pickup_key"]
            pname = resp["pickup_name"]
            label = f"{iname} ➔ {vname}"

            freq_col.extend(f_pts)
            mag_col.extend(mags)
            iid_col.extend([iid] * n_pts)
            iname_col.extend([iname] * n_pts)
            vid_col.extend([vid] * n_pts)
            vname_col.extend([vname] * n_pts)
            pkey_col.extend([pkey] * n_pts)
            pname_col.extend([pname] * n_pts)
            label_col.extend([label] * n_pts)
            topo_col.extend([v_info["topology"]] * n_pts)
            desc_col.extend([v_info["description"]] * n_pts)

    return pl.DataFrame(
        {
            "frequency": freq_col,
            "magnitude_db": mag_col,
            "instrument_id": iid_col,
            "instrument_name": iname_col,
            "voice_id": vid_col,
            "voice_name": vname_col,
            "pickup_key": pkey_col,
            "pickup_name": pname_col,
            "label": label_col,
            "topology": topo_col,
            "description": desc_col,
        }
    )


def compute_fir_csd(
    fir: Sequence[float] | np.ndarray,
    freqs: Sequence[float] | np.ndarray,
    num_slices: int = 24,
    max_time_ms: float = 10.0,
    sr: int = 48000,
    n_fft: int = 2048,
    n_rise: int = 16,
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


_BAKED_WATERFALL_CACHE: dict[tuple[int, int, float], dict[str, Any]] = {}


def build_baked_waterfall_3d_data(
    num_freqs: int = 50,
    num_slices: int = 24,
    max_time_ms: float = 10.0,
) -> dict[str, Any]:
    """
    Builds the compact 3D Cumulative Spectral Decay (CSD) and waveform data structure for
    all playable instruments and target voicings in the 1-block monolithic baked configuration.
    Results are cached for fast subsequent renders.
    """
    cache_key = (num_freqs, num_slices, float(max_time_ms))
    if cache_key in _BAKED_WATERFALL_CACHE:
        return _BAKED_WATERFALL_CACHE[cache_key]

    baked = build_baked_responses_data(step=3)
    f_orig = np.asarray(baked["frequencies"], dtype=np.float64)
    f_lin = np.asarray(FREQS, dtype=np.float64)

    f_eval = [round(float(f), 1) for f in np.geomspace(20.0, 20000.0, num_freqs)]

    # Compute time_ms using dummy impulse
    dummy_impulse = np.zeros(2048, dtype=np.float64)
    dummy_impulse[0] = 1.0
    time_ms, _ = compute_fir_csd(
        dummy_impulse, f_eval, num_slices=num_slices, max_time_ms=max_time_ms
    )

    responses_3d: dict[str, dict[str, dict[str, Any]]] = {}

    for iid, inst_resp in baked["responses"].items():
        responses_3d[iid] = {}
        for vid, resp in inst_resp.items():
            db_baked = np.asarray(resp["magnitude_db"], dtype=np.float64)
            pkey = resp["pickup_key"]
            pname = resp["pickup_name"]

            # Interpolate magnitude response onto f_eval
            mag_eval = np.interp(f_eval, f_orig, db_baked)
            mag_eval_list = [round(float(v), 1) for v in mag_eval]

            if np.allclose(db_baked, 0.0, atol=1e-2):
                # Identity voice: unit impulse at t=0, -60 dB floor elsewhere
                csd_matrix = [
                    [0.0 if m == 0 else -60.0 for _ in range(num_freqs)] for m in range(num_slices)
                ]
                fir_head = [1.0] + [0.0] * 127
            else:
                mag_lin = np.interp(
                    f_lin,
                    f_orig,
                    10.0 ** (db_baked / 20.0),
                    left=10.0 ** (db_baked[0] / 20.0),
                    right=10.0 ** (db_baked[-1] / 20.0),
                )
                fir = np.array(
                    synthesize_minimum_phase_fir(mag_lin, num_taps=2048, normalize=False),
                    dtype=np.float64,
                )
                _, csd_matrix = compute_fir_csd(
                    fir,
                    f_eval,
                    num_slices=num_slices,
                    max_time_ms=max_time_ms,
                )
                fir_head = [round(float(x), 4) for x in fir[:128]]

            responses_3d[iid][vid] = {
                "pickup_key": pkey,
                "pickup_name": pname,
                "magnitude_db": mag_eval_list,
                "csd_matrix": csd_matrix,
                "fir_waveform": fir_head,
            }

    data: dict[str, Any] = {
        "frequencies": f_eval,
        "time_ms": time_ms,
        "instruments": baked["instruments"],
        "voices": baked["voices"],
        "responses": responses_3d,
    }

    _BAKED_WATERFALL_CACHE[cache_key] = data
    return data
