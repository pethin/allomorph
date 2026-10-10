"""
Allomorph Visualizer - Frequency Response Modeling and DataFrame Generation
Calculates magnitude frequency responses for target voicings, frontend deconvolutions,
and composite signal flow stages using Polars and NumPy.
"""

from collections.abc import Sequence
from typing import Any

import numpy as np
import polars as pl

from allomorph.config.instruments import (
    load_instrument,
)
from allomorph.config.schema import InstrumentConfig
from allomorph.config.voices import VOICES

NUM_POINTS = 600
F_MIN = 20.0
F_MAX = 20000.0

log_freqs = [F_MIN * (F_MAX / F_MIN) ** (i / (NUM_POINTS - 1)) for i in range(NUM_POINTS)]

_OUTPUT_VOICE_DF_CACHE: dict[str, pl.DataFrame] = {}
_DIFF_VOICE_DF_CACHE: dict[tuple[str, str, str], pl.DataFrame] = {}


_TARGET_DFS_CACHE: dict[int, dict[str, tuple[str, np.ndarray]]] = {}


_FAST_SWEEP_EXCITATION: np.ndarray | None = None


def get_fast_sweep_excitation() -> np.ndarray:
    """Precomputes and caches the deterministic 16k logarithmic sine sweep excitation."""
    global _FAST_SWEEP_EXCITATION
    if _FAST_SWEEP_EXCITATION is None:
        from allomorph.dsp import synthesize_fast_log_sweep

        _FAST_SWEEP_EXCITATION = synthesize_fast_log_sweep(
            n_samples=16384,
            f_start=10.0,
            f_end=24000.0,
            sr=48000,
            target_dbfs=-20.5,
            fade_len=144,
            tail_len=2048,
        )
    return _FAST_SWEEP_EXCITATION


def build_voice_dataframe(
    voice_id: str,
    cfg: Any,
    instrument: InstrumentConfig | str = "30in",
    mode: str = "output",
    include_mode_col: bool = False,
) -> pl.DataFrame:
    from allomorph.circuit.forward import simulate_instrument_voicing
    from allomorph.dsp import deconvolve_log_sweep

    inst = load_instrument(instrument)
    x_sweep = get_fast_sweep_excitation()

    # Fast in-memory forward simulation evaluated in pure linear mode for visualizer curves
    y_wet = simulate_instrument_voicing(
        instrument=inst,
        voicing=voice_id,
        input_audio=x_sweep,
        return_audio=True,
        normalize="none",
        dc_block=False,
        apply_saturation=False,
        apply_dither=False,
    )

    # Regularized Farina deconvolution with causal 8192-tap impulse gating
    f_bins, H_complex, _ = deconvolve_log_sweep(y_wet, x_sweep, sr=48000, gate_taps=8192)
    H_emp = np.abs(H_complex)

    freqs = np.asarray(log_freqs, dtype=np.float64)
    mag_raw = np.interp(freqs, f_bins, H_emp)

    mag_db = 20.0 * np.log10(np.maximum(mag_raw, 1e-4))

    mid_mask = (freqs >= 100.0) & (freqs <= 800.0)
    if len(mag_db[mid_mask]) > 0:
        mag_db = mag_db - np.median(mag_db[mid_mask])

    df = pl.DataFrame(
        {
            "frequency": np.round(freqs, 1).tolist(),
            "magnitude_db": np.round(mag_db, 2).tolist(),
            "line_type": [f"Target: {cfg.name}"] * len(freqs),
            "voice_id": [voice_id] * len(freqs),
        }
    )
    if include_mode_col:
        df = df.with_columns(pl.lit(mode.capitalize()).alias("mode"))
    return df


def get_visualizer_voicing_ids() -> list[str]:
    """Discovers all target and source voicings declared across tone packs and visualizer.toml."""
    from allomorph.config.instruments import PACKS_CONFIG_DIR, load_tone_pack
    from allomorph.visualizer.schema import load_visualizer_config

    vcfg = load_visualizer_config()
    discovered: list[str] = []

    def _add_voice(v_token: str) -> None:
        if v_token in VOICES:
            vid = VOICES[v_token].id
            if vid not in discovered:
                discovered.append(vid)

    if PACKS_CONFIG_DIR.is_dir():
        for p in sorted(PACKS_CONFIG_DIR.glob("*.toml")):
            pack = load_tone_pack(p)
            for b in pack.bundles:
                _add_voice(f"{pack.instrument}:{b.source_voicing}")
                _add_voice(b.source_voicing)
                for tgt in b.targets:
                    _add_voice(tgt)

    for extra in vcfg.additional_source_voicings:
        _add_voice(extra)
    for extra in vcfg.additional_target_voicings:
        _add_voice(extra)
    for chip in vcfg.chips:
        _add_voice(chip.source)
        _add_voice(chip.target)
    _add_voice(vcfg.default_source)
    _add_voice(vcfg.default_target)

    # Ensure all registered catalog target voices are included
    for vid in sorted(VOICES.keys()):
        if vid not in discovered:
            discovered.append(vid)

    return discovered


def get_cached_target_dfs(step: int = 1) -> dict[str, tuple[str, np.ndarray]]:
    """Caches precomputed target voice responses downsampled by step."""
    if step in _TARGET_DFS_CACHE:
        return _TARGET_DFS_CACHE[step]

    target_dfs: dict[str, tuple[str, np.ndarray]] = {}
    for vid in get_visualizer_voicing_ids():
        cfg = VOICES[vid]
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
    "upright_piezo": "Upright",
    "soapbar_pair": "Soapbar",
    "soapbar_neck": "Soapbar",
    "soapbar_bridge": "Soapbar",
    "emg_soapbar_parallel": "EMG",
    "emg_soapbar_neck": "EMG",
    "emg_soapbar_bridge": "EMG",
}


def build_voicings_comparison_data(step: int = 3) -> dict[str, Any]:
    """
    Builds the compact data structure containing frequency responses and metadata
    for all target voicings in VOICES.
    Used by voicings.html to display the 3-line graph:
      - Line 1: Source Voicing (H_src)
      - Line 2: Target Voicing (H_tgt)
      - Line 3: Difference (H_diff = H_tgt - H_src)
    When Source == Target, the difference line evaluates to exact 0.00 dB.
    """
    from allomorph.config.instruments import INSTRUMENTS
    from allomorph.visualizer.schema import load_visualizer_config

    vcfg = load_visualizer_config()
    freqs = np.asarray(log_freqs[::step], dtype=np.float64)
    f_pts = np.round(freqs, 1).tolist()

    target_dfs = get_cached_target_dfs(step=step)

    voices_dict: dict[str, dict[str, Any]] = {}
    instruments_dict: dict[str, dict[str, Any]] = {}
    families_set: set[str] = set()

    for vid, (vname, db_tgt) in sorted(target_dfs.items()):
        v_obj = VOICES[vid]
        family = VOICE_FAMILIES.get(vid, "Specialty")
        families_set.add(family)
        rms_db = compute_curve_rms_db(db_tgt)

        inst_id = v_obj.instrument_id or ""
        inst = INSTRUMENTS.get(inst_id)
        inst_name = inst.name if inst else "Other"
        inst_scale = (
            float(inst.scale_length_in)
            if (inst is not None and inst.scale_length_in is not None)
            else 34.0
        )
        inst_label = f'{inst_name} ({inst_scale:.1f}")' if inst else "Other"

        if inst_id not in instruments_dict:
            instruments_dict[inst_id] = {
                "id": inst_id,
                "name": inst_name,
                "scale": inst_scale,
                "label": inst_label,
                "voice_ids": [],
            }
        instruments_dict[inst_id]["voice_ids"].append(vid)

        voices_dict[vid] = {
            "id": vid,
            "name": vname,
            "tone_name": v_obj.tone_name or vname,
            "family": family,
            "instrument_id": inst_id,
            "instrument_name": inst_name,
            "sensor_type": v_obj.sensor_type,
            "description": v_obj.description,
            "fr": float(v_obj.fr),
            "q": float(v_obj.Q),
            "alpha": float(v_obj.alpha or 0.0),
            "vsat": float(v_obj.vsat or 1.0),
            "magnitude_db": np.round(db_tgt, 2).tolist(),
            "rms_db": round(rms_db, 2),
        }

    family_order = [
        "Precision",
        "Jazz",
        "StingRay",
        "Soapbar",
        "EMG",
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

    standard_instrument_order = [
        "34in_standard_p",
        "34in_standard_jazz",
        "34in_standard_pj",
        "30in_mustang_pj",
        "34in_active_stingray",
        "34in_active_pmm",
        "34in_preamp_soapbar",
        "34in_active_emg",
        "37in_multiscale_dingwall",
        "34in_dingwall_sp1",
        "33in_rickenbacker_4003",
        "30in_gibson_eb0",
        "41in_upright_bass",
    ]
    sorted_instruments = [i for i in standard_instrument_order if i in instruments_dict] + [
        i for i in instruments_dict if i not in standard_instrument_order
    ]

    return {
        "frequencies": f_pts,
        "voices": voices_dict,
        "families": sorted_families,
        "instruments": instruments_dict,
        "instrument_order": sorted_instruments,
        "chips": [c.model_dump() for c in vcfg.chips],
        "default_source": vcfg.default_source,
        "default_target": vcfg.default_target,
    }


def compute_curve_rms_db(mag_db: np.ndarray | Sequence[float] | pl.Series) -> float:
    """Computes the equivalent broadband RMS gain in dB for a magnitude frequency response."""
    arr = np.asarray(mag_db, dtype=np.float64)
    return float(20.0 * np.log10(np.sqrt(np.mean((10.0 ** (arr / 20.0)) ** 2))))


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
        db_diff = np.round(db_tgt - db_src, 2)

    freq_col = f_pts * 3
    mag_col = np.round(db_src, 2).tolist() + np.round(db_tgt, 2).tolist() + db_diff.tolist()
    line_type_col = (
        ["1. Source Voicing"] * n_pts + ["2. Target Voicing"] * n_pts + ["3. Difference"] * n_pts
    )
    vid_col = [source_id] * n_pts + [target_id] * n_pts + [f"{source_id}_to_{target_id}"] * n_pts
    vname_col = (
        [src_name] * n_pts + [tgt_name] * n_pts + [f"Difference: {tgt_name} - {src_name}"] * n_pts
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
