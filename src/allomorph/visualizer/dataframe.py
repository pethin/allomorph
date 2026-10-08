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


def _compute_welch_psd(
    x: np.ndarray, sr: int = 48000, n_fft: int = 4096, hop_length: int = 2048
) -> tuple[np.ndarray, np.ndarray]:
    x_pad = np.pad(x, (n_fft // 2, n_fft // 2), mode="constant")
    n_frames = 1 + (len(x_pad) - n_fft) // hop_length
    frames = np.lib.stride_tricks.as_strided(
        x_pad, shape=(n_frames, n_fft), strides=(x_pad.strides[0] * hop_length, x_pad.strides[0])
    )
    window = np.hanning(n_fft)
    spectra = np.abs(np.fft.rfft(frames * window, axis=1)) ** 2
    psd = np.mean(spectra, axis=0)
    psd = psd / (np.sum(window**2) * sr)
    f_bins = np.fft.rfftfreq(n_fft, 1.0 / sr)
    return f_bins, psd


def build_voice_dataframe(
    voice_id: str,
    cfg: Any,
    instrument: InstrumentConfig | str = "30in",
    mode: str = "output",
    include_mode_col: bool = False,
) -> pl.DataFrame:
    import tempfile
    from pathlib import Path

    from allomorph.circuit.forward import simulate_instrument_voicing
    from allomorph.dsp import read_wav

    inst = load_instrument(instrument)

    sr = 48000
    n_samples = 16384
    rng = np.random.RandomState(42)
    x_white = rng.normal(0.0, 1.0, n_samples)
    x_white = x_white / np.max(np.abs(x_white)) * (10.0 ** (-20.5 / 20.0))

    with tempfile.TemporaryDirectory() as td:
        in_wav = Path(td) / "in.wav"
        out_wav = Path(td) / "out.wav"

        from allomorph.dsp import write_wav_24bit

        write_wav_24bit(in_wav, x_white, 48000)
        simulate_instrument_voicing(
            input_wav=in_wav,
            instrument=inst,
            voicing=voice_id,
            output_wav=out_wav,
            max_samples=n_samples,
            normalize="none",
        )
        y_wet, _ = read_wav(out_wav)

    f_bins, psd_y = _compute_welch_psd(y_wet, sr=sr)
    _, psd_x = _compute_welch_psd(x_white, sr=sr)

    H_emp = np.sqrt(psd_y / np.maximum(psd_x, 1e-12))
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
    "soapbar_pair": "Soapbar",
    "soapbar_neck": "Soapbar",
    "soapbar_bridge": "Soapbar",
    "active_emg_pair": "EMG",
    "active_emg_neck": "EMG",
    "active_emg_bridge": "EMG",
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
