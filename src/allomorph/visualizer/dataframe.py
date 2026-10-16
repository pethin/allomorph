"""
Allomorph Visualizer - Frequency Response Modeling and DataFrame Generation
Calculates magnitude frequency responses for instrument voicings and source-to-target
differential comparisons using Polars and NumPy.
"""

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import polars as pl

from allomorph.circuit.forward import simulate_instrument_voicing
from allomorph.config.resolvers import resolve_voicing_electroacoustics
from allomorph.config.schema import (
    InstrumentConfig,
    PreampConfig,
    StringPresetConfig,
    VoicingConfig,
)
from allomorph.dsp import deconvolve_log_sweep, synthesize_fast_log_sweep

NUM_POINTS = 600
F_MIN = 20.0
F_MAX = 20000.0

log_freqs = [F_MIN * (F_MAX / F_MIN) ** (i / (NUM_POINTS - 1)) for i in range(NUM_POINTS)]

_FAST_SWEEP_EXCITATION: np.ndarray | None = None


def get_fast_sweep_excitation() -> np.ndarray:
    """Precomputes and caches the deterministic 16k logarithmic sine sweep excitation."""
    global _FAST_SWEEP_EXCITATION
    if _FAST_SWEEP_EXCITATION is None:
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


_VOICING_CURVE_CACHE: dict[tuple[str, str], np.ndarray] = {}


def build_voice_dataframe(
    instrument: InstrumentConfig,
    voicing: VoicingConfig,
    mode: str = "output",
    include_mode_col: bool = False,
    preamps: Mapping[str, PreampConfig] | None = None,
    strings: Mapping[str, StringPresetConfig] | None = None,
) -> pl.DataFrame:
    """Computes a frequency response DataFrame for a specified instrument voicing."""
    x_sweep = get_fast_sweep_excitation()
    freqs = np.asarray(log_freqs, dtype=np.float64)

    inst_id = instrument.id or ""
    v_id = voicing.id or ""
    cache_key = (inst_id, v_id) if inst_id and v_id else None

    if cache_key is not None and cache_key in _VOICING_CURVE_CACHE:
        mag_db = _VOICING_CURVE_CACHE[cache_key]
    else:
        # Fast in-memory forward simulation evaluated in pure linear mode for visualizer curves
        y_wet = simulate_instrument_voicing(
            instrument=instrument,
            voicing=voicing,
            input_audio=x_sweep,
            return_audio=True,
            dc_block=False,
            apply_saturation=False,
            apply_dither=False,
            preamps=preamps,
            strings=strings,
        )

        # Regularized Farina deconvolution with causal 8192-tap impulse gating
        f_bins, H_complex, _ = deconvolve_log_sweep(y_wet, x_sweep, sr=48000, gate_taps=8192)
        H_emp = np.abs(H_complex)

        mag_raw = np.interp(freqs, f_bins, H_emp)
        mag_db = 20.0 * np.log10(np.maximum(mag_raw, 1e-6))

        if cache_key is not None:
            _VOICING_CURVE_CACHE[cache_key] = mag_db

    sensor_type = voicing.sensor_type

    voice_id = f"{inst_id}:{v_id}" if (inst_id and v_id) else (v_id or "default_voicing")
    vname = voicing.tone_name or voicing.name or voice_id
    df = pl.DataFrame(
        {
            "frequency": np.round(freqs, 1).tolist(),
            "magnitude_db": np.round(mag_db, 2).tolist(),
            "line_type": [f"Target: {vname}"] * len(freqs),
            "voice_id": [voice_id] * len(freqs),
            "voice_name": [vname] * len(freqs),
            "sensor_type": [sensor_type] * len(freqs),
            "description": [voicing.description or f"{voicing.name} ({instrument.name})"]
            * len(freqs),
        }
    )
    if include_mode_col:
        df = df.with_columns(pl.lit(mode.capitalize()).alias("mode"))
    return df


def build_voicings_comparison_data(
    instruments: Mapping[str, InstrumentConfig] | None = None,
    step: int = 3,
    preamps: Mapping[str, PreampConfig] | None = None,
    strings: Mapping[str, StringPresetConfig] | None = None,
) -> dict[str, Any]:
    """
    Builds the compact data structure containing frequency responses and metadata
    for all instrument voicings in the catalog, keyed by '<instrument_id>:<voicing_id>'.
    """
    from allomorph.visualizer.schema import load_visualizer_config

    if instruments is None:
        from allomorph.config.instruments import INSTRUMENTS

        instruments = INSTRUMENTS

    if preamps is None:
        from allomorph.config.preamps import PREAMPS

        preamps = PREAMPS
    if strings is None:
        from allomorph.config.strings import STRINGS

        strings = STRINGS

    vcfg = load_visualizer_config()
    freqs = np.asarray(log_freqs[::step], dtype=np.float64)
    f_pts = np.round(freqs, 1).tolist()

    voices_dict: dict[str, dict[str, Any]] = {}
    instruments_dict: dict[str, dict[str, Any]] = {}

    for inst_id, inst in sorted(instruments.items()):
        inst_name = inst.name
        inst_scale = float(inst.scale_length_in) if inst.scale_length_in is not None else 34.0
        if inst.is_multiscale and inst.scale_min_in is not None and inst.scale_max_in is not None:
            inst_label = f'{inst_name} ({inst.scale_min_in:.1f}"-{inst.scale_max_in:.1f}")'
        else:
            inst_label = f'{inst_name} ({inst_scale:.1f}")'

        instruments_dict[inst_id] = {
            "id": inst_id,
            "name": inst_name,
            "scale": inst_scale,
            "label": inst_label,
            "voice_ids": [],
        }

        for v_id, voicing_cfg in inst.voicings.items():
            canonical_vid = f"{inst_id}:{v_id}"
            vdf = build_voice_dataframe(
                inst,
                voicing_cfg,
                mode="output",
                preamps=preamps,
                strings=strings,
            )
            mag_full = np.asarray(vdf["magnitude_db"], dtype=np.float64)
            db_tgt = mag_full[::step] if step > 1 else mag_full

            electro = resolve_voicing_electroacoustics(inst, voicing_cfg)
            vname = voicing_cfg.tone_name or voicing_cfg.name or v_id
            rms_db = compute_curve_rms_db(db_tgt)

            instruments_dict[inst_id]["voice_ids"].append(canonical_vid)

            voices_dict[canonical_vid] = {
                "id": canonical_vid,
                "name": vname,
                "tone_name": vname,
                "instrument_id": inst_id,
                "instrument_name": inst_name,
                "sensor_type": voicing_cfg.sensor_type,
                "description": f"{voicing_cfg.name} ({inst_name})",
                "fr": float(electro.fr) if electro.fr is not None else None,
                "q": float(electro.q) if electro.q is not None else None,
                "alpha": round(float(electro.alpha), 3),
                "vsat": round(float(electro.vsat), 3),
                "magnitude_db": np.round(db_tgt, 2).tolist(),
                "rms_db": round(rms_db, 2),
            }

    standard_instrument_order = [
        "34in_standard_p",
        "34in_standard_jazz",
        "34in_standard_pj",
        "30in_mustang_pj",
        "34in_active_stingray",
        "34in_active_pmm",
        "32in_custom_pmm",
        "32in_fretless_pmm",
        "34in_preamp_soapbar",
        "34in_active_emg",
        "30in_emg_mmtw",
        "37in_multiscale_dingwall",
        "34in_dingwall_sp1",
        "33in_rickenbacker_4003",
        "30in_gibson_eb0",
        "30in_gibson_eb2",
        "30in_hofner_500_1",
        "41in_upright_bass",
    ]
    sorted_instruments = [i for i in standard_instrument_order if i in instruments_dict] + [
        i for i in instruments_dict if i not in standard_instrument_order
    ]

    return {
        "frequencies": f_pts,
        "voices": voices_dict,
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
    source_instrument: InstrumentConfig,
    source_voicing: VoicingConfig,
    target_instrument: InstrumentConfig,
    target_voicing: VoicingConfig,
    step: int = 1,
    preamps: Mapping[str, PreampConfig] | None = None,
    strings: Mapping[str, StringPresetConfig] | None = None,
) -> pl.DataFrame:
    """
    Constructs a 3-line Polars DataFrame comparing a Source Voicing and Target Voicing:
      1. Source Voicing (H_src)
      2. Target Voicing (H_tgt)
      3. Difference (H_diff = H_tgt - H_src)
    When source == target, the differential line evaluates to exact 0.00 dB.
    """
    freqs = np.asarray(log_freqs[::step], dtype=np.float64)
    n_pts = len(freqs)
    f_pts = np.round(freqs, 1).tolist()

    df_src = build_voice_dataframe(
        source_instrument, source_voicing, preamps=preamps, strings=strings
    )
    df_tgt = build_voice_dataframe(
        target_instrument, target_voicing, preamps=preamps, strings=strings
    )

    db_src_full = np.asarray(df_src["magnitude_db"], dtype=np.float64)
    db_tgt_full = np.asarray(df_tgt["magnitude_db"], dtype=np.float64)

    db_src = db_src_full[::step] if step > 1 else db_src_full
    db_tgt = db_tgt_full[::step] if step > 1 else db_tgt_full

    src_name = source_voicing.name or source_voicing.id or "source"
    tgt_name = target_voicing.name or target_voicing.id or "target"
    source_id = (
        f"{source_instrument.id}:{source_voicing.id}"
        if (source_instrument.id and source_voicing.id)
        else (source_voicing.id or "source")
    )
    target_id = (
        f"{target_instrument.id}:{target_voicing.id}"
        if (target_instrument.id and target_voicing.id)
        else (target_voicing.id or "target")
    )

    if source_instrument.id == target_instrument.id and source_voicing.id == target_voicing.id:
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
            "sensor_type": [source_voicing.sensor_type] * n_pts
            + [target_voicing.sensor_type] * n_pts
            + [target_voicing.sensor_type] * n_pts,
            "description": [src_name] * n_pts
            + [tgt_name] * n_pts
            + [f"{tgt_name} - {src_name}"] * n_pts,
        }
    )
