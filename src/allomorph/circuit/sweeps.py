"""
Allomorph - Instantaneous Continuous Parametric Sweeps
Evaluates exact continuous electrical parameter sweeps across frequencies in < 45 ms.
Supports tone pot, volume pot, cable capacitance, tone capacitor, and active EQ sweeps.
"""

import copy
from collections.abc import Sequence
from pathlib import Path
from typing import Self

import numpy as np
import polars as pl
from pydantic import BaseModel, ConfigDict, model_validator

from allomorph.circuit.parser import CircuitModel, load_circuit
from allomorph.circuit.schema import CircuitMetricsRecord
from allomorph.circuit.solver import (
    apply_magnet_properties_to_model,
    compute_circuit_transfer_functions,
)
from allomorph.config.schema import PreampBandConfig, VoiceConfig
from allomorph.dsp import FREQS


class ParametricSweepResult(BaseModel):
    """Represents the results of a parametric frequency response sweep with verified array invariants."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    param: str
    values: list[float]
    freqs: np.ndarray
    curves: list[np.ndarray]  # magnitude in dB for each swept value
    labels: list[str]
    voice_id: str | None = None

    @model_validator(mode="after")
    def validate_dimensional_invariants(self) -> Self:
        n_v = len(self.values)
        if len(self.curves) != n_v or len(self.labels) != n_v:
            raise ValueError(
                f"Dimensional mismatch in ParametricSweepResult: values={n_v}, "
                f"curves={len(self.curves)}, labels={len(self.labels)} must all match."
            )
        n_f = len(self.freqs)
        for i, c in enumerate(self.curves):
            if len(c) != n_f:
                raise ValueError(
                    f"Curve {i} length ({len(c)}) does not match frequencies length ({n_f})."
                )
        return self

    @property
    def curves_db(self) -> list[np.ndarray]:
        """Returns curves in decibels (alias for curves)."""
        return self.curves

    @property
    def curves_linear(self) -> list[np.ndarray]:
        """Returns linear magnitude curves."""
        return [10.0 ** (np.asarray(c, dtype=np.float64) / 20.0) for c in self.curves]

    def to_dataframe(self, include_voice_id: bool = False) -> pl.DataFrame:
        """
        Converts the sweep results to a Polars DataFrame with columns:
        ['frequency', 'magnitude_db', 'param', 'param_value', 'label']
        and optionally 'voice_id'.
        """
        n_f = len(self.freqs)
        n_v = len(self.values)
        all_freqs = np.tile(self.freqs, n_v)
        all_mags = np.concatenate([np.asarray(c, dtype=np.float64) for c in self.curves])
        all_params = [self.param] * (n_f * n_v)
        all_pvals = np.repeat(np.asarray(self.values, dtype=np.float64), n_f)
        all_labels = np.repeat(self.labels, n_f)

        data = {
            "frequency": all_freqs,
            "magnitude_db": all_mags,
            "param": all_params,
            "param_value": all_pvals,
            "label": all_labels,
        }
        if include_voice_id and self.voice_id:
            data["voice_id"] = [self.voice_id] * (n_f * n_v)
        return pl.DataFrame(data)

    def metrics_records(self) -> list[CircuitMetricsRecord]:
        """
        Extracts key analytical circuit metrics for each swept curve as CircuitMetricsRecord models:
          - f_res_hz: Resonant peak frequency (Hz) within passband (400 Hz - 12 kHz).
          - peak_db: Resonant peak magnitude (dB).
          - insertion_loss_db: Low-frequency insertion loss (dB) evaluated near 100 Hz.
          - peak_boost_db: Resonant peak boost above low-frequency insertion loss (dB).
          - q_loaded: Loaded circuit quality factor Q = f_res / delta_f.
          - bandwidth_hz: -3 dB bandwidth around resonant peak (Hz).
          - cutoff_3db_hz: -3 dB cutoff frequency relative to low-frequency baseline (Hz).
          - hf_slope_db_oct: High-frequency roll-off slope (dB/octave between 6 kHz and 12 kHz).
        """
        f = np.asarray(self.freqs, dtype=np.float64)
        idx_100 = int(np.argmin(np.abs(f - 100.0)))
        idx_6k = int(np.argmin(np.abs(f - 6000.0)))
        idx_12k = int(np.argmin(np.abs(f - 12000.0)))
        octaves_6k_12k = np.log2(f[idx_12k] / f[idx_6k]) if f[idx_12k] > f[idx_6k] else 1.0

        pb_mask = (f >= 400.0) & (f <= 12000.0)
        pb_indices = np.where(pb_mask)[0]

        records: list[CircuitMetricsRecord] = []
        for i, (val, lbl, c) in enumerate(zip(self.values, self.labels, self.curves)):
            curve = np.asarray(c, dtype=np.float64)
            loss_db = float(curve[idx_100])

            # Resonant peak search with continuous sub-bin quadratic interpolation
            if len(pb_indices) > 0:
                max_pb_idx = pb_indices[int(np.argmax(curve[pb_indices]))]
                if 0 < max_pb_idx < len(f) - 1:
                    y_m1 = float(curve[max_pb_idx - 1])
                    y_0 = float(curve[max_pb_idx])
                    y_p1 = float(curve[max_pb_idx + 1])
                    curv = y_m1 - 2.0 * y_0 + y_p1
                    if curv < -1e-9:
                        delta = float(np.clip(0.5 * (y_m1 - y_p1) / curv, -0.5, 0.5))
                        df = float(f[max_pb_idx + 1] - f[max_pb_idx])
                        f_res = float(f[max_pb_idx] + delta * df)
                        peak_db = float(y_0 - 0.25 * (y_m1 - y_p1) * delta)
                    else:
                        f_res = float(f[max_pb_idx])
                        peak_db = y_0
                else:
                    f_res = float(f[max_pb_idx])
                    peak_db = float(curve[max_pb_idx])
                peak_boost = peak_db - loss_db
            else:
                max_pb_idx = idx_100
                f_res = float(f[idx_100])
                peak_db = loss_db
                peak_boost = 0.0

            # -3 dB bandwidth around resonant peak
            bw_hz = None
            q_loaded = None
            if peak_boost >= 0.5 and 0 < max_pb_idx < len(f) - 1:
                target_3db = peak_db - 3.0

                # Left crossing (below peak)
                left_cross = None
                for j in range(max_pb_idx - 1, -1, -1):
                    if curve[j] <= target_3db:
                        denom = curve[j + 1] - curve[j]
                        frac = (target_3db - curve[j]) / denom if abs(denom) > 1e-6 else 0.5
                        left_cross = f[j] + frac * (f[j + 1] - f[j])
                        break

                # Right crossing (above peak)
                right_cross = None
                for j in range(max_pb_idx + 1, len(f)):
                    if curve[j] <= target_3db:
                        denom = curve[j - 1] - curve[j]
                        frac = (target_3db - curve[j]) / denom if abs(denom) > 1e-6 else 0.5
                        right_cross = f[j] + frac * (f[j - 1] - f[j])
                        break

                if left_cross is not None and right_cross is not None and right_cross > left_cross:
                    bw_hz = float(right_cross - left_cross)
                    q_loaded = float(f_res / bw_hz) if bw_hz > 0 else None
                elif peak_boost >= 0.2:
                    # Analytical 2nd-order lowpass loaded Q from peaking factor Mp = 10^(peak_boost/20)
                    mp = 10.0 ** (peak_boost / 20.0)
                    if mp > 1.0:
                        q_loaded = float(
                            np.sqrt((mp**2 + mp * np.sqrt(max(0.0, mp**2 - 1.0))) / 2.0)
                        )
                        bw_hz = float(f_res / q_loaded) if q_loaded > 0 else None

            # -3 dB cutoff frequency relative to low-frequency baseline (loss_db - 3.0)
            target_cutoff = loss_db - 3.0
            cutoff_3db_hz = None
            for j in range(idx_100, len(f) - 1):
                if curve[j] >= target_cutoff and curve[j + 1] < target_cutoff:
                    denom = curve[j] - curve[j + 1]
                    frac = (curve[j] - target_cutoff) / denom if abs(denom) > 1e-6 else 0.5
                    cutoff_3db_hz = float(f[j] + frac * (f[j + 1] - f[j]))
                    break

            # High-frequency slope (dB/octave) between 6 kHz and 12 kHz
            hf_slope = float((curve[idx_12k] - curve[idx_6k]) / octaves_6k_12k)

            records.append(
                CircuitMetricsRecord(
                    param=self.param,
                    param_value=float(val),
                    label=lbl,
                    f_res_hz=round(f_res, 1) if peak_boost >= 0.5 else None,
                    peak_db=round(peak_db, 2),
                    insertion_loss_db=round(loss_db, 2),
                    peak_boost_db=round(peak_boost, 2),
                    q_loaded=round(q_loaded, 2) if q_loaded is not None else None,
                    bandwidth_hz=round(bw_hz, 1) if bw_hz is not None else None,
                    cutoff_3db_hz=round(cutoff_3db_hz, 1) if cutoff_3db_hz is not None else None,
                    hf_slope_db_oct=round(hf_slope, 2),
                )
            )

        return records

    def metrics(self) -> pl.DataFrame:
        """Extracts key analytical circuit metrics for each swept curve as a Polars DataFrame."""
        recs = self.metrics_records()
        return pl.DataFrame(
            {
                "param": [r.param for r in recs],
                "param_value": [r.param_value for r in recs],
                "label": [r.label for r in recs],
                "f_res_hz": [r.f_res_hz for r in recs],
                "peak_db": [r.peak_db for r in recs],
                "insertion_loss_db": [r.insertion_loss_db for r in recs],
                "peak_boost_db": [r.peak_boost_db for r in recs],
                "q_loaded": [r.q_loaded for r in recs],
                "bandwidth_hz": [r.bandwidth_hz for r in recs],
                "cutoff_3db_hz": [r.cutoff_3db_hz for r in recs],
                "hf_slope_db_oct": [r.hf_slope_db_oct for r in recs],
            }
        )

    def summary_table(self) -> str:
        """Formats the analytical metrics into a clean terminal table string."""
        df = self.metrics()
        lines = []
        hdr = (
            f"{'Setting / Label':<24} "
            f"{'f_res (Hz)':>11} "
            f"{'Peak (dB)':>10} "
            f"{'Loss (dB)':>10} "
            f"{'Boost (dB)':>11} "
            f"{'Q loaded':>9} "
            f"{'BW (Hz)':>9} "
            f"{'-3dB Cut (Hz)':>14} "
            f"{'HF Slope':>12}"
        )
        lines.append(hdr)
        lines.append("-" * len(hdr))
        for row in df.iter_rows(named=True):
            lbl = row["label"]
            f_res = f"{row['f_res_hz']:.1f}" if row["f_res_hz"] is not None else "---"
            pk = f"{row['peak_db']:+.2f}"
            loss = f"{row['insertion_loss_db']:+.2f}"
            boost = f"{row['peak_boost_db']:+.2f}"
            q = f"{row['q_loaded']:.2f}" if row["q_loaded"] is not None else "---"
            bw = f"{row['bandwidth_hz']:.1f}" if row["bandwidth_hz"] is not None else "---"
            cut = f"{row['cutoff_3db_hz']:.1f}" if row["cutoff_3db_hz"] is not None else "---"
            slope = f"{row['hf_slope_db_oct']:.1f} dB/oct"
            lines.append(
                f"{lbl:<24} {f_res:>11} {pk:>10} {loss:>10} {boost:>11} {q:>9} {bw:>9} {cut:>14} {slope:>12}"
            )
        return "\n".join(lines)

    def print_metrics(self):
        """Prints the analytical metrics table to stdout."""
        print(self.summary_table())


def _get_default_sweep_values(param: str) -> list[float]:
    """Provides default numerical values for a given sweep parameter."""
    p = param.lower().strip()
    if (
        p in ("tone", "tone_pos", "tone_wiper", "tone_pot")
        or p in ("vol", "vol_pos", "volume", "vol_wiper", "volume_pot")
        or p in ("blend", "blend_pos", "pan", "balance")
    ):
        return [0.0, 0.25, 0.5, 0.75, 1.0]
    elif p in ("cable", "cable_pf", "ccable", "cable_capacitance"):
        return [200.0, 500.0, 750.0, 1000.0, 1500.0]
    elif p in ("tone_cap", "ctone", "cap", "tone_capacitance", "tone_cap_nf"):
        return [22.0, 33.0, 47.0, 68.0, 100.0]
    elif p in ("bass_boost", "preamp_bass", "bass") or p in (
        "treble_boost",
        "preamp_treble",
        "treble",
    ):
        return [0.0, 3.0, 6.0, 9.0, 12.0]
    return [0.0, 0.5, 1.0]


def _generate_default_labels(param: str, values: list[float]) -> list[str]:
    """Generates clean human-readable labels for sweep values."""
    p = param.lower().strip()
    if p in ("tone", "tone_pos", "tone_wiper", "tone_pot"):
        return [f"Tone {round(v * 100)}%" for v in values]
    elif p in ("vol", "vol_pos", "volume", "vol_wiper", "volume_pot"):
        return [f"Vol {round(v * 100)}%" for v in values]
    elif p in ("blend", "blend_pos", "pan", "balance"):
        lbl_map = {
            0.0: "Neck 100%",
            0.25: "Neck 75% / Bridge 25%",
            0.5: "Center (100%/100%)",
            0.75: "Neck 25% / Bridge 75%",
            1.0: "Bridge 100%",
        }
        return [lbl_map.get(round(v, 2), f"Blend {round(v * 100)}%") for v in values]
    elif p in ("cable", "cable_pf", "ccable", "cable_capacitance"):
        return [f"Cable {v:.0f} pF" if v > 1e-6 else f"Cable {v * 1e12:.0f} pF" for v in values]
    elif p in ("tone_cap", "ctone", "cap", "tone_capacitance", "tone_cap_nf"):
        return [f"Cap {v:.0f} nF" if v > 1e-6 else f"Cap {v * 1e9:.0f} nF" for v in values]
    elif p in ("bass_boost", "preamp_bass", "bass"):
        return [f"Bass {v:+.1f} dB" for v in values]
    elif p in ("treble_boost", "preamp_treble", "treble"):
        return [f"Treble {v:+.1f} dB" for v in values]
    else:
        return [f"{param}={v}" for v in values]


def compute_parametric_sweep(
    circuit_or_voice: CircuitModel | VoiceConfig | str | Path,
    param: str = "tone",
    values: Sequence[float] | np.ndarray | None = None,
    freqs: Sequence[float] | np.ndarray = FREQS,
    labels: list[str] | None = None,
    pickup_channel: int | str = 0,
    pot_taper: str = "audio",
) -> ParametricSweepResult:
    """
    Computes continuous frequency response curves across a swept electrical parameter.

    Parameters:
        circuit_or_voice: CircuitModel, VoiceConfig, or voice identifier string/path.
        param: Circuit parameter to sweep:
            - 'tone' / 'tone_pos': Pot wiper position (0.0 to 1.0).
            - 'vol' / 'vol_pos': Pot wiper position (0.0 to 1.0).
            - 'blend' / 'blend_pos': Pickup blend balance (0.0 Neck to 1.0 Bridge, 0.5 center).
            - 'cable' / 'cable_pf': Cable capacitance in pF (or Farads if < 1e-6).
            - 'tone_cap' / 'Ctone': Tone capacitance in nF (or Farads if < 1e-6).
            - 'bass_boost' / 'preamp_bass': Active preamp bass shelf gain in dB.
            - 'treble_boost' / 'preamp_treble': Active preamp treble shelf gain in dB.
            - Any direct numerical attribute on CircuitModel (e.g. 'L', 'Rdc', 'Reddy').
        values: Sequence of numerical parameter values. If None, uses smart defaults.
        freqs: Frequency vector in Hz (defaults to standard 4096-tap FREQS).
        labels: Optional custom string labels for each value.
        pickup_channel: Output channel index or 'sum' to evaluate (default 0).
        pot_taper: Potentiometer resistance curve: 'audio' (10% CTS), 'audio15' (15% Bourns), or 'linear'.

    Returns:
        ParametricSweepResult containing curves in dB, metadata, and Polars export.
    """
    if isinstance(circuit_or_voice, CircuitModel):
        model = circuit_or_voice
        voice_id = None
    elif isinstance(circuit_or_voice, VoiceConfig):
        voice_id = circuit_or_voice.id
        model = load_circuit(circuit_or_voice)
        apply_magnet_properties_to_model(model, circuit_or_voice)
    elif isinstance(circuit_or_voice, (str, Path)):
        voice_id = Path(circuit_or_voice).stem
        model = load_circuit(circuit_or_voice)
        try:
            from allomorph.config.voices import VOICES

            cfg = VOICES.get(str(circuit_or_voice)) or VOICES.get(voice_id)
            if cfg:
                apply_magnet_properties_to_model(model, cfg)
        except KeyError, ImportError, AttributeError, ValueError:
            pass
    else:
        raise TypeError(f"Invalid circuit_or_voice: {type(circuit_or_voice)}")

    f_arr = np.asarray(freqs, dtype=np.float64)

    if values is None:
        values = _get_default_sweep_values(param)
    values = [float(v) for v in values]

    if labels is None:
        labels = _generate_default_labels(param, values)
    elif len(labels) != len(values):
        raise ValueError(f"Length of labels ({len(labels)}) must match values ({len(values)})")

    ch_idx = pickup_channel if isinstance(pickup_channel, int) else 0
    p = param.lower().strip()
    curves: list[np.ndarray] = []

    if p in ("tone", "tone_pos", "tone_wiper", "tone_pot"):
        orig_tone_pos = model.tone_pos
        orig_Rtone = model.Rtone
        orig_taper = model.pot_taper
        try:
            for v in values:
                model.apply_pot_positions(tone_pos=v, pot_taper=pot_taper)
                tr = compute_circuit_transfer_functions(model, freqs=f_arr, return_numpy=True)
                ch = min(ch_idx, len(tr) - 1)
                mag_db = 20.0 * np.log10(np.maximum(tr[ch], 1e-6))
                curves.append(mag_db)
        finally:
            model.tone_pos = orig_tone_pos
            model.Rtone = orig_Rtone
            model.pot_taper = orig_taper

    elif p in ("vol", "vol_pos", "volume", "vol_wiper", "volume_pot"):
        orig_vol_pos = model.vol_pos
        orig_Rtop = model.Rtop
        orig_Rbot = model.Rbot
        orig_taper = model.pot_taper
        try:
            for v in values:
                model.apply_pot_positions(vol_pos=v, pot_taper=pot_taper)
                tr = compute_circuit_transfer_functions(model, freqs=f_arr, return_numpy=True)
                ch = min(ch_idx, len(tr) - 1)
                mag_db = 20.0 * np.log10(np.maximum(tr[ch], 1e-6))
                curves.append(mag_db)
        finally:
            model.vol_pos = orig_vol_pos
            model.Rtop = orig_Rtop
            model.Rbot = orig_Rbot
            model.pot_taper = orig_taper

    elif p in ("blend", "blend_pos", "pan", "balance"):
        orig_blend = model.blend_pos
        orig_taper = model.pot_taper
        orig_r_n = model.Rpot_n
        orig_r_b = model.Rpot_b
        try:
            for v in values:
                model.apply_pot_positions(blend_pos=v, pot_taper=pot_taper)
                tr = compute_circuit_transfer_functions(model, freqs=f_arr, return_numpy=True)
                if len(tr) > 1 and (
                    pickup_channel in ("sum", -1, 0) or str(pickup_channel).lower() == "sum"
                ):
                    combined = np.abs(tr[0] + tr[1])
                    mag_db = 20.0 * np.log10(np.maximum(combined, 1e-6))
                else:
                    ch = (
                        min(int(pickup_channel), len(tr) - 1)
                        if isinstance(pickup_channel, int)
                        else 0
                    )
                    mag_db = 20.0 * np.log10(np.maximum(tr[ch], 1e-6))
                curves.append(mag_db)
        finally:
            model.blend_pos = orig_blend
            model.pot_taper = orig_taper
            model.Rpot_n = orig_r_n
            model.Rpot_b = orig_r_b

    elif p in ("cable", "cable_pf", "ccable", "cable_capacitance"):
        orig_Ccable = model.Ccable
        try:
            for v in values:
                c_farads = v * 1e-12 if v > 1e-6 else v
                model.Ccable = c_farads
                tr = compute_circuit_transfer_functions(model, freqs=f_arr, return_numpy=True)
                ch = min(ch_idx, len(tr) - 1)
                mag_db = 20.0 * np.log10(np.maximum(tr[ch], 1e-6))
                curves.append(mag_db)
        finally:
            model.Ccable = orig_Ccable

    elif p in ("tone_cap", "ctone", "cap", "tone_capacitance", "tone_cap_nf"):
        orig_Ctone = model.Ctone
        try:
            for v in values:
                c_farads = v * 1e-9 if v > 1e-6 else v
                model.Ctone = c_farads
                tr = compute_circuit_transfer_functions(model, freqs=f_arr, return_numpy=True)
                ch = min(ch_idx, len(tr) - 1)
                mag_db = 20.0 * np.log10(np.maximum(tr[ch], 1e-6))
                curves.append(mag_db)
        finally:
            model.Ctone = orig_Ctone

    elif p in ("bass_boost", "preamp_bass", "bass"):
        orig_has_buf = model.has_active_buffer
        orig_bands = copy.deepcopy(model.preamp_bands) if model.preamp_bands is not None else None
        orig_type = model.preamp_type
        try:
            from allomorph.config.preamps import PREAMPS

            base_bands: list[PreampBandConfig] = []
            if orig_bands is not None:
                base_bands = [b.model_copy() for b in orig_bands]
            elif model.preamp_type != "none" and model.preamp_type in PREAMPS:
                base_bands = [b.model_copy() for b in PREAMPS[model.preamp_type].bands]

            shelf_idx = None
            for i, b in enumerate(base_bands):
                if b.type == "low_shelf":
                    shelf_idx = i
                    break
            if shelf_idx is None:
                base_bands.append(PreampBandConfig(type="low_shelf", freq_hz=40.0, gain_db=0.0))
                shelf_idx = len(base_bands) - 1

            model.has_active_buffer = True
            for v in values:
                bands = [b.model_copy() for b in base_bands]
                bands[shelf_idx].gain_db = float(v)
                model.preamp_bands = bands
                tr = compute_circuit_transfer_functions(model, freqs=f_arr, return_numpy=True)
                ch = min(ch_idx, len(tr) - 1)
                mag_db = 20.0 * np.log10(np.maximum(tr[ch], 1e-6))
                curves.append(mag_db)
        finally:
            model.has_active_buffer = orig_has_buf
            model.preamp_bands = orig_bands
            model.preamp_type = orig_type

    elif p in ("treble_boost", "preamp_treble", "treble"):
        orig_has_buf = model.has_active_buffer
        orig_bands = copy.deepcopy(model.preamp_bands) if model.preamp_bands is not None else None
        orig_type = model.preamp_type
        try:
            from allomorph.config.preamps import PREAMPS

            base_bands = []
            if orig_bands is not None:
                base_bands = [b.model_copy() for b in orig_bands]
            elif model.preamp_type != "none" and model.preamp_type in PREAMPS:
                base_bands = [b.model_copy() for b in PREAMPS[model.preamp_type].bands]

            shelf_idx = None
            for i, b in enumerate(base_bands):
                if b.type == "high_shelf":
                    shelf_idx = i
                    break
            if shelf_idx is None:
                base_bands.append(PreampBandConfig(type="high_shelf", freq_hz=4000.0, gain_db=0.0))
                shelf_idx = len(base_bands) - 1

            model.has_active_buffer = True
            for v in values:
                bands = [b.model_copy() for b in base_bands]
                bands[shelf_idx].gain_db = float(v)
                model.preamp_bands = bands
                tr = compute_circuit_transfer_functions(model, freqs=f_arr, return_numpy=True)
                ch = min(ch_idx, len(tr) - 1)
                mag_db = 20.0 * np.log10(np.maximum(tr[ch], 1e-6))
                curves.append(mag_db)
        finally:
            model.has_active_buffer = orig_has_buf
            model.preamp_bands = orig_bands
            model.preamp_type = orig_type

    elif hasattr(model, param):
        orig_val = getattr(model, param)
        try:
            for v in values:
                setattr(model, param, v)
                tr = compute_circuit_transfer_functions(model, freqs=f_arr, return_numpy=True)
                ch = min(ch_idx, len(tr) - 1)
                mag_db = 20.0 * np.log10(np.maximum(tr[ch], 1e-6))
                curves.append(mag_db)
        finally:
            setattr(model, param, orig_val)
    else:
        raise ValueError(f"Unsupported sweep parameter: '{param}'")

    return ParametricSweepResult(
        param=param,
        values=values,
        freqs=f_arr,
        curves=curves,
        labels=labels,
        voice_id=voice_id,
    )
