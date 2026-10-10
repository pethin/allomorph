"""Circuit stem diagnostics, causal onset evaluation, and telemetry auditing primitives.

Provides pure, testable mathematical functions for analyzing rendered audio stems:
causal onset / peak alignment, RMS/peak/crest-factor level metrics, frequency anchor
interpolation, and telemetry report aggregation.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, Field

CALIBRATION_PEAK_CEILING: float = 0.9900  # -0.087 dBFS

DEFAULT_FREQUENCY_ANCHORS: dict[str, float] = {
    "sub_bass_10hz_db": 10.0,
    "sub_bass_20hz_db": 20.0,
    "sub_bass_b0_31hz_db": 30.87,
    "sub_bass_e1_41hz_db": 41.2,
    "mid_500hz_db": 500.0,
    "mid_1khz_db": 1000.0,
    "upper_mid_2khz_db": 2000.0,
    "treble_5khz_db": 5000.0,
    "treble_10khz_db": 10000.0,
    "ultra_20khz_db": 20000.0,
}


class AudioAuditRecord(BaseModel):
    """Telemetry report for a single audio file."""

    file_path: str = Field(description="Relative or absolute path to the audio file")
    file_name: str = Field(description="Base filename")
    duration_s: float = Field(description="Audio duration in seconds")
    sample_rate: int = Field(description="Sample rate in Hz")
    peak_dbfs: float = Field(description="Max sample peak in dBFS")
    true_peak_linear: float = Field(description="4x oversampled true-peak linear amplitude")
    true_peak_dbfs: float = Field(description="4x oversampled true-peak in dBFS")
    rms_dbfs: float = Field(description="Root-mean-square loudness in dBFS")
    crest_factor_db: float = Field(description="True peak to RMS crest factor in dB")
    lufs: float | None = Field(
        default=None, description="ITU-R BS.1770-4 gated integrated loudness in LUFS"
    )
    dc_offset: float = Field(description="DC offset mean amplitude")
    has_clipping: bool = Field(description="True if true peak exceeds the 0.9900 ceiling")
    is_valid: bool = Field(description="True if audio passed all telemetry invariants")


class AudioAuditReport(BaseModel):
    """Aggregate telemetry report across a catalog of audio files."""

    total_files: int = Field(description="Total files audited")
    valid_files: int = Field(description="Files passing all telemetry invariants")
    clipped_files: int = Field(description="Files exceeding the true-peak ceiling")
    mean_lufs: float | None = Field(default=None, description="Average LUFS across valid files")
    min_lufs: float | None = Field(default=None, description="Minimum LUFS in catalog")
    max_lufs: float | None = Field(default=None, description="Maximum LUFS in catalog")
    mean_true_peak_dbfs: float = Field(description="Average true peak in dBFS")
    max_true_peak_dbfs: float = Field(description="Maximum true peak in dBFS")
    records: list[AudioAuditRecord] = Field(
        default_factory=list, description="Per-file audit records"
    )


def evaluate_causal_onset_and_peak(
    h_time: np.ndarray,
    gate_taps: int | None = 4096,
) -> tuple[int, int, bool]:
    """Evaluates the onset sample index, peak sample index, and causal zero-latency invariant.

    Invariant: Onset must begin at sample 0..4 (no artificial leading zeroes or unaligned dead latency),
    and maximum peak must reside within samples 0..80.
    """
    taps = gate_taps if gate_taps is not None and gate_taps > 0 else len(h_time)
    h_eval = np.abs(h_time[:taps])

    if len(h_eval) == 0:
        return 0, 0, False

    peak_sample = int(np.argmax(h_eval))
    peak_val = float(h_eval[peak_sample])

    # Onset threshold set at -40 dBc (1% of peak) to reliably detect acoustic onset
    thresh = 0.01 * max(peak_val, 1e-6)
    onset_candidates = np.where(h_eval >= thresh)[0]
    onset_sample = int(onset_candidates[0]) if len(onset_candidates) > 0 else peak_sample

    is_causal = (0 <= onset_sample <= 4) and (0 <= peak_sample <= 80)
    return onset_sample, peak_sample, is_causal


def compute_stem_level_metrics(audio: np.ndarray) -> dict[str, float]:
    """Computes RMS loudness, peak amplitude, and crest factor in dBFS."""
    if len(audio) == 0:
        return {
            "rms_dbfs": -120.0,
            "peak_dbfs": -120.0,
            "crest_factor_db": 0.0,
        }

    rms = float(np.sqrt(np.mean(audio**2)))
    rms_dbfs = 20.0 * math.log10(max(rms, 1e-9))
    peak = float(np.max(np.abs(audio)))
    peak_dbfs = 20.0 * math.log10(max(peak, 1e-9))
    crest_factor_db = peak_dbfs - rms_dbfs

    return {
        "rms_dbfs": round(rms_dbfs, 2),
        "peak_dbfs": round(peak_dbfs, 2),
        "crest_factor_db": round(crest_factor_db, 2),
    }


def extract_frequency_anchors(
    f_bins: np.ndarray,
    mags_db: np.ndarray,
    target_frequencies: Mapping[str, float] = DEFAULT_FREQUENCY_ANCHORS,
) -> dict[str, float]:
    """Interpolates dB magnitudes at target anchor frequencies via nearest FFT bin lookup."""
    anchors: dict[str, float] = {}
    for name, f_target in target_frequencies.items():
        idx = int(np.argmin(np.abs(f_bins - f_target)))
        anchors[name] = float(np.round(mags_db[idx], 2))
    return anchors


def is_manifest_entry_fresh(
    entry: Mapping[str, Any] | None,
    file_path: Path,
) -> bool:
    """Checks whether a cached manifest entry is valid and matches the physical file size."""
    if entry is None:
        return False
    if "true_peak_dbfs" not in entry or "lufs" not in entry:
        return False
    recorded_size = entry.get("size_bytes")
    if recorded_size is not None and file_path.exists():
        return file_path.stat().st_size == recorded_size
    return True


def aggregate_audit_report(records: Sequence[AudioAuditRecord]) -> AudioAuditReport:
    """Aggregates a sequence of AudioAuditRecord instances into an AudioAuditReport."""
    total = len(records)
    if total == 0:
        return AudioAuditReport(
            total_files=0,
            valid_files=0,
            clipped_files=0,
            mean_lufs=None,
            min_lufs=None,
            max_lufs=None,
            mean_true_peak_dbfs=-120.0,
            max_true_peak_dbfs=-120.0,
            records=[],
        )

    valid_count = sum(1 for r in records if r.is_valid)
    clipped_count = sum(1 for r in records if r.has_clipping)

    valid_lufs = [r.lufs for r in records if r.lufs is not None]
    mean_lufs = float(np.mean(valid_lufs)) if valid_lufs else None
    min_lufs = min(valid_lufs) if valid_lufs else None
    max_lufs = max(valid_lufs) if valid_lufs else None

    tp_dbs = [r.true_peak_dbfs for r in records]
    mean_tp_db = float(np.mean(tp_dbs)) if tp_dbs else -120.0
    max_tp_db = max(tp_dbs) if tp_dbs else -120.0

    return AudioAuditReport(
        total_files=total,
        valid_files=valid_count,
        clipped_files=clipped_count,
        mean_lufs=round(mean_lufs, 2) if mean_lufs is not None else None,
        min_lufs=round(min_lufs, 2) if min_lufs is not None else None,
        max_lufs=round(max_lufs, 2) if max_lufs is not None else None,
        mean_true_peak_dbfs=round(mean_tp_db, 2),
        max_true_peak_dbfs=round(max_tp_db, 2),
        records=list(records),
    )
