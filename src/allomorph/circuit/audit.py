"""
Allomorph - Audio QA Telemetry & Loudness Audit Engine.
Provides comprehensive BS.1770-4 gated loudness, 4x oversampled true-peak detection,
crest factor, DC offset, and clipping analysis across the wet audio digital twin catalog.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from pydantic import BaseModel, Field

from allomorph.config.scales import REPO_ROOT
from allomorph.dsp import (
    compute_lufs,
    compute_true_peak,
    compute_true_peak_dbfs,
    read_wav,
)

WET_AUDIO_DIR = REPO_ROOT / "audio" / "wet"
CALIBRATION_PEAK_CEILING = 0.9900  # -0.087 dBFS


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


def audit_audio_file(path: Path | str) -> AudioAuditRecord:
    """Computes comprehensive acoustic and psychoacoustic telemetry on a single audio file."""
    p = Path(path).resolve()
    if not p.exists():
        raise FileNotFoundError(f"Audio file not found: {p}")

    audio, sr = read_wav(p, dtype=np.float64)
    mono = audio[0] if audio.ndim > 1 else audio
    n = len(mono)
    duration_s = float(n) / float(sr) if sr > 0 else 0.0

    peak_linear = float(np.max(np.abs(mono))) if n > 0 else 0.0
    peak_dbfs = 20.0 * math.log10(max(peak_linear, 1e-9))

    tp_linear = compute_true_peak(mono)
    tp_dbfs = compute_true_peak_dbfs(mono)

    rms_linear = float(np.sqrt(np.mean(mono**2))) if n > 0 else 0.0
    rms_dbfs = 20.0 * math.log10(max(rms_linear, 1e-9))
    crest_factor_db = tp_dbfs - rms_dbfs

    lufs_val = compute_lufs(mono, sample_rate=sr)
    lufs_reported = (
        round(lufs_val, 2) if not (math.isinf(lufs_val) or math.isnan(lufs_val)) else None
    )

    dc_offset = float(np.mean(mono)) if n > 0 else 0.0

    # Clipping is strictly defined as true peak exceeding 0.9900 (allowing 0.0005 float margin)
    has_clipping = tp_linear > (CALIBRATION_PEAK_CEILING + 0.0005)
    is_valid = (
        not has_clipping
        and not math.isnan(peak_linear)
        and not math.isinf(peak_linear)
        and abs(dc_offset) < 0.01
    )

    return AudioAuditRecord(
        file_path=str(p),
        file_name=p.name,
        duration_s=round(duration_s, 3),
        sample_rate=sr,
        peak_dbfs=round(peak_dbfs, 2),
        true_peak_linear=round(tp_linear, 4),
        true_peak_dbfs=round(tp_dbfs, 2),
        rms_dbfs=round(rms_dbfs, 2),
        crest_factor_db=round(crest_factor_db, 2),
        lufs=lufs_reported,
        dc_offset=round(dc_offset, 6),
        has_clipping=has_clipping,
        is_valid=is_valid,
    )


def audit_wet_audio_catalog(audio_dir: Path | str | None = None) -> AudioAuditReport:
    """Audits all wet audio stems in the specified directory or audio/wet/."""
    target_dir = Path(audio_dir).resolve() if audio_dir else WET_AUDIO_DIR
    if not target_dir.exists():
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

    wav_files = sorted(target_dir.rglob("*.wav"))
    records: list[AudioAuditRecord] = []
    for wf in wav_files:
        records.append(audit_audio_file(wf))

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
        records=records,
    )
