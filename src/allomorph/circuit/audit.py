"""
Allomorph - Audio QA Telemetry & Loudness Audit Engine.
Provides comprehensive BS.1770-4 gated loudness, 4x oversampled true-peak detection,
crest factor, DC offset, and clipping analysis across the wet audio digital twin catalog.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from allomorph.circuit.diagnostics import (
    CALIBRATION_PEAK_CEILING,
    AudioAuditRecord,
    AudioAuditReport,
    aggregate_audit_report,
    is_manifest_entry_fresh,
)
from allomorph.config.scales import REPO_ROOT
from allomorph.dsp import (
    compute_lufs,
    compute_true_peak,
    compute_true_peak_dbfs,
    read_wav,
)

WET_AUDIO_DIR = REPO_ROOT / "audio" / "wet"


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


def _record_from_manifest(path: Path, entry: dict[str, Any]) -> AudioAuditRecord:
    """Constructs an AudioAuditRecord from authoritative sidecar manifest metadata."""
    sr = int(entry.get("sample_rate", 48000))
    size_bytes = int(entry.get("size_bytes", path.stat().st_size if path.exists() else 0))

    duration_s = float(entry.get("duration_s", 0.0))
    if duration_s <= 0.0 and sr > 0 and size_bytes > 44:
        # 24-bit PCM mono (3 bytes/sample) fallback
        duration_s = max(0.0, float(size_bytes - 44) / float(sr * 3))

    peak_dbfs = float(entry.get("peak_dbfs", -120.0))
    peak_linear = 10.0 ** (peak_dbfs / 20.0)

    tp_dbfs = float(entry.get("true_peak_dbfs", peak_dbfs))
    tp_linear = 10.0 ** (tp_dbfs / 20.0)

    rms_dbfs = float(entry.get("rms_dbfs", -120.0))
    crest_factor_db = tp_dbfs - rms_dbfs

    lufs_val = entry.get("lufs")
    lufs_reported = (
        round(float(lufs_val), 2)
        if lufs_val is not None and not (math.isinf(float(lufs_val)) or math.isnan(float(lufs_val)))
        else None
    )

    dc_offset = float(entry.get("dc_offset", 0.0))

    has_clipping = tp_linear > (CALIBRATION_PEAK_CEILING + 0.0005)
    is_valid = (
        not has_clipping
        and not math.isnan(peak_linear)
        and not math.isinf(peak_linear)
        and abs(dc_offset) < 0.01
    )

    return AudioAuditRecord(
        file_path=str(path),
        file_name=path.name,
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


def audit_wet_audio_catalog(
    audio_dir: Path | str | None = None,
    force: bool = False,
) -> AudioAuditReport:
    """Audits wet audio stems in the specified directory or audio/wet/.

    By default, uses authoritative telemetry recorded in sidecar manifest.json
    files when available and verified against file existence and size.
    Falls back to full audio decoding and signal analysis if manifest entries
    are missing, size-mismatched, or if force=True.
    """
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
    if not wav_files:
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

    manifest_entries: dict[Path, dict[str, Any]] = {}
    if not force:
        manifest_files = (
            [target_dir / "manifest.json"]
            if (target_dir / "manifest.json").exists()
            else sorted(target_dir.rglob("manifest.json"))
        )
        for mf in manifest_files:
            try:
                with mf.open("r", encoding="utf-8") as f:
                    mdata = json.load(f)
                files_dict = mdata.get("files", {})
                if isinstance(files_dict, dict):
                    for fname, fentry in files_dict.items():
                        if isinstance(fentry, dict):
                            p = (mf.parent / fname).resolve()
                            manifest_entries[p] = fentry
            except (json.JSONDecodeError, OSError):
                continue

    records: list[AudioAuditRecord] = []
    for wf in wav_files:
        wf_res = wf.resolve()
        entry = manifest_entries.get(wf_res)
        if is_manifest_entry_fresh(entry, wf_res):
            assert entry is not None
            records.append(_record_from_manifest(wf_res, entry))
        else:
            records.append(audit_audio_file(wf))

    return aggregate_audit_report(records)
