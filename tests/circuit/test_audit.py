"""Tests for Allomorph audio QA telemetry and loudness audit engine."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from allomorph.circuit.audit import (
    AudioAuditRecord,
    _record_from_manifest,
    audit_audio_file,
    audit_wet_audio_catalog,
)
from allomorph.dsp import FS, write_wav_24bit
from allomorph.version import write_manifest


def test_audit_audio_file(tmp_path: Path) -> None:
    """Verifies that audit_audio_file computes accurate acoustic telemetry directly from audio bytes."""
    t_arr = np.linspace(0.0, 1.0, FS, endpoint=False)
    sine = (0.5 * np.sin(2.0 * np.pi * 100.0 * t_arr)).astype(np.float32)
    wav_path = tmp_path / "test_sine.wav"
    write_wav_24bit(str(wav_path), sine, FS)

    record = audit_audio_file(wav_path)
    assert isinstance(record, AudioAuditRecord)
    assert record.file_name == "test_sine.wav"
    assert record.sample_rate == FS
    assert record.duration_s == 1.0
    assert abs(record.peak_dbfs - (-6.02)) < 0.20
    assert abs(record.true_peak_dbfs - (-6.02)) < 0.20
    assert record.has_clipping is False
    assert record.is_valid is True


def test_record_from_manifest(tmp_path: Path) -> None:
    """Verifies that _record_from_manifest constructs an AudioAuditRecord from manifest metadata."""
    wav_path = tmp_path / "test.wav"
    wav_path.write_bytes(b"\x00" * 100)

    entry = {
        "size_bytes": 100,
        "sample_rate": 48000,
        "peak_dbfs": -5.33,
        "rms_dbfs": -20.20,
        "true_peak_dbfs": -5.33,
        "lufs": -20.76,
        "duration_s": 0.5,
        "dc_offset": 0.00001,
    }
    rec = _record_from_manifest(wav_path, entry)
    assert rec.file_name == "test.wav"
    assert rec.sample_rate == 48000
    assert rec.peak_dbfs == -5.33
    assert rec.true_peak_dbfs == -5.33
    assert rec.rms_dbfs == -20.20
    assert rec.crest_factor_db == 14.87
    assert rec.lufs == -20.76
    assert rec.has_clipping is False
    assert rec.is_valid is True


def test_audit_wet_audio_catalog_manifest_and_fallback(tmp_path: Path) -> None:
    """Verifies manifest-driven catalog audit, fallback on missing/mismatched manifest, and force mode."""
    t_arr = np.linspace(0.0, 1.0, FS, endpoint=False)
    sine = (0.5 * np.sin(2.0 * np.pi * 100.0 * t_arr)).astype(np.float32)

    # 1. File A with valid manifest
    dir_a = tmp_path / "inst_a"
    dir_a.mkdir()
    wav_a = dir_a / "stem_a.wav"
    write_wav_24bit(str(wav_a), sine, FS)
    write_manifest(output_dir=dir_a, stage="test", files=[wav_a])

    # 2. File B without manifest (fallback to decoding)
    dir_b = tmp_path / "inst_b"
    dir_b.mkdir()
    wav_b = dir_b / "stem_b.wav"
    write_wav_24bit(str(wav_b), sine, FS)

    # Audit catalog
    report = audit_wet_audio_catalog(audio_dir=tmp_path, force=False)
    assert report.total_files == 2
    assert report.valid_files == 2
    assert report.clipped_files == 0
    assert report.mean_lufs is not None
    assert len(report.records) == 2

    # Verify force mode runs physical decoding on all files
    report_forced = audit_wet_audio_catalog(audio_dir=tmp_path, force=True)
    assert report_forced.total_files == 2
    assert report_forced.valid_files == 2


def test_audit_wet_audio_catalog_empty(tmp_path: Path) -> None:
    """Verifies audit_wet_audio_catalog returns an empty report when no files are present."""
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    report = audit_wet_audio_catalog(audio_dir=empty_dir)
    assert report.total_files == 0
    assert report.valid_files == 0
    assert report.records == []
