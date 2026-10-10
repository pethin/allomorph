"""Deterministic unit tests for manifest generation and audio telemetry extraction."""

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from allomorph.dsp import write_wav_24bit
from allomorph.pipeline.manifest import (
    assemble_manifest_document,
    build_dict_manifest_entry,
    build_file_manifest_entry,
    extract_wav_metrics,
    read_existing_manifest,
    write_manifest,
)
from allomorph.version import ALLOMORPH_VERSION, DSP_GENERATION


@pytest.fixture
def sample_wav(tmp_path: Path) -> Path:
    """Creates a deterministic 24-bit 48kHz WAV file for audio metric extraction testing."""
    sr = 48000
    t = np.linspace(0, 0.1, int(sr * 0.1), endpoint=False)  # 100 ms
    audio = 0.5 * np.sin(2.0 * np.pi * 440.0 * t)
    wav_path = tmp_path / "test_sine.wav"
    write_wav_24bit(wav_path, audio.astype(np.float32), sample_rate=sr)
    return wav_path


def test_extract_wav_metrics_valid(sample_wav: Path):
    """Verify audio metric extraction accurately computes duration, sample rate, peak, and RMS."""
    metrics = extract_wav_metrics(sample_wav)

    assert metrics["sample_rate"] == 48000
    assert metrics["duration_s"] == 0.1
    # 0.5 amplitude sine peak is ~ -6.02 dBFS
    assert -6.5 <= metrics["peak_dbfs"] <= -5.5
    # Sine RMS is 0.5 / sqrt(2) ~ 0.3536, ~ -9.03 dBFS
    assert -9.5 <= metrics["rms_dbfs"] <= -8.5
    assert "true_peak_dbfs" in metrics
    assert "lufs" in metrics
    assert "dc_offset" in metrics


def test_extract_wav_metrics_invalid(tmp_path: Path):
    """Verify metric extraction gracefully returns an empty dict on non-existent or corrupt files."""
    missing = tmp_path / "missing.wav"
    assert extract_wav_metrics(missing) == {}

    corrupt = tmp_path / "corrupt.wav"
    corrupt.write_bytes(b"not a valid wav header")
    assert extract_wav_metrics(corrupt) == {}


def test_build_file_manifest_entry_wav(sample_wav: Path):
    """Verify build_file_manifest_entry includes size, SHA256, and extracted audio metrics."""
    entry = build_file_manifest_entry(
        file_path=sample_wav,
        base_dry_sha256="abc123sha",
        base_dry_file="base.wav",
        version_tag="v6.1.1",
        instrument_version=1,
        voicing_version=1,
        compute_audio_metrics=True,
    )

    assert entry["size_bytes"] == sample_wav.stat().st_size
    assert len(entry["sha256"]) == 64
    assert entry["base_dry_sha256"] == "abc123sha"
    assert entry["base_dry_file"] == "base.wav"
    assert entry["version"] == "v6.1.1"
    assert entry["instrument_version"] == 1
    assert entry["voicing_version"] == 1
    assert entry["sample_rate"] == 48000


def test_build_file_manifest_entry_non_audio(tmp_path: Path):
    """Verify build_file_manifest_entry for non-WAV files omits audio metrics."""
    txt_file = tmp_path / "info.txt"
    txt_file.write_text("Allomorph documentation text", encoding="utf-8")

    entry = build_file_manifest_entry(
        file_path=txt_file,
        version_tag="v6.1.1",
        compute_audio_metrics=True,
    )

    assert entry["size_bytes"] == txt_file.stat().st_size
    assert len(entry["sha256"]) == 64
    assert entry["version"] == "v6.1.1"
    assert "sample_rate" not in entry


def test_build_dict_manifest_entry(tmp_path: Path):
    """Verify build_dict_manifest_entry merges defaults into existing caller metadata."""
    txt_file = tmp_path / "model.nam"
    txt_file.write_text('{"name": "test_model"}', encoding="utf-8")

    existing_meta: dict[str, Any] = {
        "custom_param": 42,
        "tone_name": "Precision Warm",
    }
    entry = build_dict_manifest_entry(
        file_path=txt_file,
        meta=existing_meta,
        base_dry_sha256="dry_sha_123",
        version_tag="v6.1.1",
    )

    assert entry["custom_param"] == 42
    assert entry["tone_name"] == "Precision Warm"
    assert entry["size_bytes"] == txt_file.stat().st_size
    assert len(entry["sha256"]) == 64
    assert entry["base_dry_sha256"] == "dry_sha_123"
    assert entry["version"] == "v6.1.1"


def test_read_existing_manifest(tmp_path: Path):
    """Verify read_existing_manifest handles missing, invalid, and valid JSON files."""
    missing = tmp_path / "missing_manifest.json"
    assert read_existing_manifest(missing) == {}

    corrupt = tmp_path / "corrupt_manifest.json"
    corrupt.write_text("{not valid json", encoding="utf-8")
    assert read_existing_manifest(corrupt) == {}

    valid = tmp_path / "valid_manifest.json"
    valid.write_text('{"version": "v6.1.1", "files": {}}', encoding="utf-8")
    assert read_existing_manifest(valid) == {"version": "v6.1.1", "files": {}}


def test_assemble_manifest_document():
    """Verify assemble_manifest_document builds expected top-level schema."""
    files: dict[str, Any] = {
        "test.wav": {"size_bytes": 100, "sha256": "abc"},
    }
    doc = assemble_manifest_document(
        file_entries=files,
        stage="simulation",
        version_tag="v6.1.1",
        base_dry_sha256="sha_dry",
        base_dry_file="dry.wav",
        instrument_version=1,
        voicing_version=2,
        git_commit="abcdef0",
        dsp_generation=DSP_GENERATION,
        allomorph_version=ALLOMORPH_VERSION,
        updated_at="2026-10-10T00:00:00Z",
    )

    assert doc["version"] == "v6.1.1"
    assert doc["stage"] == "simulation"
    assert doc["dsp_generation"] == DSP_GENERATION
    assert doc["allomorph_version"] == ALLOMORPH_VERSION
    assert doc["git_commit"] == "abcdef0"
    assert doc["updated_at"] == "2026-10-10T00:00:00Z"
    assert doc["file_count"] == 1
    assert doc["base_dry_sha256"] == "sha_dry"
    assert doc["base_dry_file"] == "dry.wav"
    assert doc["instrument_version"] == 1
    assert doc["voicing_version"] == 2
    assert "test.wav" in doc["files"]


def test_write_manifest_end_to_end(tmp_path: Path, sample_wav: Path):
    """Verify write_manifest writes manifest.json to disk and merges subsequent entries."""
    out_dir = tmp_path / "stage_output"
    out_dir.mkdir(parents=True, exist_ok=True)

    dest_wav1 = out_dir / "stem1.wav"
    dest_wav1.write_bytes(sample_wav.read_bytes())

    # Write initial manifest
    m_path = write_manifest(
        output_dir=out_dir,
        stage="simulation",
        files=[dest_wav1],
        version_tag="v6.1.1",
        base_dry_sha256="dry_sha_root",
        base_dry_file="root_dry.wav",
    )

    assert m_path.exists()
    data1 = read_existing_manifest(m_path)
    assert data1["file_count"] == 1
    assert "stem1.wav" in data1["files"]
    assert data1["files"]["stem1.wav"]["base_dry_sha256"] == "dry_sha_root"

    # Add second stem and write again
    dest_wav2 = out_dir / "stem2.wav"
    dest_wav2.write_bytes(sample_wav.read_bytes())

    write_manifest(
        output_dir=out_dir,
        stage="simulation",
        files=[dest_wav2],
        version_tag="v6.1.1",
    )

    data2 = read_existing_manifest(m_path)
    assert data2["file_count"] == 2
    assert "stem1.wav" in data2["files"]
    assert "stem2.wav" in data2["files"]
