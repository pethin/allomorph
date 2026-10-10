"""Manifest generation, audio telemetry extraction, and provenance auditing.

Provides pure, testable helpers for extracting WAV telemetry, building file entries,
and assembling sidecar manifest.json files.
"""

from __future__ import annotations

import contextlib
import json
import math
import threading
import wave
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from allomorph.dsp import (
    compute_lufs,
    compute_true_peak_dbfs,
    read_wav,
)
from allomorph.version import (
    ALLOMORPH_VERSION,
    DSP_GENERATION,
    compute_file_sha256,
    get_git_commit,
    resolve_tri_part_version,
)

_MANIFEST_LOCK = threading.Lock()


def extract_wav_metrics(wav_path: Path | str) -> dict[str, Any]:
    """Extracts duration, sample rate, peak, RMS, True Peak, LUFS, and DC offset for a WAV file.

    Returns an empty dict if the file cannot be read or is not a valid audio file.
    """
    path = Path(wav_path)
    metrics: dict[str, Any] = {}
    with contextlib.suppress(wave.Error, OSError, ValueError, RuntimeError, EOFError):
        audio, sr = read_wav(path)
        peak = float(np.max(np.abs(audio)))
        rms = float(np.sqrt(np.mean(audio**2)))
        mono = audio[0] if audio.ndim > 1 else audio
        metrics["duration_s"] = round(float(len(mono)) / float(sr), 3) if sr > 0 else 0.0
        metrics["sample_rate"] = sr
        metrics["peak_dbfs"] = round(20.0 * np.log10(max(peak, 1e-9)), 2)
        metrics["rms_dbfs"] = round(20.0 * np.log10(max(rms, 1e-9)), 2)
        metrics["true_peak_dbfs"] = round(compute_true_peak_dbfs(audio), 2)
        lufs_val = compute_lufs(audio, sample_rate=sr)
        metrics["lufs"] = (
            round(lufs_val, 2)
            if not (math.isinf(lufs_val) or math.isnan(lufs_val))
            else None
        )
        metrics["dc_offset"] = round(float(np.mean(mono)), 6)
    return metrics


def build_file_manifest_entry(
    file_path: Path | str,
    base_dry_sha256: str | None = None,
    base_dry_file: str | None = None,
    version_tag: str | None = None,
    instrument_version: int | None = None,
    voicing_version: int | None = None,
    compute_audio_metrics: bool = True,
) -> dict[str, Any]:
    """Builds an authoritative metadata entry for a single physical file on disk."""
    path = Path(file_path)
    entry: dict[str, Any] = {
        "size_bytes": path.stat().st_size,
        "sha256": compute_file_sha256(path),
    }
    if base_dry_sha256 is not None:
        entry["base_dry_sha256"] = base_dry_sha256
    if base_dry_file is not None:
        entry["base_dry_file"] = base_dry_file
    if version_tag is not None:
        entry["version"] = version_tag
    if instrument_version is not None:
        entry["instrument_version"] = instrument_version
    if voicing_version is not None:
        entry["voicing_version"] = voicing_version

    if compute_audio_metrics and path.suffix.lower() == ".wav":
        audio_metrics = extract_wav_metrics(path)
        entry.update(audio_metrics)

    return entry


def build_dict_manifest_entry(
    file_path: Path | str,
    meta: Mapping[str, Any],
    base_dry_sha256: str | None = None,
    base_dry_file: str | None = None,
    version_tag: str | None = None,
    instrument_version: int | None = None,
    voicing_version: int | None = None,
) -> dict[str, Any]:
    """Merges provenance and file size defaults into caller-provided metadata dictionary."""
    path = Path(file_path)
    entry: dict[str, Any] = dict(meta)
    if path.exists():
        entry.setdefault("size_bytes", path.stat().st_size)
        entry.setdefault("sha256", compute_file_sha256(path))
    if base_dry_sha256 is not None:
        entry.setdefault("base_dry_sha256", base_dry_sha256)
    if base_dry_file is not None:
        entry.setdefault("base_dry_file", base_dry_file)
    if version_tag is not None:
        entry.setdefault("version", version_tag)
    if instrument_version is not None:
        entry.setdefault("instrument_version", instrument_version)
    if voicing_version is not None:
        entry.setdefault("voicing_version", voicing_version)
    return entry


def read_existing_manifest(manifest_path: Path | str) -> dict[str, Any]:
    """Safely reads an existing manifest.json file, returning an empty dict on error."""
    path = Path(manifest_path)
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
            return data if isinstance(data, dict) else {}
    except (json.JSONDecodeError, OSError):
        return {}


def assemble_manifest_document(
    file_entries: Mapping[str, Any],
    stage: str,
    version_tag: str | None = None,
    base_dry_sha256: str | None = None,
    base_dry_file: str | None = None,
    instrument_version: int | None = None,
    voicing_version: int | None = None,
    git_commit: str | None = None,
    dsp_generation: int = DSP_GENERATION,
    allomorph_version: str = ALLOMORPH_VERSION,
    updated_at: str | None = None,
) -> dict[str, Any]:
    """Assembles the complete manifest document dictionary."""
    manifest_data: dict[str, Any] = {
        "version": version_tag,
        "stage": stage,
        "allomorph_version": allomorph_version,
        "dsp_generation": dsp_generation,
        "git_commit": git_commit if git_commit is not None else get_git_commit(),
        "updated_at": updated_at if updated_at is not None else datetime.now(UTC).isoformat(),
        "file_count": len(file_entries),
    }
    if base_dry_sha256 is not None:
        manifest_data["base_dry_sha256"] = base_dry_sha256
    if base_dry_file is not None:
        manifest_data["base_dry_file"] = base_dry_file
    if instrument_version is not None:
        manifest_data["instrument_version"] = instrument_version
    if voicing_version is not None:
        manifest_data["voicing_version"] = voicing_version

    manifest_data["files"] = dict(sorted(file_entries.items()))
    return manifest_data


def write_manifest(
    output_dir: Path | str,
    stage: str,
    files: dict[str, dict[str, Any]] | Sequence[Path | str],
    version_tag: str | None = None,
    base_dry_sha256: str | None = None,
    base_dry_file: str | None = None,
    instrument_version: int | None = None,
    voicing_version: int | None = None,
) -> Path:
    """Emits an authoritative manifest.json sidecar file in the output directory.

    Records file SHA256 digests, size in bytes, audio metrics (if applicable),
    base dry excitation SHA256 provenance, engine metadata, and versioning provenance.
    """
    out_p = Path(output_dir)
    out_p.mkdir(parents=True, exist_ok=True)
    manifest_path = out_p / "manifest.json"

    tag = version_tag or resolve_tri_part_version()

    with _MANIFEST_LOCK:
        existing_manifest = read_existing_manifest(manifest_path)
        file_entries: dict[str, Any] = dict(existing_manifest.get("files", {}))

        eff_base_dry_sha = base_dry_sha256 or existing_manifest.get("base_dry_sha256")
        eff_base_dry_file = base_dry_file or existing_manifest.get("base_dry_file")

        if isinstance(files, dict):
            for fname, meta in files.items():
                f_path = out_p / fname if not Path(fname).is_absolute() else Path(fname)
                entry = build_dict_manifest_entry(
                    file_path=f_path,
                    meta=meta,
                    base_dry_sha256=eff_base_dry_sha,
                    base_dry_file=eff_base_dry_file,
                    version_tag=tag,
                    instrument_version=instrument_version,
                    voicing_version=voicing_version,
                )
                file_entries[f_path.name] = entry
        else:
            for f_item in files:
                f_path = out_p / f_item if not Path(f_item).is_absolute() else Path(f_item)
                if f_path.exists() and f_path.name != "manifest.json":
                    entry = build_file_manifest_entry(
                        file_path=f_path,
                        base_dry_sha256=eff_base_dry_sha,
                        base_dry_file=eff_base_dry_file,
                        version_tag=tag,
                        instrument_version=instrument_version,
                        voicing_version=voicing_version,
                        compute_audio_metrics=True,
                    )
                    file_entries[f_path.name] = entry

        manifest_data = assemble_manifest_document(
            file_entries=file_entries,
            stage=stage,
            version_tag=tag,
            base_dry_sha256=eff_base_dry_sha,
            base_dry_file=eff_base_dry_file,
            instrument_version=instrument_version,
            voicing_version=voicing_version,
        )

        with manifest_path.open("w", encoding="utf-8") as f:
            json.dump(manifest_data, f, indent=2)

    return manifest_path
