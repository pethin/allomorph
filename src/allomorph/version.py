"""
Allomorph - Engine Versioning, Provenance & Sidecar Manifest System.

Defines the tri-part semantic versioning specification v[dsp].[inst].[voice],
Git commit provenance tracking, and authoritative sidecar manifest.json generation.
"""

from __future__ import annotations

import contextlib
import functools
import hashlib
import json
import subprocess
import wave
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ALLOMORPH_VERSION: str = "0.3.0"
DSP_GENERATION: int = 3
DEFAULT_INST_VERSION: int = 1
DEFAULT_VOICE_VERSION: int = 1


@functools.lru_cache(maxsize=1)
def get_git_commit() -> str | None:
    """Extracts short Git commit hash for provenance auditing."""
    try:
        res = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=2.0,
            check=False,
        )
        if res.returncode == 0:
            commit = res.stdout.strip()
            if commit:
                return commit
    except subprocess.SubprocessError, OSError:
        pass
    return None


def resolve_tri_part_version(
    dsp: int = DSP_GENERATION,
    inst_version: int | None = DEFAULT_INST_VERSION,
    voice_version: int | None = DEFAULT_VOICE_VERSION,
) -> str:
    """
    Constructs the tri-part semantic version token v[dsp].[inst].[voice].
    If voice_version is None, formats as two-part: v[dsp].[inst].
    If both inst_version and voice_version are None, formats as single-part: v[dsp].
    Example: resolve_tri_part_version(2, 1, 1) -> 'v2.1.1'
    """
    if inst_version is None and voice_version is None:
        return f"v{dsp}"
    if voice_version is None:
        return f"v{dsp}.{inst_version}"
    return f"v{dsp}.{inst_version or 1}.{voice_version}"


def get_version_info(
    dsp: int = DSP_GENERATION,
    inst_version: int = DEFAULT_INST_VERSION,
    voice_version: int = DEFAULT_VOICE_VERSION,
) -> dict[str, Any]:
    """Returns comprehensive version and provenance metadata dictionary."""
    return {
        "version": resolve_tri_part_version(dsp, inst_version, voice_version),
        "dsp_version": dsp,
        "instrument_version": inst_version,
        "voice_version": voice_version,
        "allomorph_version": ALLOMORPH_VERSION,
        "git_commit": get_git_commit(),
        "generated_at": datetime.now(UTC).isoformat(),
    }


@functools.lru_cache(maxsize=32)
def _compute_file_sha256_cached(resolved_str: str, mtime_ns: int, size_bytes: int) -> str:
    p = Path(resolved_str)
    h = hashlib.sha256()
    with p.open("rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def compute_file_sha256(filepath: Path | str) -> str:
    """Computes SHA256 hexadecimal digest for a file with mtime/size cache validation."""
    p = Path(filepath).resolve()
    if not p.exists():
        raise FileNotFoundError(f"File not found: {p}")
    stat = p.stat()
    return _compute_file_sha256_cached(str(p), stat.st_mtime_ns, stat.st_size)


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

    existing_manifest: dict[str, Any] = {}
    if manifest_path.exists():
        try:
            with manifest_path.open("r", encoding="utf-8") as f:
                existing_manifest = json.load(f)
        except json.JSONDecodeError, OSError:
            existing_manifest = {}

    file_entries: dict[str, Any] = existing_manifest.get("files", {})

    eff_base_dry_sha = base_dry_sha256 or existing_manifest.get("base_dry_sha256")
    eff_base_dry_file = base_dry_file or existing_manifest.get("base_dry_file")

    if isinstance(files, dict):
        for fname, meta in files.items():
            f_path = out_p / fname if not Path(fname).is_absolute() else Path(fname)
            entry: dict[str, Any] = dict(meta)
            if f_path.exists():
                entry.setdefault("size_bytes", f_path.stat().st_size)
                entry.setdefault("sha256", compute_file_sha256(f_path))
            if eff_base_dry_sha is not None:
                entry.setdefault("base_dry_sha256", eff_base_dry_sha)
            if eff_base_dry_file is not None:
                entry.setdefault("base_dry_file", eff_base_dry_file)
            if tag is not None:
                entry.setdefault("version", tag)
            if instrument_version is not None:
                entry.setdefault("instrument_version", instrument_version)
            if voicing_version is not None:
                entry.setdefault("voicing_version", voicing_version)
            file_entries[f_path.name] = entry
    else:
        for f_item in files:
            f_path = out_p / f_item if not Path(f_item).is_absolute() else Path(f_item)
            if f_path.exists() and f_path.name != "manifest.json":
                entry = {
                    "size_bytes": f_path.stat().st_size,
                    "sha256": compute_file_sha256(f_path),
                }
                if eff_base_dry_sha is not None:
                    entry["base_dry_sha256"] = eff_base_dry_sha
                if eff_base_dry_file is not None:
                    entry["base_dry_file"] = eff_base_dry_file
                if tag is not None:
                    entry["version"] = tag
                if instrument_version is not None:
                    entry["instrument_version"] = instrument_version
                if voicing_version is not None:
                    entry["voicing_version"] = voicing_version
                if f_path.suffix.lower() == ".wav":
                    with contextlib.suppress(
                        wave.Error, OSError, ValueError, RuntimeError, EOFError
                    ):
                        import numpy as np

                        from allomorph.dsp import read_wav

                        audio, sr = read_wav(f_path)
                        peak = float(np.max(np.abs(audio)))
                        rms = float(np.sqrt(np.mean(audio**2)))
                        entry["sample_rate"] = sr
                        entry["peak_dbfs"] = round(20.0 * np.log10(max(peak, 1e-9)), 2)
                        entry["rms_dbfs"] = round(20.0 * np.log10(max(rms, 1e-9)), 2)
                file_entries[f_path.name] = entry

    manifest_data: dict[str, Any] = {
        "version": tag,
        "stage": stage,
        "allomorph_version": ALLOMORPH_VERSION,
        "dsp_generation": DSP_GENERATION,
        "git_commit": get_git_commit(),
        "updated_at": datetime.now(UTC).isoformat(),
        "file_count": len(file_entries),
    }
    if eff_base_dry_sha is not None:
        manifest_data["base_dry_sha256"] = eff_base_dry_sha
    if eff_base_dry_file is not None:
        manifest_data["base_dry_file"] = eff_base_dry_file
    if instrument_version is not None:
        manifest_data["instrument_version"] = instrument_version
    if voicing_version is not None:
        manifest_data["voicing_version"] = voicing_version

    manifest_data["files"] = dict(sorted(file_entries.items()))

    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2)

    return manifest_path


def is_wet_stem_valid(
    stem_path: Path | str,
    base_dry_path: Path | str | None = None,
    expected_version: str | None = None,
) -> bool:
    """Validates that a compiled wet audio stem exists and its manifest matches the expected base dry SHA256.

    Returns False if:
    - The stem file does not exist.
    - The sidecar manifest.json does not exist or cannot be parsed.
    - The stem is not registered in the manifest.
    - The manifest's recorded base_dry_sha256 does not match the expected base dry audio's SHA256.
    - An expected_version is provided and neither the stem entry nor the manifest version matches it.
    """
    stem_p = Path(stem_path)
    if not stem_p.exists():
        return False

    manifest_p = stem_p.parent / "manifest.json"
    if not manifest_p.exists():
        return False

    try:
        with manifest_p.open("r", encoding="utf-8") as f:
            manifest = json.load(f)
    except json.JSONDecodeError, OSError:
        return False

    files = manifest.get("files", {})
    if stem_p.name not in files:
        return False

    file_entry = files[stem_p.name]
    recorded_sha = file_entry.get("base_dry_sha256") or manifest.get("base_dry_sha256")
    if not recorded_sha:
        return False

    if expected_version is not None:
        recorded_ver = file_entry.get("version") or manifest.get("version")
        if recorded_ver != expected_version:
            return False

    # Resolve expected base dry file
    if base_dry_path is not None:
        base_p = Path(base_dry_path)
    else:
        from allomorph.naming import get_optimal_dry_path

        base_p = get_optimal_dry_path()

    if not base_p.exists():
        return False

    try:
        expected_sha = compute_file_sha256(base_p)
    except FileNotFoundError:
        return False

    return recorded_sha == expected_sha
