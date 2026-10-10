"""Allomorph - Engine Versioning, Provenance & Sidecar Manifest System.

Defines the tri-part semantic versioning specification v[dsp].[inst].[voice],
Git commit provenance tracking, and authoritative sidecar manifest.json generation.
"""

from __future__ import annotations

import functools
import hashlib
import json
import subprocess
import threading
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ALLOMORPH_VERSION: str = "0.5.0"
DSP_GENERATION: int = 6
DEFAULT_INST_VERSION: int = 1
DEFAULT_VOICE_VERSION: int = 1

_MANIFEST_LOCK = threading.Lock()


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
    except (subprocess.SubprocessError, OSError):
        pass
    return None


def resolve_tri_part_version(
    dsp: int = DSP_GENERATION,
    inst_version: int | None = DEFAULT_INST_VERSION,
    voice_version: int | None = DEFAULT_VOICE_VERSION,
) -> str:
    """Constructs the tri-part semantic version token v[dsp].[inst].[voice].

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

    Delegates to allomorph.pipeline.manifest.write_manifest.
    """
    from allomorph.pipeline.manifest import write_manifest as _write_manifest

    return _write_manifest(
        output_dir=output_dir,
        stage=stage,
        files=files,
        version_tag=version_tag,
        base_dry_sha256=base_dry_sha256,
        base_dry_file=base_dry_file,
        instrument_version=instrument_version,
        voicing_version=voicing_version,
    )


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
    except (json.JSONDecodeError, OSError):
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
        from allomorph.naming import get_default_input_path

        base_p = get_default_input_path()

    if not base_p.exists():
        return False

    try:
        expected_sha = compute_file_sha256(base_p)
    except FileNotFoundError:
        return False

    return recorded_sha == expected_sha


__all__ = [
    "ALLOMORPH_VERSION",
    "DEFAULT_INST_VERSION",
    "DEFAULT_VOICE_VERSION",
    "DSP_GENERATION",
    "_MANIFEST_LOCK",
    "compute_file_sha256",
    "get_git_commit",
    "get_version_info",
    "is_wet_stem_valid",
    "resolve_tri_part_version",
    "write_manifest",
]
