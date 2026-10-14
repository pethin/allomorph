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
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ALLOMORPH_VERSION: str = "0.5.0"
DSP_GENERATION: int = 7
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
    except subprocess.SubprocessError, OSError:
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
    Example: resolve_tri_part_version(6, 1, 1) -> 'v6.1.1'
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


_DRY_TOKEN_CACHE: dict[str, tuple[int, str]] = {}


def compute_excitation_recipe_token(
    dsp_generation: int = DSP_GENERATION,
    sample_rate: int = 48000,
    duration_sec: float = 240.0,
    recipe_tag: str | None = None,
) -> str:
    """Computes a deterministic 256-bit recipe token for synthetic dry excitation.

    Returns: 'alo_dryrec_<64 hex chars>'.
    """
    tag = recipe_tag or f"synthetic_optimal_bass_dry_v{dsp_generation}"
    payload = {
        "dsp_generation": dsp_generation,
        "sample_rate": sample_rate,
        "duration_sec": duration_sec,
        "recipe_tag": tag,
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return f"alo_dryrec_{hashlib.sha256(raw).hexdigest()}"


def compute_excitation_digest_token(
    input_path: Path | str | None = None,
    recipe_tag: str | None = None,
    force: bool = False,
) -> str:
    """Retrieves or computes the dry excitation digest token (alo_dry_<64 hex>).

    16-bit PCM quantized audio sample frames + telemetry digest.
    """
    return get_dry_excitation_token(input_path=input_path, recipe_tag=recipe_tag, force=force)


def compute_pcm_stem_digest(
    audio: Any,
    sample_rate: int = 48000,
) -> str:
    """Computes a robust 256-bit audio digest from 16-bit PCM quantized frames and audio telemetry.

    Resistant to sub-LSB float roundoff across architectures.
    Returns: 'alo_stem_<64 hex chars>'.
    """
    import math

    import numpy as np

    from allomorph.dsp.io import compute_true_peak_dbfs
    from allomorph.dsp.loudness import compute_lufs

    audio_arr = np.asarray(audio, dtype=np.float64)
    pcm16 = np.clip(np.round(audio_arr * 32767.0), -32768, 32767).astype(np.int16)

    mono = audio_arr[0] if audio_arr.ndim > 1 else audio_arr
    dur = round(float(len(mono)) / float(sample_rate), 3) if sample_rate > 0 else 0.0
    lufs_val = compute_lufs(audio_arr, sample_rate=sample_rate)
    lufs_rounded = (
        round(lufs_val, 2) if not (math.isinf(lufs_val) or math.isnan(lufs_val)) else -100.0
    )
    tp_rounded = round(compute_true_peak_dbfs(audio_arr, sample_rate=sample_rate), 2)
    rms = float(np.sqrt(np.mean(audio_arr**2)))
    rms_rounded = round(20.0 * float(np.log10(max(rms, 1e-9))), 2)

    header_str = f"{sample_rate}:{dur}:{lufs_rounded}:{tp_rounded}:{rms_rounded}:"

    hasher = hashlib.sha256()
    hasher.update(header_str.encode("utf-8"))
    hasher.update(pcm16.tobytes())
    return f"alo_stem_{hasher.hexdigest()}"


def compute_stem_digest_token(
    stem_audio: Path | str | Any,
    sample_rate: int = 48000,
) -> str:
    """Computes or retrieves the stem digest token (alo_stem_<64 hex>) for a wet audio stem."""
    try:
        import numpy as np

        if isinstance(stem_audio, np.ndarray):
            return compute_pcm_stem_digest(stem_audio, sample_rate=sample_rate)
    except ImportError:
        pass

    if hasattr(stem_audio, "ndim") and hasattr(stem_audio, "shape"):
        return compute_pcm_stem_digest(stem_audio, sample_rate=sample_rate)

    p = Path(stem_audio)
    sidecar_p = p.parent / "manifest.json"
    if sidecar_p.exists():
        try:
            with sidecar_p.open("r", encoding="utf-8") as sf:
                sdata = json.load(sf)
            file_entry = sdata.get("files", {}).get(p.name, {})
            tok = file_entry.get("digest_token")
            if (
                tok
                and isinstance(tok, str)
                and tok.startswith("alo_stem_")
                and file_entry.get("size_bytes") == p.stat().st_size
            ):
                return str(tok)
        except (json.JSONDecodeError, OSError):
            pass

    from allomorph.dsp.io import read_wav

    audio, sr = read_wav(p)
    return compute_pcm_stem_digest(audio, sample_rate=sr)


def compute_robust_audio_digest(
    audio_path: Path | str,
    recipe_tag: str | None = None,
) -> str:
    """Computes a robust 256-bit audio content digest resistant to sub-LSB float roundoff across architectures.

    Combines 16-bit PCM quantized sample frames (header-stripped) with audio telemetry
    (sample rate, duration, LUFS, True Peak, RMS) and parametric recipe tag.
    Returns: 'alo_dry_<64 hex chars>'.
    """
    import math

    import numpy as np

    from allomorph.dsp.io import compute_true_peak_dbfs, read_wav
    from allomorph.dsp.loudness import compute_lufs

    p = Path(audio_path)
    audio, sr = read_wav(p)
    pcm16 = np.clip(np.round(audio * 32767.0), -32768, 32767).astype(np.int16)

    mono = audio[0] if audio.ndim > 1 else audio
    dur = round(float(len(mono)) / float(sr), 3) if sr > 0 else 0.0
    lufs_val = compute_lufs(audio, sample_rate=sr)
    lufs_rounded = (
        round(lufs_val, 2) if not (math.isinf(lufs_val) or math.isnan(lufs_val)) else -100.0
    )
    tp_rounded = round(compute_true_peak_dbfs(audio, sample_rate=sr), 2)
    rms = float(np.sqrt(np.mean(audio**2)))
    rms_rounded = round(20.0 * float(np.log10(max(rms, 1e-9))), 2)

    tag = recipe_tag or f"synthetic_optimal_bass_dry_v{DSP_GENERATION}"
    header_str = f"{tag}:{sr}:{dur}:{lufs_rounded}:{tp_rounded}:{rms_rounded}:"

    hasher = hashlib.sha256()
    hasher.update(header_str.encode("utf-8"))
    hasher.update(pcm16.tobytes())
    return f"alo_dry_{hasher.hexdigest()}"


def get_dry_excitation_token(
    input_path: Path | str | None = None,
    recipe_tag: str | None = None,
    force: bool = False,
) -> str:
    """Retrieves or computes the dry excitation token with robust, timestamp-independent caching.

    Checks an in-process cache, then an on-disk sidecar manifest in audio/manifest.json.
    Only computes the full 16-bit PCM digest if the file size changes or the manifest is missing.
    """
    if input_path is not None:
        p = Path(input_path)
    else:
        from allomorph.circuit.audio import find_default_input_audio

        p = find_default_input_audio()

    if not p.exists():
        from allomorph.naming import get_default_input_path

        if p.resolve() == get_default_input_path().resolve() or p.name == "input.wav":
            from allomorph.circuit.audio import find_default_input_audio

            p = find_default_input_audio()

    if not p.exists():
        raise FileNotFoundError(f"Dry excitation audio file not found: {p}")

    resolved_str = str(p.resolve())
    size = p.stat().st_size

    tag = recipe_tag or f"synthetic_optimal_bass_dry_v{DSP_GENERATION}"

    sidecar_p = p.parent / "manifest.json"

    if not force:
        # Layer 1: In-process memory cache
        if resolved_str in _DRY_TOKEN_CACHE:
            cached_size, cached_tok = _DRY_TOKEN_CACHE[resolved_str]
            if cached_size == size:
                return cached_tok

        # Layer 2: Persistent on-disk sidecar (audio/manifest.json)
        if sidecar_p.exists():
            try:
                with sidecar_p.open("r", encoding="utf-8") as sf:
                    sdata = json.load(sf)
                file_entry = sdata.get("files", {}).get(p.name, {})
                tok = file_entry.get("digest_token") or file_entry.get("dry_token")
                if (
                    tok
                    and file_entry.get("size_bytes") == size
                    and file_entry.get("recipe_tag") == tag
                ):
                    token = str(tok)
                    _DRY_TOKEN_CACHE[resolved_str] = (size, token)
                    return token
            except (json.JSONDecodeError, OSError):
                pass

    # Cold path: Compute robust audio digest and persist to sidecar
    token = compute_robust_audio_digest(p, recipe_tag=tag)
    _DRY_TOKEN_CACHE[resolved_str] = (size, token)

    rec_token = compute_excitation_recipe_token(dsp_generation=DSP_GENERATION, recipe_tag=tag)

    with _MANIFEST_LOCK:
        try:
            sidecar_data: dict[str, Any] = {}
            if sidecar_p.exists():
                with sidecar_p.open("r", encoding="utf-8") as sf:
                    sidecar_data = json.load(sf)
            files_dict = sidecar_data.setdefault("files", {})
            files_dict[p.name] = {
                "recipe_token": rec_token,
                "digest_token": token,
                "dry_token": token,
                "recipe_tag": tag,
                "size_bytes": size,
            }
            with sidecar_p.open("w", encoding="utf-8") as sf:
                json.dump(sidecar_data, sf, indent=2)
        except (json.JSONDecodeError, OSError):
            pass

    return token


def compute_recipe_token(
    instrument: Any,
    voicing: Any,
    base_dry_path: Path | str | None = None,
    base_dry_token: str | None = None,
    dsp_generation: int = DSP_GENERATION,
    preamps: Any = None,
    strings: Any = None,
) -> str:
    """Computes a deterministic 256-bit recipe token for an instrument voicing digital twin.

    Returns: 'alo_rec_<64 hex chars>'.
    """
    excitation_recipe_token = compute_excitation_recipe_token(dsp_generation=dsp_generation)

    eff_dry_token = (
        base_dry_token if base_dry_token is not None else get_dry_excitation_token(base_dry_path)
    )

    if hasattr(instrument, "model_dump"):
        inst_dict = instrument.model_dump(mode="json", exclude_none=True)
    elif isinstance(instrument, Mapping):
        inst_dict = dict(instrument)
    else:
        from allomorph.config.instruments import load_instrument

        inst_dict = load_instrument(instrument).model_dump(mode="json", exclude_none=True)

    if hasattr(voicing, "model_dump"):
        voice_dict = voicing.model_dump(mode="json", exclude_none=True)
    elif isinstance(voicing, Mapping):
        voice_dict = dict(voicing)
    else:
        voice_dict = {"id": str(voicing)}

    payload: dict[str, Any] = {
        "dsp_generation": dsp_generation,
        "excitation_recipe_token": excitation_recipe_token,
        "base_dry_token": eff_dry_token,
        "instrument": inst_dict,
        "voicing": voice_dict,
    }

    # Resolve harness and preamp
    harnesses = inst_dict.get("harnesses", {})
    h_id = voice_dict.get("harness")
    if not h_id and harnesses:
        h_id = next(iter(harnesses.keys()))

    h_cfg = harnesses.get(h_id) if h_id else None
    h_type = h_cfg.get("type") if isinstance(h_cfg, Mapping) else getattr(h_cfg, "type", None)

    preamp_id = (
        voice_dict.get("preamp")
        or (h_cfg.get("preamp") if isinstance(h_cfg, Mapping) else getattr(h_cfg, "preamp", None))
        or inst_dict.get("preamp")
    )
    if h_type == "active_preamp" and not preamp_id:
        preamp_id = "flat_buffer"

    if preamps is None:
        from allomorph.config.preamps import PREAMPS

        preamps = PREAMPS

    if preamp_id and preamps is not None:
        p_cfg = preamps.get(preamp_id) if hasattr(preamps, "get") else None
        if p_cfg is not None:
            if hasattr(p_cfg, "model_dump"):
                payload["preamp"] = p_cfg.model_dump(mode="json", exclude_none=True)
            elif isinstance(p_cfg, Mapping):
                payload["preamp"] = dict(p_cfg)

    raw_bytes = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    digest = hashlib.sha256(raw_bytes).hexdigest()
    return f"alo_rec_{digest}"


def compute_bundle_recipe_token(
    dry_digest_token: str,
    stem_tokens: Sequence[str | tuple[str, str]] | Mapping[str, str],
    *,
    dry_recipe_token: str | None = None,
) -> str:
    """Computes a deterministic 256-bit recipe token for an upload bundle aggregating audio digests.

    Returns: 'alo_bun_<64 hex chars>'.
    """
    eff_dry = dry_recipe_token if dry_recipe_token is not None else dry_digest_token
    hasher = hashlib.sha256()
    hasher.update(f"dry:{eff_dry}\n".encode())

    if isinstance(stem_tokens, Mapping):
        sorted_items = sorted(stem_tokens.items())
    elif stem_tokens and isinstance(stem_tokens[0], tuple):
        sorted_items = sorted(stem_tokens)  # type: ignore[arg-type]
    else:
        sorted_items = [(f"stem_{i}", str(tok)) for i, tok in enumerate(sorted(stem_tokens))]  # type: ignore[type-var]

    for name, tok in sorted_items:
        hasher.update(f"{name}:{tok}\n".encode())

    return f"alo_bun_{hasher.hexdigest()}"


def compute_pack_recipe_token(
    instrument_id: str,
    bundle_tokens: Mapping[str, str] | Sequence[tuple[str, str]],
    dsp_generation: int = DSP_GENERATION,
) -> str:
    """Computes a deterministic 256-bit recipe token for a complete Tone Pack.

    Returns: 'alo_pck_<64 hex chars>'.
    """
    hasher = hashlib.sha256()
    hasher.update(f"pack:{instrument_id}:dsp{dsp_generation}\n".encode())

    sorted_bundles = (
        sorted(bundle_tokens.items())
        if isinstance(bundle_tokens, Mapping)
        else sorted(bundle_tokens)
    )
    for b_name, b_tok in sorted_bundles:
        hasher.update(f"bundle:{b_name}:{b_tok}\n".encode())

    return f"alo_pck_{hasher.hexdigest()}"


def write_manifest(
    output_dir: Path | str,
    stage: str,
    files: dict[str, dict[str, Any]] | Sequence[Path | str],
    version_tag: str | None = None,
    base_dry_token: str | None = None,
    base_dry_file: str | None = None,
    instrument_version: int | None = None,
    voicing_version: int | None = None,
    recipe_token: str | None = None,
    digest_token: str | None = None,
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
        base_dry_token=base_dry_token,
        base_dry_file=base_dry_file,
        instrument_version=instrument_version,
        voicing_version=voicing_version,
        recipe_token=recipe_token,
        digest_token=digest_token,
    )


def check_wet_stem_freshness(
    stem_path: Path | str,
    expected_recipe_token: str,
    base_dry_path: Path | str | None = None,
    expected_version: str | None = None,
) -> tuple[bool, str]:
    """Validates that a compiled wet audio stem exists and matches the expected recipe token.

    Returns (is_fresh, reason) where reason is 'cached & fresh' or diagnostic explanation.
    """
    stem_p = Path(stem_path)
    if not stem_p.exists():
        return False, "audio file missing"

    manifest_p = stem_p.parent / "manifest.json"
    if not manifest_p.exists():
        return False, "manifest.json missing"

    try:
        with manifest_p.open("r", encoding="utf-8") as f:
            manifest = json.load(f)
    except (json.JSONDecodeError, OSError):
        return False, "manifest.json unreadable"

    files = manifest.get("files", {})
    if stem_p.name not in files:
        return False, "not registered in manifest.json"

    file_entry = files[stem_p.name]
    recorded_token = file_entry.get("recipe_token") or manifest.get("recipe_token")
    if recorded_token != expected_recipe_token:
        return False, "recipe token mismatch"

    if expected_version is not None:
        recorded_ver = file_entry.get("version") or manifest.get("version")
        if recorded_ver != expected_version:
            return False, f"version mismatch ({recorded_ver} != {expected_version})"

    return True, "cached & fresh"


def is_wet_stem_valid(
    stem_path: Path | str,
    expected_recipe_token: str,
    base_dry_path: Path | str | None = None,
    expected_version: str | None = None,
) -> bool:
    """Validates that a compiled wet audio stem exists and matches the expected recipe token.

    Returns False if:
    - The stem file does not exist.
    - The sidecar manifest.json does not exist or cannot be parsed.
    - The stem is not registered in the manifest.
    - The manifest's recorded recipe_token does not match expected_recipe_token.
    - An expected_version is provided and the manifest version does not match it.
    """
    valid, _ = check_wet_stem_freshness(
        stem_path=stem_path,
        expected_recipe_token=expected_recipe_token,
        base_dry_path=base_dry_path,
        expected_version=expected_version,
    )
    return valid


__all__ = [
    "ALLOMORPH_VERSION",
    "DEFAULT_INST_VERSION",
    "DEFAULT_VOICE_VERSION",
    "DSP_GENERATION",
    "_MANIFEST_LOCK",
    "check_wet_stem_freshness",
    "compute_bundle_recipe_token",
    "compute_excitation_digest_token",
    "compute_excitation_recipe_token",
    "compute_file_sha256",
    "compute_pack_recipe_token",
    "compute_pcm_stem_digest",
    "compute_recipe_token",
    "compute_robust_audio_digest",
    "compute_stem_digest_token",
    "get_dry_excitation_token",
    "get_git_commit",
    "get_version_info",
    "is_wet_stem_valid",
    "resolve_tri_part_version",
    "write_manifest",
]
