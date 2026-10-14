"""
Allomorph - Tone3000 Tone Pack Pipeline
Automates creation, partitioning, packaging, and local training of Tone3000 upload bundles
respecting Tone3000's strict '1 Dry File + Multiple Wet Stems' batch upload constraint.
"""

import hashlib
import json
import shutil
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from allomorph.circuit.forward import simulate_instrument_voicing
from allomorph.config.instruments import (
    VoicingBundle,
    load_instrument,
    load_tone_pack,
    partition_instrument_bundles,
)
from allomorph.config.scales import REPO_ROOT
from allomorph.config.schema import InstrumentConfig
from allomorph.dsp import read_wav
from allomorph.naming import VOICE_SLUG_ALIASES, get_default_input_path
from allomorph.pipeline.bundle import (
    assemble_root_pack_manifest,
    build_bundle_manifest_entry,
    generate_bundle_upload_instructions,
    resolve_bundle_target_file_descriptors,
)
from allomorph.trainer.constants import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_ETA_MIN,
    DEFAULT_LR_SCHEDULER,
    DEFAULT_LR_T_MAX,
    DEFAULT_MAX_EPOCHS,
    DEFAULT_MIN_DELTA,
    DEFAULT_MIN_EPOCHS,
    DEFAULT_MRSTFT_WEIGHT,
    DEFAULT_MSE_WEIGHT,
    DEFAULT_NUM_WORKERS,
    DEFAULT_PATIENCE,
    DEFAULT_PRECISION,
)
from allomorph.version import (
    DSP_GENERATION,
    compute_bundle_recipe_token,
    compute_excitation_recipe_token,
    compute_pack_recipe_token,
    compute_recipe_token,
    compute_stem_digest_token,
    get_dry_excitation_token,
    is_wet_stem_valid,
    resolve_tri_part_version,
)

AUDIO_DIR = REPO_ROOT / "audio"
WET_AUDIO_DIR = AUDIO_DIR / "wet"
PACKS_DIR = REPO_ROOT / "tone3000" / "packs"
ASSETS_DIR = REPO_ROOT / "tone3000" / "assets"


def _sha256_file(path: Path) -> str:
    """Calculates SHA256 hex digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def check_pack_needs_repacking(
    instrument: InstrumentConfig | str | Path,
    output_dir: Path | str | None = None,
    input_wav: Path | str | None = None,
    catalog_targets: Sequence[tuple[str, str]] | None = None,
) -> bool:
    """Checks whether a tone pack requires repacking based on directory existence,
    bundle file integrity, and cryptographic recipe tokens."""
    inst = (
        load_instrument(instrument) if not isinstance(instrument, InstrumentConfig) else instrument
    )
    pack_dir = Path(output_dir) if output_dir else PACKS_DIR / inst.id
    bundles_dir = pack_dir / "bundles"
    manifest_path = pack_dir / "manifest.json"

    if not manifest_path.exists() or not bundles_dir.exists():
        return True

    bundles = partition_instrument_bundles(inst, catalog_targets=catalog_targets)
    is_multi_pickup = len(bundles) > 1

    try:
        with open(manifest_path, "r", encoding="utf-8") as mf:
            existing_pack_data = json.load(mf)
        recorded_pack_token = existing_pack_data.get("recipe_token")

        if existing_pack_data.get("dsp_generation") != DSP_GENERATION:
            return True
        if existing_pack_data.get("instrument_version") != inst.version:
            return True
        if not (pack_dir / "storefront_description.txt").exists():
            return True

        artwork_candidates = [
            ASSETS_DIR / f"allomorph_{inst.id}.jpg",
            ASSETS_DIR
            / f"allomorph_{inst.id.replace('34in_', '').replace('30in_', '').replace('32in_', '')}.jpg",
            ASSETS_DIR / f"{inst.id}.jpg",
        ]
        if any(c.exists() for c in artwork_candidates) and not (pack_dir / "artwork.jpg").exists():
            return True

        existing_bundles = existing_pack_data.get("bundles", {})
        if set(existing_bundles.keys()) != set(bundles.keys()):
            return True

        expected_bundle_tokens: dict[str, str] = {}
        for b_name, b_bundle in bundles.items():
            b_dir = bundles_dir / b_name
            b_man_p = b_dir / "manifest.json"
            if not b_man_p.exists() or not (b_dir / "upload_instructions.txt").exists():
                return True
            with open(b_man_p, "r", encoding="utf-8") as bmf:
                b_man = json.load(bmf)

            dry_f = b_man.get("dry_file")
            if not dry_f or not (b_dir / dry_f).exists():
                return True

            b_src_path = WET_AUDIO_DIR / inst.id / f"{b_bundle.source_voicing}.wav"
            if not b_src_path.exists():
                return True
            master_dry_digest = compute_stem_digest_token(b_src_path)
            if b_man.get("dry_digest_token") != master_dry_digest:
                return True

            b_descs = resolve_bundle_target_file_descriptors(b_bundle, is_multi_pickup)

            # Check for extraneous .wav files in the bundle directory
            expected_wav_names = {dry_f} | {desc["stem_filename"] for desc in b_descs}
            actual_wav_names = {p.name for p in b_dir.glob("*.wav")}
            if actual_wav_names != expected_wav_names:
                return True

            b_stem_toks = []
            stems_by_tgt = {s["target"]: s for s in b_man.get("stems", []) if "target" in s}
            stems_by_tone = {s["tone_name"]: s for s in b_man.get("stems", []) if "tone_name" in s}
            for desc in b_descs:
                t_ref = desc["target_ref"]
                t_str = f"{t_ref.instrument_id}:{t_ref.voicing_id}"
                s_info = stems_by_tgt.get(t_str) or stems_by_tone.get(desc["tone_name"])
                if not s_info:
                    return True
                s_fn = s_info.get("filename", desc["stem_filename"])
                if not (b_dir / s_fn).exists():
                    return True

                t_stem_path = WET_AUDIO_DIR / t_ref.instrument_id / f"{desc['target_slug']}.wav"
                if not t_stem_path.exists():
                    return True

                master_wet_digest = compute_stem_digest_token(t_stem_path)
                if s_info.get("digest_token") != master_wet_digest:
                    return True

                if s_info.get("version") != desc["target_v_tag"]:
                    return True
                if s_info.get("filename") != desc["stem_filename"]:
                    return True

                b_stem_toks.append((desc["stem_filename"], master_wet_digest))

            expected_bundle_tokens[b_name] = compute_bundle_recipe_token(
                dry_digest_token=master_dry_digest,
                stem_tokens=b_stem_toks,
            )
            if b_man.get("recipe_token") != expected_bundle_tokens[b_name]:
                return True

        expected_pack_token = compute_pack_recipe_token(
            inst.id, expected_bundle_tokens, dsp_generation=DSP_GENERATION
        )

        all_bundles_exist = all(
            (bundles_dir / b_name / "manifest.json").exists()
            and (bundles_dir / b_name / "upload_instructions.txt").exists()
            for b_name in bundles
        )
        return not (recorded_pack_token == expected_pack_token and all_bundles_exist)
    except (json.JSONDecodeError, KeyError, OSError):
        return True


def export_tone_pack(
    instrument: InstrumentConfig | str | Path,
    output_dir: Path | str | None = None,
    input_wav: Path | str | None = None,
    max_samples: int | None = None,
    catalog_targets: Sequence[tuple[str, str]] | None = None,
    jobs: int | None = None,
    train: bool = False,
) -> Path:
    """
    Exports a complete Tone3000 upload pack for an instrument partitioned into
    self-contained bundles matching the '1 Dry + Multiple Wet Stems' upload constraint.
    Copies existing pre-compiled wet stems from audio/wet/, or compiles then copies them.
    """
    inst = (
        load_instrument(instrument) if not isinstance(instrument, InstrumentConfig) else instrument
    )

    pack_dir = Path(output_dir) if output_dir else PACKS_DIR / inst.id
    bundles_dir = pack_dir / "bundles"
    nam_dir = pack_dir / "nam"

    if not check_pack_needs_repacking(
        inst, output_dir=pack_dir, input_wav=input_wav, catalog_targets=catalog_targets
    ):
        print(
            f"\n[Tone Pack Exporter] Tone Pack already exists and is fresh at '{pack_dir}'."
        )
        return pack_dir

    # 1. Partition instrument into pickup bundles first to validate before creating directories
    bundles = partition_instrument_bundles(inst, catalog_targets=catalog_targets)
    is_multi_pickup = len(bundles) > 1

    pack_dir.mkdir(parents=True, exist_ok=True)
    bundles_dir.mkdir(parents=True, exist_ok=True)
    if train:
        nam_dir.mkdir(parents=True, exist_ok=True)

    # Prune any stale/extraneous bundle directories not declared in bundles
    for existing_item in bundles_dir.iterdir():
        if existing_item.is_dir() and existing_item.name not in bundles:
            shutil.rmtree(existing_item)

    manifest_path = pack_dir / "manifest.json"

    # 2. Process each bundle
    # 2. Process each bundle
    def _process_bundle(item: tuple[str, VoicingBundle]) -> tuple[str, dict[str, Any]]:
        b_name, bundle = item
        b_dir = bundles_dir / b_name
        b_dir.mkdir(parents=True, exist_ok=True)

        pos_label = bundle.position_name or b_name.capitalize()

        # Determine source voicing version for this bundle
        src_v_id = bundle.source_voicing
        if src_v_id not in inst.voicings:
            raise KeyError(f"Source voicing '{src_v_id}' not found on instrument '{inst.id}'.")
        src_v = inst.voicings[src_v_id]
        source_voice_ver = getattr(src_v, "version", 1)

        dry_recipe_token = compute_recipe_token(
            instrument=inst,
            voicing=src_v,
            base_dry_path=input_wav,
            dsp_generation=DSP_GENERATION,
        )

        # Check for existing bundle manifest
        existing_b_man: dict[str, Any] | None = None
        old_stems_by_target: dict[str, dict[str, Any]] = {}
        old_stems_by_tone: dict[str, dict[str, Any]] = {}
        b_manifest_p = b_dir / "manifest.json"
        if b_manifest_p.exists():
            try:
                with open(b_manifest_p, "r", encoding="utf-8") as bf:
                    loaded = json.load(bf)
                if isinstance(loaded, dict):
                    existing_b_man = loaded
                    for s in existing_b_man.get("stems", []):
                        if "target" in s:
                            old_stems_by_target[s["target"]] = s
                        if "tone_name" in s:
                            old_stems_by_tone[s["tone_name"]] = s
                else:
                    existing_b_man = None
            except (json.JSONDecodeError, OSError):
                existing_b_man = None

        # A. Synthesize / Copy Source Audio for this bundle
        current_dry_v_tag = resolve_tri_part_version(DSP_GENERATION, inst.version, source_voice_ver)
        source_wet_path = WET_AUDIO_DIR / inst.id / f"{src_v_id}.wav"

        if max_samples is None:
            if not is_wet_stem_valid(
                source_wet_path,
                expected_recipe_token=dry_recipe_token,
                base_dry_path=input_wav,
                expected_version=current_dry_v_tag,
            ):
                source_wet_path.parent.mkdir(parents=True, exist_ok=True)
                simulate_instrument_voicing(
                    instrument=inst,
                    voicing=src_v,
                    input_wav=input_wav,
                    output_wav=source_wet_path,
                    max_samples=None,
                    force=False,
                )
            source_digest_token = compute_stem_digest_token(source_wet_path)
        else:
            source_digest_token = (
                compute_stem_digest_token(source_wet_path)
                if source_wet_path.exists()
                else "alo_stem_bounded"
            )

        if (
            existing_b_man
            and existing_b_man.get("dry_digest_token") == source_digest_token
            and existing_b_man.get("dry_file")
            and (b_dir / existing_b_man["dry_file"]).exists()
        ):
            dry_filename = existing_b_man["dry_file"]
            dry_v_tag = existing_b_man.get("dry_version", current_dry_v_tag)
        else:
            dry_v_tag = current_dry_v_tag
            dry_filename = f"dry {dry_v_tag}.wav"

        dry_dest = b_dir / dry_filename

        # B. Wet Audio Stems (Copy cached wet audio from audio/wet or compile then copy)
        bundle_stems: list[dict[str, Any]] = []
        stem_filenames: list[str] = []
        descriptors = resolve_bundle_target_file_descriptors(bundle, is_multi_pickup)

        temp_stems: list[tuple[dict[str, Any], Path, Path, dict[str, Any] | None]] = []
        for desc in descriptors:
            target_ref = desc["target_ref"]
            tone_name = desc["tone_name"]
            target_v_tag_current = desc["target_v_tag"]
            target_slug = desc["target_slug"]
            target_wet_path = WET_AUDIO_DIR / target_ref.instrument_id / f"{target_slug}.wav"

            target_recipe_token = compute_recipe_token(
                instrument=target_ref.instrument,
                voicing=target_ref.voicing,
                base_dry_path=input_wav,
                dsp_generation=DSP_GENERATION,
            )

            if max_samples is None:
                if not is_wet_stem_valid(
                    target_wet_path,
                    expected_recipe_token=target_recipe_token,
                    base_dry_path=input_wav,
                    expected_version=target_v_tag_current,
                ):
                    target_wet_path.parent.mkdir(parents=True, exist_ok=True)
                    simulate_instrument_voicing(
                        instrument=target_ref.instrument,
                        voicing=target_ref.voicing,
                        input_wav=input_wav,
                        output_wav=target_wet_path,
                        max_samples=None,
                        force=False,
                    )
                target_digest_token = compute_stem_digest_token(target_wet_path)
            else:
                target_digest_token = (
                    compute_stem_digest_token(target_wet_path)
                    if target_wet_path.exists()
                    else "alo_stem_bounded"
                )

            target_str = f"{target_ref.instrument_id}:{target_ref.voicing_id}"
            old_s = old_stems_by_target.get(target_str) or old_stems_by_tone.get(tone_name)

            if (
                old_s
                and old_s.get("digest_token") == target_digest_token
                and old_s.get("filename")
                and (b_dir / old_s["filename"]).exists()
            ):
                stem_filename = old_s["filename"]
                target_v_tag = old_s.get("version", target_v_tag_current)
            else:
                target_v_tag = target_v_tag_current
                stem_filename = desc["stem_filename"]

            stem_path = b_dir / stem_filename
            stem_filenames.append(stem_filename)

            s_entry = {
                "filename": stem_filename,
                "target": target_str,
                "target_instrument_version": target_ref.instrument.version,
                "target_voicing_version": target_ref.voicing.version,
                "version": target_v_tag,
                "tone_name": tone_name,
                "recipe_token": target_recipe_token,
                "digest_token": target_digest_token,
            }
            bundle_stems.append(s_entry)
            temp_stems.append((s_entry, stem_path, target_wet_path, old_s))

        # Clean up any extraneous or stale wav files in bundle directory
        needed_wavs = {dry_filename} | set(stem_filenames)
        for old_wav in b_dir.glob("*.wav"):
            if old_wav.name not in needed_wavs:
                old_wav.unlink(missing_ok=True)

        # Copy dry file
        if max_samples is None:
            if (
                not dry_dest.exists()
                or not existing_b_man
                or existing_b_man.get("dry_digest_token") != source_digest_token
            ):
                shutil.copyfile(source_wet_path, dry_dest)
        else:
            simulate_instrument_voicing(
                instrument=inst,
                voicing=src_v,
                input_wav=input_wav,
                output_wav=dry_dest,
                max_samples=max_samples,
                force=True,
            )

        # Copy wet stems
        for s_dict, s_path, s_wet_p, stem_old_s in temp_stems:
            if max_samples is None:
                if (
                    not s_path.exists()
                    or not stem_old_s
                    or stem_old_s.get("digest_token") != s_dict["digest_token"]
                ):
                    shutil.copyfile(s_wet_p, s_path)
            else:
                tgt_desc = next(
                    d
                    for d in descriptors
                    if d["stem_filename"] == s_dict["filename"]
                    or d["tone_name"] == s_dict["tone_name"]
                )
                simulate_instrument_voicing(
                    instrument=tgt_desc["target_ref"].instrument,
                    voicing=tgt_desc["target_ref"].voicing,
                    input_wav=input_wav,
                    output_wav=s_path,
                    max_samples=max_samples,
                    force=True,
                )

            # Tone3000 Zero-Collision Guardrail: verify wet stem is not bit-exact identical to dry file
            dry_audio, _ = read_wav(dry_dest)
            wet_audio, _ = read_wav(s_path)
            min_len = min(len(dry_audio), len(wet_audio))
            max_diff = float(np.max(np.abs(wet_audio[:min_len] - dry_audio[:min_len])))
            if np.array_equal(wet_audio[:min_len], dry_audio[:min_len]) or max_diff < 1e-10:
                raise ValueError(
                    f"Identity stem collision detected: target '{s_dict['target']}' on '{inst.id}' "
                    f"is bit-exact identical to dry stem in bundle '{b_name}' (max_diff={max_diff:.8f}). "
                    "Tone3000 will reject training on identical dry/wet stems."
                )

        # C. Write Bundle Upload Instructions
        instructions_path = b_dir / "upload_instructions.txt"
        instr_content = generate_bundle_upload_instructions(
            bundle_name=b_name,
            position_name=pos_label,
            instrument_name=inst.name,
            dry_filename=dry_filename,
            stem_filenames=stem_filenames,
        )
        instructions_path.write_text(instr_content, encoding="utf-8")

        base_dry_p = Path(input_wav) if input_wav else get_default_input_path()
        base_dry_token = get_dry_excitation_token(base_dry_p)
        base_dry_rec = compute_excitation_recipe_token(dsp_generation=DSP_GENERATION)

        bundle_recipe_token = compute_bundle_recipe_token(
            dry_digest_token=source_digest_token,
            stem_tokens=[(s["filename"], s["digest_token"]) for s in bundle_stems],
        )

        source_token = f"{inst.id}:{src_v_id}"
        bundle_manifest = build_bundle_manifest_entry(
            bundle_name=b_name,
            position_name=pos_label,
            source=source_token,
            dry_filename=dry_filename,
            dry_version=dry_v_tag,
            dry_recipe_token=dry_recipe_token,
            recipe_token=bundle_recipe_token,
            instrument_id=inst.id,
            instrument_version=inst.version,
            source_voicing_version=source_voice_ver,
            bundle_stems=bundle_stems,
            base_dry_file=base_dry_p.name,
            base_dry_token=base_dry_token,
            dry_digest_token=source_digest_token,
            base_dry_recipe_token=base_dry_rec,
        )
        with open(b_dir / "manifest.json", "w", encoding="utf-8") as bf:
            json.dump(bundle_manifest, bf, indent=2)

        bundle_entry = {
            "recipe_token": bundle_recipe_token,
            "source": source_token,
            "position_name": pos_label,
            "base_dry_file": base_dry_p.name,
            "base_dry_token": base_dry_token,
            "dry_file": dry_filename,
            "dry_version": dry_v_tag,
            "dry_recipe_token": dry_recipe_token,
            "dry_digest_token": source_digest_token,
            "source_voicing_version": source_voice_ver,
            "stem_count": len(bundle_stems),
        }
        return b_name, bundle_entry

    eff_jobs = jobs if jobs is not None and jobs > 0 else 1
    if eff_jobs > 1 and len(bundles) > 1:
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=eff_jobs) as executor:
            bundle_results = list(executor.map(_process_bundle, bundles.items()))
    else:
        bundle_results = [_process_bundle(item) for item in bundles.items()]

    pack_bundle_tokens = {b_name: b_meta["recipe_token"] for b_name, b_meta in bundle_results}
    pack_recipe_token = compute_pack_recipe_token(
        inst.id, pack_bundle_tokens, dsp_generation=DSP_GENERATION
    )

    manifest_entries = assemble_root_pack_manifest(
        inst=inst,
        bundle_entries={b_name: b_meta for b_name, b_meta in bundle_results},
        recipe_token=pack_recipe_token,
        dsp_generation=DSP_GENERATION,
    )

    # 3. Storefront Description
    desc_path = pack_dir / "storefront_description.txt"
    pack_cfg = load_tone_pack(inst.id)
    desc_text = pack_cfg.full_storefront_description
    desc_path.write_text(desc_text, encoding="utf-8")

    # 4. Artwork
    artwork_candidates = [
        ASSETS_DIR / f"allomorph_{inst.id}.jpg",
        ASSETS_DIR
        / f"allomorph_{inst.id.replace('34in_', '').replace('30in_', '').replace('32in_', '')}.jpg",
        ASSETS_DIR / f"{inst.id}.jpg",
    ]
    if "jazz" in inst.id:
        artwork_candidates.append(ASSETS_DIR / "allomorph_standard_jazz_bass.jpg")
    elif "standard_p" in inst.id:
        artwork_candidates.append(ASSETS_DIR / "allomorph_standard_precision_bass.jpg")
    elif "stingray" in inst.id:
        artwork_candidates.append(ASSETS_DIR / "allomorph_active_stingray_bass.jpg")
    elif "mustang" in inst.id:
        artwork_candidates.append(ASSETS_DIR / "allomorph_mustang_pj_bass.jpg")
    elif "pj" in inst.id:
        artwork_candidates.append(ASSETS_DIR / "allomorph_standard_pj_bass.jpg")
    elif "emg" in inst.id:
        artwork_candidates.append(ASSETS_DIR / "allomorph_active_emg_bass.jpg")
    elif "soapbar" in inst.id:
        artwork_candidates.append(ASSETS_DIR / "allomorph_preamp_soapbar_bass.jpg")

    dest_artwork = pack_dir / "artwork.jpg"
    for cand in artwork_candidates:
        if cand.exists():
            shutil.copyfile(cand, dest_artwork)
            break

    # 5. Manifest JSON: preserve and migrate any existing trained models
    manifest_path = pack_dir / "manifest.json"
    if manifest_path.exists():
        try:
            with open(manifest_path, "r", encoding="utf-8") as mf:
                old_m = json.load(mf)
            if (old_models := old_m.get("models")) and isinstance(old_models, dict):
                stem_lookup: dict[tuple[str, str] | str, tuple[dict[str, Any], dict[str, Any]]] = {}
                for b_name in manifest_entries["bundles"]:
                    b_man_p = pack_dir / "bundles" / b_name / "manifest.json"
                    if b_man_p.exists():
                        try:
                            with open(b_man_p, "r", encoding="utf-8") as bf:
                                b_man = json.load(bf)
                            for s_info in b_man.get("stems", []):
                                stem_lookup[(b_name, s_info["filename"])] = (b_man, s_info)
                                stem_lookup[s_info["filename"]] = (b_man, s_info)
                                if "target" in s_info:
                                    stem_lookup[(b_name, s_info["target"])] = (b_man, s_info)
                                    stem_lookup[s_info["target"]] = (b_man, s_info)
                                if "tone_name" in s_info:
                                    stem_lookup[(b_name, s_info["tone_name"])] = (b_man, s_info)
                                    stem_lookup[s_info["tone_name"]] = (b_man, s_info)
                        except (json.JSONDecodeError, OSError):
                            pass

                updated_models: dict[str, Any] = {}
                for m_name, m_entry in old_models.items():
                    entry = dict(m_entry)
                    entry.pop("dry_sha256", None)
                    entry.pop("wet_sha256", None)
                    entry.pop("target_instrument", None)
                    entry.pop("target_voicing", None)
                    b_name = entry.get("bundle", "")
                    wet_file = entry.get("wet_file", "")
                    target = entry.get("target", "")
                    tone_name = entry.get("tone_name", "")
                    lookup_hit = (
                        stem_lookup.get((b_name, wet_file))
                        or stem_lookup.get(wet_file)
                        or (stem_lookup.get((b_name, target)) if target else None)
                        or (stem_lookup.get(target) if target else None)
                        or (stem_lookup.get((b_name, tone_name)) if tone_name else None)
                        or (stem_lookup.get(tone_name) if tone_name else None)
                    )
                    if lookup_hit:
                        b_info, s_info = lookup_hit
                        m_dry_digest = entry.get("dry_digest_token")
                        m_wet_digest = entry.get("wet_digest_token")
                        curr_dry_digest = b_info.get("dry_digest_token")
                        curr_wet_digest = s_info.get("digest_token")

                        digests_match = (
                            bool(m_dry_digest)
                            and bool(m_wet_digest)
                            and m_dry_digest == curr_dry_digest
                            and m_wet_digest == curr_wet_digest
                        )

                        if digests_match:
                            if "filename" in s_info:
                                entry["wet_file"] = s_info["filename"]
                            if "dry_file" in b_info:
                                entry["dry_file"] = b_info["dry_file"]
                            if "dry_recipe_token" in b_info:
                                entry["dry_recipe_token"] = b_info["dry_recipe_token"]
                            if "recipe_token" in s_info:
                                entry["wet_recipe_token"] = s_info["recipe_token"]
                            if "version" in s_info:
                                entry["version"] = s_info["version"]
                            if "target_instrument_version" in s_info:
                                entry["target_instrument_version"] = s_info["target_instrument_version"]
                            if "target_voicing_version" in s_info:
                                entry["target_voicing_version"] = s_info["target_voicing_version"]

                        if "target" in s_info and not entry.get("target"):
                            entry["target"] = s_info["target"]
                        if "tone_name" in s_info and not entry.get("tone_name"):
                            entry["tone_name"] = s_info["tone_name"]
                        if not entry.get("bundle") and b_name:
                            entry["bundle"] = b_name
                    updated_models[m_name] = entry
                manifest_entries["models"] = updated_models
        except (json.JSONDecodeError, OSError):
            pass

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_entries, f, indent=2)

    print(f"\n[Tone Pack Exporter] Successfully exported Tone Pack for '{inst.name}' to:")
    print(f"  -> {pack_dir}")
    print(f"  -> Bundles: {list(bundles.keys())}")
    print(f"  -> Total Stems: {sum(b['stem_count'] for b in manifest_entries['bundles'].values())}")

    if train:
        train_tone_pack(
            pack=inst,
        )

    return pack_dir


def _populate_manifest_model_entry(
    models_dict: dict[str, Any],
    nam_path: Path,
    stem_info: dict[str, Any],
    bundle_name: str,
    dry_info: dict[str, Any] | None = None,
) -> None:
    """Reads .nam container and populates metadata entry in pack manifest."""
    sha256 = _sha256_file(nam_path)
    target_str = stem_info.get("target") or (
        f"{stem_info.get('target_instrument')}:{stem_info.get('target_voicing')}"
        if stem_info.get("target_instrument") and stem_info.get("target_voicing")
        else None
    )
    entry: dict[str, Any] = {
        "filename": nam_path.name,
        "bundle": bundle_name,
        "target": target_str,
        "target_instrument_version": stem_info.get("target_instrument_version"),
        "target_voicing_version": stem_info.get("target_voicing_version"),
        "version": stem_info.get("version"),
        "tone_name": stem_info.get("tone_name"),
        "sha256": sha256,
    }
    if dry_info:
        if "dry_file" in dry_info:
            entry["dry_file"] = dry_info["dry_file"]
        if "dry_recipe_token" in dry_info:
            entry["dry_recipe_token"] = dry_info["dry_recipe_token"]
        if "dry_digest_token" in dry_info:
            entry["dry_digest_token"] = dry_info["dry_digest_token"]
    if "filename" in stem_info:
        entry["wet_file"] = stem_info["filename"]
    if "recipe_token" in stem_info:
        entry["wet_recipe_token"] = stem_info["recipe_token"]
    if "digest_token" in stem_info:
        entry["wet_digest_token"] = stem_info["digest_token"]
    try:
        with open(nam_path, "r", encoding="utf-8") as f:
            nam_data = json.load(f)
        meta = nam_data.get("metadata", {})
        training_meta = meta.get("training", {})
        if isinstance(training_meta, dict):
            for field in [
                "validation_esr",
                "validation_esr_a2_full",
                "validation_esr_a2_lite",
                "validation_esr_ch8",
                "validation_esr_ch3",
                "validation_esr_aggregate",
                "esr",
                "differential_esr",
                "differential_esr_ch3",
                "mrstft_loss",
                "mrstft_loss_ch3",
                "baseline_mrstft",
                "differential_mrstft",
                "epochs_trained",
                "stop_reason",
                "batch_size",
                "precision",
                "num_workers",
                "device_name",
                "engine",
                "training_time_seconds",
            ]:
                if field in training_meta and training_meta[field] is not None:
                    entry[field] = training_meta[field]
    except (json.JSONDecodeError, OSError) as e:
        print(
            f"[Tone Pack Trainer] Warning: Could not read training metadata from {nam_path.name}: {e}"
        )

    # Preserve any training metadata already recorded in the manifest
    existing_entry = models_dict.get(nam_path.name, {})
    if isinstance(existing_entry, dict):
        for field in [
            "validation_esr",
            "validation_esr_a2_full",
            "validation_esr_a2_lite",
            "validation_esr_ch8",
            "validation_esr_ch3",
            "validation_esr_aggregate",
            "esr",
            "differential_esr",
            "differential_esr_ch3",
            "mrstft_loss",
            "mrstft_loss_ch3",
            "baseline_mrstft",
            "differential_mrstft",
            "epochs_trained",
            "stop_reason",
            "batch_size",
            "precision",
            "num_workers",
            "device_name",
            "engine",
            "training_time_seconds",
        ]:
            if field in existing_entry and field not in entry:
                entry[field] = existing_entry[field]

    models_dict[nam_path.name] = entry


def train_tone_pack(
    pack: InstrumentConfig | str | Path,
    voice: str = "all",
    overwrite: bool = False,
    epochs: int = DEFAULT_MAX_EPOCHS,
    min_epochs: int = DEFAULT_MIN_EPOCHS,
    patience: int = DEFAULT_PATIENCE,
    min_delta: float = DEFAULT_MIN_DELTA,
    mse_weight: float = DEFAULT_MSE_WEIGHT,
    mrstft_weight: float = DEFAULT_MRSTFT_WEIGHT,
    batch_size: int | str = DEFAULT_BATCH_SIZE,
    precision: str = DEFAULT_PRECISION,
    num_workers: int | str = DEFAULT_NUM_WORKERS,
    lr_scheduler: str = DEFAULT_LR_SCHEDULER,
    eta_min: float = DEFAULT_ETA_MIN,
    lr_t_max: int = DEFAULT_LR_T_MAX,
    fast_dev_run: bool = False,
    engine: str = "auto",
    seed: int | None = None,
) -> Path:
    """Trains a complete Tone3000 upload pack, outputting flat .nam files to tone3000/packs/[pack]/nam/.

    Reads each bundle in the pack manifest, training its target wet stems against the bundle's dry excitation.
    Automatically generates missing or incomplete bundle files before training.
    Skips already trained models unless overwrite=True.
    Updates the pack manifest with a top-level 'models' section indexing all trained model containers.
    """

    from allomorph.config.voices import VOICES

    if (
        isinstance(pack, (str, Path))
        and Path(pack).is_dir()
        and (Path(pack) / "manifest.json").exists()
    ):
        with open(Path(pack) / "manifest.json", "r", encoding="utf-8") as mf:
            pack_manifest = json.load(mf)
        inst = load_instrument(pack_manifest.get("instrument_id", Path(pack).name))
        pack_dir = Path(pack)
    else:
        inst = load_instrument(pack) if not isinstance(pack, InstrumentConfig) else pack
        pack_dir = PACKS_DIR / inst.id

    manifest_path = pack_dir / "manifest.json"
    if not manifest_path.exists():
        print(
            f"[Tone Pack Trainer] Manifest missing for '{inst.name}' at '{pack_dir}'. Auto-exporting pack..."
        )
        export_tone_pack(inst, output_dir=pack_dir)

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest_data = json.load(f)

    # Check for missing bundle stems and auto-export if needed
    bundles_dict = manifest_data.get("bundles", {})
    missing_stems = False
    if not bundles_dict:
        missing_stems = True
    else:
        for b_name, b_info in bundles_dict.items():
            b_dir = pack_dir / "bundles" / b_name
            dry_p = b_dir / b_info.get("dry_file", "")
            if not dry_p.exists():
                missing_stems = True
                break
            b_manifest_path = b_dir / "manifest.json"
            if not b_manifest_path.exists():
                missing_stems = True
                break
            try:
                with open(b_manifest_path, "r", encoding="utf-8") as bmf:
                    b_manifest = json.load(bmf)
                b_stems = b_manifest.get("stems", [])
            except (json.JSONDecodeError, OSError):
                missing_stems = True
                break
            for s_info in b_stems:
                stem_p = b_dir / s_info.get("filename", "")
                if not stem_p.exists():
                    missing_stems = True
                    break
            if missing_stems:
                break

    if missing_stems:
        print(
            f"[Tone Pack Trainer] Pack bundles missing or incomplete in '{pack_dir}'. Auto-exporting..."
        )
        export_tone_pack(inst, output_dir=pack_dir)
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest_data = json.load(f)

    nam_dir = pack_dir / "nam"
    nam_dir.mkdir(parents=True, exist_ok=True)

    selected_voices: set[str] | None = None
    if voice and str(voice).strip().lower() != "all":
        from allomorph.naming import resolve_voices

        selected_voices = set(resolve_voices(voice))

    if "models" not in manifest_data or not isinstance(manifest_data["models"], dict):
        manifest_data["models"] = {}

    from allomorph.trainer import train_voice

    total_trained = 0
    total_skipped = 0

    for b_name, b_info in manifest_data.get("bundles", {}).items():
        b_dir = pack_dir / "bundles" / b_name
        dry_p = b_dir / b_info["dry_file"]
        if not dry_p.exists():
            raise FileNotFoundError(f"Bundle dry file not found: {dry_p}")

        b_manifest_path = b_dir / "manifest.json"
        if not b_manifest_path.exists():
            raise FileNotFoundError(f"Bundle manifest not found: {b_manifest_path}")
        with open(b_manifest_path, "r", encoding="utf-8") as bmf:
            b_manifest = json.load(bmf)

        for s_info in b_manifest.get("stems", []):
            stem_filename = s_info["filename"]
            stem_p = b_dir / stem_filename
            if not stem_p.exists():
                raise FileNotFoundError(f"Bundle wet stem not found: {stem_p}")

            raw_tone = s_info.get("tone_name", "")
            raw_slug = raw_tone.lower().replace(" ", "_").replace("∕", "_").replace("/", "_")
            target_str = s_info.get("target", "")
            target_voicing = (
                target_str.split(":", 1)[1] if ":" in target_str else s_info.get("target_voicing", "")
            )
            canonical_voice = VOICE_SLUG_ALIASES.get(raw_slug, raw_slug)
            if canonical_voice not in VOICES:
                canonical_voice = VOICE_SLUG_ALIASES.get(target_voicing, target_voicing)

            if selected_voices is not None:
                matches_filter = raw_slug in selected_voices or canonical_voice in selected_voices
                if not matches_filter:
                    continue

            model_basename = stem_p.stem
            target_nam = nam_dir / f"{model_basename}.nam"

            # Check if this stem has an existing model in manifest_data["models"]
            existing_nam_path = None
            existing_m_entry = None
            if target_nam.exists():
                existing_nam_path = target_nam
                existing_m_entry = manifest_data["models"].get(target_nam.name)
            else:
                for m_k, m_v in manifest_data["models"].items():
                    m_b = m_v.get("bundle", "")
                    m_t = m_v.get("target", "")
                    m_tone = m_v.get("tone_name", "")
                    if m_b == b_name and (m_t == target_str or m_tone == raw_tone):
                        cand_path = nam_dir / m_k
                        if cand_path.exists():
                            existing_nam_path = cand_path
                            existing_m_entry = m_v
                            break

            if existing_nam_path is not None and not overwrite:
                # Stage 2: Compare digest tokens
                recorded_dry_digest = existing_m_entry.get("dry_digest_token") if existing_m_entry else None
                recorded_wet_digest = existing_m_entry.get("wet_digest_token") if existing_m_entry else None
                current_dry_digest = b_info.get("dry_digest_token")
                current_wet_digest = s_info.get("digest_token")

                if (
                    recorded_dry_digest
                    and recorded_wet_digest
                    and recorded_dry_digest == current_dry_digest
                    and recorded_wet_digest == current_wet_digest
                ):
                    print(
                        f"[Tone Pack Trainer] Audio digests match, skipping retraining: {existing_nam_path.name}"
                    )
                    total_skipped += 1
                    continue
                elif existing_nam_path == target_nam:
                    # Fallback check for legacy models before dual-tokens
                    is_invalid = False
                    try:
                        with open(target_nam, "r", encoding="utf-8") as f:
                            data = json.load(f)
                        arch = data.get("architecture")
                        if arch == "SlimmableContainer":
                            submodels = data.get("config", {}).get("submodels", [])
                            if len(submodels) < 2:
                                is_invalid = True
                        elif arch == "WaveNet":
                            if not data.get("weights"):
                                is_invalid = True
                        else:
                            is_invalid = True
                    except (json.JSONDecodeError, OSError):
                        is_invalid = True

                    if not is_invalid and not recorded_dry_digest and not recorded_wet_digest:
                        print(
                            f"[Tone Pack Trainer] Model already exists, skipping: {target_nam.name}"
                        )
                        total_skipped += 1
                        if target_nam.name not in manifest_data["models"]:
                            _populate_manifest_model_entry(
                                manifest_data["models"], target_nam, s_info, b_name, dry_info=b_info
                            )
                            with open(manifest_path, "w", encoding="utf-8") as f:
                                json.dump(manifest_data, f, indent=2)
                        continue

            print(
                f"\n[Tone Pack Trainer] Training model for {model_basename} (Bundle: {b_name})..."
            )
            src_vid = (
                b_info["source"].split(":", 1)[1]
                if "source" in b_info and ":" in b_info["source"]
                else b_info.get("source_voicing")
            )
            ok = train_voice(
                instrument=inst,
                voice=canonical_voice if canonical_voice in VOICES else target_voicing,
                source_voicing=src_vid,
                input_wav=dry_p,
                output_wav=stem_p,
                reference_wav=dry_p,
                models_dir=nam_dir,
                basename=model_basename,
                epochs=epochs,
                min_epochs=min_epochs,
                patience=patience,
                min_delta=min_delta,
                mse_weight=mse_weight,
                mrstft_weight=mrstft_weight,
                batch_size=batch_size,
                precision=precision,
                num_workers=num_workers,
                lr_scheduler=lr_scheduler,
                eta_min=eta_min,
                lr_t_max=lr_t_max,
                fast_dev_run=fast_dev_run,
                version_tag=None,
                no_manifest=False,
                include_identity=False,
                engine=engine,
                seed=seed,
                silent=False,
            )
            if not ok:
                print(f"[Tone Pack Trainer] Warning: Training failed for {target_nam.name}")
                continue

            total_trained += 1
            if target_nam.exists():
                if manifest_path.exists():
                    try:
                        with open(manifest_path, "r", encoding="utf-8") as f:
                            manifest_data = json.load(f)
                    except json.JSONDecodeError, OSError:
                        pass
                if "models" not in manifest_data or not isinstance(manifest_data["models"], dict):
                    manifest_data["models"] = {}
                _populate_manifest_model_entry(
                    manifest_data["models"], target_nam, s_info, b_name, dry_info=b_info
                )
                with open(manifest_path, "w", encoding="utf-8") as f:
                    json.dump(manifest_data, f, indent=2)

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2)

    print(f"\n[Tone Pack Trainer] Training completed for pack '{inst.name}':")
    print(f"  -> Model Directory: {nam_dir}")
    print(
        f"  -> Trained: {total_trained}, Skipped: {total_skipped}, Total Models Indexed: {len(manifest_data['models'])}"
    )
    return nam_dir
