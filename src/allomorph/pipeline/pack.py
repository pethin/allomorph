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
    partition_instrument_bundles,
)
from allomorph.config.scales import REPO_ROOT
from allomorph.config.schema import InstrumentConfig
from allomorph.dsp import read_wav
from allomorph.naming import get_default_input_path
from allomorph.pipeline.bundle import (
    assemble_root_pack_manifest,
    build_bundle_manifest_entry,
    generate_bundle_upload_instructions,
    resolve_bundle_target_file_descriptors,
)
from allomorph.pipeline.storefront import (
    VOICE_SLUG_ALIASES,
    generate_storefront_description,
)
from allomorph.version import (
    DSP_GENERATION,
    compute_file_sha256,
    is_wet_stem_valid,
    resolve_tri_part_version,
)

AUDIO_DIR = REPO_ROOT / "audio"
WET_AUDIO_DIR = AUDIO_DIR / "wet"
PACKS_DIR = REPO_ROOT / "tone3000" / "packs"
ASSETS_DIR = REPO_ROOT / "tone3000" / "assets"
DOCS_DIR = REPO_ROOT / "tone3000" / "docs"


def _sha256_file(path: Path) -> str:
    """Calculates SHA256 hex digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()



def export_tone_pack(
    instrument: InstrumentConfig | str | Path,
    output_dir: Path | str | None = None,
    input_wav: Path | str | None = None,
    max_samples: int | None = None,
    catalog_targets: Sequence[tuple[str, str]] | None = None,
    jobs: int | None = None,
    train: bool = False,
    overwrite: bool = False,
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
    pack_dir.mkdir(parents=True, exist_ok=True)
    bundles_dir.mkdir(parents=True, exist_ok=True)
    if train:
        nam_dir.mkdir(parents=True, exist_ok=True)

    # 1. Partition instrument into pickup bundles
    bundles = partition_instrument_bundles(inst, catalog_targets=catalog_targets)

    manifest_path = pack_dir / "manifest.json"
    if not overwrite and manifest_path.exists():
        all_bundles_exist = all(
            (bundles_dir / b_name / "manifest.json").exists()
            and (bundles_dir / b_name / "upload_instructions.txt").exists()
            for b_name in bundles
        )
        if all_bundles_exist:
            print(
                f"\n[Tone Pack Exporter] Tone Pack already exists at '{pack_dir}'. Pass overwrite=True (or --force) to force repack."
            )
            return pack_dir

    # 2. Process each bundle
    is_multi_pickup = len(bundles) > 1

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

        # A. Synthesize / Copy Source Audio for this bundle (Tone3000 upload contract requires dry v[dsp].[inst].[voicing].wav)
        dry_v_tag = resolve_tri_part_version(DSP_GENERATION, inst.version, source_voice_ver)
        dry_filename = f"dry {dry_v_tag}.wav"
        dry_dest = b_dir / dry_filename
        source_wet_path = WET_AUDIO_DIR / inst.id / f"{src_v_id}.wav"

        # Clean up any previous wav files in bundle directory to avoid stale stems
        for old_wav in b_dir.glob("*.wav"):
            old_wav.unlink(missing_ok=True)

        if max_samples is None and is_wet_stem_valid(
            source_wet_path,
            base_dry_path=input_wav,
            expected_version=dry_v_tag,
        ):
            shutil.copyfile(source_wet_path, dry_dest)
        elif max_samples is None:
            source_wet_path.parent.mkdir(parents=True, exist_ok=True)
            simulate_instrument_voicing(
                instrument=inst,
                voicing=src_v,
                input_wav=input_wav,
                output_wav=source_wet_path,
                max_samples=None,
                force=False,
            )
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

        dry_sha256 = _sha256_file(dry_dest)

        # B. Wet Audio Stems (Copy cached wet audio from audio/wet or compile then copy)
        bundle_stems: list[dict[str, Any]] = []
        stem_filenames: list[str] = []
        descriptors = resolve_bundle_target_file_descriptors(bundle, is_multi_pickup)

        for desc in descriptors:
            target_ref = desc["target_ref"]
            tone_name = desc["tone_name"]
            target_v_tag = desc["target_v_tag"]
            stem_filename = desc["stem_filename"]
            stem_path = b_dir / stem_filename
            target_slug = desc["target_slug"]
            target_wet_path = WET_AUDIO_DIR / target_ref.instrument_id / f"{target_slug}.wav"

            if max_samples is None:
                if not is_wet_stem_valid(
                    target_wet_path,
                    base_dry_path=input_wav,
                    expected_version=target_v_tag,
                ):
                    target_wet_path.parent.mkdir(parents=True, exist_ok=True)
                    simulate_instrument_voicing(
                        instrument=target_ref.instrument,
                        voicing=target_ref.voicing,
                        input_wav=input_wav,
                        output_wav=target_wet_path,
                        max_samples=None,
                    )
                shutil.copyfile(target_wet_path, stem_path)
            else:
                simulate_instrument_voicing(
                    instrument=target_ref.instrument,
                    voicing=target_ref.voicing,
                    input_wav=input_wav,
                    output_wav=stem_path,
                    max_samples=max_samples,
                )

            # Tone3000 Zero-Collision Guardrail: verify wet stem is not bit-exact identical to dry file
            dry_audio, _ = read_wav(dry_dest)
            wet_audio, _ = read_wav(stem_path)
            min_len = min(len(dry_audio), len(wet_audio))
            max_diff = float(np.max(np.abs(wet_audio[:min_len] - dry_audio[:min_len])))
            if np.array_equal(wet_audio[:min_len], dry_audio[:min_len]) or max_diff < 1e-10:
                raise ValueError(
                    f"Identity stem collision detected: target '{target_ref.voicing_id}' on '{target_ref.instrument_id}' "
                    f"is bit-exact identical to dry stem in bundle '{b_name}' on '{inst.id}' (max_diff={max_diff:.8f}). "
                    "Tone3000 will reject training on identical dry/wet stems."
                )

            stem_sha256 = _sha256_file(stem_path)
            stem_filenames.append(stem_filename)
            bundle_stems.append(
                {
                    "filename": stem_filename,
                    "target_instrument": target_ref.instrument_id,
                    "target_instrument_version": target_ref.instrument.version,
                    "target_voicing": target_ref.voicing_id,
                    "target_voicing_version": target_ref.voicing.version,
                    "version": target_v_tag,
                    "tone_name": tone_name,
                    "sha256": stem_sha256,
                }
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
        base_dry_sha = compute_file_sha256(base_dry_p) if base_dry_p.exists() else None

        bundle_manifest = build_bundle_manifest_entry(
            bundle_name=b_name,
            position_name=pos_label,
            source_voicing_id=src_v_id,
            dry_filename=dry_filename,
            dry_version=dry_v_tag,
            dry_sha256=dry_sha256,
            instrument_id=inst.id,
            instrument_version=inst.version,
            source_voicing_version=source_voice_ver,
            bundle_stems=bundle_stems,
            base_dry_file=base_dry_p.name,
            base_dry_sha256=base_dry_sha,
        )
        with open(b_dir / "manifest.json", "w", encoding="utf-8") as bf:
            json.dump(bundle_manifest, bf, indent=2)

        bundle_entry = {
            "source_voicing": src_v_id,
            "position_name": pos_label,
            "base_dry_file": base_dry_p.name,
            "base_dry_sha256": base_dry_sha,
            "dry_file": dry_filename,
            "dry_version": dry_v_tag,
            "dry_sha256": dry_sha256,
            "instrument_id": inst.id,
            "instrument_version": inst.version,
            "source_voicing_version": source_voice_ver,
            "stem_count": len(bundle_stems),
            "stems": bundle_stems,
        }
        return b_name, bundle_entry

    eff_jobs = jobs if jobs is not None and jobs > 0 else 1
    if eff_jobs > 1 and len(bundles) > 1:
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=eff_jobs) as executor:
            bundle_results = list(executor.map(_process_bundle, bundles.items()))
    else:
        bundle_results = [_process_bundle(item) for item in bundles.items()]

    manifest_entries = assemble_root_pack_manifest(
        inst=inst,
        bundle_entries={b_name: b_meta for b_name, b_meta in bundle_results},
        dsp_generation=DSP_GENERATION,
    )

    # 3. Storefront Description
    desc_path = pack_dir / "storefront_description.txt"
    desc_text = generate_storefront_description(inst, bundles)
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

    # 5. Manifest JSON
    manifest_path = pack_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_entries, f, indent=2)

    print(f"\n[Tone Pack Exporter] Successfully exported Tone Pack for '{inst.name}' to:")
    print(f"  -> {pack_dir}")
    print(f"  -> Bundles: {list(bundles.keys())}")
    print(f"  -> Total Stems: {sum(len(b['stems']) for b in manifest_entries['bundles'].values())}")

    if train:
        train_tone_pack(
            pack=inst,
            overwrite=overwrite,
        )

    return pack_dir


def _populate_manifest_model_entry(
    models_dict: dict[str, Any],
    nam_path: Path,
    stem_info: dict[str, Any],
    bundle_name: str,
) -> None:
    """Reads .nam container and populates metadata entry in pack manifest."""
    sha256 = _sha256_file(nam_path)
    entry: dict[str, Any] = {
        "filename": nam_path.name,
        "bundle": bundle_name,
        "target_instrument": stem_info.get("target_instrument"),
        "target_instrument_version": stem_info.get("target_instrument_version"),
        "target_voicing": stem_info.get("target_voicing"),
        "target_voicing_version": stem_info.get("target_voicing_version"),
        "version": stem_info.get("version"),
        "tone_name": stem_info.get("tone_name"),
        "sha256": sha256,
    }
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
            ]:
                if field in training_meta and training_meta[field] is not None:
                    entry[field] = training_meta[field]
    except (json.JSONDecodeError, OSError) as e:
        print(
            f"[Tone Pack Trainer] Warning: Could not read training metadata from {nam_path.name}: {e}"
        )

    models_dict[nam_path.name] = entry


def train_tone_pack(
    pack: InstrumentConfig | str | Path,
    voice: str = "all",
    overwrite: bool = False,
    epochs: int = 40,
    min_epochs: int = 20,
    patience: int = 8,
    min_delta: float = 1.0e-6,
    batch_size: int | str = "auto",
    precision: str = "auto",
    num_workers: int | str = "auto",
    lr_scheduler: str = "cosine",
    eta_min: float = 1e-5,
    lr_t_max: int = 40,
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
        export_tone_pack(inst, output_dir=pack_dir, overwrite=overwrite)

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
            for s_info in b_info.get("stems", []):
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
        export_tone_pack(inst, output_dir=pack_dir, overwrite=True)
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

        for s_info in b_info.get("stems", []):
            stem_filename = s_info["filename"]
            stem_p = b_dir / stem_filename
            if not stem_p.exists():
                raise FileNotFoundError(f"Bundle wet stem not found: {stem_p}")

            raw_tone = s_info.get("tone_name", "")
            raw_slug = raw_tone.lower().replace(" ", "_").replace("∕", "_").replace("/", "_")
            target_voicing = s_info.get("target_voicing", "")
            canonical_voice = VOICE_SLUG_ALIASES.get(target_voicing, target_voicing)
            if canonical_voice not in VOICES:
                canonical_voice = VOICE_SLUG_ALIASES.get(raw_slug, raw_slug)

            if selected_voices is not None:
                matches_filter = (
                    target_voicing in selected_voices
                    or canonical_voice in selected_voices
                    or raw_slug in selected_voices
                )
                if not matches_filter:
                    continue

            model_basename = stem_p.stem
            target_nam = nam_dir / f"{model_basename}.nam"

            if target_nam.exists() and not overwrite:
                # Verify that existing model is a valid dual-tier slimmable container
                is_legacy = False
                try:
                    with open(target_nam, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    submodels = data.get("config", {}).get("submodels", [])
                    if len(submodels) < 2:
                        is_legacy = True
                except (json.JSONDecodeError, OSError):
                    is_legacy = True

                if is_legacy:
                    print(
                        f"[Tone Pack Trainer] Removing legacy single-tier model to upgrade to official slimmable A2: {target_nam.name}"
                    )
                    try:
                        target_nam.unlink(missing_ok=True)
                    except OSError:
                        pass
                else:
                    print(f"[Tone Pack Trainer] Model already exists, skipping: {target_nam.name}")
                    total_skipped += 1
                    if target_nam.name not in manifest_data["models"]:
                        _populate_manifest_model_entry(
                            manifest_data["models"], target_nam, s_info, b_name
                        )
                        with open(manifest_path, "w", encoding="utf-8") as f:
                            json.dump(manifest_data, f, indent=2)
                    continue

            print(
                f"\n[Tone Pack Trainer] Training model for {model_basename} (Bundle: {b_name})..."
            )
            ok = train_voice(
                instrument=inst,
                voice=canonical_voice if canonical_voice in VOICES else target_voicing,
                source_voicing=b_info.get("source_voicing"),
                input_wav=dry_p,
                output_wav=stem_p,
                reference_wav=dry_p,
                models_dir=nam_dir,
                basename=model_basename,
                epochs=epochs,
                min_epochs=min_epochs,
                patience=patience,
                min_delta=min_delta,
                batch_size=batch_size,
                precision=precision,
                num_workers=num_workers,
                lr_scheduler=lr_scheduler,
                eta_min=eta_min,
                lr_t_max=lr_t_max,
                fast_dev_run=fast_dev_run,
                version_tag=None,
                no_manifest=True,
                include_identity=False,
                engine=engine,
                seed=seed,
            )
            if not ok:
                print(f"[Tone Pack Trainer] Warning: Training failed for {target_nam.name}")
                continue

            total_trained += 1
            if target_nam.exists():
                _populate_manifest_model_entry(manifest_data["models"], target_nam, s_info, b_name)
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
