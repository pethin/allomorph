"""
Allomorph - Tone3000 Tone Pack Pipeline (DSP Gen 3 / v0.3.0)
Automates creation, partitioning, packaging, and local training of Tone3000 upload bundles
respecting Tone3000's strict '1 Dry File + Multiple Wet Stems' batch upload constraint.
"""

import hashlib
import json
import shutil
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from allomorph.circuit.forward import simulate_instrument_voicing
from allomorph.circuit.staging import export_instrument_pickup_wav
from allomorph.config.instruments import (
    PickupBundle,
    load_instrument,
    partition_instrument_bundles,
)
from allomorph.config.scales import REPO_ROOT
from allomorph.config.schema import InstrumentConfig
from allomorph.naming import get_optimal_dry_path, get_t3k_basename
from allomorph.version import (
    ALLOMORPH_VERSION,
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


def generate_storefront_description(
    inst: InstrumentConfig,
    bundles: dict[str, PickupBundle],
) -> str:
    """Generates standard Tone3000 storefront product listing description."""
    title = f"ALLOMORPH: {inst.name.upper()} EDITION"
    total_targets = sum(len(b.targets) for b in bundles.values())

    lines: list[str] = [
        title,
        "=" * len(title),
        "",
        f"Transform your {inst.name} into {total_targets} legendary active, vintage, modern, and acoustic pickup configurations.",
        "",
        "OVERVIEW",
        "--------",
        f"Allomorph turns your {inst.name} into a versatile tonal chameleon.",
        "Instead of modeling an amplifier or speaker cabinet, Allomorph sits at the very beginning",
        "of your signal chain (Block 1 on the Darkglass Anagram) to reshape the actual sound,",
        "pickup response, and feel of your instrument.",
        "",
        "RECOMMENDED SIGNAL CHAIN",
        "------------------------",
        f"[{inst.name}]",
        "  -> [Block 1: Allomorph NAM Preamp]  (High-Impedance Pickup Front-End)",
        "  -> [Block 2: Amp / Drive / Preamp]   (Darkglass B7K, SVT, B-15, etc.)",
        "  -> [Block 3: Speaker Cabinet IR]     (8x10, 4x10, 1x15, etc.)",
        "  -> [FOH / Audio Interface / DAW]",
        "",
        "QUICK INSTRUMENT SETUP",
        "----------------------",
        "• Master Tone Knob: 100% (Wide Open across all models)",
    ]

    if len(bundles) > 1:
        lines.append("• Pickup Selector: Set your bass switch to match the bracketed tag:")
        for b_name, b in bundles.items():
            pos_label = b.pickup.position_name or b_name.capitalize()
            lines.append(f"   - [{pos_label}]: Select {b.pickup.name}")
    else:
        lines.append(
            "• Single-Pickup Bass: Plug straight in. All pickup placement and pot loading are modeled."
        )

    lines.extend(
        [
            "",
            f"INCLUDED VOICINGS ({total_targets} MODELS)",
            "-----------------------------------------",
        ]
    )

    for b_name, b in bundles.items():
        pos_label = b.pickup.position_name or b_name.capitalize()
        lines.append(f"\n### Bundle: {pos_label} ({len(b.targets)} Voicings)")
        for t in b.targets:
            t_name = t.voicing.tone_name or t.voicing.name
            tag = f" [{pos_label}]" if len(bundles) > 1 else ""
            t_ver = resolve_tri_part_version(
                DSP_GENERATION, t.instrument.version, t.voicing.version
            )
            lines.append(f"• {t_name}{tag} {t_ver}: {t.voicing.name} ({t.instrument.name})")

    lines.extend(
        [
            "",
            "TECHNICAL SPECIFICATIONS",
            "------------------------",
            f"• DSP Generation: {DSP_GENERATION} (v{ALLOMORPH_VERSION})",
            "• Sample Rate: 48 kHz / 24-bit PCM",
            "• Latency: 0 samples (Pure causal minimum-phase FIR)",
            "• Headroom: Calibrated to -0.09 dBFS ceiling",
            "• Host Compatibility: Darkglass Anagram Block 1, Neural Amp Modeler (NAM), DAW Plugin Hosts",
        ]
    )

    return "\n".join(lines) + "\n"


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
    models_dir = pack_dir / "models"
    pack_dir.mkdir(parents=True, exist_ok=True)
    bundles_dir.mkdir(parents=True, exist_ok=True)
    if train:
        models_dir.mkdir(parents=True, exist_ok=True)

    # 1. Partition instrument into pickup bundles
    bundles = partition_instrument_bundles(inst, catalog_targets=catalog_targets)

    # 2. Process each bundle
    manifest_entries: dict[str, Any] = {
        "instrument_id": inst.id,
        "instrument_name": inst.name,
        "instrument_version": inst.version,
        "version": resolve_tri_part_version(DSP_GENERATION, inst.version, 1),
        "allomorph_version": ALLOMORPH_VERSION,
        "dsp_generation": DSP_GENERATION,
        "bundles": {},
    }

    is_multi_pickup = len(bundles) > 1

    for b_name, bundle in bundles.items():
        b_dir = bundles_dir / b_name
        b_dir.mkdir(parents=True, exist_ok=True)

        pos_label = bundle.pickup.position_name or b_name.capitalize()

        # Determine source voicing version for this bundle
        source_voice_ver = 1
        for v in inst.voicings.values():
            if v.pickup == bundle.pickup_key:
                source_voice_ver = getattr(v, "version", 1)
                break

        # A. Synthesize / Copy Source Audio for this bundle (Tone3000 upload contract requires dry v[dsp].[inst].[voicing].wav)
        dry_v_tag = resolve_tri_part_version(DSP_GENERATION, inst.version, source_voice_ver)
        dry_filename = f"dry {dry_v_tag}.wav"
        dry_dest = b_dir / dry_filename
        canonical_source_path = WET_AUDIO_DIR / inst.id / f"{bundle.pickup_key}.wav"

        # If overwriting, clean up any previous wav files in bundle directory
        if overwrite:
            for old_wav in b_dir.glob("*.wav"):
                old_wav.unlink(missing_ok=True)

        if (
            max_samples is None
            and is_wet_stem_valid(
                canonical_source_path,
                base_dry_path=input_wav,
                expected_version=dry_v_tag,
            )
            and not overwrite
        ):
            shutil.copyfile(canonical_source_path, dry_dest)
        elif max_samples is None:
            raw_source_path = export_instrument_pickup_wav(
                inst_id=inst.id,
                pickup_key=bundle.pickup_key,
                input_wav=input_wav,
                output_dir=WET_AUDIO_DIR / inst.id,
                version_tag=dry_v_tag,
                no_manifest=False,
                max_samples=None,
            )
            shutil.copyfile(raw_source_path, dry_dest)
        else:
            source_tmp_dir = b_dir / "_source_tmp"
            raw_source_path = export_instrument_pickup_wav(
                inst_id=inst.id,
                pickup_key=bundle.pickup_key,
                input_wav=input_wav,
                output_dir=source_tmp_dir,
                version_tag=dry_v_tag,
                no_manifest=True,
                max_samples=max_samples,
            )
            shutil.copyfile(raw_source_path, dry_dest)
            shutil.rmtree(source_tmp_dir, ignore_errors=True)

        dry_sha256 = _sha256_file(dry_dest)

        # B. Wet Audio Stems (Copy cached wet audio from audio/wet or compile then copy)
        bundle_stems: list[dict[str, Any]] = []
        stem_filenames: list[str] = []

        for target_ref in bundle.targets:
            tone_name = target_ref.voicing.tone_name or target_ref.voicing.name
            pos_tag = pos_label if is_multi_pickup else None
            target_v_tag = resolve_tri_part_version(
                DSP_GENERATION,
                target_ref.instrument.version,
                target_ref.voicing.version,
            )
            stem_base = get_t3k_basename(
                tone_name=tone_name,
                position_name=pos_tag,
                version_tag=target_v_tag,
            )
            stem_filename = f"{stem_base}.wav"
            stem_path = b_dir / stem_filename

            target_slug = (
                target_ref.voicing.tone_name.lower()
                .replace(" ", "_")
                .replace("∕", "_")
                .replace("/", "_")
                if target_ref.voicing.tone_name
                else (target_ref.voicing.id or "default_voicing")
            )
            canonical_wet_path = WET_AUDIO_DIR / target_ref.instrument_id / f"{target_slug}.wav"

            if max_samples is None:
                if overwrite or not is_wet_stem_valid(
                    canonical_wet_path,
                    base_dry_path=input_wav,
                    expected_version=target_v_tag,
                ):
                    canonical_wet_path.parent.mkdir(parents=True, exist_ok=True)
                    simulate_instrument_voicing(
                        instrument=target_ref.instrument,
                        voicing=target_ref.voicing,
                        input_wav=input_wav,
                        output_wav=canonical_wet_path,
                        max_samples=None,
                    )
                shutil.copyfile(canonical_wet_path, stem_path)
            else:
                simulate_instrument_voicing(
                    instrument=target_ref.instrument,
                    voicing=target_ref.voicing,
                    input_wav=input_wav,
                    output_wav=stem_path,
                    max_samples=max_samples,
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
        instr_content = [
            "=" * 78,
            f"TONE3000 UPLOAD BUNDLE: {b_name.upper()}",
            "=" * 78,
            f"Source Instrument: {inst.name}",
            f"Pickup Position: {pos_label}",
            f"Physical Switch Setting on Bass: {pos_label}",
            "Tone Knob Setting: 100% (Wide Open)",
            "",
            "STEP-BY-STEP UPLOAD INSTRUCTIONS (Tone3000 Studio Trainer):",
            "1. Open Tone3000 Trainer in your browser or local batch tool.",
            f"2. DRAG DRY FILE: Drag '{dry_filename}' from this directory into the '1 Dry File' slot.",
            f"3. DRAG WET STEMS: Select and drag the following {len(stem_filenames)} .wav files into the 'Wet Stems' slot:",
        ]
        for fn in stem_filenames:
            instr_content.append(f"   • {fn}")
        instr_content.extend(
            [
                "",
                "4. START TRAINING: Tone3000 will batch-train all models against this single dry baseline.",
                "=" * 78,
            ]
        )
        instructions_path.write_text("\n".join(instr_content) + "\n", encoding="utf-8")

        base_dry_p = Path(input_wav) if input_wav else get_optimal_dry_path()
        base_dry_sha = compute_file_sha256(base_dry_p) if base_dry_p.exists() else None

        bundle_manifest = {
            "bundle": b_name,
            "pickup_key": bundle.pickup_key,
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
        with open(b_dir / "manifest.json", "w", encoding="utf-8") as bf:
            json.dump(bundle_manifest, bf, indent=2)

        manifest_entries["bundles"][b_name] = {
            "pickup_key": bundle.pickup_key,
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

    return pack_dir
