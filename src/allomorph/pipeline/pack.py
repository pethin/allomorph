"""
Allomorph - Tone3000 Tone Pack Pipeline (DSP Gen 4 / v0.4.0)
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
    PickupBundle,
    load_instrument,
    partition_instrument_bundles,
)
from allomorph.config.scales import REPO_ROOT
from allomorph.config.schema import InstrumentConfig
from allomorph.dsp import read_wav
from allomorph.naming import get_default_input_path, get_t3k_basename
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


VOICE_CATALOG_DESCRIPTIONS: dict[str, dict[str, str]] = {
    # Precision Bass & Tone Shaper Family
    "precision_vintage": {
        "family": "Precision Bass & Tone Shaper Family",
        "name": "Precision Vintage",
        "desc": (
            "The gold standard: a vintage 1962 Precision Bass with tone wide open. "
            "Warm, woody low-mids, open harmonic bloom, and organic touch sensitivity "
            "that sits perfectly in any mix."
        ),
    },
    "precision_active": {
        "family": "Precision Bass & Tone Shaper Family",
        "name": "Precision Active",
        "desc": (
            "Modern active ceramic split-coil P-bass tone with 2-band preamp. "
            "Punchy, articulate, and wideband with authoritative low-end weight "
            "and bright transient clarity—built for modern rock, gospel, and mix cutting authority."
        ),
    },
    "precision_mids": {
        "family": "Precision Bass & Tone Shaper Family",
        "name": "Precision Mids",
        "desc": (
            "Vintage P-bass with a modern vocal low-mid punch. "
            "Cuts finger clatter and high-end sizzle while keeping a muscular, "
            "forward presence in a busy mix."
        ),
    },
    "precision_warm": {
        "family": "Precision Bass & Tone Shaper Family",
        "name": "Precision Warm",
        "desc": (
            "The legendary Motown flatwound sound (James Jamerson / Pino Palladino). "
            "Fat, pillowy sub-bass thump with rolled-off highs that glues the groove together."
        ),
    },
    "precision_dub": {
        "family": "Precision Bass & Tone Shaper Family",
        "name": "Precision Dub",
        "desc": (
            "Ultra-deep vintage '50s dub spec. "
            "Pure, earth-shaking sub-bass thump with zero top-end click—perfect "
            "for reggae, dub, hip-hop, and classic R&B."
        ),
    },
    # Jazz Bass Family
    "jazz_pair_active": {
        "family": "Jazz Bass Family",
        "name": "Jazz Pair Active",
        "desc": (
            "The quintessential NYC active slap tone: tight, massive sub-bass "
            "and crisp, glassy highs with a scooped midrange that pops right out of a dense mix."
        ),
    },
    "jazz_pair_open": {
        "family": "Jazz Bass Family",
        "name": "Jazz Pair Open",
        "desc": (
            "The golden-era '60s Jazz Bass sound. Both pickups wide open deliver "
            "that iconic hollow midrange scoop, warm low-mids, and smooth, singing top-end."
        ),
    },
    "jazz_pair_mids": {
        "family": "Jazz Bass Family",
        "name": "Jazz Pair Mids",
        "desc": (
            "Jazz Bass pair with a vocal midrange punch. Thickens up your fingerstyle attack "
            "and smooths out harsh fret clank while keeping every note articulate."
        ),
    },
    "jazz_bridge_growl": {
        "family": "Jazz Bass Family",
        "name": "Jazz Bridge Growl",
        "desc": (
            "The unmistakable Jaco Pastorius tone: rolling the neck volume back just a touch "
            "unlocks that singing, vocal, honky bridge growl without losing low-end body."
        ),
    },
    "jazz_bridge_open": {
        "family": "Jazz Bass Family",
        "name": "Jazz Bridge Open",
        "desc": (
            "Solo '60s bridge single-coil. Tight, laser-focused midrange bite and percussive "
            "attack, perfect for 16th-note funk and fast staccato fingerstyle."
        ),
    },
    "jazz_neck_warm": {
        "family": "Jazz Bass Family",
        "name": "Jazz Neck Warm",
        "desc": (
            "Solo '60s neck single-coil. Deep, warm, and round with a woody low-end bloom "
            "that rivals a P-bass while retaining single-coil openness."
        ),
    },
    # Music Man StingRay Family
    "stingray_parallel": {
        "family": "Music Man StingRay Family",
        "name": "StingRay Parallel",
        "desc": (
            "The iconic Music Man StingRay sound: thunderous low end, aggressive metallic slap "
            "clank, and that hollow midrange scoop heard on thousands of hit records."
        ),
    },
    "stingray_series": {
        "family": "Music Man StingRay Family",
        "name": "StingRay Series",
        "desc": (
            "StingRay wired in series for a hotter, thicker signal with punchy low-mids "
            "and an aggressive, in-your-face rock bark."
        ),
    },
    # P/J Bass Family
    "pj_active": {
        "family": "P/J Bass Family",
        "name": "PJ Active",
        "desc": (
            "Modern active P/J punch (Sadowsky/Spector style). Combines thick split-coil body "
            "with bridge single-coil cut, boosted by a powerful active 2-band EQ."
        ),
    },
    "pj_passive": {
        "family": "P/J Bass Family",
        "name": "PJ Passive",
        "desc": (
            "Classic '80s rock P/J tone (Fender Precision Special / Yamaha BB). "
            "Thick hard-rock punch with aggressive midrange bite and authoritative low-end grunt."
        ),
    },
    # P/MM Modern Hybrids
    "p_mm_parallel": {
        "family": "P/MM Modern Hybrids",
        "name": "P∕MM Parallel",
        "desc": (
            "Modern boutique session hybrid (Sandberg California / Lakland style). "
            "Blends the warm body of a P-bass neck with the punchy slap bite of a Music Man bridge humbucker."
        ),
    },
    "p_mm_series": {
        "family": "P/MM Modern Hybrids",
        "name": "P∕MM Series",
        "desc": (
            "P-bass neck and Music Man bridge wired in series for a massive wall of sound "
            "with immense low-mid thrust and heavy overdrive sustain."
        ),
    },
    # Progressive & Classic Rock Legends
    "rickenbacker_clank": {
        "family": "Progressive & Classic Rock Legends",
        "name": "Rickenbacker Clank",
        "desc": (
            "The quintessential prog-rock clank (Chris Squire / Geddy Lee). "
            "Tight, lean low end with biting, aggressive pick attack and cutting upper-mid grit."
        ),
    },
    "rickenbacker_open": {
        "family": "Progressive & Classic Rock Legends",
        "name": "Rickenbacker Open",
        "desc": (
            "Modern 4003 bridge tone with the vintage capacitor bypassed. "
            "Fuller low-end body and punch while retaining that signature biting Rickenbacker top-end cut."
        ),
    },
    "dingwall_bridge": {
        "family": "Progressive & Classic Rock Legends",
        "name": "Dingwall Bridge",
        "desc": (
            "Modern progressive metal tone: razor-sharp pick clank, extreme note clarity, "
            "and piano-like low-B definition tailored for drop tunings and heavy distortion."
        ),
    },
    "dingwall_middle": {
        "family": "Progressive & Classic Rock Legends",
        "name": "Dingwall Middle",
        "desc": (
            "FD3 middle pickup solo: focused, aggressive low-mid grunt and punchy growl "
            "with exceptional articulation across all registers."
        ),
    },
    "dingwall_parallel": {
        "family": "Progressive & Classic Rock Legends",
        "name": "Dingwall Parallel",
        "desc": (
            "FD3 bridge and middle coils in parallel: modern scooped aggression with ultra-fast "
            "transient attack, tight sub-bass, and pristine harmonic sparkle."
        ),
    },
    "mudbucker_deep": {
        "family": "Progressive & Classic Rock Legends",
        "name": "Mudbucker Deep",
        "desc": (
            "The legendary Gibson \"Mudbucker\": a dark, colossal wall of pure vintage "
            "bass rumble with zero top-end harshness—pure '60s and '70s British blues-rock."
        ),
    },
    # Acoustic Transducers
    "upright_acoustic": {
        "family": "Acoustic Transducers",
        "name": "Upright Acoustic",
        "desc": (
            "Acoustic upright double bass: transforms your electric bass into a woody, "
            "resonant acoustic upright with natural body thump and organic finger feel. "
            "Pair with 3 Sigma Audio \"Acoustic Upright Standard\" AST IRs in your cabinet "
            "loader for authentic soundboard acoustic bloom."
        ),
    },
}

VOICE_SLUG_ALIASES: dict[str, str] = {
    "vintage_open": "precision_vintage",
    "vintage_mids": "precision_mids",
    "vintage_warm": "precision_warm",
    "neck_active": "precision_active",
    "slab_dub": "precision_dub",
    "pair_open": "jazz_pair_open",
    "pair_mids": "jazz_pair_mids",
    "pair_active": "jazz_pair_active",
    "bridge_growl": "jazz_bridge_growl",
    "bridge_open": "jazz_bridge_open",
    "neck_warm": "jazz_neck_warm",
    "parallel_classic": "stingray_parallel",
    "series_punch": "stingray_series",
    "middle_growl": "dingwall_middle",
    "pmm_parallel": "p_mm_parallel",
    "pmm_series": "p_mm_series",
    "neck_deep": "mudbucker_deep",
    "bridge_piezo_acoustic": "upright_acoustic",
}

FAMILY_ORDER: list[str] = [
    "Precision Bass & Tone Shaper Family",
    "Jazz Bass Family",
    "Music Man StingRay Family",
    "P/J Bass Family",
    "P/MM Modern Hybrids",
    "Progressive & Classic Rock Legends",
    "Acoustic Transducers",
    "Studio Buffers & Dynamics",
]

VOICE_ORDER_IN_FAMILY: dict[str, list[str]] = {
    "Precision Bass & Tone Shaper Family": [
        "precision_vintage",
        "precision_active",
        "precision_mids",
        "precision_warm",
        "precision_dub",
    ],
    "Jazz Bass Family": [
        "jazz_pair_active",
        "jazz_pair_open",
        "jazz_pair_mids",
        "jazz_bridge_growl",
        "jazz_bridge_open",
        "jazz_neck_warm",
    ],
    "Music Man StingRay Family": [
        "stingray_parallel",
        "stingray_series",
    ],
    "P/J Bass Family": [
        "pj_active",
        "pj_passive",
    ],
    "P/MM Modern Hybrids": [
        "p_mm_parallel",
        "p_mm_series",
    ],
    "Progressive & Classic Rock Legends": [
        "rickenbacker_clank",
        "rickenbacker_open",
        "dingwall_bridge",
        "dingwall_middle",
        "dingwall_parallel",
        "mudbucker_deep",
    ],
    "Acoustic Transducers": [
        "upright_acoustic",
    ],
    "Studio Buffers & Dynamics": [
        "active_character",
        "passive_character",
    ],
}


def _family_sort_key(fam: str, inst: InstrumentConfig) -> tuple[int, int]:
    is_native = (
        ("jazz" in inst.id and "Jazz" in fam)
        or ("precision" in inst.id and "Precision" in fam)
        or ("standard_p" in inst.id and "Precision" in fam)
        or ("stingray" in inst.id and "StingRay" in fam)
    )
    order_idx = FAMILY_ORDER.index(fam) if fam in FAMILY_ORDER else 99
    return (1 if is_native else 0, order_idx)


def _target_sort_key(target_item: tuple[str, str, str, str]) -> int:
    slug, _, _, fam = target_item
    order_list = VOICE_ORDER_IN_FAMILY.get(fam, [])
    return order_list.index(slug) if slug in order_list else 99


def generate_storefront_description(
    inst: InstrumentConfig,
    bundles: dict[str, PickupBundle],
) -> str:
    """Generates standard Tone3000 storefront product listing description in raw text format."""
    total_targets = sum(len(b.targets) for b in bundles.values())

    target_items: list[tuple[str, str, str, str]] = []
    is_multi_pickup = len(bundles) > 1
    for b_name, b in bundles.items():
        pos_label = b.pickup.position_name or b_name.capitalize()
        pos_tag = pos_label if is_multi_pickup else None
        for t in b.targets:
            raw_tone = t.voicing.tone_name or t.voicing.name
            raw_slug = raw_tone.lower().replace(" ", "_").replace("∕", "_").replace("/", "_")
            slug = (
                raw_slug
                if raw_slug in VOICE_CATALOG_DESCRIPTIONS
                else VOICE_SLUG_ALIASES.get(
                    raw_slug, VOICE_SLUG_ALIASES.get(t.voicing_id, raw_slug)
                )
            )

            info = VOICE_CATALOG_DESCRIPTIONS.get(slug)
            if info:
                family = info["family"]
                display_name = info["name"]
                desc = info["desc"]
            else:
                family = "Other Target Voicings"
                display_name = raw_tone
                desc = f"{t.voicing.name} ({t.instrument.name})"

            target_v_tag = resolve_tri_part_version(
                DSP_GENERATION,
                t.instrument.version,
                t.voicing.version,
            )
            stem_base = get_t3k_basename(
                tone_name=display_name,
                position_name=pos_tag,
                version_tag=target_v_tag,
            )

            target_items.append((slug, stem_base, desc, family))

    grouped: dict[str, list[tuple[str, str, str, str]]] = {}
    for item in target_items:
        grouped.setdefault(item[3], []).append(item)

    sorted_families = sorted(grouped.keys(), key=lambda f: _family_sort_key(f, inst))

    for fam in sorted_families:
        grouped[fam].sort(key=_target_sort_key)

    lines: list[str] = [
        f"ALLOMORPH: {inst.name.upper()} EDITION",
        "",
        f"Transform your {inst.name} into {total_targets} legendary active, vintage, modern, and acoustic pickup configurations.",
        "",
        "",
        "OVERVIEW",
        "",
        (
            f"Allomorph turns your {inst.name} into a versatile tonal chameleon. "
            "Instead of modeling an amplifier or speaker cabinet, Allomorph sits at the very beginning "
            "of your signal chain (Block 1 on the Darkglass Anagram) to reshape the actual sound, "
            "pickup response, and feel of your instrument."
        ),
        "",
        (
            f"With this pack, your {inst.name} can instantly deliver:\n"
            "• The iconic scoop of vintage '60s Jazz Basses and aggressive punch of Music Man StingRays\n"
            "• Modern active slap preamps, boutique P/MM hybrids, and cutting progressive rock clank\n"
            "• Warm Motown flatwound thump, deep reggae dub sub-bass, and pristine modern ceramic punch\n"
            "• Acoustic upright double bass resonance for jazz and acoustic sessions\n"
            "• Pure studio buffer clarity or organic vintage tube-friendly passive warmth"
        ),
        "",
    ]

    if len(inst.pickups) == 1:
        p_name = next(iter(inst.pickups.values())).name
        lines.append(
            f"Calibrated specifically for the {inst.name} with {p_name}. "
            f"Plug in, select a model, and hear your {inst.name} transform into a completely different instrument in real time with zero latency."
        )
    else:
        lines.append(
            f"Calibrated specifically for the {inst.name}. "
            f"Plug in, select a model, and hear your {inst.name} transform into a completely different instrument in real time with zero latency."
        )

    lines.extend(
        [
            "",
            "",
            "RECOMMENDED SIGNAL CHAIN",
            "",
            f"[{inst.name}]",
            "  -> [Block 1: Allomorph Pickup Twin] (Input / Pickup Front-End)",
            "  -> [Block 2: Amp / Drive / Preamp]  (Darkglass B7K, SVT, B-15, etc.)",
            "  -> [Block 3: Speaker Cabinet IR]    (8x10, 4x10, 1x15, etc.)",
            "  -> [FOH / Audio Interface / DAW]",
            "",
            (
                "Why Block 1? Overdrive pedals and amps react directly to the resonant character of your pickups. "
                "Putting Allomorph first ensures downstream pedals and amp models distort the authentic growl of a Jazz bridge pickup, "
                "the clank of a StingRay, or the scoop of dual single-coils, rather than a generic DI signal."
            ),
            "",
            "",
            "QUICK INSTRUMENT SETUP",
            "",
        ]
    )

    is_active = (inst.electronics == "active") or any(
        getattr(p, "type", "") == "active" for p in inst.pickups.values()
    )

    if len(bundles) > 1:
        unique_tags = list(
            dict.fromkeys(
                f"[{b.pickup.position_name or b_name.capitalize()}]"
                for b_name, b in bundles.items()
            )
        )
        tags_str = ", ".join(unique_tags)
        lines.append(
            f"For each voicing, set your bass's physical pickup controls to the recommended bracketed setting {tags_str}:"
        )
        lines.append("")
        seen_positions: set[str] = set()
        for b_name, b in bundles.items():
            pos_label = b.pickup.position_name or b_name.capitalize()
            if pos_label not in seen_positions:
                seen_positions.add(pos_label)
                lines.append(f"• [{pos_label}]: Select {b.pickup.name}")
        if is_active:
            lines.append("• Onboard Active EQ (Bass, Mid, Treble): Set to Center Detents (Flat / 0 dB)")
            lines.append("• Master Volume Knob: 100% (Wide Open)")
        else:
            lines.append("• Master Tone Knob: 100% (Wide Open across all models)")
            lines.append("• Physical Volume Knob: 100% (Wide Open)")
    else:
        lines.append("• Physical Volume Knob: 100% (Wide Open)")
        if is_active:
            lines.append("• Onboard Active EQ (Bass, Mid, Treble): Set to Center Detents (Flat / 0 dB)")
        else:
            lines.append("• Physical Tone Knob: 100% (Wide Open)")

    lines.extend(
        [
            "",
            "Plug straight in. All pickup positioning, pot loading, tone capacitor shaping, and active preamp EQs are handled automatically by the models.",
            "",
            "",
            f"THE {total_targets} DIGITAL TWIN VOICINGS",
            "",
        ]
    )

    count = 1
    for fam in sorted_families:
        lines.append("")
        lines.append(fam)
        lines.append("")
        for slug, stem_base, desc, _ in grouped[fam]:
            num_str = f"{count:02d}."
            lines.append(f"{num_str} {stem_base}")
            lines.append(desc)
            lines.append("")
            count += 1

    lines.extend(
        [
            "",
            "LICENSE & DISCLAIMER",
            "",
            "License:",
            (
                "Licensed for personal and commercial musical performances, recording, and production. "
                "Redistribution, sublicensing, or resale of raw model files or derived profiles without "
                "permission is strictly prohibited."
            ),
            "",
            "Trademark Disclaimer:",
            (
                "All product names, trademarks, and registered trademarks (such as Fender, Precision Bass, "
                "P-Bass, Jazz Bass, J-Bass, Music Man, StingRay, Sterling, Rickenbacker, Gibson, Dingwall, "
                "Bartolini, Sadowsky, Spector, ToneStyler, EMG, and Darkglass) are the property of their "
                "respective owners. Used solely to identify historical instruments and analog circuits "
                "analyzed in this project. Allomorph is an independent project and is not affiliated with or "
                "endorsed by any of these manufacturers."
            ),
            "",
        ]
    )

    return "\n".join(lines).strip() + "\n"


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

    def _process_bundle(item: tuple[str, PickupBundle]) -> tuple[str, dict[str, Any]]:
        b_name, bundle = item
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
        source_wet_path = WET_AUDIO_DIR / inst.id / f"{bundle.pickup_key}.wav"

        # Clean up any previous wav files in bundle directory to avoid stale stems
        for old_wav in b_dir.glob("*.wav"):
            old_wav.unlink(missing_ok=True)

        if (
            max_samples is None
            and is_wet_stem_valid(
                source_wet_path,
                base_dry_path=input_wav,
                expected_version=dry_v_tag,
            )
        ):
            shutil.copyfile(source_wet_path, dry_dest)
        elif max_samples is None:
            source_wet_path.parent.mkdir(parents=True, exist_ok=True)
            simulate_instrument_voicing(
                instrument=inst,
                voicing=bundle.pickup_key,
                input_wav=input_wav,
                output_wav=source_wet_path,
                max_samples=None,
                force=overwrite,
            )
            shutil.copyfile(source_wet_path, dry_dest)
        else:
            simulate_instrument_voicing(
                instrument=inst,
                voicing=bundle.pickup_key,
                input_wav=input_wav,
                output_wav=dry_dest,
                max_samples=max_samples,
                force=True,
            )

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

        base_dry_p = Path(input_wav) if input_wav else get_default_input_path()
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

        bundle_entry = {
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
        return b_name, bundle_entry

    eff_jobs = jobs if jobs is not None and jobs > 0 else 1
    if eff_jobs > 1 and len(bundles) > 1:
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=eff_jobs) as executor:
            bundle_results = list(executor.map(_process_bundle, bundles.items()))
    else:
        bundle_results = [_process_bundle(item) for item in bundles.items()]

    for b_name, b_meta in bundle_results:
        manifest_entries["bundles"][b_name] = b_meta

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
