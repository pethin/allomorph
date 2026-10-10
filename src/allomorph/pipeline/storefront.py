"""Storefront description generation and copy formatting for Tone3000 storefronts.

Provides pure, testable section builders for Tone3000 product descriptions:
overview, Darkglass Anagram Block 1 signal chain, instrument physical setup,
catalog listings with zero-scroll basenames, and legal disclaimers.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

from allomorph.naming import get_t3k_basename
from allomorph.version import DSP_GENERATION, resolve_tri_part_version

if TYPE_CHECKING:
    from allomorph.config.instruments import VoicingBundle
    from allomorph.config.schema import InstrumentConfig


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
            'The legendary Gibson "Mudbucker": a dark, colossal wall of pure vintage '
            "bass rumble with zero top-end harshness—pure '60s and '70s British blues-rock."
        ),
    },
    # Acoustic Transducers
    "upright_piezo": {
        "family": "Acoustic Transducers",
        "name": "Upright Piezo",
        "desc": (
            "Acoustic upright double bass bridge piezo: captures the authentic high-tension attack bite, "
            "bridge rocking compliance, and wood-mass roll-off of an acoustic double bass bridge piezo transducer, "
            "designed specifically as an ultra-high-Z front-end before acoustic IRs (such as 3 Sigma Audio AST "
            "Double Bass IRs) and preamp stages."
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
    "bridge_piezo": "upright_piezo",
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
        "upright_piezo",
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


def build_storefront_overview(inst: InstrumentConfig, total_targets: int) -> list[str]:
    """Builds the header and overview section of the Tone3000 storefront description."""
    lines = [
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
            "• Acoustic upright double bass bridge piezo bite for jazz and acoustic sessions\n"
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

    return lines


def build_storefront_signal_chain(inst: InstrumentConfig) -> list[str]:
    """Builds the Darkglass Anagram signal chain recommendation section."""
    return [
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


def build_storefront_instrument_setup(
    inst: InstrumentConfig,
    bundles: Mapping[str, VoicingBundle],
) -> list[str]:
    """Builds physical knob and switch setup guidance based on electronics and pickup count."""
    lines: list[str] = []
    is_active = (inst.electronics == "active") or any(
        getattr(p, "type", "") == "active" for p in inst.pickups.values()
    )

    if len(bundles) > 1:
        unique_tags = list(
            dict.fromkeys(
                f"[{b.position_name or b_name.capitalize()}]" for b_name, b in bundles.items()
            )
        )
        tags_str = ", ".join(unique_tags)
        lines.append(
            f"For each voicing, set your bass's physical pickup controls to the recommended bracketed setting {tags_str}:"
        )
        lines.append("")
        seen_positions: set[str] = set()
        for b_name, b in bundles.items():
            pos_label = b.position_name or b_name.capitalize()
            if pos_label not in seen_positions:
                seen_positions.add(pos_label)
                src_v = inst.voicings.get(b.source_voicing)
                src_name = src_v.name if src_v else pos_label
                lines.append(f"• [{pos_label}]: Select {src_name}")
        if is_active:
            lines.append(
                "• Onboard Active EQ (Bass, Mid, Treble): Set to Center Detents (Flat / 0 dB)"
            )
            lines.append("• Master Volume Knob: 100% (Wide Open)")
        else:
            lines.append("• Master Tone Knob: 100% (Wide Open across all models)")
            lines.append("• Physical Volume Knob: 100% (Wide Open)")
    else:
        lines.append("• Physical Volume Knob: 100% (Wide Open)")
        if is_active:
            lines.append(
                "• Onboard Active EQ (Bass, Mid, Treble): Set to Center Detents (Flat / 0 dB)"
            )
        else:
            lines.append("• Physical Tone Knob: 100% (Wide Open)")

    lines.extend(
        [
            "",
            "Plug straight in. All pickup positioning, pot loading, tone capacitor shaping, and active preamp EQs are handled automatically by the models.",
        ]
    )
    return lines


def build_storefront_voicings_catalog(
    target_items: Sequence[tuple[str, str, str, str]],
    sorted_families: Sequence[str],
    grouped: Mapping[str, Sequence[tuple[str, str, str, str]]],
) -> list[str]:
    """Builds the numbered target voicings catalog list."""
    total_targets = len(target_items)
    lines: list[str] = [
        "",
        "",
        f"THE {total_targets} DIGITAL TWIN VOICINGS",
        "",
    ]
    count = 1
    for fam in sorted_families:
        lines.append("")
        lines.append(fam)
        lines.append("")
        for _slug, stem_base, desc, _ in grouped[fam]:
            num_str = f"{count:02d}."
            lines.append(f"{num_str} {stem_base}")
            lines.append(desc)
            lines.append("")
            count += 1
    return lines


def build_storefront_legal_disclaimer() -> list[str]:
    """Builds the standardized license and trademark disclaimer section."""
    return [
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


def generate_storefront_description(
    inst: InstrumentConfig,
    bundles: Mapping[str, VoicingBundle],
) -> str:
    """Generates standard Tone3000 storefront product listing description in raw text format."""
    total_targets = sum(len(b.targets) for b in bundles.values())

    target_items: list[tuple[str, str, str, str]] = []
    is_multi_pickup = len(bundles) > 1
    for b_name, b in bundles.items():
        pos_label = b.position_name or b_name.capitalize()
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

    lines: list[str] = []
    lines.extend(build_storefront_overview(inst, total_targets))
    lines.extend(build_storefront_signal_chain(inst))
    lines.extend(build_storefront_instrument_setup(inst, bundles))
    lines.extend(build_storefront_voicings_catalog(target_items, sorted_families, grouped))
    lines.extend(build_storefront_legal_disclaimer())

    return "\n".join(lines).strip() + "\n"
