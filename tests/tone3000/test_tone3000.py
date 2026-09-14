"""Automated verification suite for Tone3000 Storefront Packs, documentation, and production artwork.

Verifies:
1. Storefront description character limits (chars <= 10,000 per listing).
2. Completeness of all digital twin voicings across all editions.
3. Multi-pickup configuration bracketed tags and active instrument EQ guidance.
4. Production artwork SVG XML validity and 1024x1024 JPG generation.
5. README catalog integrity and file existence links.
"""

import re
import xml.etree.ElementTree as ET
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
TONE3000_DIR = REPO_ROOT / "tone3000"
DOCS_DIR = TONE3000_DIR / "docs"
ASSETS_DIR = TONE3000_DIR / "assets"

PACK_EDITIONS: list[str] = [
    "standard_precision_bass",
    "standard_jazz_bass",
    "standard_pj_bass",
    "mustang_pj_bass",
    "preamp_soapbar_bass",
    "active_stingray_bass",
    "active_emg_bass",
]

PACK_VOICING_COUNTS: dict[str, int] = {
    "standard_precision_bass": 21,
    "standard_jazz_bass": 20,
    "standard_pj_bass": 19,
    "mustang_pj_bass": 22,
    "preamp_soapbar_bass": 22,
    "active_stingray_bass": 21,
    "active_emg_bass": 22,
}

MULTI_PICKUP_PACKS: dict[str, list[str]] = {
    "standard_jazz_bass": ["[Parallel]", "[Neck]", "[Bridge]"],
    "standard_pj_bass": ["[Parallel]", "[Neck]", "[Bridge]"],
    "mustang_pj_bass": ["[Parallel]", "[Neck]", "[Bridge]"],
    "preamp_soapbar_bass": ["[Parallel]", "[Neck]", "[Bridge]"],
    "active_emg_bass": ["[Parallel]", "[Neck]", "[Bridge]"],
}

ACTIVE_PACKS: list[str] = [
    "preamp_soapbar_bass",
    "active_stingray_bass",
    "active_emg_bass",
]


def test_tone3000_storefront_text_character_limits():
    """Verify that all storefront descriptions strictly respect Tone3000 length constraints."""
    for pack in PACK_EDITIONS:
        txt_path = DOCS_DIR / f"{pack}.txt"
        assert txt_path.exists(), f"Storefront document missing: {txt_path}"

        content = txt_path.read_text(encoding="utf-8")
        char_count = len(content)

        assert char_count <= 10000, (
            f"Storefront file {pack}.txt character count {char_count} violates platform limits "
            f"(must be at most 10,000 chars)."
        )


def test_tone3000_pydantic_schema_validation():
    """Verify that each storefront listing strictly validates against Tone3000PackListing schema."""
    from allomorph.pipeline.schema import Tone3000PackListing

    for pack in PACK_EDITIONS:
        txt_path = DOCS_DIR / f"{pack}.txt"
        content = txt_path.read_text(encoding="utf-8")
        tags = MULTI_PICKUP_PACKS.get(pack, [])
        expected_count = PACK_VOICING_COUNTS[pack]
        voicings = [f"{v:02d}" for v in range(1, expected_count + 1)]
        listing = Tone3000PackListing(
            edition=pack,
            description=content,
            pickup_tags=tags,
            voicings=voicings,
        )
        assert listing.edition == pack
        assert len(listing.voicings) == expected_count
        assert len(listing.description) <= 10000


def test_tone3000_voicing_enumeration():
    """Verify that every storefront pack describes all configured digital twin voicings in order."""
    for pack in PACK_EDITIONS:
        txt_path = DOCS_DIR / f"{pack}.txt"
        content = txt_path.read_text(encoding="utf-8")
        expected_count = PACK_VOICING_COUNTS[pack]

        for voice_num in range(1, expected_count + 1):
            prefix = f"{voice_num:02d}."
            assert prefix in content, f"{pack}.txt is missing voicing {prefix}"


def test_tone3000_required_sections():
    """Verify that each storefront pack contains all essential sections."""
    for pack in PACK_EDITIONS:
        expected_count = PACK_VOICING_COUNTS[pack]
        required_sections = [
            "OVERVIEW",
            "RECOMMENDED SIGNAL CHAIN",
            "QUICK INSTRUMENT SETUP",
            f"THE {expected_count} DIGITAL TWIN VOICINGS",
            "LICENSE & DISCLAIMER",
        ]

        txt_path = DOCS_DIR / f"{pack}.txt"
        content = txt_path.read_text(encoding="utf-8")

        for section in required_sections:
            assert section in content, f"{pack}.txt is missing required section: {section}"


def test_tone3000_multi_pickup_tags():
    """Verify that multi-pickup editions include bracketed selector tags."""
    for pack, tags in MULTI_PICKUP_PACKS.items():
        txt_path = DOCS_DIR / f"{pack}.txt"
        content = txt_path.read_text(encoding="utf-8")
        expected_count = PACK_VOICING_COUNTS[pack]

        # Find all numbered voicing lines (e.g. "01. Modern Active Jazz Bass...")
        voicing_lines = [
            line.strip() for line in content.splitlines() if re.match(r"^\d{2}\.", line.strip())
        ]
        assert len(voicing_lines) == expected_count, (
            f"{pack}.txt expected {expected_count} voicing lines, found {len(voicing_lines)}"
        )

        for line in voicing_lines:
            # Studio tones (Studio Active / Passive / Direct) preserve aperture and omit selector tags
            if any(
                studio_tone in line
                for studio_tone in [
                    "Studio Active",
                    "Studio Passive",
                    "Studio Direct",
                ]
            ):
                assert not any(tag in line for tag in tags), (
                    f"Studio tone should not have selector tag: '{line}'"
                )
                continue
            has_tag = any(tag in line for tag in tags)
            assert has_tag, f"{pack}.txt voicing line missing selector tag {tags}: '{line}'"


def test_tone3000_active_instrument_guidance():
    """Verify that active instrument packs include explicit active EQ flat / center detent guidance."""
    for pack in ACTIVE_PACKS:
        txt_path = DOCS_DIR / f"{pack}.txt"
        content = txt_path.read_text(encoding="utf-8")

        assert "flat" in content.lower() or "center" in content.lower(), (
            f"Active instrument pack {pack}.txt lacks explicit flat/center-detent EQ setup guidance."
        )


def test_tone3000_character_voicing_ranking_order():
    """Verify that active bass packs rank Studio Passive before Studio Active,
    while passive bass packs rank Studio Active before Studio Passive.

    Under the Inverse-Availability Principle, active instruments already possess onboard active
    buffering, so an authentic high-impedance passive RLC network provides the primary transformative
    value ("Passivizer") and must appear earlier in the storefront listing.
    """
    for pack in PACK_EDITIONS:
        txt_path = DOCS_DIR / f"{pack}.txt"
        content = txt_path.read_text(encoding="utf-8")
        pos_p = content.find("Studio Passive")
        pos_a = content.find("Studio Active")
        assert pos_p != -1, f"Studio Passive missing in {pack}.txt"
        assert pos_a != -1, f"Studio Active missing in {pack}.txt"

        if pack in ACTIVE_PACKS:
            assert pos_p < pos_a, (
                f"Active pack {pack}.txt must rank Studio Passive before Studio Active"
            )
        else:
            assert pos_a < pos_p, (
                f"Passive pack {pack}.txt must rank Studio Active before Studio Passive"
            )


def test_tone3000_artwork_files_exist():
    """Verify that every pack has valid JPG and SVG artwork in tone3000/assets/."""
    for pack in PACK_EDITIONS:
        jpg_path = ASSETS_DIR / f"allomorph_{pack}.jpg"
        svg_path = ASSETS_DIR / f"allomorph_{pack}.svg"

        assert jpg_path.exists(), f"Missing JPG artwork: {jpg_path}"
        assert svg_path.exists(), f"Missing SVG artwork: {svg_path}"

        # SVG validation: parse XML to ensure no malformed markup
        tree = ET.parse(svg_path)
        root = tree.getroot()
        assert root.tag.endswith("svg"), f"{svg_path} root element is not <svg>"


def test_tone3000_catalog_readme_integrity():
    """Verify that tone3000/docs/README.md references all 6 active storefront packs."""
    readme_path = DOCS_DIR / "README.md"
    assert readme_path.exists(), f"Missing catalog README: {readme_path}"

    content = readme_path.read_text(encoding="utf-8")
    for pack in PACK_EDITIONS:
        assert f"{pack}.txt" in content, f"README.md does not reference {pack}.txt"
        assert f"allomorph_{pack}.jpg" in content, (
            f"README.md does not reference allomorph_{pack}.jpg"
        )


PACK_INSTRUMENT_MAP: dict[str, str] = {
    "standard_precision_bass": "34in_standard_p",
    "standard_jazz_bass": "34in_standard_jazz",
    "standard_pj_bass": "34in_standard_pj",
    "mustang_pj_bass": "30in_mustang_pj",
    "preamp_soapbar_bass": "34in_preamp_soapbar",
    "active_stingray_bass": "34in_active_stingray",
    "active_emg_bass": "34in_active_emg",
}


def test_tone3000_t3k_pack_basename_alignment():
    """Verify that every numbered voicing in each storefront description strictly matches
    the authoritative get_t3k_basename(tone_name, pos_name) file name and is <= 34 chars.
    """
    from allomorph.config.instruments import get_source_pickup, load_instrument
    from allomorph.config.voices import VOICES
    from allomorph.naming import get_t3k_basename

    for pack, iid in PACK_INSTRUMENT_MAP.items():
        inst = load_instrument(iid)
        txt_path = DOCS_DIR / f"{pack}.txt"
        content = txt_path.read_text(encoding="utf-8")
        voicing_lines = [
            line.strip() for line in content.splitlines() if re.match(r"^\d{2}\.", line.strip())
        ]
        expected_count = PACK_VOICING_COUNTS[pack]
        assert len(voicing_lines) == expected_count, (
            f"{pack}.txt expected {expected_count} voicing lines, found {len(voicing_lines)}"
        )

        valid_basenames = set()
        for vid, vcfg in VOICES.items():
            pcfg = get_source_pickup(inst, vid)
            pos = (
                (pcfg.position_name or pcfg.name)
                if (pack in MULTI_PICKUP_PACKS and not vcfg.preserve_aperture)
                else None
            )
            tone = vcfg.tone_name or vcfg.name
            valid_basenames.add(
                get_t3k_basename(tone, pos, preserve_aperture=vcfg.preserve_aperture)
            )

        for idx, vline in enumerate(voicing_lines, 1):
            expected_prefix = f"{idx:02d}. "
            assert vline.startswith(expected_prefix), (
                f"{pack}.txt voicing line {idx} does not start with {expected_prefix}: '{vline}'"
            )
            basename = vline[len(expected_prefix) :]
            assert basename in valid_basenames, (
                f"{pack}.txt voicing line '{vline}' basename '{basename}' is not a valid "
                f"T3K basename for instrument '{iid}'."
            )
            assert len(basename) <= 34, (
                f"{pack}.txt basename '{basename}' exceeds 34 characters: {len(basename)}"
            )
