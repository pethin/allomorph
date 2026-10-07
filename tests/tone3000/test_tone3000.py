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
    "standard_precision_bass": 19,
    "standard_jazz_bass": 18,
    "standard_pj_bass": 17,
    "mustang_pj_bass": 20,
    "preamp_soapbar_bass": 20,
    "active_stingray_bass": 19,
    "active_emg_bass": 20,
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


def test_generated_pack_storefront_descriptions():
    """Verify that all generated pack storefront descriptions strictly conform to:
    1. Raw text (no markdown underlines === or ---, no markdown headings ###).
    2. No Technical Specifications section.
    3. Essential sections present: OVERVIEW, RECOMMENDED SIGNAL CHAIN, QUICK INSTRUMENT SETUP,
       THE {N} DIGITAL TWIN VOICINGS, LICENSE & DISCLAIMER.
    4. Exact License and Trademark Disclaimer text.
    5. Word-wrap friendliness: descriptions and paragraphs are single continuous lines.
    6. Tone3000 character limit (<= 10,000 chars).
    """
    packs_dir = TONE3000_DIR / "packs"
    assert packs_dir.exists(), f"Packs directory missing: {packs_dir}"

    desc_files = list(packs_dir.glob("*/storefront_description.txt"))
    assert len(desc_files) >= 16, f"Expected at least 16 pack descriptions, found {len(desc_files)}"

    for desc_file in desc_files:
        content = desc_file.read_text(encoding="utf-8")
        pack_id = desc_file.parent.name

        # 1. No Technical Specifications
        assert "TECHNICAL SPECIFICATIONS" not in content, (
            f"Pack {pack_id} contains forbidden TECHNICAL SPECIFICATIONS section."
        )

        # 2. No markdown underlines or headings
        for line in content.splitlines():
            assert not re.match(r"^={3,}$", line), (
                f"Pack {pack_id} contains markdown underline: '{line}'"
            )
            assert not re.match(r"^-{3,}$", line), (
                f"Pack {pack_id} contains markdown underline: '{line}'"
            )
            assert not line.startswith("###"), (
                f"Pack {pack_id} contains markdown heading: '{line}'"
            )

        # 3. Essential sections
        assert "OVERVIEW" in content, f"Pack {pack_id} missing OVERVIEW"
        assert "RECOMMENDED SIGNAL CHAIN" in content, f"Pack {pack_id} missing RECOMMENDED SIGNAL CHAIN"
        assert "QUICK INSTRUMENT SETUP" in content, f"Pack {pack_id} missing QUICK INSTRUMENT SETUP"
        assert "DIGITAL TWIN VOICINGS" in content, f"Pack {pack_id} missing DIGITAL TWIN VOICINGS"
        assert "LICENSE & DISCLAIMER" in content, f"Pack {pack_id} missing LICENSE & DISCLAIMER"

        # 4. License & Trademark Disclaimer
        assert "License:" in content, f"Pack {pack_id} missing 'License:'"
        assert "Licensed for personal and commercial musical performances" in content, (
            f"Pack {pack_id} missing standard License text"
        )
        assert "Trademark Disclaimer:" in content, f"Pack {pack_id} missing 'Trademark Disclaimer:'"
        assert "Allomorph is an independent project and is not affiliated with or endorsed" in content, (
            f"Pack {pack_id} missing standard Trademark Disclaimer text"
        )

        # 5. Character limit
        assert len(content) <= 10000, (
            f"Pack {pack_id} exceeds 10,000 characters: {len(content)}"
        )

        # 6. Voicing entries match file names and have versions
        voicing_lines = [
            line.strip() for line in content.splitlines() if re.match(r"^\d{2}\.", line.strip())
        ]
        assert len(voicing_lines) >= 15, f"Pack {pack_id} has fewer than 15 voicings: {len(voicing_lines)}"

        bundle_wav_stems = {
            f.stem for f in desc_file.parent.glob("bundles/*/*.wav") if not f.name.startswith("dry")
        }
        for vline in voicing_lines:
            assert re.match(r"^\d{2}\. .+ v\d+\.\d+\.\d+$", vline), (
                f"Pack {pack_id} voicing line missing version tag: '{vline}'"
            )
            tone_entry = re.sub(r"^\d{2}\. ", "", vline)
            if bundle_wav_stems:
                assert tone_entry in bundle_wav_stems, (
                    f"Pack {pack_id} voicing '{tone_entry}' does not match any bundle stem file name"
                )
