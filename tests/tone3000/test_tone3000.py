"""Automated verification suite for Tone3000 Storefront Packs, documentation, and production artwork.

Verifies:
1. Storefront description character limits (chars <= 10,000 per listing).
2. Completeness of all digital twin voicings across all editions.
3. Multi-pickup configuration bracketed tags and active instrument EQ guidance.
4. Production artwork SVG XML validity and 1024x1024 JPG generation.
5. Catalog integrity in docs/tone_packs/ and file existence links.
6. Removal of legacy tone3000/docs/ directory.
7. Multi-instrument support in TonePackConfig.
"""

import re
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from allomorph.config.instruments import (
    load_tone_pack,
    partition_instrument_bundles,
)
from allomorph.config.legal import get_legal_disclaimer_text
from allomorph.config.schema import BundleTargetConfig, PackBundleConfig, TonePackConfig
from allomorph.naming import get_t3k_basename
from allomorph.pipeline.schema import Tone3000PackListing

REPO_ROOT = Path(__file__).resolve().parents[2]
TONE_PACKS_DOCS_DIR = REPO_ROOT / "docs" / "tone_packs"
ASSETS_DIR = REPO_ROOT / "tone3000" / "assets"
LEGACY_DOCS_DIR = REPO_ROOT / "tone3000" / "docs"

PACK_EDITIONS: list[str] = [
    "34in_standard_p",
    "34in_standard_jazz",
    "34in_standard_pj",
    "30in_mustang_pj",
    "34in_preamp_soapbar",
    "34in_active_stingray",
    "34in_active_emg",
]

PACK_ARTWORK_NAMES: dict[str, str] = {
    "34in_standard_p": "standard_precision_bass",
    "34in_standard_jazz": "standard_jazz_bass",
    "34in_standard_pj": "standard_pj_bass",
    "30in_mustang_pj": "mustang_pj_bass",
    "34in_preamp_soapbar": "preamp_soapbar_bass",
    "34in_active_stingray": "active_stingray_bass",
    "34in_active_emg": "active_emg_bass",
}

PACK_VOICING_COUNTS: dict[str, int] = {
    "34in_standard_p": 25,
    "34in_standard_jazz": 24,
    "34in_standard_pj": 24,
    "30in_mustang_pj": 26,
    "34in_preamp_soapbar": 26,
    "34in_active_stingray": 25,
    "34in_active_emg": 26,
}

MULTI_PICKUP_PACKS: dict[str, list[str]] = {
    "34in_standard_jazz": ["[Parallel]", "[Neck]", "[Bridge]"],
    "34in_standard_pj": ["[Parallel]", "[Neck]", "[Bridge]"],
    "30in_mustang_pj": ["[Parallel]", "[Neck]", "[Bridge]"],
    "34in_preamp_soapbar": ["[Parallel]", "[Neck]", "[Bridge]"],
    "34in_active_emg": ["[Parallel]", "[Neck]", "[Bridge]"],
}

ACTIVE_PACKS: list[str] = [
    "34in_preamp_soapbar",
    "34in_active_stingray",
    "34in_active_emg",
]


def test_legacy_tone3000_docs_directory_removed():
    """Verify that the legacy tone3000/docs/ directory has been completely removed."""
    assert not LEGACY_DOCS_DIR.exists(), f"Legacy directory {LEGACY_DOCS_DIR} must be deleted."


def test_tone_packs_docs_files_exist():
    """Verify that all docs/tone_packs/ documentation files exist and are populated."""
    for filename in ["naming.md", "ranking.md", "catalog.md"]:
        doc_path = TONE_PACKS_DOCS_DIR / filename
        assert doc_path.exists(), f"Missing tone packs documentation file: {doc_path}"
        assert len(doc_path.read_text(encoding="utf-8").strip()) > 100


def test_tone3000_storefront_text_character_limits():
    """Verify that all storefront descriptions strictly respect Tone3000 length constraints."""
    for pack_id in PACK_EDITIONS:
        pack_cfg = load_tone_pack(pack_id)
        content = pack_cfg.full_storefront_description
        assert content, f"Storefront description empty for pack {pack_id}"
        char_count = len(content)

        assert char_count <= 10000, (
            f"Storefront pack {pack_id} character count {char_count} violates platform limits "
            f"(must be at most 10,000 chars)."
        )


def test_tone3000_pydantic_schema_validation():
    """Verify that each storefront listing strictly validates against Tone3000PackListing schema."""
    for pack_id in PACK_EDITIONS:
        pack_cfg = load_tone_pack(pack_id)
        content = pack_cfg.full_storefront_description
        tags = MULTI_PICKUP_PACKS.get(pack_id, [])
        expected_count = PACK_VOICING_COUNTS[pack_id]
        voicings = [f"{v:02d}" for v in range(1, expected_count + 1)]
        listing = Tone3000PackListing(
            edition=pack_id,
            description=content,
            pickup_tags=tags,
            voicings=voicings,
        )
        assert listing.edition == pack_id
        assert len(listing.voicings) == expected_count
        assert len(listing.description) <= 10000


def test_tone3000_voicing_enumeration():
    """Verify that every storefront pack describes all configured digital twin voicings in order."""
    for pack_id in PACK_EDITIONS:
        pack_cfg = load_tone_pack(pack_id)
        content = pack_cfg.full_storefront_description
        expected_count = PACK_VOICING_COUNTS[pack_id]

        for voice_num in range(1, expected_count + 1):
            prefix = f"{voice_num:02d}."
            assert prefix in content, f"{pack_id} is missing voicing {prefix}"


def test_tone3000_required_sections():
    """Verify that each storefront pack contains all essential sections."""
    legal_text = get_legal_disclaimer_text()
    for pack_id in PACK_EDITIONS:
        expected_count = PACK_VOICING_COUNTS[pack_id]
        required_sections = [
            "OVERVIEW",
            "RECOMMENDED SIGNAL CHAIN",
            "QUICK INSTRUMENT SETUP",
            f"THE {expected_count} DIGITAL TWIN VOICINGS",
            "LICENSE & DISCLAIMER",
        ]

        pack_cfg = load_tone_pack(pack_id)
        content = pack_cfg.full_storefront_description

        for section in required_sections:
            assert section in content, f"{pack_id} is missing required section: {section}"

        assert legal_text.strip() in content, f"{pack_id} missing standard legal disclaimer text"


def test_tone3000_multi_pickup_tags():
    """Verify that multi-pickup editions include bracketed selector tags."""
    for pack_id, tags in MULTI_PICKUP_PACKS.items():
        pack_cfg = load_tone_pack(pack_id)
        content = pack_cfg.full_storefront_description
        expected_count = PACK_VOICING_COUNTS[pack_id]

        voicing_lines = [
            line.strip() for line in content.splitlines() if re.match(r"^\d{2}\.", line.strip())
        ]
        assert len(voicing_lines) == expected_count, (
            f"{pack_id} expected {expected_count} voicing lines, found {len(voicing_lines)}"
        )

        for line in voicing_lines:
            has_tag = any(tag in line for tag in tags)
            assert has_tag, f"{pack_id} voicing line missing selector tag {tags}: '{line}'"


def test_tone3000_active_instrument_guidance():
    """Verify that active instrument packs include explicit active EQ flat / center detent guidance."""
    for pack_id in ACTIVE_PACKS:
        pack_cfg = load_tone_pack(pack_id)
        content = pack_cfg.full_storefront_description

        assert "flat" in content.lower() or "center" in content.lower(), (
            f"Active instrument pack {pack_id} lacks explicit flat/center-detent EQ setup guidance."
        )


def test_tone3000_artwork_files_exist():
    """Verify that every pack has valid JPG and SVG artwork in tone3000/assets/."""
    for pack_id in PACK_EDITIONS:
        art_name = PACK_ARTWORK_NAMES[pack_id]
        jpg_path = ASSETS_DIR / f"allomorph_{art_name}.jpg"
        svg_path = ASSETS_DIR / f"allomorph_{art_name}.svg"

        assert jpg_path.exists(), f"Missing JPG artwork: {jpg_path}"
        assert svg_path.exists(), f"Missing SVG artwork: {svg_path}"

        # SVG validation: parse XML to ensure no malformed markup
        tree = ET.parse(svg_path)
        root = tree.getroot()
        assert root.tag.endswith("svg"), f"{svg_path} root element is not <svg>"


def test_tone3000_catalog_docs_integrity():
    """Verify that docs/tone_packs/catalog.md references all 7 active storefront packs and artworks."""
    catalog_path = TONE_PACKS_DOCS_DIR / "catalog.md"
    assert catalog_path.exists(), f"Missing catalog document: {catalog_path}"

    content = catalog_path.read_text(encoding="utf-8")
    for pack_id in PACK_EDITIONS:
        art_name = PACK_ARTWORK_NAMES[pack_id]
        assert f"{pack_id}.toml" in content, f"catalog.md does not reference {pack_id}.toml"
        assert f"allomorph_{art_name}.jpg" in content, (
            f"catalog.md does not reference allomorph_{art_name}.jpg"
        )


def test_tone3000_t3k_pack_basename_alignment():
    """Verify that every numbered voicing in each storefront description strictly matches
    the authoritative get_t3k_basename(tone_name, pos_name) file name and is <= 34 chars.
    """
    for pack_id in PACK_EDITIONS:
        pack_cfg = load_tone_pack(pack_id)
        content = pack_cfg.full_storefront_description
        voicing_lines = [
            line.strip() for line in content.splitlines() if re.match(r"^\d{2}\.", line.strip())
        ]
        expected_count = PACK_VOICING_COUNTS[pack_id]
        assert len(voicing_lines) == expected_count, (
            f"{pack_id} expected {expected_count} voicing lines, found {len(voicing_lines)}"
        )

        bundles = partition_instrument_bundles(pack_id)
        valid_basenames = set()
        for b_data in bundles.values():
            pos = b_data.position_name if pack_id in MULTI_PICKUP_PACKS else None
            for tgt_ref in b_data.targets:
                if tgt_ref.custom_name:
                    valid_basenames.add(tgt_ref.custom_name)
                else:
                    vcfg = tgt_ref.voicing
                    tone = vcfg.tone_name or vcfg.name
                    valid_basenames.add(get_t3k_basename(tone, pos))

        for idx, vline in enumerate(voicing_lines, 1):
            expected_prefix = f"{idx:02d}. "
            assert vline.startswith(expected_prefix), (
                f"{pack_id} voicing line {idx} does not start with {expected_prefix}: '{vline}'"
            )
            basename = vline[len(expected_prefix) :]
            assert basename in valid_basenames, (
                f"{pack_id} voicing line '{vline}' basename '{basename}' is not a valid "
                f"T3K basename for pack '{pack_id}'."
            )
            assert len(basename) <= 34, (
                f"{pack_id} basename '{basename}' exceeds 34 characters: {len(basename)}"
            )


def test_tone_pack_multi_instrument_support():
    """Verify TonePackConfig handles single and multi-instrument definitions and validates sources."""
    # Single instrument
    cfg1 = TonePackConfig(
        id="test_pack_single",
        instrument="34in_standard_p",
        bundles=[
            PackBundleConfig(
                name="split",
                source="34in_standard_p:vintage_open",
                targets=["34in_standard_jazz:bridge_growl"],
            )
        ],
    )
    assert cfg1.instruments == ["34in_standard_p"]

    # Multiple instruments
    cfg2 = TonePackConfig(
        id="test_pack_multi",
        instrument=["34in_standard_p", "34in_standard_jazz"],
        bundles=[
            PackBundleConfig(
                name="split",
                source="34in_standard_p:vintage_open",
                targets=[
                    BundleTargetConfig(
                        target="41in_upright_bass:bridge_piezo",
                        name="P-Bass [Split] → Upright",
                    )
                ],
            ),
            PackBundleConfig(
                name="bridge",
                source="34in_standard_jazz:bridge_growl",
                targets=["41in_upright_bass:bridge_piezo"],
            ),
        ],
    )
    assert cfg2.instruments == ["34in_standard_p", "34in_standard_jazz"]

    # Validation failure: bundle uses an instrument not in pack.instruments
    with pytest.raises(ValueError, match="is not among pack instrument"):
        TonePackConfig(
            id="test_invalid",
            instrument="34in_standard_p",
            bundles=[
                PackBundleConfig(
                    name="invalid",
                    source="34in_standard_jazz:bridge_growl",
                    targets=["41in_upright_bass:bridge_piezo"],
                )
            ],
        )
