"""Deterministic unit tests for Tone3000 storefront copy generation."""

from allomorph.config.instruments import load_instrument, partition_instrument_bundles
from allomorph.pipeline.storefront import (
    _family_sort_key,
    _target_sort_key,
    build_storefront_instrument_setup,
    build_storefront_legal_disclaimer,
    build_storefront_overview,
    build_storefront_signal_chain,
    build_storefront_voicings_catalog,
    generate_storefront_description,
)


def test_family_sort_key_native_priority():
    """Verify that native instrument families sort with higher priority."""
    p_inst = load_instrument("34in_standard_p")
    # For a Precision bass, non-native families (transformations) sort before native family
    p_key = _family_sort_key("Precision Bass & Tone Shaper Family", p_inst)
    j_key = _family_sort_key("Jazz Bass Family", p_inst)
    assert j_key < p_key

    j_inst = load_instrument("34in_standard_jazz")
    p_key_j = _family_sort_key("Precision Bass & Tone Shaper Family", j_inst)
    j_key_j = _family_sort_key("Jazz Bass Family", j_inst)
    assert p_key_j < j_key_j


def test_target_sort_key_ordering():
    """Verify target ordering inside a family matches the curated order."""
    item1 = ("precision_vintage", "Precision Vintage", "desc", "Precision Bass & Tone Shaper Family")
    item2 = ("precision_dub", "Precision Dub", "desc", "Precision Bass & Tone Shaper Family")
    assert _target_sort_key(item1) < _target_sort_key(item2)

    item_unknown = ("unknown_slug", "Unknown", "desc", "Precision Bass & Tone Shaper Family")
    assert _target_sort_key(item_unknown) == 99


def test_build_storefront_overview_single_pickup():
    """Verify overview contains instrument name and single-pickup calibration note."""
    inst = load_instrument("34in_standard_p")
    lines = build_storefront_overview(inst, 24)
    text = "\n".join(lines)
    assert f"ALLOMORPH: {inst.name.upper()} EDITION" in text
    assert "Transform your" in text
    assert "24" in text
    assert "Block 1 on the Darkglass Anagram" in text
    p_name = next(iter(inst.pickups.values())).name
    assert f"Calibrated specifically for the {inst.name} with {p_name}" in text


def test_build_storefront_overview_multi_pickup():
    """Verify overview contains multi-pickup instrument calibration phrasing."""
    inst = load_instrument("34in_standard_jazz")
    lines = build_storefront_overview(inst, 20)
    text = "\n".join(lines)
    assert f"Calibrated specifically for the {inst.name}." in text


def test_build_storefront_signal_chain():
    """Verify recommended signal chain contains all four required blocks."""
    inst = load_instrument("34in_standard_p")
    lines = build_storefront_signal_chain(inst)
    text = "\n".join(lines)
    assert "RECOMMENDED SIGNAL CHAIN" in text
    assert "Block 1: Allomorph Pickup Twin" in text
    assert "Block 2: Amp / Drive / Preamp" in text
    assert "Block 3: Speaker Cabinet IR" in text
    assert "FOH / Audio Interface / DAW" in text


def test_build_storefront_instrument_setup_passive_single():
    """Verify instrument setup for a single-pickup passive instrument."""
    inst = load_instrument("34in_standard_p")
    bundles = partition_instrument_bundles(inst)
    lines = build_storefront_instrument_setup(inst, bundles)
    text = "\n".join(lines)
    assert "Physical Volume Knob: 100% (Wide Open)" in text
    assert "Physical Tone Knob: 100% (Wide Open)" in text
    assert "Active EQ" not in text


def test_build_storefront_instrument_setup_active():
    """Verify active instruments get active EQ center detent instructions."""
    inst = load_instrument("34in_active_emg")
    bundles = partition_instrument_bundles(inst)
    lines = build_storefront_instrument_setup(inst, bundles)
    text = "\n".join(lines)
    assert "Onboard Active EQ (Bass, Mid, Treble): Set to Center Detents (Flat / 0 dB)" in text
    assert "Master Volume Knob: 100% (Wide Open)" in text


def test_build_storefront_instrument_setup_multi_bundle():
    """Verify multi-bundle setup includes physical selector switch settings."""
    inst = load_instrument("34in_standard_jazz")
    bundles = partition_instrument_bundles(inst)
    lines = build_storefront_instrument_setup(inst, bundles)
    text = "\n".join(lines)
    assert "set your bass's physical pickup controls to the recommended bracketed setting" in text
    assert "• [" in text


def test_build_storefront_voicings_catalog():
    """Verify numbering and grouped rendering of the target catalog."""
    target_items = [
        ("precision_vintage", "Precision Vintage", "Classic P sound", "Precision Bass & Tone Shaper Family"),
        ("jazz_pair_open", "Jazz Pair Open", "Classic J pair", "Jazz Bass Family"),
    ]
    sorted_families = ["Precision Bass & Tone Shaper Family", "Jazz Bass Family"]
    grouped = {
        "Precision Bass & Tone Shaper Family": [target_items[0]],
        "Jazz Bass Family": [target_items[1]],
    }
    lines = build_storefront_voicings_catalog(target_items, sorted_families, grouped)
    text = "\n".join(lines)
    assert "THE 2 DIGITAL TWIN VOICINGS" in text
    assert "01. Precision Vintage" in text
    assert "Classic P sound" in text
    assert "02. Jazz Pair Open" in text
    assert "Classic J pair" in text


def test_build_storefront_legal_disclaimer():
    """Verify standard legal disclaimer clauses are present."""
    lines = build_storefront_legal_disclaimer()
    text = "\n".join(lines)
    assert "LICENSE & DISCLAIMER" in text
    assert "Commercial musical performances" in text or "commercial musical performances" in text
    assert "Trademark Disclaimer:" in text
    assert "Darkglass" in text
    assert "Allomorph is an independent project" in text


def test_generate_storefront_description_integration():
    """Verify end-to-end storefront description generation for standard P bass."""
    import re

    inst = load_instrument("34in_standard_p")
    bundles = partition_instrument_bundles(inst)
    desc = generate_storefront_description(inst, bundles)
    assert "ALLOMORPH" in desc
    assert "OVERVIEW" in desc
    assert "RECOMMENDED SIGNAL CHAIN" in desc
    assert "QUICK INSTRUMENT SETUP" in desc
    assert "DIGITAL TWIN VOICINGS" in desc
    assert "LICENSE & DISCLAIMER" in desc
    assert desc.endswith("\n")

    # Verify no version tags are present in numbered voicing lines
    voicing_lines = [line.strip() for line in desc.splitlines() if re.match(r"^\d{2}\.", line.strip())]
    assert len(voicing_lines) == 19
    for vline in voicing_lines:
        assert not re.search(r" v\d+\.\d+\.\d+", vline), f"Found version tag in storefront line: '{vline}'"

