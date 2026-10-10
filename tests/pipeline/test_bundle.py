"""Deterministic unit tests for Tone3000 upload bundle packaging logic."""

from allomorph.config.instruments import load_instrument, partition_instrument_bundles
from allomorph.pipeline.bundle import (
    assemble_root_pack_manifest,
    build_bundle_manifest_entry,
    generate_bundle_upload_instructions,
    resolve_bundle_target_file_descriptors,
)
from allomorph.version import ALLOMORPH_VERSION, DSP_GENERATION


def test_resolve_bundle_target_file_descriptors_single_pickup():
    """Verify descriptors for a single-pickup instrument omit bracketed position tags."""
    inst = load_instrument("34in_standard_p")
    bundles = partition_instrument_bundles(inst)
    bundle = next(iter(bundles.values()))

    descriptors = resolve_bundle_target_file_descriptors(bundle, is_multi_pickup=False)
    assert len(descriptors) > 0

    first = descriptors[0]
    assert "target_ref" in first
    assert "tone_name" in first
    assert first["pos_tag"] is None
    assert first["stem_filename"].endswith(".wav")
    assert first["stem_base"] + ".wav" == first["stem_filename"]
    assert "[" not in first["stem_base"]
    assert "v" in first["target_v_tag"]


def test_resolve_bundle_target_file_descriptors_multi_pickup():
    """Verify descriptors for a multi-pickup instrument include bracketed position tags."""
    inst = load_instrument("34in_standard_jazz")
    bundles = partition_instrument_bundles(inst)
    bridge_bundle = bundles["bridge"]

    descriptors = resolve_bundle_target_file_descriptors(bridge_bundle, is_multi_pickup=True)
    assert len(descriptors) > 0

    first = descriptors[0]
    assert first["pos_tag"] == bridge_bundle.position_name or "Bridge"
    assert f"[{first['pos_tag']}]" in first["stem_base"]


def test_generate_bundle_upload_instructions():
    """Verify upload instructions contain clear step-by-step instructions and file lists."""
    instr = generate_bundle_upload_instructions(
        bundle_name="bridge",
        position_name="Bridge",
        instrument_name="Fender Jazz Bass",
        dry_filename="dry v2.1.1.wav",
        stem_filenames=["stem_1.wav", "stem_2.wav"],
    )

    assert "TONE3000 UPLOAD BUNDLE: BRIDGE" in instr
    assert "Fender Jazz Bass" in instr
    assert "Pickup Position: Bridge" in instr
    assert "DRAG DRY FILE: Drag 'dry v2.1.1.wav'" in instr
    assert "DRAG WET STEMS: Select and drag the following 2 .wav files" in instr
    assert "• stem_1.wav" in instr
    assert "• stem_2.wav" in instr
    assert "4. START TRAINING" in instr
    assert instr.endswith("\n")


def test_build_bundle_manifest_entry():
    """Verify bundle manifest entry dictionary schema and contents."""
    stems = [
        {
            "filename": "stem1.wav",
            "target_instrument": "34in_standard_p",
            "target_instrument_version": 1,
            "target_voicing": "vintage_open",
            "target_voicing_version": 1,
            "version": "v2.1.1",
            "tone_name": "Precision Vintage",
            "sha256": "abc12345",
        }
    ]
    entry = build_bundle_manifest_entry(
        bundle_name="neck",
        position_name="Neck",
        source_voicing_id="vintage_open",
        dry_filename="dry v2.1.1.wav",
        dry_version="v2.1.1",
        dry_sha256="deadbeef",
        instrument_id="34in_standard_p",
        instrument_version=1,
        source_voicing_version=1,
        bundle_stems=stems,
        base_dry_file="test_input.wav",
        base_dry_sha256="feedface",
    )

    assert entry["bundle"] == "neck"
    assert entry["position_name"] == "Neck"
    assert entry["source_voicing"] == "vintage_open"
    assert entry["dry_file"] == "dry v2.1.1.wav"
    assert entry["dry_sha256"] == "deadbeef"
    assert entry["base_dry_file"] == "test_input.wav"
    assert entry["base_dry_sha256"] == "feedface"
    assert entry["stem_count"] == 1
    assert entry["stems"] == stems


def test_assemble_root_pack_manifest():
    """Verify root pack manifest structure, version tagging, and bundle containment."""
    inst = load_instrument("34in_standard_p")
    mock_stems: list[dict[str, str]] = []
    mock_bundles = {
        "split": {
            "source_voicing": "vintage_open",
            "dry_file": "dry v2.1.1.wav",
            "stem_count": 0,
            "stems": mock_stems,
        }
    }
    manifest = assemble_root_pack_manifest(
        inst=inst,
        bundle_entries=mock_bundles,
        dsp_generation=DSP_GENERATION,
    )

    assert manifest["instrument_id"] == inst.id
    assert manifest["instrument_name"] == inst.name
    assert manifest["instrument_version"] == inst.version
    assert manifest["allomorph_version"] == ALLOMORPH_VERSION
    assert manifest["dsp_generation"] == DSP_GENERATION
    assert manifest["bundles"] == mock_bundles
    assert manifest["version"] == f"v{DSP_GENERATION}.{inst.version}.1"
