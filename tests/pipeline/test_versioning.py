"""
Unit tests for Allomorph 3-part semantic versioning (v[dsp].[inst].[voice]),
Tone3000 length bounds, anti-collision guarantees, sidecar manifest generation,
and NAM container provenance metadata.
"""

import json
from pathlib import Path

import pytest

from allomorph.config.instruments import INSTRUMENTS
from allomorph.config.voices import VOICES
from allomorph.naming import get_t3k_basename
from allomorph.pipeline.schema import (
    NamExportMetadata,
    NamSourceInstrumentMeta,
    NamSourcePickupMeta,
    NamTargetVoiceMeta,
    NamTrainingConfig,
    NamTrainingMetadata,
    PipelineCliConfig,
)
from allomorph.version import (
    ALLOMORPH_VERSION,
    DEFAULT_INST_VERSION,
    DEFAULT_VOICE_VERSION,
    DSP_GENERATION,
    compute_file_sha256,
    get_git_commit,
    get_version_info,
    resolve_tri_part_version,
    write_manifest,
)


def test_resolve_tri_part_version():
    """Verify default and custom tri-part semantic version strings."""
    assert resolve_tri_part_version() == "v3.1.1"
    assert resolve_tri_part_version(2, 1, 1) == "v2.1.1"
    assert resolve_tri_part_version(2, 2, 1) == "v2.2.1"
    assert resolve_tri_part_version(2, 1, 2) == "v2.1.2"
    assert resolve_tri_part_version(3, 1, 1) == "v3.1.1"
    assert resolve_tri_part_version(1, 3, 5) == "v1.3.5"


def test_get_version_info():
    """Verify provenance metadata dictionary structure and git commit detection."""
    info = get_version_info()
    assert info["version"] == "v3.1.1"
    assert info["dsp_version"] == DSP_GENERATION
    assert info["instrument_version"] == DEFAULT_INST_VERSION
    assert info["voice_version"] == DEFAULT_VOICE_VERSION
    assert info["allomorph_version"] == ALLOMORPH_VERSION
    assert "generated_at" in info
    commit = get_git_commit()
    assert info["git_commit"] == commit


def test_anti_collision_guarantee():
    """Verify that incrementing any version component produces distinct filenames,

    preventing tone overwrites or collisions within tone packs.
    """
    v_base = resolve_tri_part_version(2, 1, 1)
    v_dsp = resolve_tri_part_version(3, 1, 1)
    v_inst = resolve_tri_part_version(2, 2, 1)
    v_voice = resolve_tri_part_version(2, 1, 2)

    assert len({v_base, v_dsp, v_inst, v_voice}) == 4

    # Tone3000 filenames
    t3k_base = get_t3k_basename("Precision Vintage", "Split", version_tag=v_base)
    t3k_dsp = get_t3k_basename("Precision Vintage", "Split", version_tag=v_dsp)
    t3k_inst = get_t3k_basename("Precision Vintage", "Split", version_tag=v_inst)
    t3k_voice = get_t3k_basename("Precision Vintage", "Split", version_tag=v_voice)

    assert len({t3k_base, t3k_dsp, t3k_inst, t3k_voice}) == 4
    assert t3k_base == "Precision Vintage [Split] v2.1.1"
    assert t3k_dsp == "Precision Vintage [Split] v3.1.1"
    assert t3k_inst == "Precision Vintage [Split] v2.2.1"
    assert t3k_voice == "Precision Vintage [Split] v2.1.2"


def test_t3k_filename_length_ceiling():
    """Verify that all target voices across all instruments with version_tag stay strictly <= 64 chars.

    Tone3000 has a hard 64-character upload ceiling for the tone model name.
    """
    version_tag = "v2.1.1"
    for inst_id, inst_cfg in INSTRUMENTS.items():
        for vid, vcfg in VOICES.items():
            tone_name = vcfg.tone_name or vcfg.name
            if len(inst_cfg.pickups) <= 1 or vcfg.preserve_aperture:
                pos_name = None
            else:
                from allomorph.config import get_source_pickup

                try:
                    pcfg = get_source_pickup(inst_cfg, vid)
                    pos_name = pcfg.position_name or pcfg.name
                except KeyError, ValueError:
                    pos_name = None

            basename = get_t3k_basename(
                tone_name,
                pos_name,
                preserve_aperture=vcfg.preserve_aperture,
                version_tag=version_tag,
            )
            assert len(basename) <= 64, (
                f"Tone '{basename}' for {inst_id} -> {vid} exceeds 64-character ceiling ({len(basename)} > 64)"
            )


def test_t3k_basename_exceeding_ceiling_raises_error():
    """Verify that get_t3k_basename raises ValueError if tone name exceeds 64 chars."""
    with pytest.raises(ValueError, match="exceeds 64 characters"):
        get_t3k_basename(
            "Super Long Ridiculously Verbose Target Voicing Name That Will Overrun",
            "Neck",
            version_tag="v2.1.1",
        )


def test_write_manifest_and_sha256(tmp_path: Path):
    """Verify manifest.json generation, SHA256 calculation, and audio metrics."""
    import numpy as np

    from allomorph.dsp import write_wav_24bit

    # Create dummy audio and text file
    dummy_wav = tmp_path / "dummy.wav"
    audio_data = np.full(48000, 0.5, dtype=np.float32)
    write_wav_24bit(str(dummy_wav), audio_data, 48000)

    dummy_nam = tmp_path / "dummy.nam"
    dummy_nam.write_text('{"mock": true}', encoding="utf-8")

    manifest_p = write_manifest(
        output_dir=tmp_path,
        stage="test_stage",
        files=[dummy_wav, dummy_nam],
        version_tag="v2.1.1",
    )

    assert manifest_p.exists()
    with manifest_p.open("r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["version"] == "v2.1.1"
    assert data["stage"] == "test_stage"
    assert data["dsp_generation"] == 3
    assert data["file_count"] == 2
    assert "dummy.wav" in data["files"]
    assert "dummy.nam" in data["files"]

    wav_entry = data["files"]["dummy.wav"]
    assert wav_entry["sha256"] == compute_file_sha256(dummy_wav)
    assert wav_entry["size_bytes"] == dummy_wav.stat().st_size
    assert wav_entry["sample_rate"] == 48000
    assert "peak_dbfs" in wav_entry
    assert "rms_dbfs" in wav_entry

    nam_entry = data["files"]["dummy.nam"]
    assert nam_entry["sha256"] == compute_file_sha256(dummy_nam)
    assert nam_entry["size_bytes"] == dummy_nam.stat().st_size


def test_export_instrument_pickup_wav_with_version_tag(tmp_path: Path):
    """Verify export_instrument_pickup_wav embeds version tag and creates manifest when requested."""
    from allomorph.circuit.staging import export_instrument_pickup_wav

    wet_p = export_instrument_pickup_wav(
        inst_id="30in_emg_mmtw",
        pickup_key="mmtw_dual",
        output_dir=tmp_path,
        version_tag="auto",
        max_samples=2048,
    )
    assert wet_p.exists()
    assert wet_p.name == "mmtw_dual.wav"

    manifest_p = wet_p.parent / "manifest.json"
    assert manifest_p.exists()
    with manifest_p.open("r", encoding="utf-8") as f:
        m = json.load(f)
    assert m["version"] == "v3.1.1"
    assert wet_p.name in m["files"]
    assert "base_dry_sha256" in m
    assert m["files"][wet_p.name]["base_dry_sha256"] == m["base_dry_sha256"]


def test_wet_stem_manifest_validation_and_invalidation(tmp_path: Path):
    """Verify is_wet_stem_valid detects valid stems and catches stale/mismatched base dry audio."""
    from allomorph.version import compute_file_sha256, is_wet_stem_valid, write_manifest

    # 1. Setup mock base dry excitation file
    base_dry_file = tmp_path / "test_base_dry.wav"
    base_dry_file.write_bytes(b"ORIGINAL_BASE_DRY_EXCITATION_DATA_12345")
    original_sha = compute_file_sha256(base_dry_file)

    # 2. Setup mock wet stem
    wet_stem = tmp_path / "mock_wet.wav"
    wet_stem.write_bytes(b"MOCK_WET_CONVOLVED_AUDIO_DATA_67890")

    # 3. Initially without manifest -> invalid
    assert not is_wet_stem_valid(wet_stem, base_dry_path=base_dry_file)

    # 4. Write authoritative manifest recording base_dry_sha256
    manifest_p = write_manifest(
        output_dir=tmp_path,
        stage="voicing",
        files=[wet_stem],
        base_dry_sha256=original_sha,
        base_dry_file=base_dry_file.name,
    )
    assert manifest_p.exists()

    # 5. Now it must be valid
    assert is_wet_stem_valid(wet_stem, base_dry_path=base_dry_file)
    assert is_wet_stem_valid(wet_stem, base_dry_path=base_dry_file, expected_version="v3.1.1")
    assert not is_wet_stem_valid(wet_stem, base_dry_path=base_dry_file, expected_version="v3.2.1")

    # 6. Mutate base dry file (simulating re-generated / updated optimal_bass_dry)
    base_dry_file.write_bytes(b"UPDATED_MUTATED_BASE_DRY_EXCITATION_99999")
    mutated_sha = compute_file_sha256(base_dry_file)
    assert mutated_sha != original_sha

    # 7. Must immediately invalidate stale wet stem cache!
    assert not is_wet_stem_valid(wet_stem, base_dry_path=base_dry_file)

    # 8. Non-existent wet stem returns False
    missing_stem = tmp_path / "missing_stem.wav"
    assert not is_wet_stem_valid(missing_stem, base_dry_path=base_dry_file)


def test_manifest_tracks_instrument_and_voicing_versions(tmp_path: Path):
    """Verify write_manifest records instrument_version and voicing_version at root and in file entries."""
    from allomorph.version import write_manifest

    mock_stem = tmp_path / "mock_stem.wav"
    mock_stem.write_bytes(b"MOCK_DATA_123")

    m_path = write_manifest(
        output_dir=tmp_path,
        stage="voicing",
        files=[mock_stem],
        version_tag="v3.1.2",
        base_dry_sha256="abc123sha",
        base_dry_file="base.wav",
        instrument_version=1,
        voicing_version=2,
    )
    assert m_path.exists()
    with m_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["version"] == "v3.1.2"
    assert data["instrument_version"] == 1
    assert data["voicing_version"] == 2
    assert "mock_stem.wav" in data["files"]
    fentry = data["files"]["mock_stem.wav"]
    assert fentry["instrument_version"] == 1
    assert fentry["voicing_version"] == 2
    assert fentry["version"] == "v3.1.2"


def test_all_catalog_instruments_and_voicings_declare_version():
    """Verify all instruments in config/instruments declare version >= 1 at instrument and voicing level."""
    from allomorph.config.instruments import load_all_instruments

    instruments = load_all_instruments()
    assert len(instruments) >= 17
    for iid, inst in instruments.items():
        assert inst.version >= 1, f"Instrument {iid} has invalid version {inst.version}"
        for vid, v in inst.voicings.items():
            assert v.version >= 1, f"Voicing {iid}:{vid} has invalid version {v.version}"


def test_nam_export_metadata_schema_version_fields():
    """Verify NamExportMetadata validates full tri-part version and engine provenance."""
    meta = NamExportMetadata(
        training=NamTrainingMetadata(
            validation_esr=0.0075,
            training_loss=0.0050,
            best_epoch=120,
            total_epochs=150,
        ),
        license="PolyForm Noncommercial License 1.0.0",
        copyright="Copyright 2026 Peter Nguyen",
        author="Peter Nguyen",
        version="v2.1.1",
        dsp_version=2,
        instrument_version=1,
        voice_version=1,
        allomorph_version="0.2.0",
        git_commit="abc1234",
        generated_at="2026-09-13T12:00:00Z",
        source_instrument=NamSourceInstrumentMeta(
            id="30in_emg_mmtw",
            name='30" Short Scale MM',
            scale_length_in=30.0,
            pickup=NamSourcePickupMeta(name="EMG MM Dual Coil"),
        ),
        target_voice=NamTargetVoiceMeta(
            id="precision_vintage",
            name="Vintage 62 P (Alnico V)",
        ),
    )
    d = meta.model_dump()
    assert d["version"] == "v2.1.1"
    assert d["dsp_version"] == 2
    assert d["instrument_version"] == 1
    assert d["voice_version"] == 1
    assert d["allomorph_version"] == "0.2.0"
    assert d["git_commit"] == "abc1234"


def test_cli_config_schemas_support_version_and_manifest_flags():
    """Verify that NamTrainingConfig and PipelineCliConfig include version_tag and no_manifest."""
    n_cfg = NamTrainingConfig(version_tag="v2.1.1", no_manifest=True)
    assert n_cfg.version_tag == "v2.1.1"
    p_cfg = PipelineCliConfig(version_tag="auto", no_manifest=False)
    assert p_cfg.version_tag == "auto"
    assert p_cfg.no_manifest is False


def test_resolve_tri_part_version_flexible_none():
    """Verify resolve_tri_part_version handles None for voice_version and inst_version."""
    assert resolve_tri_part_version(2, 1, 1) == "v2.1.1"
    assert resolve_tri_part_version(2, 1, None) == "v2.1"
    assert resolve_tri_part_version(2, None, None) == "v2"
    assert resolve_tri_part_version(inst_version=None, voice_version=None) == "v3"
    assert resolve_tri_part_version(inst_version=2, voice_version=None) == "v3.2"


def test_dry_naming_helpers():
    """Verify naming helpers for optimal dry file."""
    from allomorph.naming import (
        get_optimal_dry_basename,
        get_optimal_dry_path,
    )

    assert get_optimal_dry_basename() == f"optimal_bass_dry_v{DSP_GENERATION}"
    assert get_optimal_dry_basename("v2") == "optimal_bass_dry_v2"

    dry_p = get_optimal_dry_path(audio_dir="/tmp/test_allomorph")
    assert dry_p == Path(f"/tmp/test_allomorph/canonical/optimal_bass_dry_v{DSP_GENERATION}.wav")


def test_ensure_optimal_dry_wav_versioning(tmp_path: Path):
    """Verify ensure_optimal_dry_wav synthesizes to versioned output path."""
    from allomorph.dsp import FS, ensure_optimal_dry_wav, read_wav

    out_file = tmp_path / "canonical" / "optimal_bass_dry_v2.wav"
    res = ensure_optimal_dry_wav(output_path=out_file, duration_sec=1.0)
    assert res == out_file
    assert out_file.exists()

    audio, sr = read_wav(out_file)
    assert sr == FS
    assert len(audio) == int(1.0 * FS)
