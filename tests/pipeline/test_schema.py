"""
Unit tests for pipeline schemas (CLI arguments, storefront listings, NAM model metadata).
"""

import pytest
from pydantic import ValidationError

from allomorph.pipeline.schema import (
    ArtworkPackConfig,
    NamExportMetadata,
    NamSourceInstrumentMeta,
    NamSourcePickupMeta,
    NamTargetVoiceMeta,
    NamTrainingConfig,
    NamTrainingMetadata,
    PipelineCliConfig,
    Tone3000PackListing,
)


def test_pipeline_cli_config_validation():
    """Verify PipelineCliConfig validation of command line options."""
    cfg = PipelineCliConfig(stage="sim", instrument="30in", cable_pf=750.0)
    assert cfg.stage == "sim"
    assert cfg.instrument == "30in"

    with pytest.raises(ValidationError):
        PipelineCliConfig(stage="unsupported_stage")  # type: ignore[arg-type]


def test_tone3000_listing_validation():
    """Verify Tone3000PackListing minimum and maximum length bounds."""
    valid_desc = "A" * 7500
    valid_voicings = [f"voicing_{i:02d}" for i in range(22)]
    listing = Tone3000PackListing(
        edition="standard_precision_bass",
        description=valid_desc,
        pickup_tags=["split_coil", "alnico_v"],
        voicings=valid_voicings,
    )
    assert listing.edition == "standard_precision_bass"

    # Too long description (> 10000 chars)
    with pytest.raises(ValidationError):
        Tone3000PackListing(
            edition="test",
            description="A" * 10001,
            voicings=valid_voicings,
        )

    # Incorrect number of voicings (< 18 or > 32)
    with pytest.raises(ValidationError):
        Tone3000PackListing(
            edition="test",
            description=valid_desc,
            voicings=valid_voicings[:15],
        )
    with pytest.raises(ValidationError):
        Tone3000PackListing(
            edition="test",
            description=valid_desc,
            voicings=[f"voicing_{i:02d}" for i in range(33)],
        )


def test_nam_export_metadata_validation():
    """Verify NamExportMetadata and nested instrument/voice metadata validation."""
    meta = NamExportMetadata(
        training=NamTrainingMetadata(esr=0.0004, epochs=100),
        license="PolyForm Noncommercial License 1.0.0",
        copyright="Copyright 2026 Peter Nguyen",
        author="Peter Nguyen",
        source_instrument=NamSourceInstrumentMeta(
            id="30in",
            name='30" Short Scale Bass',
            scale_length_in=30.0,
            scale_length_m=0.762,
            string_wave_speeds=[58.0, 75.0, 99.0, 130.0],
            pickup=NamSourcePickupMeta(name="EMG MMTW", position_from_bridge_m=0.0775),
        ),
        target_voice=NamTargetVoiceMeta(
            id="precision_vintage",
            name="'62 Precision Bass (Alnico V)",
            topology="single",
            resonant_frequency_hz=3200.0,
            q_factor=1.8,
        ),
    )
    assert meta.source_instrument.id == "30in"
    assert meta.target_voice.id == "precision_vintage"
    assert meta.training.esr == 0.0004
    assert meta.source_instrument.pickup.name == "EMG MMTW"

    # Rejection of missing required fields
    with pytest.raises(ValidationError):
        NamExportMetadata.model_validate({"training": {}})


def test_nam_training_config_validation():
    """Verify NamTrainingConfig defaults and hyperparameter bounds."""
    cfg = NamTrainingConfig()
    assert cfg.instrument == "all"
    assert cfg.voice == "all"
    assert cfg.epochs == 400
    assert cfg.batch_size == 32
    assert cfg.goal_esr == 0.0080
    assert cfg.a2_lite_only is False

    # Valid custom configuration
    custom = NamTrainingConfig(
        instrument="30in",
        voice="precision_vintage",
        epochs=50,
        batch_size=64,
        fast_dev_run=True,
        a2_lite_only=True,
    )
    assert custom.epochs == 50
    assert custom.batch_size == 64
    assert custom.a2_lite_only is True

    # Negative epochs rejection
    with pytest.raises(ValidationError):
        NamTrainingConfig.model_validate({"epochs": 0})


def test_artwork_pack_config_validation():
    """Verify ArtworkPackConfig hex color pattern and required metadata."""

    def dummy_renderer(accent: str) -> str:
        return f"<svg color='{accent}'></svg>"

    pack = ArtworkPackConfig(
        accent="#38bdf8",
        title="TEST PACK",
        desc_line1="Line 1",
        desc_line2="Line 2",
        scale="34in",
        badge2="STD",
        badge3="PASSIVE",
        content=dummy_renderer,
    )
    assert pack.accent == "#38bdf8"
    assert pack.content(pack.accent) == "<svg color='#38bdf8'></svg>"

    # Invalid hex color code pattern rejection
    with pytest.raises(ValidationError):
        ArtworkPackConfig(
            accent="not-a-hex",
            title="TEST",
            desc_line1="1",
            desc_line2="2",
            scale="34in",
            badge2="STD",
            badge3="PASSIVE",
            content=dummy_renderer,
        )
