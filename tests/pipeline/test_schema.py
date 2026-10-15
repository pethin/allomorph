"""
Unit tests for pipeline schemas (storefront listings, NAM model metadata, artwork configs).
"""

import pytest
from pydantic import ValidationError

from allomorph.pipeline.schema import (
    ArtworkPackConfig,
    NamExportMetadata,
    NamSourceInstrumentMeta,
    NamSourceVoicingMeta,
    NamTargetVoicingMeta,
    NamTrainingMetadata,
    Tone3000PackListing,
)


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

    # Incorrect number of voicings (< 15 or > 32)
    with pytest.raises(ValidationError):
        Tone3000PackListing(
            edition="test",
            description=valid_desc,
            voicings=valid_voicings[:14],
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
        training=NamTrainingMetadata(
            validation_spectral_loss=0.0004,
            validation_spectral_loss_a2_full=0.0004,
            validation_spectral_loss_a2_lite=0.0018,
            esr_pre_loss=0.0003,
            mrstft_loss=0.0008,
            mrstft_loss_ch3=0.0012,
            epochs_trained=85,
            stop_reason="Dual-Gate Loss Met",
            epochs=100,
        ),
        license="PolyForm Noncommercial License 1.0.0",
        copyright="Copyright 2026 Peter Nguyen",
        author="Peter Nguyen",
        source_instrument=NamSourceInstrumentMeta(
            id="30in_emg_mmtw",
            name='30" Short Scale Bass',
            scale_length_in=30.0,
            scale_length_m=0.762,
            string_wave_speeds=[58.0, 75.0, 99.0, 130.0],
            voicing=NamSourceVoicingMeta(name="EMG MMTWX", position_from_bridge_m=0.0775),
        ),
        target_voicing=NamTargetVoicingMeta(
            id="vintage_open",
            name="'62 Precision Bass (Alnico V)",
            sensor_type="magnetic",
            resonant_frequency_hz=3200.0,
            q_factor=1.8,
        ),
    )
    assert meta.source_instrument.id == "30in_emg_mmtw"
    assert meta.target_voicing.id == "vintage_open"
    assert meta.training.validation_spectral_loss == 0.0004
    assert meta.training.validation_spectral_loss_a2_full == 0.0004
    assert meta.training.validation_spectral_loss_a2_lite == 0.0018
    assert meta.training.esr_pre_loss == 0.0003
    assert meta.training.mrstft_loss == 0.0008
    assert meta.training.mrstft_loss_ch3 == 0.0012
    assert meta.training.epochs_trained == 85
    assert meta.training.stop_reason == "Dual-Gate Loss Met"
    assert meta.source_instrument.voicing.name == "EMG MMTWX"

    # Rejection of missing required fields
    with pytest.raises(ValidationError):
        NamExportMetadata.model_validate({"training": {}})


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
