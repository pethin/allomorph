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
    cfg = PipelineCliConfig(
        stage="sim",
        instrument="30in",
        pack="34in_active_stingray",
        cable_pf=750.0,
        overwrite=True,
        force=True,
    )
    assert cfg.stage == "sim"
    assert cfg.instrument == "30in"
    assert cfg.pack == "34in_active_stingray"
    assert cfg.overwrite is True
    assert cfg.force is True

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
            esr=0.0004,
            validation_esr=0.0004,
            validation_esr_a2_full=0.0004,
            validation_esr_a2_lite=0.0018,
            differential_esr=0.015,
            mrstft_loss=0.0008,
            epochs_trained=85,
            stop_reason="Dual-Gate ESR & Delta Met",
            epochs=100,
        ),
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
    assert meta.training.validation_esr_a2_full == 0.0004
    assert meta.training.validation_esr_a2_lite == 0.0018
    assert meta.training.differential_esr == 0.015
    assert meta.training.mrstft_loss == 0.0008
    assert meta.training.epochs_trained == 85
    assert meta.training.stop_reason == "Dual-Gate ESR & Delta Met"
    assert meta.source_instrument.pickup.name == "EMG MMTW"

    # Rejection of missing required fields
    with pytest.raises(ValidationError):
        NamExportMetadata.model_validate({"training": {}})


def test_nam_training_config_validation():
    """Verify NamTrainingConfig defaults and hyperparameter bounds."""
    cfg = NamTrainingConfig()
    assert cfg.instrument == "all"
    assert cfg.pack is None
    assert cfg.overwrite is False
    assert cfg.voice == "all"
    assert cfg.epochs == 40
    assert cfg.min_epochs == 20
    assert cfg.batch_size == "auto"
    assert cfg.precision == "auto"
    assert cfg.num_workers == "auto"
    assert cfg.patience == 8
    assert cfg.min_delta == 1.0e-6
    assert cfg.pre_emph_weight == 0.25
    assert cfg.pre_emph_coef == 0.85
    assert cfg.mrstft_weight == 0.0010
    assert cfg.lr_scheduler == "cosine"
    assert cfg.eta_min == 1e-5
    assert cfg.lr_t_max == 40
    assert cfg.engine == "auto"

    # Valid custom configuration
    custom = NamTrainingConfig(
        instrument="30in",
        pack="30in_emg_mmtw",
        overwrite=True,
        voice="precision_vintage",
        epochs=50,
        min_epochs=10,
        patience=8,
        min_delta=1e-5,
        batch_size=64,
        fast_dev_run=True,
    )
    assert custom.pack == "30in_emg_mmtw"
    assert custom.overwrite is True
    assert custom.epochs == 50
    assert custom.min_epochs == 10
    assert custom.patience == 8
    assert custom.min_delta == 1e-5
    assert custom.batch_size == 64

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
