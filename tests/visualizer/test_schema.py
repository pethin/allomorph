"""
Tests for visualizer configuration schemas (PortalInstrumentMeta and VisualizerCliConfig).
"""

import pytest
from pydantic import ValidationError

from allomorph.visualizer.schema import PortalInstrumentMeta, VisualizerCliConfig


def test_portal_instrument_meta_validation():
    """Verify PortalInstrumentMeta schema properties and serialization."""
    meta = PortalInstrumentMeta(
        id="30in",
        name='30" Short Scale Bass',
        scale_in=30.0,
        scale_m=0.762,
        speeds_str="58.0 m/s, 75.0 m/s, 99.0 m/s, 130.0 m/s",
        pickups_summary="EMG MM (@ 77.5mm)",
    )
    assert meta.id == "30in"
    assert meta.scale_in == 30.0
    dump = meta.model_dump()
    assert dump["id"] == "30in"
    assert dump["scale_m"] == 0.762


def test_visualizer_cli_config_validation():
    """Verify VisualizerCliConfig defaults and CLI mode validation."""
    default_cfg = VisualizerCliConfig()
    assert default_cfg.instrument == "all"
    assert default_cfg.mode == "voicings"
    assert not default_cfg.all
    assert default_cfg.out is None

    # Valid custom modes
    for m in (
        "voicings",
        "unified",
        "output",
        "difference",
    ):
        cfg = VisualizerCliConfig(instrument="32in", mode=m, all=True, out="/tmp/test.html")  # type: ignore[arg-type]
        assert cfg.mode == m
        assert cfg.all is True

    # Invalid mode rejection (including obsolete composite and 3D IR modes)
    with pytest.raises(ValidationError):
        VisualizerCliConfig(mode="composite")  # type: ignore[arg-type]
    with pytest.raises(ValidationError):
        VisualizerCliConfig(mode="voicing_ir_3d")  # type: ignore[arg-type]
    with pytest.raises(ValidationError):
        VisualizerCliConfig(mode="waterfall3d")  # type: ignore[arg-type]
    with pytest.raises(ValidationError):
        VisualizerCliConfig(mode="bogus_mode")  # type: ignore[arg-type]

    # Extra arguments rejection
    with pytest.raises(ValidationError):
        VisualizerCliConfig.model_validate({"instrument": "30in", "unexpected_option": True})


def test_visualizer_config_and_chips_schema():
    """Verify VisualizerChipConfig and VisualizerConfig serialization and defaults."""
    from allomorph.visualizer.schema import (
        VisualizerChipConfig,
        VisualizerConfig,
        load_visualizer_config,
    )

    chip = VisualizerChipConfig(
        label="P vs J", source="precision_vintage", target="jazz_bridge_growl"
    )
    assert chip.label == "P vs J"
    assert chip.source == "precision_vintage"
    assert chip.target == "jazz_bridge_growl"

    cfg = VisualizerConfig(
        default_source="precision_vintage",
        default_target="jazz_bridge_growl",
        chips=[chip],
    )
    assert len(cfg.chips) == 1
    assert cfg.chips[0].label == "P vs J"

    # Test load_visualizer_config loading from default config/visualizer.toml
    loaded = load_visualizer_config()
    assert loaded.default_source == "precision_vintage"
    assert loaded.default_target == "jazz_bridge_growl"
    assert len(loaded.chips) >= 8
    chip_labels = [c.label for c in loaded.chips]
    assert "P vs J Bridge" in chip_labels
    assert "Identity (Reset)" in chip_labels
