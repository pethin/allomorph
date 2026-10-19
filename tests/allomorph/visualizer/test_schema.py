"""
Tests for visualizer configuration schemas (VisualizerChipConfig and VisualizerConfig).
"""

from allomorph.visualizer.schema import (
    VisualizerChipConfig,
    VisualizerConfig,
    load_visualizer_config,
)


def test_visualizer_config_and_chips_schema():
    """Verify VisualizerChipConfig and VisualizerConfig serialization and defaults."""
    chip = VisualizerChipConfig(
        label="P vs J",
        source="34in_standard_p:vintage_open",
        target="34in_standard_jazz:bridge_growl",
    )
    assert chip.label == "P vs J"
    assert chip.source == "34in_standard_p:vintage_open"
    assert chip.target == "34in_standard_jazz:bridge_growl"

    cfg = VisualizerConfig(
        default_source="34in_standard_p:vintage_open",
        default_target="34in_standard_jazz:bridge_growl",
        chips=[chip],
    )
    assert len(cfg.chips) == 1
    assert cfg.chips[0].label == "P vs J"

    # Test load_visualizer_config loading from default config/visualizer.toml
    loaded = load_visualizer_config()
    assert loaded.default_source == "34in_standard_p:vintage_open"
    assert loaded.default_target == "34in_standard_jazz:bridge_growl"
    assert len(loaded.chips) >= 8
    chip_labels = [c.label for c in loaded.chips]
    assert "P vs J Bridge" in chip_labels
    assert "Identity (Reset)" in chip_labels
