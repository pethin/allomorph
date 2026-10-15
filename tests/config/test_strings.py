"""
Tests for physical strings catalog, presets, and instrument/voice string resolution.
"""

import pytest

from allomorph.config.instruments import load_instrument
from allomorph.config.schema import ResolvedStringConfig, StringPresetConfig
from allomorph.config.strings import (
    STRINGS,
    get_instrument_string,
    load_strings_config,
)


def test_strings_catalog_loading():
    """Verify that all standard physical string presets exist and have valid physics."""
    assert len(STRINGS) >= 5
    assert "roundwound_nickel_standard" in STRINGS
    assert "flatwound_vintage_heavy" in STRINGS

    for s in STRINGS.values():
        assert isinstance(s, StringPresetConfig)
        assert s.name
        assert s.wrap in ("nickel", "steel", "monel", "nylon", "gut") or len(s.wrap) > 0
        assert s.core in ("steel", "nylon", "gut") or len(s.core) > 0
        assert s.tension_lbs > 0
        assert s.damping_cutoff_hz > 0
        assert s.damping_order > 0


def test_load_strings_config_missing_file():
    """Verify loading non-existent strings catalog returns empty dictionary."""
    assert load_strings_config("/non_existent_path/strings.toml") == {}


def test_get_instrument_string():
    """Verify string resolution for standard instruments and override blocks."""
    inst = load_instrument("34in_standard_p")
    res = get_instrument_string(inst)
    assert isinstance(res, ResolvedStringConfig)
    assert res.preset == "roundwound_nickel_standard"
    assert res.tension_lbs > 0

    # Custom override on instrument strings block
    custom_inst = inst.model_copy(deep=True)
    custom_inst.strings.preset = "flatwound_vintage_heavy"
    custom_inst.strings.tension_lbs = 45.0
    res_custom = get_instrument_string(custom_inst)
    assert res_custom.preset == "flatwound_vintage_heavy"
    assert res_custom.tension_lbs == 45.0

    # Unknown string preset raises KeyError
    custom_inst.strings.preset = "non_existent_preset_xyz"
    with pytest.raises(KeyError, match="String preset 'non_existent_preset_xyz' not found"):
        get_instrument_string(custom_inst)
