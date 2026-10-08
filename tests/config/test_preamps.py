"""
Tests for reusable onboard active preamps and buffer catalog configuration.
"""

from typing import Any, cast

import pytest

from allomorph.config import (
    PREAMPS,
    PreampBandConfig,
    PreampConfig,
    PreampOverrideConfig,
    get_preamp,
)


def test_preamps_catalog_loading():
    """Verify that all standard active preamp presets exist and have valid structure."""
    expected_presets = [
        "flat_buffer",
        "sadowsky_2band",
        "stingray_2band",
        "aguilar_3band",
        "dingwall_active",
    ]
    for pid in expected_presets:
        assert pid in PREAMPS, f"Missing preamp preset '{pid}'"
        cfg = PREAMPS[pid]
        assert isinstance(cfg, PreampConfig)
        assert cfg.name
        assert cfg.input_impedance_meg >= 0.5
        assert cfg.output_impedance_ohm <= 1000.0
        assert isinstance(cfg.bands, list)

        for b in cfg.bands:
            assert b.type in ["low_shelf", "high_shelf", "bell", "low_pass", "high_pass"]
            assert 20.0 <= b.freq_hz <= 20000.0
            assert -20.0 <= b.gain_db <= 20.0


def test_get_preamp_resolution():
    """Verify get_preamp resolves presets, default fallbacks, and inline overrides."""
    # 1. None/empty returns flat buffer
    default_pre = get_preamp(None)
    assert default_pre.name == "Flat Studio Active Buffer"
    assert len(default_pre.bands) == 0

    # 2. String preset lookup
    sad = get_preamp("sadowsky_2band")
    assert sad.name == "Sadowsky 2-Band Boost-Only Preamp"
    assert len(sad.bands) == 2
    assert sad.bands[0].freq_hz == 60.0
    assert sad.bands[0].gain_db == 3.5
    assert sad.bands[1].freq_hz == 3500.0
    assert sad.bands[1].gain_db == 3.5

    # 3. Model with preset override
    overridden = get_preamp(PreampOverrideConfig(preset="sadowsky_2band", gain_db=2.0))
    assert overridden.gain_db == 2.0
    assert len(overridden.bands) == 2

    # 4. Pure custom PreampConfig model
    custom = get_preamp(
        PreampConfig(
            id="custom_1band",
            name="Custom 1-Band",
            output_impedance_ohm=150.0,
            bands=[PreampBandConfig(type="low_shelf", freq_hz=50.0, gain_db=4.0)],
        )
    )
    assert custom.name == "Custom 1-Band"
    assert len(custom.bands) == 1


def test_get_preamp_edge_cases_and_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    from allomorph.config.preamps import get_preamp, load_preamps_config

    # None, empty string, 'none', 'flat'
    for spec in ("", "none", "flat"):
        p = get_preamp(spec)
        assert p.id == "flat_buffer"

    # PreampOverrideConfig with all optional fields
    override = PreampOverrideConfig(
        preset="sadowsky_2band",
        gain_db=1.5,
        input_impedance_meg=2.0,
        output_impedance_ohm=50.0,
        bands=[PreampBandConfig(type="bell", freq_hz=500.0, gain_db=-3.0, q=0.7)],
    )
    res = get_preamp(override)
    assert res.gain_db == 1.5
    assert res.input_impedance_meg == 2.0
    assert res.output_impedance_ohm == 50.0
    assert len(res.bands) == 1
    assert res.bands[0].freq_hz == 500.0

    # Unknown string preset raises KeyError
    with pytest.raises(KeyError, match="Unknown preamp preset 'non_existent_preset'"):
        get_preamp("non_existent_preset")

    # Override with unknown preset raises KeyError
    with pytest.raises(KeyError, match="Unknown preamp preset 'bad_preset'"):
        get_preamp(PreampOverrideConfig(preset="bad_preset"))

    # Invalid specification type raises TypeError
    with pytest.raises(TypeError, match="Invalid preamp specification type"):
        get_preamp(cast(Any, 12345))

    # Missing config file returns empty dict
    assert load_preamps_config("/non_existent_dir/preamps.toml") == {}

    # Flat buffer fallback when flat_buffer not in PREAMPS
    monkeypatch.setattr("allomorph.config.preamps.PREAMPS", {})
    fallback = get_preamp(None)
    assert fallback.id == "flat_buffer"
    assert fallback.name == "Flat Active Buffer"
    assert fallback.input_impedance_meg == 1.0
