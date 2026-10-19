from typing import Any, cast

import pytest

from allomorph.config import SCALES, ScaleConfig


def test_scales_structure():
    assert "30in" in SCALES
    assert "32in" in SCALES
    assert "34in" in SCALES
    assert "multiscale" in SCALES
    assert "upright" in SCALES

    for key, cfg in SCALES.items():
        assert isinstance(cfg, ScaleConfig), f"Scale {key} is not a ScaleConfig instance"
        assert len(cfg.speeds) in (4, 5)
        assert cfg.scale_length_m is not None and cfg.scale_length_m > 0

    # 30" wave speeds should be lower than 34" wave speeds, and 34" lower than upright
    for s30, s34, sup in zip(
        SCALES["30in"].speeds, SCALES["34in"].speeds, SCALES["upright"].speeds
    ):
        assert s30 < s34 < sup


def test_resolve_scale_range_defaults_and_numeric():
    from allomorph.config.scales import resolve_scale_range

    # None defaults to canonical 34" reference scale
    assert resolve_scale_range(None) == pytest.approx((0.8636, 0.8636))

    # Numeric inches (> 5.0)
    assert resolve_scale_range(34.0) == pytest.approx((0.8636, 0.8636))
    assert resolve_scale_range(30.0) == pytest.approx((0.762, 0.762))

    # Numeric meters (<= 5.0)
    assert resolve_scale_range(0.8636) == pytest.approx((0.8636, 0.8636))


def test_resolve_scale_range_sequence():
    from allomorph.config.scales import resolve_scale_range

    # Valid sequence in meters (<= 5.0)
    res = resolve_scale_range([0.8636, 0.9398])
    assert res == pytest.approx((0.8636, 0.9398))

    # Sequence with invalid length
    with pytest.raises(ValueError, match="Invalid scale length tuple/list"):
        resolve_scale_range([0.8636])

    # Sequence with values > 5.0
    with pytest.raises(ValueError, match="Invalid scale length tuple/list"):
        resolve_scale_range([34.0, 37.0])


def test_resolve_scale_range_string():
    from allomorph.config.scales import resolve_scale_range

    # Direct scale ID
    assert resolve_scale_range("34in") == pytest.approx((0.8636, 0.8636))
    # Multiscale ID
    min_m, max_m = resolve_scale_range("multiscale")
    assert min_m == pytest.approx(34.0 * 0.0254)
    assert max_m == pytest.approx(37.0 * 0.0254)

    # Instrument ID resolution
    p_min, p_max = resolve_scale_range("34in_standard_p")
    assert p_min == pytest.approx(0.8636)
    assert p_max == pytest.approx(0.8636)

    # Unknown identifier
    with pytest.raises(ValueError, match="Unknown scale or instrument identifier"):
        resolve_scale_range("non_existent_scale_id_xyz")


def test_resolve_scale_range_config_models():
    from allomorph.config.scales import resolve_scale_range
    from allomorph.config.schema import ScaleConfig

    # Multiscale with scale_min_in and scale_max_in
    cfg_ms = ScaleConfig(
        name="Test MS",
        scale_length_in=34.0,
        is_multiscale=True,
        scale_min_in=34.0,
        scale_max_in=37.0,
    )
    assert resolve_scale_range(cfg_ms) == pytest.approx((34.0 * 0.0254, 37.0 * 0.0254))

    # Scale with scale_length_in
    cfg_in = ScaleConfig(
        name="Test In",
        scale_length_in=32.0,
    )
    assert resolve_scale_range(cfg_in) == pytest.approx((32.0 * 0.0254, 32.0 * 0.0254))

    # Scale with only scale_length_m
    cfg_m = ScaleConfig(
        name="Test M",
        scale_length_m=0.85,
    )
    object.__setattr__(cfg_m, "scale_length_in", None)
    assert resolve_scale_range(cfg_m) == pytest.approx((0.85, 0.85))

    # Missing all scale specifications on ScaleConfig
    object.__setattr__(cfg_m, "scale_length_m", None)
    with pytest.raises(ValueError, match="has no valid scale specification"):
        resolve_scale_range(cfg_m)


def test_resolve_scale_range_invalid_type():
    from allomorph.config.scales import resolve_scale_range

    with pytest.raises(TypeError, match="Cannot resolve scale range from object"):
        resolve_scale_range(cast(Any, {"scale": 34.0}))
