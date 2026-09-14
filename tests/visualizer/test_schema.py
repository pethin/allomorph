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
        default_pickup="emg_mm",
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
        "voicing_ir_3d",
        "waterfall3d",
        "unified",
        "output",
        "difference",
    ):
        cfg = VisualizerCliConfig(instrument="32in", mode=m, all=True, out="/tmp/test.html")  # type: ignore[arg-type]
        assert cfg.mode == m
        assert cfg.all is True

    # Invalid mode rejection (including obsolete composite mode)
    with pytest.raises(ValidationError):
        VisualizerCliConfig(mode="composite")  # type: ignore[arg-type]
    with pytest.raises(ValidationError):
        VisualizerCliConfig(mode="bogus_mode")  # type: ignore[arg-type]

    # Extra arguments rejection
    with pytest.raises(ValidationError):
        VisualizerCliConfig.model_validate({"instrument": "30in", "unexpected_option": True})
