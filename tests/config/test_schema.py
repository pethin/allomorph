"""
Unit tests for Pydantic configuration schemas and validation behavior.
Ensures fail-fast declarative integrity (Guardrail 5.3.5) with extra="forbid",
type checking, strict attribute access, and bounds validation.
"""

import pytest
from pydantic import ValidationError

from allomorph.base import AllomorphBaseModel
from allomorph.config.schema import (
    CoilConfig,
    ControlElementConfig,
    HarnessConfig,
    InstrumentConfig,
    InstrumentStringsConfig,
    PackBundleConfig,
    PickupConfig,
    PreampBandConfig,
    PreampConfig,
    ScaleConfig,
    StringPresetConfig,
    SwitchConfig,
    SwitchPositionConfig,
    TonePackConfig,
    VoiceCoilConfig,
    VoiceConfig,
    VoicePickupConfig,
    VoicingConfig,
)


class SampleModel(AllomorphBaseModel):
    name: str
    value: float = 1.0


def test_allomorph_base_model_attribute_access_and_strictness():
    """Verify strongly-typed attribute access and rejection of subscripting on AllomorphBaseModel."""
    m = SampleModel(name="test", value=42.0)

    # Direct attribute reading
    assert m.name == "test"
    assert m.value == 42.0

    # Attribute mutation with validate_assignment=True
    m.value = 100.0
    assert m.value == 100.0

    # Subscripting rejected
    with pytest.raises(TypeError):
        _ = m["name"]  # type: ignore[index]

    with pytest.raises(TypeError):
        m["value"] = 50.0  # type: ignore[index]

    # Does not have custom dict-like __contains__
    assert "__contains__" not in AllomorphBaseModel.__dict__

    # Dict methods do not exist on model
    assert not hasattr(m, "get")
    assert not hasattr(m, "keys")
    assert not hasattr(m, "values")
    assert not hasattr(m, "items")

    # Copy
    m_copy = m.model_copy()
    assert m_copy.value == 100.0


def test_allomorph_base_model_extra_forbid():
    """Verify that unknown/unexpected keys are strictly rejected across all models."""
    with pytest.raises(ValidationError) as exc_info:
        SampleModel.model_validate({"name": "test", "unexpected_field": "illegal"})
    assert "extra_forbidden" in str(exc_info.value)


def test_scale_config_validation():
    """Verify ScaleConfig validation and bounds."""
    scale = ScaleConfig(
        name='34" Standard',
        scale_length_in=34.0,
        string_wave_speeds=[77.4, 103.4, 137.9, 184.2],
    )
    assert scale.scale_length_in == 34.0
    assert scale.scale_length_m == pytest.approx(34.0 * 0.0254)
    assert len(scale.speeds) == 4

    # Extra fields forbidden
    with pytest.raises(ValidationError):
        ScaleConfig.model_validate(
            {
                "name": "Bad",
                "scale_length_in": 34.0,
                "string_wave_speeds": [77.4, 103.4, 137.9, 184.2],
                "bogus_key": 123,
            }
        )


def test_string_preset_config_validation():
    """Verify StringPresetConfig validation."""
    preset = StringPresetConfig(
        name="Standard Nickel Roundwound",
        type="roundwound",
        wrap="nickel",
        core="steel",
        tension_lbs=42.8,
        damping_cutoff_hz=1200.0,
        damping_order=1.5,
    )
    assert preset.wrap == "nickel"
    assert preset.core == "steel"
    assert preset.tension_lbs == 42.8

    # Disallow unexpected field
    with pytest.raises(ValidationError):
        StringPresetConfig.model_validate(
            {
                "name": "Bad",
                "type": "roundwound",
                "wrap": "nickel",
                "core": "steel",
                "tension_lbs": 42.8,
                "damping_cutoff_hz": 1200.0,
                "damping_order": 1.5,
                "extra_field": True,
            }
        )


def test_preamp_band_and_catalog_validation():
    """Verify PreampBandConfig and PreampConfig validation."""
    band = PreampBandConfig(
        type="low_shelf",
        freq_hz=40.0,
        gain_db=4.0,
        q=0.707,
    )
    assert band.type == "low_shelf"
    assert band.freq_hz == 40.0

    preamp = PreampConfig(
        name="Test Preamp",
        input_impedance_meg=1.0,
        output_impedance_ohm=100.0,
        bands=[band],
    )
    assert preamp.name == "Test Preamp"
    assert len(preamp.bands) == 1

    # Disallow invalid extra key in band
    with pytest.raises(ValidationError):
        PreampBandConfig.model_validate(
            {
                "type": "bell",
                "freq_hz": 800.0,
                "gain_db": 2.0,
                "unknown": "not_allowed",
            }
        )


def test_pickup_and_instrument_config_validation():
    """Verify CoilConfig, PickupConfig, and InstrumentConfig validation."""
    coil = CoilConfig(
        position_from_bridge_m=0.0775,
        aperture_width_in=0.75,
    )
    assert coil.position_from_bridge_m == 0.0775

    pickup = PickupConfig(
        name="Test Bridge Pickup",
        position_from_bridge_m=0.0775,
        aperture_width_in=0.90,
        coils=[coil],
    )
    assert pickup.name == "Test Bridge Pickup"
    assert len(pickup.coils) == 1

    inst = InstrumentConfig(
        id="test_bass",
        name="Test Bass",
        scale_length_in=34.0,
        scale_length_m=0.8636,
        string_wave_speeds=[77.4, 103.4, 137.9, 184.2],
        pickups={"bridge": pickup},
        strings=InstrumentStringsConfig(preset="roundwound_nickel_standard"),
    )
    assert inst.id == "test_bass"
    assert "bridge" in inst.pickups

    # Extra field forbidden on InstrumentConfig
    with pytest.raises(ValidationError):
        InstrumentConfig.model_validate(
            {
                "id": "test_bass",
                "name": "Test Bass",
                "scale_length_in": 34.0,
                "scale_length_m": 0.8636,
                "string_wave_speeds": [77.4, 103.4, 137.9, 184.2],
                "pickups": {"bridge": pickup},
                "unregistered_custom_attr": "forbidden",
            }
        )


def test_voice_config_validation():
    """Verify VoiceConfig, VoicePickupConfig, and VoiceCoilConfig validation."""
    coil = VoiceCoilConfig(
        position_from_bridge_m=0.0635,
        aperture_width_in=0.75,
    )
    voice_pickup = VoicePickupConfig(
        name="Bridge Single",
        fr=3200.0,
        Q=1.6,
        coils=[coil],
    )
    voice = VoiceConfig(
        id="03_test_voice",
        name="Test Voice",
        description="A test target voice",
        fr=3200.0,
        Q=1.6,
        pickups=[voice_pickup],
    )
    assert voice.id == "03_test_voice"
    assert voice.pickups is not None and len(voice.pickups) == 1

    # Extra field forbidden on VoiceConfig
    with pytest.raises(ValidationError):
        VoiceConfig.model_validate(
            {
                "id": "03_test_voice",
                "name": "Test Voice",
                "description": "A test target voice",
                "fr": 3200.0,
                "Q": 1.6,
                "circuit": {"topology": "single", "L": 3.2, "Rdc": 8000.0, "Reddy": 150000.0},
                "pickups": [{"name": "Bridge Single", "fr": 3200.0, "Q": 1.6}],
                "extra_bad_arg": "bad",
            }
        )


def test_coil_config_rlc_parameters():
    """Verify CoilConfig with intrinsic RLC parameters and validation."""
    coil = CoilConfig(
        id="neck",
        position_from_bridge_m=0.0771,
        aperture_width_in=0.45,
        L=2.4,
        Rdc=4400.0,
        Reddy=75000.0,
        Ccoil=1.8e-10,
    )
    assert coil.id == "neck"
    assert coil.L == 2.4
    assert coil.Rdc == 4400.0
    assert coil.Reddy == 75000.0
    assert coil.Ccoil == 1.8e-10

    # Negative inductance rejected
    with pytest.raises(ValidationError):
        CoilConfig(position_from_bridge_m=0.0771, L=-1.0)


def test_pickup_config_internal_buffer():
    """Verify PickupConfig with internal sealed active buffer."""
    p = PickupConfig(
        name="EMG CSX",
        has_internal_buffer=True,
        buffer_output_impedance=2000.0,
        active_variant="x_series",
    )
    assert p.has_internal_buffer is True
    assert p.buffer_output_impedance == 2000.0
    assert p.active_variant == "x_series"


def test_harness_and_control_element_config():
    """Verify ControlElementConfig, SwitchConfig, and HarnessConfig models."""
    vol = ControlElementConfig(
        name="Neck Volume",
        type="pot",
        resistance=250000.0,
        taper="audio_15",
        default=1.0,
    )
    tone = ControlElementConfig(
        name="Master Tone",
        type="pot",
        resistance=250000.0,
        taper="audio_15",
        cap=4.7e-8,
        default=1.0,
    )
    sw = SwitchConfig(
        name="Series/Parallel",
        type="toggle",
        default="parallel",
        positions={
            "parallel": SwitchPositionConfig(
                connect=[["neck.hot", "out"], ["bridge.hot", "out"]],
                k_mutual=0.20,
            ),
            "series": SwitchPositionConfig(
                connect=[["neck.hot", "out"], ["neck.cold", "mid"], ["bridge.hot", "mid"]],
                k_mutual=0.20,
            ),
        },
    )
    harness = HarnessConfig(
        name="Passive V/V/T with Series Switch",
        type="passive",
        controls={"vol": vol, "tone": tone},
        switches={"mode": sw},
        cable_pf=750.0,
        load_resistance=1000000.0,
    )
    assert harness.name == "Passive V/V/T with Series Switch"
    assert "vol" in harness.controls
    assert "mode" in harness.switches
    assert harness.switches["mode"].positions["parallel"].k_mutual == 0.20


def test_instrument_config_harness_and_voicing_validation():
    """Verify InstrumentConfig validates controls, switches, and bounds against its harnesses."""
    harness_passive = HarnessConfig(
        name="Passive",
        type="passive",
        controls={
            "neck_vol": ControlElementConfig(name="Neck Vol", default=1.0),
            "bridge_vol": ControlElementConfig(name="Bridge Vol", default=1.0),
            "tone": ControlElementConfig(name="Tone", cap=4.7e-8, default=1.0),
        },
        switches={
            "pickup_selector": SwitchConfig(
                default="both",
                positions={
                    "neck": SwitchPositionConfig(connect=[["neck_wiper", "out"]]),
                    "both": SwitchPositionConfig(connect=[["neck_wiper", "out"], ["bridge_wiper", "out"]]),
                    "bridge": SwitchPositionConfig(connect=[["bridge_wiper", "out"]]),
                },
            )
        },
    )

    # Valid instrument
    inst = InstrumentConfig(
        id="test_bass",
        name="Test Bass",
        harnesses={"passive": harness_passive},
        voicings={
            "warm": VoicingConfig(
                name="Warm",
                harness="passive",
                controls={"neck_vol": 1.0, "tone": 0.5},
                switches={"pickup_selector": "neck"},
            ),
            "bridge_solo": VoicingConfig(
                name="Bridge Solo",
                harness="passive",
                switch="bridge",  # Shorthand resolves to pickup_selector
                controls={"bridge_vol": 1.0},
            ),
        },
    )
    assert inst.voicings["warm"].controls["tone"] == 0.5
    assert inst.voicings["bridge_solo"].switches["pickup_selector"] == "bridge"

    # Unknown harness raises KeyError
    with pytest.raises(KeyError, match="unknown harness 'active'"):
        InstrumentConfig(
            id="bad_harness",
            harnesses={"passive": harness_passive},
            voicings={"v": VoicingConfig(name="Bad", harness="active")},
        )

    # Unknown control name raises KeyError
    with pytest.raises(KeyError, match="unknown control 'bogus_knob'"):
        InstrumentConfig(
            id="bad_ctl",
            harnesses={"passive": harness_passive},
            voicings={"v": VoicingConfig(name="Bad", harness="passive", controls={"bogus_knob": 0.5})},
        )

    # Control value out of [0.0, 1.0] raises ValueError
    with pytest.raises(ValueError, match="out of range"):
        InstrumentConfig(
            id="bad_range",
            harnesses={"passive": harness_passive},
            voicings={"v": VoicingConfig(name="Bad", harness="passive", controls={"neck_vol": 1.5})},
        )

    # Unknown switch name raises KeyError
    with pytest.raises(KeyError, match="unknown switch 'coil_tap'"):
        InstrumentConfig(
            id="bad_sw",
            harnesses={"passive": harness_passive},
            voicings={"v": VoicingConfig(name="Bad", harness="passive", switches={"coil_tap": "single"})},
        )

    # Unknown switch position raises KeyError
    with pytest.raises(KeyError, match="unknown position 'sideways'"):
        InstrumentConfig(
            id="bad_pos",
            harnesses={"passive": harness_passive},
            voicings={"v": VoicingConfig(name="Bad", harness="passive", switches={"pickup_selector": "sideways"})},
        )


def test_tone_pack_and_bundle_config():
    """Verify PackBundleConfig and TonePackConfig models."""
    bundle = PackBundleConfig(
        name="neck",
        source_voicing="neck_warm",
        targets=["precision_vintage", "mudbucker_deep"],
    )
    pack = TonePackConfig(
        id="34in_standard_jazz",
        instrument="34in_standard_jazz",
        name="Standard Jazz Digital Twin Pack",
        bundles=[bundle],
    )
    assert pack.id == "34in_standard_jazz"
    assert pack.bundles[0].source_voicing == "neck_warm"
    assert len(pack.bundles[0].targets) == 2

    # Extra fields forbidden
    with pytest.raises(ValidationError):
        PackBundleConfig.model_validate(
            {"name": "neck", "source_voicing": "warm", "illegal_arg": 123}
        )


def test_voicing_config_rejects_misplaced_hardware_fields():
    """Verify that VoicingConfig strictly rejects misplaced hardware transducer fields."""
    # Voicing is purely a cavity state vector; hardware fields like magnet_type, alpha, vsat, fr, Q are forbidden
    for forbidden_field in [
        "hpf",
        "resonant_frequency_hz",
        "q_factor",
        "magnet_type",
        "alpha",
        "vsat",
    ]:
        with pytest.raises(ValidationError) as exc_info:
            VoicingConfig.model_validate(
                {
                    "name": "Test",
                    "harness": "passive",
                    forbidden_field: 1.0 if "hz" in forbidden_field or forbidden_field == "q_factor" else "alnico_v",
                }
            )
        assert "extra_forbidden" in str(exc_info.value)


def test_voice_config_rejects_dead_metallurgy_fields():
    """Verify that VoiceConfig rejects the 15 eliminated dead micro-metallurgy fields."""
    dead_fields = [
        "alpha3",
        "eta_hyst",
        "k_sag",
        "k_eddy",
        "kappa_orbit",
        "k_body",
        "beta_curv",
        "k_pull",
        "tau_touch",
        "chi_mu",
        "k_dist",
        "kappa_geom",
        "k_stein",
        "k_emf",
        "lambda_L",
    ]
    for dead_field in dead_fields:
        with pytest.raises(ValidationError) as exc_info:
            VoiceConfig.model_validate(
                {
                    "id": "v_test",
                    "name": "Test Voice",
                    "description": "Test",
                    "fr": 3000.0,
                    "Q": 1.5,
                    dead_field: 0.1,
                }
            )
        assert "extra_forbidden" in str(exc_info.value)


def test_pickup_config_rejects_bundle_name():
    """Verify that PickupConfig rejects the legacy bundle_name field."""
    with pytest.raises(ValidationError) as exc_info:
        PickupConfig.model_validate(
            {
                "name": "Test Pickup",
                "bundle_name": "legacy_neck",
            }
        )
    assert "extra_forbidden" in str(exc_info.value)

