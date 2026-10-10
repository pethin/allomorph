import os
import sys
import types
from collections.abc import Generator
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from allomorph.config.schema import InstrumentConfig, VoiceConfig


# Ensure headless matplotlib raster backend is active
try:
    import matplotlib

    matplotlib.use("Agg")
except ImportError:
    pass

# Headless Tkinter fallback for neural-amp-modeler in environments without python3-tk
if "tkinter" not in sys.modules:
    try:
        import tkinter  # noqa: F401
    except ModuleNotFoundError:

        def _dummy_mainloop(*args: object, **kwargs: object) -> None:
            pass

        class _DummyTkMisc:
            mainloop = _dummy_mainloop

        class _DummyTkWidget:
            def __init__(self, *args: object, **kwargs: object) -> None:
                pass

            def __getattr__(self, name: str) -> Any:
                return _dummy_mainloop

        class _DummyTkModule(types.ModuleType):
            Tk = _DummyTkWidget
            Toplevel = _DummyTkWidget
            Label = _DummyTkWidget
            Button = _DummyTkWidget
            Misc = _DummyTkMisc
            mainloop = _dummy_mainloop

        sys.modules["tkinter"] = _DummyTkModule("tkinter")

import pytest
from hypothesis import Verbosity, settings

from allomorph.config.scales import REPO_ROOT

# Register standard Hypothesis profiles
settings.register_profile("ci", max_examples=100, deadline=None)
settings.register_profile("dev", max_examples=25, deadline=None)
settings.register_profile("debug", max_examples=10, verbosity=Verbosity.verbose, deadline=None)
settings.register_profile("smt", max_examples=10, backend="crosshair", deadline=None)

# Load profile from environment (default to 'dev' for fast local test runs)
settings.load_profile(os.getenv("HYPOTHESIS_PROFILE", "dev"))


@pytest.fixture(scope="session", autouse=True)
def guard_no_audio_pollution() -> Generator[None]:
    """
    Session-level guard fixture verifying that running the test suite does not pollute
    or leave behind newly generated files or directories in the project's audio/ directory.
    """
    audio_dir = REPO_ROOT / "audio"
    before_items = (
        {p for p in audio_dir.rglob("*") if p.name != ".DS_Store"} if audio_dir.exists() else set()
    )
    yield
    after_items = (
        {p for p in audio_dir.rglob("*") if p.name != ".DS_Store"} if audio_dir.exists() else set()
    )
    new_items = sorted(
        str(p.relative_to(REPO_ROOT))
        for p in (after_items - before_items)
        if p != audio_dir / "input.wav"
    )
    assert not new_items, (
        f"Test suite polluted the audio directory with {len(new_items)} item(s):\n"
        + "\n".join(new_items)
    )


def make_generic_instrument_config() -> InstrumentConfig:
    """Returns a synthetic, unbranded single-pickup instrument configuration for SUT tests."""
    from allomorph.config.schema import (
        CoilConfig,
        ControlElementConfig,
        HarnessConfig,
        InstrumentConfig,
        PickupConfig,
        SwitchConfig,
        SwitchPositionConfig,
        VoicingConfig,
    )

    return InstrumentConfig(
        id="generic_test_bass",
        name="Generic Test Bass",
        scale_length_in=34.0,
        string_wave_speeds=[71.16, 95.0, 126.81, 169.27],
        pickups={
            "main": PickupConfig(
                name="Main Pickup",
                position_from_bridge_m=0.10,
                type="single_coil",
                coils=[
                    CoilConfig(
                        id="main_c",
                        position_from_bridge_m=0.10,
                        aperture_width_in=0.75,
                        L=4.0,
                        Rdc=6000.0,
                        Reddy=200000.0,
                        Ccoil=5e-11,
                        weight=1.0,
                        polarity=1.0,
                        strings=[1, 2, 3, 4],
                    )
                ],
            ),
        },
        harnesses={
            "passive": HarnessConfig(
                name="Standard 1V/1T",
                type="passive",
                load_resistance=1000000.0,
                cable_pf=750.0,
                wiring=[
                    ["pickups.main.cold", "GND"],
                    ["pickups.main.hot", "controls.vol.in"],
                    ["controls.vol.wiper", "out"],
                    ["controls.vol.gnd", "GND"],
                    ["controls.vol.in", "controls.tone.in"],
                    ["controls.tone.gnd", "GND"],
                ],
                controls={
                    "vol": ControlElementConfig(
                        name="Master Volume", resistance=250000.0, taper="audio_15", default=1.0
                    ),
                    "tone": ControlElementConfig(
                        name="Master Tone",
                        resistance=250000.0,
                        taper="audio_15",
                        cap=4.7e-8,
                        default=1.0,
                    ),
                },
            ),
            "direct": HarnessConfig(
                name="Direct",
                type="passive",
                load_resistance=1000000.0,
                cable_pf=750.0,
                wiring=[
                    ["pickups.main.hot", "out"],
                    ["pickups.main.cold", "GND"],
                ],
            ),
            "active": HarnessConfig(
                name="Active Buffer Preamp",
                type="active_preamp",
                load_resistance=1000000.0,
                cable_pf=30.0,
                wiring=[
                    ["pickups.main.cold", "GND"],
                    ["pickups.main.hot", "preamp.in"],
                    ["preamp.out", "controls.vol.in"],
                    ["controls.vol.wiper", "out"],
                    ["controls.vol.gnd", "GND"],
                ],
                controls={
                    "vol": ControlElementConfig(
                        name="Master Volume", resistance=25000.0, taper="audio_15", default=1.0
                    ),
                },
            ),
            "series_hpf": HarnessConfig(
                name="Passive HPF",
                type="passive",
                load_resistance=1000000.0,
                cable_pf=750.0,
                wiring=[
                    ["pickups.main.cold", "GND"],
                    ["controls.vol.wiper", "out"],
                    ["controls.vol.gnd", "GND"],
                ],
                controls={
                    "vol": ControlElementConfig(
                        name="Master Volume", resistance=250000.0, taper="audio_15", default=1.0
                    ),
                },
                switches={
                    "hpf": SwitchConfig(
                        name="HPF Switch",
                        default="engage",
                        positions={
                            "bypass": SwitchPositionConfig(
                                connect=[["pickups.main.hot", "controls.vol.in"]]
                            ),
                            "engage": SwitchPositionConfig(
                                connect=[
                                    ["pickups.main.hot", "hpf_cap.in"],
                                    ["hpf_cap.out", "controls.vol.in"],
                                ],
                                components={"hpf_cap": 4.7e-9},
                            ),
                        },
                    )
                },
            ),
        },
        voicings={
            "generic_voice": VoicingConfig(
                id="generic_voice",
                name="Generic Voicing",
                harness="passive",
                controls={"vol": 1.0, "tone": 1.0},
            ),
            "passive_open": VoicingConfig(
                id="passive_open",
                name="Passive Open",
                harness="passive",
                controls={"vol": 1.0, "tone": 1.0},
            ),
            "passive_warm": VoicingConfig(
                id="passive_warm",
                name="Passive Warm",
                harness="passive",
                controls={"vol": 1.0, "tone": 0.0},
            ),
            "direct": VoicingConfig(
                id="direct",
                name="Direct Voicing",
                harness="direct",
            ),
            "active": VoicingConfig(
                id="active",
                name="Active Voicing",
                harness="active",
                controls={"vol": 1.0},
            ),
            "series_hpf": VoicingConfig(
                id="series_hpf",
                name="Series HPF",
                harness="series_hpf",
                switches={"hpf": "engage"},
                controls={"vol": 1.0},
            ),
        },
    )


@pytest.fixture
def generic_instrument_config() -> InstrumentConfig:
    """Returns a synthetic, unbranded single-pickup instrument configuration for SUT tests."""
    return make_generic_instrument_config()


def make_generic_dual_pickup_instrument() -> InstrumentConfig:
    """Returns a synthetic, unbranded dual-pickup instrument configuration for multi-pickup SUT tests."""
    from allomorph.config.schema import (
        CoilConfig,
        ControlElementConfig,
        HarnessConfig,
        InstrumentConfig,
        PickupConfig,
        VoicingConfig,
    )

    pos_neck = 0.14
    pos_bridge = 0.06
    return InstrumentConfig(
        id="generic_dual_bass",
        name="Generic Dual Bass",
        scale_length_in=34.0,
        string_wave_speeds=[71.16, 95.0, 126.81, 169.27],
        pickups={
            "neck": PickupConfig(
                name="Neck Pickup",
                position_from_bridge_m=pos_neck,
                type="single_coil",
                magnet_type="alnico_v",
                coils=[
                    CoilConfig(
                        id="neck_c",
                        position_from_bridge_m=pos_neck,
                        aperture_width_in=0.5,
                        L=3.0,
                        Rdc=6000.0,
                        Reddy=100000.0,
                        Ccoil=5e-11,
                        weight=1.0,
                        polarity=1.0,
                        strings=[1, 2, 3, 4],
                    )
                ],
            ),
            "bridge": PickupConfig(
                name="Bridge Pickup",
                position_from_bridge_m=pos_bridge,
                type="single_coil",
                magnet_type="ceramic",
                coils=[
                    CoilConfig(
                        id="bridge_c",
                        position_from_bridge_m=pos_bridge,
                        aperture_width_in=0.5,
                        L=3.0,
                        Rdc=6000.0,
                        Reddy=100000.0,
                        Ccoil=5e-11,
                        weight=1.0,
                        polarity=1.0,
                        strings=[1, 2, 3, 4],
                    )
                ],
            ),
        },
        harnesses={
            "passive": HarnessConfig(
                name="Direct Parallel",
                type="passive",
                load_resistance=1000000.0,
                cable_pf=750.0,
                wiring=[
                    ["pickups.neck.hot", "out"],
                    ["pickups.neck.cold", "GND"],
                    ["pickups.bridge.hot", "out"],
                    ["pickups.bridge.cold", "GND"],
                ],
            ),
            "parallel_controls": HarnessConfig(
                name="2V/1T Passive Parallel",
                type="passive",
                load_resistance=1000000.0,
                cable_pf=750.0,
                wiring=[
                    ["pickups.neck.hot", "controls.neck_vol.wiper"],
                    ["controls.neck_vol.in", "out"],
                    ["controls.neck_vol.gnd", "GND"],
                    ["pickups.neck.cold", "GND"],
                    ["pickups.bridge.hot", "controls.bridge_vol.wiper"],
                    ["controls.bridge_vol.in", "out"],
                    ["controls.bridge_vol.gnd", "GND"],
                    ["pickups.bridge.cold", "GND"],
                    ["out", "controls.tone.in"],
                    ["controls.tone.gnd", "GND"],
                ],
                controls={
                    "neck_vol": ControlElementConfig(
                        name="Neck Volume", resistance=250000.0, taper="audio_15", default=1.0
                    ),
                    "bridge_vol": ControlElementConfig(
                        name="Bridge Volume", resistance=250000.0, taper="audio_15", default=1.0
                    ),
                    "tone": ControlElementConfig(
                        name="Tone", resistance=250000.0, taper="audio_15", cap=4.7e-8, default=1.0
                    ),
                },
            ),
            "series": HarnessConfig(
                name="Passive Series",
                type="passive",
                load_resistance=1000000.0,
                cable_pf=750.0,
                wiring=[
                    ["pickups.bridge.cold", "GND"],
                    ["pickups.bridge.hot", "pickups.neck.cold"],
                    ["pickups.neck.hot", "controls.vol.in"],
                    ["controls.vol.wiper", "out"],
                    ["controls.vol.gnd", "GND"],
                ],
                controls={
                    "vol": ControlElementConfig(
                        name="Master Volume", resistance=250000.0, taper="audio_15", default=1.0
                    ),
                },
            ),
            "active": HarnessConfig(
                name="Active Buffer Parallel",
                type="active_preamp",
                load_resistance=1000000.0,
                cable_pf=30.0,
                wiring=[
                    ["pickups.neck.cold", "GND"],
                    ["pickups.bridge.cold", "GND"],
                    ["pickups.neck.hot", "preamp.in"],
                    ["pickups.bridge.hot", "preamp.in"],
                    ["preamp.out", "controls.vol.in"],
                    ["controls.vol.wiper", "out"],
                    ["controls.vol.gnd", "GND"],
                ],
                controls={
                    "vol": ControlElementConfig(
                        name="Master Volume", resistance=25000.0, taper="audio_15", default=1.0
                    ),
                },
            ),
        },
        voicings={
            "blend": VoicingConfig(
                id="blend",
                name="Direct Parallel Blend",
                harness="passive",
            ),
            "blend_controls": VoicingConfig(
                id="blend_controls",
                name="Parallel Blend with Controls",
                harness="parallel_controls",
                controls={"neck_vol": 1.0, "bridge_vol": 1.0, "tone": 1.0},
            ),
            "neck_solo": VoicingConfig(
                id="neck_solo",
                name="Neck Solo",
                harness="parallel_controls",
                controls={"neck_vol": 1.0, "bridge_vol": 0.0, "tone": 1.0},
            ),
            "bridge_solo": VoicingConfig(
                id="bridge_solo",
                name="Bridge Solo",
                harness="parallel_controls",
                controls={"neck_vol": 0.0, "bridge_vol": 1.0, "tone": 1.0},
            ),
            "series": VoicingConfig(
                id="series",
                name="Series",
                harness="series",
                controls={"vol": 1.0},
            ),
            "active": VoicingConfig(
                id="active",
                name="Active",
                harness="active",
                controls={"vol": 1.0},
            ),
        },
    )


@pytest.fixture
def generic_dual_pickup_instrument() -> InstrumentConfig:
    """Returns a synthetic, unbranded dual-pickup instrument configuration for multi-pickup SUT tests."""
    return make_generic_dual_pickup_instrument()


@pytest.fixture
def generic_piezo_instrument() -> InstrumentConfig:
    """Returns a synthetic piezo-based acoustic instrument configuration for SUT tests."""
    from allomorph.config.schema import (
        CoilConfig,
        HarnessConfig,
        InstrumentConfig,
        PickupConfig,
        VoicingConfig,
    )

    return InstrumentConfig(
        id="generic_piezo_bass",
        name="Generic Piezo Bass",
        scale_length_in=41.0,
        electronics="piezo",
        string_wave_speeds=[50.0, 70.0, 90.0, 110.0],
        pickups={
            "piezo": PickupConfig(
                name="Bridge Piezo",
                position_from_bridge_m=0.01,
                type="piezo_bridge",
                magnet_type="piezo",
                coils=[
                    CoilConfig(
                        id="piezo_c",
                        position_from_bridge_m=0.01,
                        aperture_width_in=0.25,
                        weight=1.0,
                        L=1e-06,
                        Rdc=50.0,
                        Reddy=1000000.0,
                        Ccoil=1.2e-09,
                    )
                ],
            ),
        },
        harnesses={
            "direct": HarnessConfig(
                name="Direct Piezo",
                type="passive",
                load_resistance=10000000.0,
                cable_pf=50.0,
                wiring=[
                    ["pickups.piezo.hot", "out"],
                    ["pickups.piezo.cold", "GND"],
                ],
            ),
        },
        voicings={
            "piezo_voice": VoicingConfig(
                id="piezo_voice",
                name="Piezo Voice",
                sensor_type="bridge_force",
                harness="direct",
            ),
        },
    )


@pytest.fixture
def generic_voice_config() -> VoiceConfig:
    """Returns a synthetic, unbranded target voice configuration for SUT tests."""
    from allomorph.config.schema import VoiceCoilConfig, VoiceConfig

    return VoiceConfig(
        id="generic_voice",
        name="Generic Voice",
        description="Generic synthetic voice",
        fr=3000.0,
        Q=1.5,
        coils=[
            VoiceCoilConfig(
                position_from_bridge_m=0.10,
                aperture_width_in=0.75,
                weight=1.0,
                polarity=1.0,
                strings=[1, 2, 3, 4],
            )
        ],
    )
