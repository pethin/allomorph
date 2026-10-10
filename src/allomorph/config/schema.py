"""
Allomorph - Pydantic Configuration Schemas.

Defines formal, strictly-typed Pydantic v2 schemas for all Allomorph physical instruments,
target voices, RLC circuit digital twins, coils, active onboard preamps, physical scales,
and string presets.
"""

import warnings
from pathlib import Path
from typing import Literal, Self

from pydantic import Field, model_validator

from allomorph.base import AllomorphBaseModel, SpiceFloat, parse_spice_unit

warnings.filterwarnings(
    "ignore",
    message=r'.*Field name "register".*shadows an attribute in parent.*',
    category=UserWarning,
)

__all__ = [
    "AllomorphBaseModel",
    "CoilConfig",
    "ControlElementConfig",
    "HarnessConfig",
    "InstrumentConfig",
    "InstrumentStringsConfig",
    "PackBundleConfig",
    "PickupComponentConfig",
    "PickupConfig",
    "PreampBandConfig",
    "PreampConfig",
    "PreampOverrideConfig",
    "PreampsCatalog",
    "ResolvedStringConfig",
    "ScaleConfig",
    "ScalesCatalog",
    "SpiceFloat",
    "StringPresetConfig",
    "StringsCatalog",
    "SwitchConfig",
    "SwitchPositionConfig",
    "TonePackConfig",
    "VoiceCoilConfig",
    "VoiceConfig",
    "VoicePickupConfig",
    "VoicingConfig",
    "parse_spice_unit",
]


# ==============================================================================
# 1. PHYSICAL SCALES SCHEMAS
# ==============================================================================


class ScaleConfig(AllomorphBaseModel):
    """Vibrating string scale length and physical wave speeds."""

    name: str
    scale_length_in: float | None = None
    scale_length_m: float | None = None
    scale_min_in: float | None = None
    scale_max_in: float | None = None
    is_multiscale: bool = False
    string_wave_speeds: list[float] = Field(default_factory=list)

    @model_validator(mode="after")
    def compute_scale_m(self) -> Self:
        if self.scale_length_m is None and self.scale_length_in is not None:
            self.scale_length_m = self.scale_length_in * 0.0254
        elif self.scale_length_in is None and self.scale_length_m is not None:
            self.scale_length_in = self.scale_length_m / 0.0254
        elif self.scale_length_m is None and self.scale_length_in is None:
            raise ValueError("ScaleConfig must specify either scale_length_in or scale_length_m")
        return self

    @property
    def scale_m(self) -> float:
        if self.scale_length_m is not None:
            return self.scale_length_m
        if self.scale_length_in is not None:
            return self.scale_length_in * 0.0254
        return 0.8636

    @property
    def speeds(self) -> list[float]:
        return self.string_wave_speeds


class ScalesCatalog(AllomorphBaseModel):
    """Catalog of standard vibrating scale lengths."""

    scales: dict[str, ScaleConfig] = Field(default_factory=dict)


# ==============================================================================
# 2. PHYSICAL STRINGS SCHEMAS
# ==============================================================================


class StringPresetConfig(AllomorphBaseModel):
    """Physical string core, wrap, tension, and viscoelastic damping preset."""

    name: str
    type: str = "roundwound"
    wrap: str
    core: str
    tension_lbs: float = Field(..., gt=0.0)
    damping_cutoff_hz: float = Field(..., gt=0.0)
    damping_order: float = Field(..., gt=0.0)
    k_long: float = Field(0.20, ge=0.0)


class InstrumentStringsConfig(AllomorphBaseModel):
    """Source instrument string setup block referencing a preset."""

    preset: str = "roundwound_nickel_standard"
    brand: str | None = None
    model: str | None = None
    gauge: str | None = None
    core: str | None = None
    wrap: str | None = None
    tension_lbs: float | None = None


class ResolvedStringConfig(StringPresetConfig):
    """Fully resolved string configuration combining preset physics and instrument setup."""

    preset: str = "roundwound_nickel_standard"
    brand: str | None = None
    model: str | None = None
    gauge: str | None = None

    @classmethod
    def from_preset_and_overrides(
        cls, preset: StringPresetConfig, overrides: InstrumentStringsConfig
    ) -> Self:
        """Constructs a fully resolved string configuration directly from preset and setup models."""
        return cls(
            name=preset.name,
            type=preset.type,
            wrap=overrides.wrap if overrides.wrap is not None else preset.wrap,
            core=overrides.core if overrides.core is not None else preset.core,
            tension_lbs=(
                overrides.tension_lbs if overrides.tension_lbs is not None else preset.tension_lbs
            ),
            damping_cutoff_hz=preset.damping_cutoff_hz,
            damping_order=preset.damping_order,
            k_long=preset.k_long,
            preset=overrides.preset,
            brand=overrides.brand,
            model=overrides.model,
            gauge=overrides.gauge,
        )


class StringsCatalog(AllomorphBaseModel):
    """Catalog of physical string presets."""

    strings: dict[str, StringPresetConfig] = Field(default_factory=dict)


# ==============================================================================
# 3. ONBOARD ACTIVE PREAMP SCHEMAS
# ==============================================================================


class PreampBandConfig(AllomorphBaseModel):
    """Active onboard preamp equalizer band specification."""

    type: Literal["low_shelf", "high_shelf", "bell", "low_pass", "high_pass"]
    freq_hz: float = Field(..., gt=0.0, le=20000.0)
    gain_db: float = Field(..., ge=-30.0, le=30.0)
    q: float | None = None


class PreampConfig(AllomorphBaseModel):
    """Active onboard preamp or buffer configuration."""

    id: str | None = None
    name: str
    description: str | None = None
    input_impedance_meg: float = Field(1.0, gt=0.0)
    output_impedance_ohm: float = Field(100.0, ge=0.0)
    gain_db: float = 0.0
    bands: list[PreampBandConfig] = Field(default_factory=list)


class PreampsCatalog(AllomorphBaseModel):
    """Catalog of active preamp and buffer presets."""

    preamps: dict[str, PreampConfig] = Field(default_factory=dict)


class PreampOverrideConfig(AllomorphBaseModel):
    """Configuration model for inline or preset-based preamp overrides."""

    preset: str
    gain_db: float | None = None
    input_impedance_meg: float | None = None
    output_impedance_ohm: float | None = None
    bands: list[PreampBandConfig] = Field(default_factory=list)


# ==============================================================================
# 5. COIL & PICKUP SCHEMAS
# ==============================================================================


# ==============================================================================
# 5. COIL & PICKUP SCHEMAS
# ==============================================================================


class CoilConfig(AllomorphBaseModel):
    """Physical sensing coil aperture geometry, string routing, and intrinsic RLC parameters."""

    id: str | None = None
    position_from_bridge_m: float = Field(..., gt=0.0)
    aperture_width_in: float = Field(0.75, gt=0.0)
    weight: float = Field(1.0, gt=0.0)
    polarity: float = 1.0
    strings: list[int | str] = Field(default_factory=lambda: ["all"])
    pole_type: str | None = None
    L: float | None = Field(None, gt=0.0, description="Coil self-inductance in Henries")
    Rdc: float | None = Field(None, gt=0.0, description="Coil DC resistance in Ohms")
    Reddy: float | None = Field(None, gt=0.0, description="Core eddy damping resistance in Ohms")
    Ccoil: float | None = Field(None, ge=0.0, description="Inter-turn coil capacitance in Farads")


class PickupComponentConfig(AllomorphBaseModel):
    """Component coil in a composite blended pickup configuration."""

    pickup: str | None = None
    position_from_bridge_m: float | None = None
    aperture_width_in: float = 0.75
    weight: float = 1.0
    polarity: float = 1.0
    strings: list[int | str] = Field(default_factory=lambda: ["all"])
    pole_type: str | None = None


class PickupConfig(AllomorphBaseModel):
    """Source instrument pickup configuration."""

    id: str | None = None
    name: str
    position_name: str | None = None
    position_from_bridge_m: float | None = None
    aperture_width_in: float = 0.75
    coil_spacing_in: float = 0.0
    type: str = "single_coil"
    magnet_type: str | None = None
    pole_type: str | None = None
    resonant_frequency_hz: float | None = None
    q_factor: float | None = None
    alpha: float | None = None
    vsat: float | None = None
    has_internal_buffer: bool = False
    buffer_output_impedance: float = Field(2000.0, ge=0.0)
    active_variant: Literal["x_series", "classic"] | None = None
    coils: list[CoilConfig] = Field(default_factory=list)
    components: list[PickupComponentConfig] = Field(default_factory=list)


# ==============================================================================
# 6. NODAL CONTROL HARNESS SCHEMAS
# ==============================================================================


class ControlElementConfig(AllomorphBaseModel):
    """Analog potentiometer, blend control, or active EQ band in the control harness."""

    name: str = ""
    type: Literal["pot", "preamp_band"] = "pot"
    resistance: float = Field(250000.0, gt=0.0)
    taper: Literal["audio_15", "audio_10", "linear", "mn_blend", "reverse_audio"] = "audio_15"
    cap: float | None = Field(None, gt=0.0, description="Attached capacitor value in Farads (e.g. tone cap)")
    band: str | None = Field(None, description="Preamp band name if type == 'preamp_band'")
    default: float = Field(1.0, ge=0.0, le=1.0, description="Default normalized wiper position")


class SwitchPositionConfig(AllomorphBaseModel):
    """Circuit node connections and mutual coupling for a discrete switch position."""

    connect: list[list[str]] = Field(
        default_factory=list,
        description="List of [terminal_a, terminal_b] pairs connected in this position.",
    )
    k_mutual: float | None = Field(None, ge=0.0, le=1.0)
    components: dict[str, float] = Field(default_factory=dict)


class SwitchConfig(AllomorphBaseModel):
    """Discrete multi-position switch (toggle, blade, rotary, push-pull)."""

    name: str = ""
    type: Literal["toggle", "push_pull", "rotary", "blade"] = "toggle"
    default: str
    positions: dict[str, SwitchPositionConfig] = Field(default_factory=dict)


class HarnessConfig(AllomorphBaseModel):
    """Control cavity electrical topology, potentiometers, switches, and active preamp."""

    name: str = ""
    type: Literal["passive", "active_preamp", "direct"] = "passive"
    preamp: str | None = None
    preamp_gain: float = Field(1.0, gt=0.0)
    wiring: list[list[str]] = Field(
        default_factory=list,
        description="Base permanent terminal-to-node netlist connections.",
    )
    k_mutual: float | None = Field(None, ge=0.0, le=1.0)
    controls: dict[str, ControlElementConfig] = Field(default_factory=dict)
    switches: dict[str, SwitchConfig] = Field(default_factory=dict)
    cable_pf: float = Field(750.0, ge=0.0)
    load_resistance: float = Field(1000000.0, gt=0.0)


# ==============================================================================
# 7. SOURCE INSTRUMENT & VOICING SCHEMAS
# ==============================================================================


class VoicingConfig(AllomorphBaseModel):
    """
    Declarative physical instrument voicing setting.
    Parameterized state vector setting arbitrary control knobs and switches.
    """

    id: str | None = None
    name: str
    tone_name: str | None = None
    description: str | None = None
    harness: str = Field(
        ...,
        description="Target harness ID in instrument.harnesses.",
    )
    controls: dict[str, float] = Field(
        default_factory=dict,
        description="Continuous control knob settings: knob_id -> normalized position in [0.0, 1.0].",
    )
    switches: dict[str, str] = Field(
        default_factory=dict,
        description="Discrete switch settings: switch_id -> position_name.",
    )
    switch: str | None = Field(
        default=None,
        description="Shorthand switch setting when the selected harness has a single switch.",
    )
    components: dict[str, float] = Field(
        default_factory=dict,
        description="Strict dotted-path parameter overrides (e.g. 'controls.tone.cap' = 100e-9).",
    )
    string_preset_override: str | None = None
    sensor_type: Literal["magnetic", "bridge_force", "direct"] = "magnetic"
    gain_db: float = 0.0
    preserve_aperture: bool = False
    preamp_bands: list[PreampBandConfig] = Field(default_factory=list)
    version: int = Field(1, ge=1, description="Instrument voicing configuration version")


class InstrumentConfig(AllomorphBaseModel):
    """Complete source instrument definition."""

    id: str = "custom"
    name: str = ""
    scale_length_in: float | None = Field(34.0, gt=0.0)
    scale_length_m: float | None = None
    scale_min_in: float | None = None
    scale_max_in: float | None = None
    is_multiscale: bool = False
    electronics: str = "passive"
    version: int = Field(1, ge=1, description="Instrument physical datum configuration version")
    string_wave_speeds: list[float] = Field(default_factory=list)
    strings: InstrumentStringsConfig = Field(default_factory=InstrumentStringsConfig)
    pickups: dict[str, PickupConfig] = Field(default_factory=dict)
    harnesses: dict[str, HarnessConfig] = Field(default_factory=dict)
    voicings: dict[str, VoicingConfig] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_instrument(self) -> Self:
        if not self.name:
            self.name = self.id
        if self.scale_length_m is None and self.scale_length_in is not None:
            self.scale_length_m = self.scale_length_in * 0.0254
        elif self.scale_length_in is None and self.scale_length_m is not None:
            self.scale_length_in = self.scale_length_m / 0.0254
        elif self.scale_length_m is None and self.scale_length_in is None:
            self.scale_length_in = 34.0
            self.scale_length_m = 0.8636

        # Validate harnesses and voicings
        for vid, v in self.voicings.items():
            if v.id is None:
                v.id = vid
            if self.harnesses:
                h_name = v.harness
                if not h_name or h_name not in self.harnesses:
                    raise KeyError(
                        f"Voicing '{vid}' on instrument '{self.id}' references unknown harness '{h_name}'. "
                        f"Available harnesses: {list(self.harnesses.keys())}"
                    )
                h_cfg = self.harnesses[h_name]
                for ctl_name, ctl_val in v.controls.items():
                    if ctl_name not in h_cfg.controls:
                        raise KeyError(
                            f"Voicing '{vid}' sets unknown control '{ctl_name}' on harness '{h_name}'. "
                            f"Available controls: {list(h_cfg.controls.keys())}"
                        )
                    if not (0.0 <= ctl_val <= 1.0):
                        raise ValueError(
                            f"Voicing '{vid}' control '{ctl_name}' value {ctl_val} out of range [0.0, 1.0]."
                        )
                if v.switch is not None:
                    if len(h_cfg.switches) == 1:
                        sw_name = next(iter(h_cfg.switches.keys()))
                        if v.switch not in h_cfg.switches[sw_name].positions:
                            raise KeyError(
                                f"Voicing '{vid}' switch shorthand '{v.switch}' not found in switch '{sw_name}'. "
                                f"Available positions: {list(h_cfg.switches[sw_name].positions.keys())}"
                            )
                        v.switches[sw_name] = v.switch
                    elif len(h_cfg.switches) == 0:
                        raise KeyError(
                            f"Voicing '{vid}' specifies switch='{v.switch}' but harness '{h_name}' has no switches."
                        )
                    else:
                        raise KeyError(
                            f"Voicing '{vid}' uses switch shorthand '{v.switch}' but harness '{h_name}' has multiple switches: "
                            f"{list(h_cfg.switches.keys())}. Specify explicit switches = {{ ... }}."
                        )
                for sw_name, pos_name in v.switches.items():
                    if sw_name not in h_cfg.switches:
                        raise KeyError(
                            f"Voicing '{vid}' sets unknown switch '{sw_name}' on harness '{h_name}'. "
                            f"Available switches: {list(h_cfg.switches.keys())}"
                        )
                    if pos_name not in h_cfg.switches[sw_name].positions:
                        raise KeyError(
                            f"Voicing '{vid}' sets unknown position '{pos_name}' for switch '{sw_name}'. "
                            f"Available positions: {list(h_cfg.switches[sw_name].positions.keys())}"
                        )

        return self

    @classmethod
    def load(cls, identifier_or_path: str | Path) -> InstrumentConfig:
        """Loads and validates an instrument configuration."""
        from allomorph.config.instruments import load_instrument

        return load_instrument(identifier_or_path)


class PackBundleConfig(AllomorphBaseModel):
    """
    Tone3000 upload bundle (folder containing 1 dry stem + target wet stems).
    Supports canonical position affinities ('neck', 'bridge', 'parallel', 'direct')
    as well as custom character bundle names (e.g. 'bridge_clank', 'slab_dub').
    """

    name: str = Field(..., description="Bundle folder name (e.g. 'neck', 'bridge', 'bridge_clank')")
    source_voicing: str = Field(
        ..., description="Voicing ID on the source instrument generating the single dry WAV"
    )
    targets: list[str] = Field(default_factory=list, description="Target voice IDs or tone slugs to simulate")


class TonePackConfig(AllomorphBaseModel):
    """Declarative storefront pack configuration for an instrument."""

    id: str
    instrument: str = Field(..., description="ID of source instrument in config/instruments/")
    name: str = ""
    description: str = ""
    version: int = Field(1, ge=1)
    bundles: list[PackBundleConfig] = Field(default_factory=list)


# ==============================================================================
# 7. TARGET VOICE SCHEMAS
# ==============================================================================


class VoiceCoilConfig(AllomorphBaseModel):
    """Target voice coil sensing aperture geometry."""

    position_from_bridge_m: float = Field(..., gt=0.0)
    aperture_width_in: float = Field(0.75, gt=0.0)
    weight: float = Field(1.0, gt=0.0)
    polarity: float = 1.0
    strings: list[int | str] = Field(default_factory=lambda: ["all"])
    pole_type: str | None = None


class VoicePickupConfig(AllomorphBaseModel):
    """Multi-pickup target voice sub-pickup configuration."""

    name: str
    type: str = "single_coil"
    magnet_type: str | None = None
    alpha: float | None = None
    fr: float = Field(..., gt=0.0)
    Q: float = Field(..., gt=0.0)
    weight: float = Field(1.0, gt=0.0)
    polarity: float = 1.0
    coils: list[VoiceCoilConfig] = Field(default_factory=list)


class VoiceConfig(AllomorphBaseModel):
    """Complete target digital twin voice specification."""

    id: str
    name: str
    version: int = Field(1, ge=1, description="Target voice RLC netlist configuration version")
    tone_name: str | None = None
    description: str
    magnet_type: str | None = None
    sensor_type: Literal["magnetic", "bridge_force", "direct"] = "magnetic"
    target_string: str | None = None
    alpha: float = 0.0
    vsat: float | None = None
    fr: float = Field(..., gt=0.0)
    Q: float = Field(..., gt=0.0)
    gain_db: float = 0.0
    scale: str = "34in"
    preserve_aperture: bool = False
    coils: list[VoiceCoilConfig] = Field(default_factory=list)
    pickups: list[VoicePickupConfig] | None = None
    instrument_id: str | None = None

    @classmethod
    def load(cls, identifier_or_path: str | Path) -> VoiceConfig:
        """Loads and validates a target voice configuration."""
        from allomorph.config.voices import VOICES, load_voice_config

        if isinstance(identifier_or_path, str) and identifier_or_path in VOICES:
            return VOICES[identifier_or_path]
        return load_voice_config(identifier_or_path)
