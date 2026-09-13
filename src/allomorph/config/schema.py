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
from allomorph.circuit.schema import CircuitConfig

warnings.filterwarnings(
    "ignore",
    message=r'.*Field name "register".*shadows an attribute in parent.*',
    category=UserWarning,
)

__all__ = [
    "AllomorphBaseModel",
    "CoilConfig",
    "InstrumentConfig",
    "InstrumentStringsConfig",
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
    "VoiceCoilConfig",
    "VoiceConfig",
    "VoicePickupConfig",
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
    bloom_db: float = 0.0
    pluck_excursion_factor: float = Field(1.0, gt=0.0)
    k_long: float = Field(0.20, ge=0.0)
    bridge_rocking_compliance: float | None = None


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
            bloom_db=preset.bloom_db,
            pluck_excursion_factor=preset.pluck_excursion_factor,
            k_long=preset.k_long,
            bridge_rocking_compliance=preset.bridge_rocking_compliance,
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


class CoilConfig(AllomorphBaseModel):
    """Physical sensing coil aperture geometry and string routing."""

    position_from_bridge_m: float = Field(..., gt=0.0)
    aperture_width_in: float = Field(0.75, gt=0.0)
    weight: float = Field(1.0, gt=0.0)
    polarity: float = 1.0
    strings: list[int | str] = Field(default_factory=lambda: ["all"])
    pole_type: str | None = None


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
    coils: list[CoilConfig] = Field(default_factory=list)
    circuit: CircuitConfig | None = None
    components: list[PickupComponentConfig] = Field(default_factory=list)


# ==============================================================================
# 6. SOURCE INSTRUMENT SCHEMAS
# ==============================================================================


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
    default_pickup: str | None = None
    strings: InstrumentStringsConfig = Field(default_factory=InstrumentStringsConfig)
    pickups: dict[str, PickupConfig] = Field(default_factory=dict)
    pickup_mapping: dict[str, str] = Field(default_factory=dict)

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
        if self.default_pickup and self.pickups and self.default_pickup not in self.pickups:
            raise KeyError(
                f"Instrument '{self.id}' default_pickup '{self.default_pickup}' "
                f"not found in pickups: {list(self.pickups.keys())}"
            )
        return self

    @classmethod
    def load(cls, identifier_or_path: str | Path) -> InstrumentConfig:
        """Loads and validates an instrument configuration."""
        from allomorph.config.instruments import load_instrument

        return load_instrument(identifier_or_path)


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
    topology: str
    description: str
    blend_mode: str | None = None
    magnet_type: str | None = None
    sensor_type: Literal["magnetic", "bridge_force", "direct"] = "magnetic"
    target_string: str | None = None
    alpha: float = 0.0
    alpha3: float | None = None
    eta_hyst: float | None = None
    k_sag: float | None = None
    vsat: float | None = None
    k_eddy: float | None = None
    kappa_orbit: float | None = None
    k_body: float | None = None
    beta_curv: float | None = None
    k_pull: float | None = None
    tau_touch: float | None = None
    chi_mu: float | None = None
    k_dist: float | None = None
    kappa_geom: float | None = None
    k_stein: float | None = None
    k_emf: float | None = None
    lambda_L: float | None = None
    fr: float = Field(..., gt=0.0)
    Q: float = Field(..., gt=0.0)
    gain_db: float = 0.0
    scale: str = "34in"
    hpf: float | None = None
    preserve_aperture: bool = False
    coils: list[VoiceCoilConfig] = Field(default_factory=list)
    pickups: list[VoicePickupConfig] | None = None
    circuit: CircuitConfig

    @classmethod
    def load(cls, identifier_or_path: str | Path) -> VoiceConfig:
        """Loads and validates a target voice configuration."""
        from allomorph.config.voices import VOICES, load_voice_config

        if isinstance(identifier_or_path, str) and identifier_or_path in VOICES:
            return VOICES[identifier_or_path]
        return load_voice_config(identifier_or_path)
