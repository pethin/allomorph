"""
Physical string catalog presets and instrument/voice string configuration resolution.
"""

import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
CONFIG_DIR = REPO_ROOT / "config"
STRINGS_FILE = CONFIG_DIR / "strings.toml"


from collections.abc import Mapping

from allomorph.config.schema import (
    InstrumentConfig,
    ResolvedStringConfig,
    StringPresetConfig,
    StringsCatalog,
    VoiceConfig,
)


def load_strings_config(config_path: str | Path | None = None) -> dict[str, StringPresetConfig]:
    """Loads physical string catalog from TOML into validated StringPresetConfig models."""
    path = Path(config_path) if config_path else STRINGS_FILE
    if not path.exists():
        empty_res: dict[str, StringPresetConfig] = {}
        return empty_res
    with open(path, "rb") as f:
        data = tomllib.load(f)
    catalog = StringsCatalog.model_validate(data)
    return catalog.strings


STRINGS: dict[str, StringPresetConfig] = load_strings_config()


def get_instrument_string(
    instrument: InstrumentConfig,
    strings: Mapping[str, StringPresetConfig] | None = None,
) -> ResolvedStringConfig:
    """Resolves string configuration for a source instrument."""
    s_block = instrument.strings
    preset = s_block.preset or "roundwound_nickel_standard"
    strings_map = strings if strings is not None else STRINGS

    if preset not in strings_map:
        raise KeyError(
            f"String preset '{preset}' not found in strings catalog ({STRINGS_FILE}). "
            f"Available presets: {list(strings_map.keys())}"
        )

    return ResolvedStringConfig.from_preset_and_overrides(strings_map[preset], s_block)


def get_voice_string(
    voice_cfg: VoiceConfig,
    strings: Mapping[str, StringPresetConfig] | None = None,
) -> StringPresetConfig:
    """Resolves target string configuration for a target voice."""
    preset = voice_cfg.target_string or "roundwound_nickel_standard"
    strings_map = strings if strings is not None else STRINGS

    if preset not in strings_map:
        raise KeyError(
            f"String preset '{preset}' not found in strings catalog ({STRINGS_FILE}). "
            f"Available presets: {list(strings_map.keys())}"
        )
    return strings_map[preset]
