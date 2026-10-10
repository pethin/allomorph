"""
Scale lengths, wave speeds, and physical scale range resolution.
"""

import tomllib
from collections.abc import Sequence
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
CONFIG_DIR = REPO_ROOT / "config"
SCALES_FILE = CONFIG_DIR / "scales.toml"


from allomorph.config.schema import InstrumentConfig, ScaleConfig, ScalesCatalog


def load_scales(config_path: str | Path | None = None) -> dict[str, ScaleConfig]:
    """Loads scale lengths and wave speeds from TOML into validated ScaleConfig models."""
    path = Path(config_path) if config_path else SCALES_FILE
    with open(path, "rb") as f:
        data = tomllib.load(f)
    catalog = ScalesCatalog.model_validate(data)
    return catalog.scales


SCALES: dict[str, ScaleConfig] = load_scales()


def resolve_scale_range(
    inst_or_scale: ScaleConfig
    | InstrumentConfig
    | float
    | tuple[float, float]
    | list[float]
    | Sequence[float]
    | str
    | None = None,
) -> tuple[float, float]:
    """
    Resolves the vibrating scale length range (scale_min_m, scale_max_m) in meters.
    Returns (L, L) for standard single-scale instruments, or (min_m, max_m) for multi-scale.
    If None is provided, defaults to the canonical standard 34" reference scale (0.8636 m).
    """
    if inst_or_scale is None:
        return 0.8636, 0.8636

    if not isinstance(inst_or_scale, (str, bytes, InstrumentConfig, ScaleConfig)) and isinstance(
        inst_or_scale, Sequence
    ):
        seq = list(inst_or_scale)
        if len(seq) == 2 and all(float(v) <= 5.0 for v in seq):
            return float(min(seq)), float(max(seq))
        raise ValueError(f"Invalid scale length tuple/list: {inst_or_scale}")

    if isinstance(inst_or_scale, (int, float)):
        val = float(inst_or_scale)
        val_m = val * 0.0254 if val > 5.0 else val
        return val_m, val_m

    if isinstance(inst_or_scale, str):
        clean_s = inst_or_scale.strip().lower()
        import re

        m_multi = re.match(r"^(\d+(?:\.\d+)?)(?:in)?\s*[-/]\s*(\d+(?:\.\d+)?)\s*in$", clean_s)
        if m_multi:
            return float(m_multi.group(1)) * 0.0254, float(m_multi.group(2)) * 0.0254
        m_single = re.match(r"^(\d+(?:\.\d+)?)\s*in$", clean_s)
        if m_single:
            val_m = float(m_single.group(1)) * 0.0254
            return val_m, val_m

        if inst_or_scale in SCALES:
            s_info = SCALES[inst_or_scale]
            if s_info.is_multiscale:
                min_m = (s_info.scale_min_in or 34.0) * 0.0254
                max_m = (s_info.scale_max_in or 37.0) * 0.0254
                return min_m, max_m
            l_m = s_info.scale_length_m or (
                s_info.scale_length_in * 0.0254 if s_info.scale_length_in else 0.8636
            )
            return l_m, l_m
        try:
            from allomorph.config.instruments import load_instrument

            inst = load_instrument(inst_or_scale)
            return resolve_scale_range(inst)
        except (FileNotFoundError, ValueError, KeyError) as e:
            raise ValueError(f"Unknown scale or instrument identifier '{inst_or_scale}': {e}")

    if isinstance(inst_or_scale, (ScaleConfig, InstrumentConfig)):
        if inst_or_scale.is_multiscale:
            min_in = inst_or_scale.scale_min_in
            max_in = inst_or_scale.scale_max_in or inst_or_scale.scale_length_in or 37.0
            if min_in is not None and max_in is not None:
                return float(min_in) * 0.0254, float(max_in) * 0.0254
        l_in = inst_or_scale.scale_length_in
        if l_in is not None:
            return float(l_in) * 0.0254, float(l_in) * 0.0254
        l_m = inst_or_scale.scale_length_m
        if l_m is not None:
            return float(l_m), float(l_m)
        raise ValueError(f"Configuration object has no valid scale specification: {inst_or_scale}")

    raise TypeError(
        f"Cannot resolve scale range from object of type {type(inst_or_scale)}: {inst_or_scale}"
    )
