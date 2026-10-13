"""Legal disclaimer loader and formatter for Allomorph storefront descriptions."""

from __future__ import annotations

import functools
import tomllib
from pathlib import Path

from allomorph.config.scales import REPO_ROOT

LEGAL_CONFIG_PATH: Path = REPO_ROOT / "config" / "legal.toml"


@functools.lru_cache(maxsize=1)
def get_legal_disclaimer_text(config_path: Path | None = None) -> str:
    """Loads and formats the canonical license and trademark disclaimer text."""
    path = config_path or LEGAL_CONFIG_PATH
    with open(path, "rb") as f:
        data = tomllib.load(f)

    license_terms = data.get("license", {}).get("terms", "").strip()
    trademark_disc = data.get("trademarks", {}).get("disclaimer", "").strip()

    return (
        "LICENSE & DISCLAIMER\n\n"
        f"License:\n{license_terms}\n\n"
        f"Trademark Disclaimer:\n{trademark_disc}\n"
    )
