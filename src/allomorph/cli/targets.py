"""
Allomorph CLI - Target Grammar Parser & Resolver
Resolves structured target expressions for stems generation, packing, and neural training.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from allomorph.config.instruments import INSTRUMENTS
from allomorph.pipeline.pack import PACKS_DIR

TargetKind = Literal["pack", "instrument"]


@dataclass(frozen=True)
class ResolvedTarget:
    """A fully resolved target specification."""

    kind: TargetKind
    id: str
    sub_target: str | None = None  # tone slug, voicing id, or None for all


def get_available_pack_ids() -> list[str]:
    """Returns sorted list of available tone pack IDs."""
    if not PACKS_DIR.exists():
        return sorted(INSTRUMENTS.keys())
    packs = [p.name for p in PACKS_DIR.iterdir() if p.is_dir() and not p.name.startswith(".")]
    return sorted(packs) if packs else sorted(INSTRUMENTS.keys())


def get_available_instrument_ids() -> list[str]:
    """Returns sorted list of available instrument IDs."""
    return sorted(INSTRUMENTS.keys())


def parse_target_token(token: str, allow_shorthand: bool = False) -> list[ResolvedTarget]:
    """
    Parses a single target token according to the Allomorph target grammar.

    Grammar:
      - 'packs': all tone packs
      - 'pack:<name>': specific pack
      - 'pack:<name>:<tone>': specific tone in pack
      - 'instruments': all instruments
      - 'instrument:<name>': all voicings for instrument
      - 'instrument:<name>:<voicing>': specific voicing on instrument
      - 'all': if allow_shorthand, maps to 'instruments'
      - '<name>:<voicing>': if allow_shorthand, maps to 'instrument:<name>:<voicing>'
      - '<name>': if allow_shorthand and <name> in INSTRUMENTS, maps to 'instrument:<name>'
    """
    raw = token.strip()
    if not raw:
        return []

    available_packs = get_available_pack_ids()
    available_insts = get_available_instrument_ids()

    if raw == "packs":
        return [ResolvedTarget(kind="pack", id=p) for p in available_packs]

    if raw == "instruments":
        return [ResolvedTarget(kind="instrument", id=i) for i in available_insts]

    if allow_shorthand and raw == "all":
        return [ResolvedTarget(kind="instrument", id=i) for i in available_insts]

    parts = raw.split(":")
    prefix = parts[0]

    if prefix == "pack":
        if len(parts) == 1:
            raise ValueError("Invalid target syntax 'pack'. Expected 'pack:<name>' or 'packs'.")
        pack_name = parts[1]
        if pack_name not in available_packs and pack_name not in available_insts:
            raise ValueError(
                f"Unknown tone pack '{pack_name}'. Available packs: {', '.join(available_packs)}"
            )
        tone = parts[2] if len(parts) > 2 else None
        return [ResolvedTarget(kind="pack", id=pack_name, sub_target=tone)]

    if prefix == "instrument":
        if len(parts) == 1:
            raise ValueError(
                "Invalid target syntax 'instrument'. Expected 'instrument:<name>' or 'instruments'."
            )
        inst_name = parts[1]
        if inst_name not in available_insts:
            raise ValueError(
                f"Unknown instrument '{inst_name}'. Available instruments: {', '.join(available_insts)}"
            )
        voicing = parts[2] if len(parts) > 2 else None
        return [ResolvedTarget(kind="instrument", id=inst_name, sub_target=voicing)]

    if allow_shorthand:
        # Check <instrument>:<voicing> shorthand
        if len(parts) == 2 and parts[0] in available_insts:
            return [ResolvedTarget(kind="instrument", id=parts[0], sub_target=parts[1])]
        # Check <instrument> shorthand
        if len(parts) == 1 and parts[0] in available_insts:
            return [ResolvedTarget(kind="instrument", id=parts[0])]

    raise ValueError(
        f"Unrecognized target expression '{raw}'. Expected 'packs', 'pack:<name>[:<tone>]', "
        f"'instruments', or 'instrument:<name>[:<voicing>]'."
    )


def resolve_targets(
    target_tokens: Sequence[str],
    default: str = "packs",
    allow_shorthand: bool = False,
) -> list[ResolvedTarget]:
    """Resolves a collection of target tokens into a deduplicated list of ResolvedTargets."""
    tokens = list(target_tokens) if target_tokens else [default]
    resolved: list[ResolvedTarget] = []
    seen: set[tuple[TargetKind, str, str | None]] = set()

    for tok in tokens:
        items = parse_target_token(tok, allow_shorthand=allow_shorthand)
        for item in items:
            key = (item.kind, item.id, item.sub_target)
            if key not in seen:
                seen.add(key)
                resolved.append(item)

    return resolved
