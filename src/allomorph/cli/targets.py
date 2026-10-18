"""
Allomorph CLI - Target Grammar Parser & Resolver
Resolves structured target expressions for stems generation, packing, and neural training.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from allomorph.config.instruments import INSTRUMENTS, TONE_PACKS

TargetKind = Literal["pack", "instrument", "excitation"]


@dataclass(frozen=True)
class ResolvedTarget:
    """A fully resolved target specification."""

    kind: TargetKind
    id: str
    sub_target: str | None = None  # voicing id, <inst_id>:<voicing_id>, or None for all


def get_available_pack_ids() -> list[str]:
    """Returns sorted list of available tone pack IDs."""
    return sorted(TONE_PACKS.keys()) if TONE_PACKS else sorted(INSTRUMENTS.keys())


def get_available_instrument_ids() -> list[str]:
    """Returns sorted list of available instrument IDs."""
    return sorted(INSTRUMENTS.keys())


def parse_target_token(
    token: str,
    allow_shorthand: bool = False,
    shorthand_kind: Literal["pack", "instrument"] = "instrument",
) -> list[ResolvedTarget]:
    """
    Parses a single target token according to the Allomorph target grammar.

    Grammar:
      - 'target1,target2,...': comma-separated composite targets
      - 'packs': all tone packs
      - 'pack:<name>': specific pack
      - 'pack:<name>:<voicing>': specific voicing in pack
      - 'pack:<name>:<inst>:<voicing>': specific cross-instrument target voicing in pack
      - 'instruments': all instruments
      - 'instrument:<name>': all voicings for instrument
      - 'instrument:<name>:<voicing>': specific voicing on instrument
      - 'all': if allow_shorthand, maps to 'packs' (when shorthand_kind=='pack') or 'instruments'
      - '<name>:<voicing>': if allow_shorthand, maps to '<shorthand_kind>:<name>:<voicing>'
      - '<name>': if allow_shorthand, maps to '<shorthand_kind>:<name>'
    """
    raw = token.strip()
    if not raw:
        return []

    if "," in raw:
        sub_tokens = [part.strip() for part in raw.split(",") if part.strip()]
        res: list[ResolvedTarget] = []
        seen_keys: set[tuple[TargetKind, str, str | None]] = set()
        for sub_tok in sub_tokens:
            for item in parse_target_token(
                sub_tok,
                allow_shorthand=allow_shorthand,
                shorthand_kind=shorthand_kind,
            ):
                key = (item.kind, item.id, item.sub_target)
                if key not in seen_keys:
                    seen_keys.add(key)
                    res.append(item)
        return res

    available_packs = get_available_pack_ids()
    available_insts = get_available_instrument_ids()

    if raw in ("excitation", "dry"):
        return [ResolvedTarget(kind="excitation", id="default")]

    if raw == "packs":
        return [ResolvedTarget(kind="pack", id=p) for p in available_packs]

    if raw == "instruments":
        return [ResolvedTarget(kind="instrument", id=i) for i in available_insts]

    if allow_shorthand and raw == "all":
        if shorthand_kind == "pack":
            return [ResolvedTarget(kind="pack", id=p) for p in available_packs]
        return [ResolvedTarget(kind="instrument", id=i) for i in available_insts]

    parts = raw.split(":")
    prefix = parts[0]

    if prefix in ("excitation", "dry"):
        sub = parts[1] if len(parts) > 1 else None
        return [ResolvedTarget(kind="excitation", id="default", sub_target=sub)]

    if prefix == "pack":
        if len(parts) == 1 or not parts[1]:
            raise ValueError("Invalid target syntax 'pack'. Expected 'pack:<name>' or 'packs'.")
        pack_name = parts[1]
        if pack_name not in available_packs and pack_name not in available_insts:
            raise ValueError(
                f"Unknown tone pack '{pack_name}'. Available packs: {', '.join(available_packs)}"
            )
        if len(parts) == 2:
            tone = None
        elif len(parts) == 3:
            tone = parts[2]
        elif len(parts) == 4:
            tone = f"{parts[2]}:{parts[3]}"
        else:
            raise ValueError(
                f"Invalid pack target syntax '{raw}'. Expected 'pack:<pack_id>[:<inst_id>:<voicing_id>]'."
            )
        return [ResolvedTarget(kind="pack", id=pack_name, sub_target=tone)]

    if prefix == "instrument":
        if len(parts) == 1 or not parts[1]:
            raise ValueError(
                "Invalid target syntax 'instrument'. Expected 'instrument:<name>' or 'instruments'."
            )
        inst_name = parts[1]
        if inst_name not in available_insts:
            raise ValueError(
                f"Unknown instrument '{inst_name}'. Available instruments: {', '.join(available_insts)}"
            )
        if len(parts) > 3:
            raise ValueError(
                f"Invalid instrument target syntax '{raw}'. Expected 'instrument:<inst_id>[:<voicing_id>]'."
            )
        voicing = parts[2] if len(parts) == 3 else None
        return [ResolvedTarget(kind="instrument", id=inst_name, sub_target=voicing)]

    if allow_shorthand:
        if shorthand_kind == "pack":
            if len(parts) == 3 and parts[0] in available_packs:
                return [
                    ResolvedTarget(kind="pack", id=parts[0], sub_target=f"{parts[1]}:{parts[2]}")
                ]
            if len(parts) == 2 and parts[0] in available_packs:
                return [ResolvedTarget(kind="pack", id=parts[0], sub_target=parts[1])]
            if len(parts) == 1 and parts[0] in available_packs:
                return [ResolvedTarget(kind="pack", id=parts[0])]
        # Check <instrument>:<voicing> shorthand
        if len(parts) == 2 and parts[0] in available_insts:
            return [ResolvedTarget(kind="instrument", id=parts[0], sub_target=parts[1])]
        # Check <instrument> shorthand
        if len(parts) == 1 and parts[0] in available_insts:
            return [ResolvedTarget(kind="instrument", id=parts[0])]

    raise ValueError(
        f"Unrecognized target expression '{raw}'. Expected 'excitation', 'packs', 'pack:<name>[:<tone>]', "
        f"'instruments', or 'instrument:<name>[:<voicing>]'."
    )


def resolve_targets(
    target_tokens: Sequence[str],
    default: str = "packs",
    allow_shorthand: bool = False,
    shorthand_kind: Literal["pack", "instrument"] = "instrument",
) -> list[ResolvedTarget]:
    """Resolves a collection of target tokens into a deduplicated list of ResolvedTargets."""
    tokens = list(target_tokens) if target_tokens else [default]
    resolved: list[ResolvedTarget] = []
    seen: set[tuple[TargetKind, str, str | None]] = set()

    for tok in tokens:
        items = parse_target_token(
            tok, allow_shorthand=allow_shorthand, shorthand_kind=shorthand_kind
        )
        for item in items:
            key = (item.kind, item.id, item.sub_target)
            if key not in seen:
                seen.add(key)
                resolved.append(item)

    return resolved
