"""
Tests for target parsing, validation, and resolution in allomorph.cli.targets.
"""

import pytest

from allomorph.cli.targets import (
    ResolvedTarget,
    get_available_instrument_ids,
    get_available_pack_ids,
    parse_target_token,
    resolve_targets,
)


def test_get_available_ids() -> None:
    packs = get_available_pack_ids()
    assert len(packs) > 0
    assert "34in_standard_p" in packs

    insts = get_available_instrument_ids()
    assert len(insts) > 0
    assert "34in_standard_p" in insts


def test_parse_target_token_packs() -> None:
    targets = parse_target_token("packs")
    assert len(targets) > 0
    assert all(t.kind == "pack" for t in targets)
    assert any(t.id == "34in_standard_p" for t in targets)


def test_parse_target_token_instruments() -> None:
    targets = parse_target_token("instruments")
    assert len(targets) > 0
    assert all(t.kind == "instrument" for t in targets)
    assert any(t.id == "34in_standard_p" for t in targets)


def test_parse_target_token_specific_pack() -> None:
    targets = parse_target_token("pack:34in_standard_p")
    assert len(targets) == 1
    assert targets[0] == ResolvedTarget(kind="pack", id="34in_standard_p", sub_target=None)


def test_parse_target_token_pack_with_tone() -> None:
    targets = parse_target_token("pack:34in_standard_p:precision_vintage")
    assert len(targets) == 1
    assert targets[0] == ResolvedTarget(
        kind="pack", id="34in_standard_p", sub_target="precision_vintage"
    )


def test_parse_target_token_specific_instrument() -> None:
    targets = parse_target_token("instrument:34in_standard_p")
    assert len(targets) == 1
    assert targets[0] == ResolvedTarget(kind="instrument", id="34in_standard_p", sub_target=None)


def test_parse_target_token_instrument_with_voicing() -> None:
    targets = parse_target_token("instrument:34in_standard_p:precision_vintage")
    assert len(targets) == 1
    assert targets[0] == ResolvedTarget(
        kind="instrument", id="34in_standard_p", sub_target="precision_vintage"
    )


def test_parse_target_token_shorthand() -> None:
    # All shorthand
    targets = parse_target_token("all", allow_shorthand=True)
    assert len(targets) > 0
    assert all(t.kind == "instrument" for t in targets)

    # Instrument:voicing shorthand
    targets = parse_target_token("34in_standard_p:precision_vintage", allow_shorthand=True)
    assert len(targets) == 1
    assert targets[0] == ResolvedTarget(
        kind="instrument", id="34in_standard_p", sub_target="precision_vintage"
    )

    # Instrument shorthand
    targets = parse_target_token("34in_standard_p", allow_shorthand=True)
    assert len(targets) == 1
    assert targets[0] == ResolvedTarget(kind="instrument", id="34in_standard_p", sub_target=None)


def test_parse_target_token_invalid_pack() -> None:
    with pytest.raises(ValueError, match="Unknown tone pack 'invalid_pack_xyz'"):
        parse_target_token("pack:invalid_pack_xyz")

    with pytest.raises(ValueError, match="Invalid target syntax 'pack'"):
        parse_target_token("pack")


def test_parse_target_token_invalid_instrument() -> None:
    with pytest.raises(ValueError, match="Unknown instrument 'invalid_inst_xyz'"):
        parse_target_token("instrument:invalid_inst_xyz")

    with pytest.raises(ValueError, match="Invalid target syntax 'instrument'"):
        parse_target_token("instrument")


def test_parse_target_token_unrecognized() -> None:
    with pytest.raises(ValueError, match="Unrecognized target expression 'not_a_valid_target'"):
        parse_target_token("not_a_valid_target")


def test_resolve_targets_deduplication() -> None:
    resolved = resolve_targets(
        ["pack:34in_standard_p", "pack:34in_standard_p"],
        default="packs",
    )
    assert len(resolved) == 1
    assert resolved[0].id == "34in_standard_p"
