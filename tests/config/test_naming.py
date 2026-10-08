"""
Tests for naming authority, Tone3000 basenames, and CLI voice/instrument resolvers in allomorph.naming.
"""

from pathlib import Path

import pytest

from allomorph.naming import (
    VOICE_CONCISE_SLUGS,
    get_default_input_path,
    get_instrument_pickup_basename,
    get_t3k_basename,
    resolve_instruments,
    resolve_voices,
)


def test_voice_concise_slugs():
    """Verify standard concise slug mappings exist and are non-empty."""
    assert len(VOICE_CONCISE_SLUGS) >= 20
    assert VOICE_CONCISE_SLUGS["precision_vintage"] == "p_vintage"
    assert VOICE_CONCISE_SLUGS["jazz_pair_open"] == "jazz_pair"
    assert VOICE_CONCISE_SLUGS["dingwall_bridge"] == "dingwall_brg"


def test_get_t3k_basename_single_pickup():
    """Verify single-pickup basename formatting without position brackets."""
    name = get_t3k_basename(tone_name="Precision Vintage", position_name=None, version_tag="v2.1.1")
    assert name == "Precision Vintage v2.1.1"


def test_get_t3k_basename_multi_pickup():
    """Verify multi-pickup basename formatting includes bracketed position."""
    name = get_t3k_basename(tone_name="Jazz Growl", position_name="Bridge", version_tag="v2.1.1")
    assert name == "Jazz Growl [Bridge] v2.1.1"


def test_get_t3k_basename_preserve_aperture():
    """Verify character tone with preserve_aperture=True omits position bracket."""
    name = get_t3k_basename(
        tone_name="Acoustic Upright",
        position_name="Bridge",
        version_tag="v2.1.1",
        preserve_aperture=True,
    )
    assert name == "Acoustic Upright v2.1.1"


def test_get_t3k_basename_slash_sanitization():
    """Verify slashes and backslashes are replaced with Unicode Division Slash."""
    name = get_t3k_basename(tone_name="P/MM Parallel", position_name=r"Bridge\Solo")
    assert "/" not in name
    assert "\\" not in name
    assert "\u2215" in name
    assert name == "P\u2215MM Parallel [Bridge\u2215Solo]"


def test_get_t3k_basename_exceeds_max_length_raises():
    """Verify exceeding max_length raises diagnostic ValueError."""
    long_tone = "A" * 60
    with pytest.raises(ValueError, match="exceeds 64 characters"):
        get_t3k_basename(tone_name=long_tone, version_tag="v2.1.1", max_length=64)


def test_resolve_voices_all_and_none():
    """Verify resolve_voices with 'all' or None returns all configured voices."""
    all_v1 = resolve_voices(None)
    all_v2 = resolve_voices("all")
    assert len(all_v1) >= 20
    assert all_v1 == all_v2


def test_resolve_voices_comma_separated():
    """Verify comma-separated list of voices parses correctly."""
    voices = resolve_voices("precision_vintage,jazz_pair_open")
    assert voices == ["precision_vintage", "jazz_pair_open"]


def test_resolve_voices_prefix_match():
    """Verify prefix matching for voice identifiers."""
    voices = resolve_voices("precision_vin")
    assert "precision_vintage" in voices


def test_resolve_voices_invalid_raises():
    """Verify unknown voice token raises ValueError with close match hint."""
    with pytest.raises(ValueError, match="Unknown target voice identifier"):
        resolve_voices("non_existent_voice_xyz")


def test_resolve_instruments_all_and_none():
    """Verify resolve_instruments returns all playable instruments."""
    all_inst1 = resolve_instruments(None)
    all_inst2 = resolve_instruments("all")
    assert len(all_inst1) >= 8
    assert all_inst1 == all_inst2

    # Comma-separated token containing 'all'
    all_inst3 = resolve_instruments("30in,all")
    assert len(all_inst3) == len(all_inst1)


def test_resolve_instruments_partial_match():
    """Verify partial substring matching for instrument tokens."""
    matches = resolve_instruments("standard_jazz")
    assert "34in_standard_jazz" in matches


def test_resolve_instruments_comma_and_alias():
    """Verify comma-separated aliases resolve to canonical IDs."""
    insts = resolve_instruments("30in,34in")
    assert "30in_emg_mmtw" in insts
    assert "34in_standard_p" in insts


def test_resolve_instruments_invalid_raises():
    """Verify unknown instrument identifier raises ValueError."""
    with pytest.raises(ValueError, match="Unknown source instrument identifier"):
        resolve_instruments("unknown_bass_xyz123")


def test_get_default_input_path():
    """Verify default dry input audio path resolution."""
    default_p = get_default_input_path()
    assert default_p.name == "input.wav"
    assert "audio" in str(default_p)

    custom_p = get_default_input_path("/custom/dir")
    assert custom_p == Path("/custom/dir/input.wav")


def test_get_instrument_pickup_basename():
    """Verify instrument pickup basename formatting."""
    assert get_instrument_pickup_basename("34in_p", "split_p") == "split_p"
    assert get_instrument_pickup_basename("34in_p", None) == "34in_p"
