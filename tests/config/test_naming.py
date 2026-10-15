"""
Tests for naming authority and Tone3000 basenames in allomorph.naming.
"""

from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st

from allomorph.naming import (
    get_default_input_path,
    get_t3k_basename,
)


def test_get_t3k_basename_single_pickup():
    """Verify single-pickup basename formatting without position brackets."""
    name = get_t3k_basename(tone_name="Precision Vintage", position_name=None, version_tag="v2.1.1")
    assert name == "Precision Vintage v2.1.1"


def test_get_t3k_basename_multi_pickup():
    """Verify multi-pickup basename formatting includes bracketed position."""
    name = get_t3k_basename(tone_name="Jazz Growl", position_name="Bridge", version_tag="v2.1.1")
    assert name == "Jazz Growl [Bridge] v2.1.1"


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


@given(
    st.text(min_size=1, max_size=25),
    st.text(min_size=1, max_size=15),
    st.text(min_size=1, max_size=10),
)
def test_get_t3k_basename_sanitization_and_length_property(tone: str, pos: str, ver: str) -> None:
    """Property test verifying slash sanitization and length bounds on get_t3k_basename."""
    try:
        res = get_t3k_basename(tone_name=tone, position_name=pos, version_tag=ver, max_length=64)
        assert "/" not in res
        assert "\\" not in res
        assert len(res) <= 64
    except ValueError as e:
        assert "exceeds 64 characters" in str(e)


def test_get_default_input_path():
    """Verify default dry input audio path resolution."""
    default_p = get_default_input_path()
    assert default_p.name == "input.wav"
    assert "audio" in str(default_p)

    custom_p = get_default_input_path("/custom/dir")
    assert custom_p == Path("/custom/dir/input.wav")
