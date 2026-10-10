"""
Property-based tests for naming primitives and Tone3000 zero-scroll formatting
in allomorph.naming.
Verifies invariants:
1. Token sanitization idempotence: sanitize(sanitize(x)) == sanitize(x).
2. Filesystem safety: no forward or backward slashes survive sanitization.
3. Tri-part version format: format_tri_part_tag produces canonical 'vX.Y.Z'.
4. Display ceiling bounding: compose_zero_scroll_model_name strictly enforces character limits.
"""

from hypothesis import given
from hypothesis import strategies as st

from allomorph.naming import (
    compose_zero_scroll_model_name,
    format_tri_part_tag,
    sanitize_tone_name_token,
)
from tests.strategies import st_naming_tokens


@given(token=st_naming_tokens())
def test_token_sanitization_idempotence_and_slash_safety(token: str) -> None:
    """Sanitizing a token is idempotent and completely eliminates '/' and '\\'."""
    s1 = sanitize_tone_name_token(token)
    s2 = sanitize_tone_name_token(s1)

    assert s1 == s2, f"Idempotence failed: {s1!r} != {s2!r}"
    assert "/" not in s1
    assert "\\" not in s1


@given(
    dsp_gen=st.integers(min_value=1, max_value=20),
    inst_ver=st.integers(min_value=1, max_value=50),
    voice_ver=st.integers(min_value=1, max_value=50),
)
def test_format_tri_part_tag(dsp_gen: int, inst_ver: int, voice_ver: int) -> None:
    """format_tri_part_tag matches canonical 'v{dsp}.{inst}.{voice}' pattern."""
    tag = format_tri_part_tag(dsp_gen, inst_ver, voice_ver)
    assert tag == f"v{dsp_gen}.{inst_ver}.{voice_ver}"
    assert tag.startswith("v")


@given(
    tone=st.text(
        min_size=1,
        max_size=30,
        alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd", "Zs")),
    ),
    pos=st.one_of(
        st.none(),
        st.text(min_size=1, max_size=15, alphabet=st.characters(whitelist_categories=("Lu", "Ll"))),
    ),
    ver=st.one_of(
        st.none(),
        st.text(
            min_size=1,
            max_size=10,
            alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd", "Po")),
        ),
    ),
    max_len=st.integers(min_value=10, max_value=70),
)
def test_model_name_zero_scroll_ceiling(
    tone: str, pos: str | None, ver: str | None, max_len: int
) -> None:
    """compose_zero_scroll_model_name strictly enforces max_len and raises on overflow."""
    try:
        name = compose_zero_scroll_model_name(
            tone_name=tone, position_tag=pos, version_tag=ver, max_len=max_len
        )
        assert len(name) <= max_len
        assert "/" not in name
        assert "\\" not in name
    except ValueError as exc:
        assert "exceeds" in str(exc)
