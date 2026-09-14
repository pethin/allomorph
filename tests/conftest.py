"""
Pytest configuration and global session fixtures.
"""

from collections.abc import Generator

import pytest

from allomorph.config.scales import REPO_ROOT


@pytest.fixture(scope="session", autouse=True)
def guard_no_audio_pollution() -> Generator[None]:
    """
    Session-level guard fixture verifying that running the test suite does not pollute
    or leave behind newly generated files or directories in the project's audio/ directory.
    """
    audio_dir = REPO_ROOT / "audio"
    before_items = (
        {p for p in audio_dir.rglob("*") if p.name != ".DS_Store"} if audio_dir.exists() else set()
    )
    yield
    after_items = (
        {p for p in audio_dir.rglob("*") if p.name != ".DS_Store"} if audio_dir.exists() else set()
    )
    new_items = sorted(
        str(p.relative_to(REPO_ROOT))
        for p in (after_items - before_items)
        if not (p == audio_dir / "canonical" or p.is_relative_to(audio_dir / "canonical"))
    )
    assert not new_items, (
        f"Test suite polluted the audio directory with {len(new_items)} item(s):\n"
        + "\n".join(new_items)
    )
