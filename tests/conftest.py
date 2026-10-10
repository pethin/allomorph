import os
import sys
import types
from collections.abc import Generator
from typing import Any

# Ensure headless matplotlib raster backend is active
try:
    import matplotlib

    matplotlib.use("Agg")
except ImportError:
    pass

# Headless Tkinter fallback for neural-amp-modeler in environments without python3-tk
if "tkinter" not in sys.modules:
    try:
        import tkinter  # noqa: F401
    except ModuleNotFoundError:

        def _dummy_mainloop(*args: object, **kwargs: object) -> None:
            pass

        class _DummyTkMisc:
            mainloop = _dummy_mainloop

        class _DummyTkWidget:
            def __init__(self, *args: object, **kwargs: object) -> None:
                pass

            def __getattr__(self, name: str) -> Any:
                return _dummy_mainloop

        class _DummyTkModule(types.ModuleType):
            Tk = _DummyTkWidget
            Toplevel = _DummyTkWidget
            Label = _DummyTkWidget
            Button = _DummyTkWidget
            Misc = _DummyTkMisc
            mainloop = _dummy_mainloop

        sys.modules["tkinter"] = _DummyTkModule("tkinter")

import pytest
from hypothesis import Verbosity, settings

from allomorph.config.scales import REPO_ROOT

# Register standard Hypothesis profiles
settings.register_profile("ci", max_examples=100, deadline=None)
settings.register_profile("dev", max_examples=25, deadline=None)
settings.register_profile("debug", max_examples=10, verbosity=Verbosity.verbose, deadline=None)
settings.register_profile("smt", max_examples=10, backend="crosshair", deadline=None)

# Load profile from environment (default to 'dev' for fast local test runs)
settings.load_profile(os.getenv("HYPOTHESIS_PROFILE", "dev"))


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
        if p != audio_dir / "input.wav"
    )
    assert not new_items, (
        f"Test suite polluted the audio directory with {len(new_items)} item(s):\n"
        + "\n".join(new_items)
    )
