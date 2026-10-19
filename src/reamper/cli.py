"""Cyclopts CLI for the Allomorph Hardware Reamper & NAM Trainer (`reamper`)."""

from __future__ import annotations

from cyclopts import App

from reamper.capture import capture_idle_noise, reamp
from reamper.generate import generate_reamp_input
from reamper.train import train_reamp

app = App(
    name="reamper",
    help="Allomorph Hardware Re-Amping & Native NAM A2 Trainer CLI.",
    version="1.0.0",
)

app.command(generate_reamp_input, name=["generate", "generate-input"])
app.command(reamp, name=["capture", "reamp"])
app.command(capture_idle_noise, name=["capture-noise"])
app.command(train_reamp, name=["train"])


def main() -> None:
    """Entry point for the `reamper` CLI."""
    app()
