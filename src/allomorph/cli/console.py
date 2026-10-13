"""
Allomorph CLI - Rich Console Logging & Terminal Telemetry
Unified terminal styling, formatted tables, badges, and automated TTY detection.
"""

import os
import sys
from collections.abc import Sequence
from typing import Any

from rich.console import Console
from rich.table import Table
from rich.text import Text

_CONSOLE: Console | None = None


def get_console() -> Console:
    """Returns a singleton Rich Console configured for interactive or headless execution."""
    global _CONSOLE
    if _CONSOLE is None:
        is_interactive = sys.stdout.isatty() and "NO_COLOR" not in os.environ
        _CONSOLE = Console(
            no_color=not is_interactive,
            highlight=False,
            force_terminal=is_interactive if is_interactive else False,
        )
    return _CONSOLE


def log_step(badge: str, message: str, style: str = "bold cyan") -> None:
    """Logs a formatted step with bracketed badge."""
    c = get_console()
    badge_text = Text(f"[{badge}]", style=style)
    c.print(badge_text, message)


def log_success(message: str) -> None:
    """Logs a success message with green badge."""
    log_step("SUCCESS", message, style="bold green")


def log_info(message: str) -> None:
    """Logs an informational message."""
    c = get_console()
    c.print(message)


def log_warning(message: str) -> None:
    """Logs a warning message with yellow badge."""
    log_step("WARNING", message, style="bold yellow")


def log_error(message: str) -> None:
    """Logs an error message with red badge."""
    log_step("ERROR", message, style="bold red")


def render_table(
    title: str,
    headers: Sequence[str],
    rows: Sequence[Sequence[Any]],
    border_style: str = "dim",
) -> None:
    """Renders a styled Rich table."""
    c = get_console()
    table = Table(title=title, border_style=border_style, show_header=True)
    for h in headers:
        table.add_column(h, style="bold" if h == headers[0] else None)
    for row in rows:
        table.add_row(*[str(cell) for cell in row])
    c.print(table)
