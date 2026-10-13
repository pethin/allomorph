"""
Allomorph - Modern Modular CLI Package
Powered by Cyclopts, Rich, and First-Principles Electroacoustic Modeling.
"""

import sys
from collections.abc import Sequence

from cyclopts import App, CycloptsError

from allomorph.cli.audit_cmd import audit
from allomorph.cli.console import log_error
from allomorph.cli.list_cmd import list_app
from allomorph.cli.pack_cmd import pack
from allomorph.cli.stems_cmd import generate_stems, prune_stems
from allomorph.cli.train_cmd import native_group, stock_group, train
from allomorph.cli.visualize_cmd import visualize

# ------------------------------------------------------------------------------
# Root Cyclopts Application
# ------------------------------------------------------------------------------
app = App(
    name="allomorph",
    help="Allomorph - Analog digital twin & universal pickup/transducer modeling engine.",
    version="0.5.0",
)

# 1. Visualize Command (with 'viz' alias)
app.command(visualize, name=["visualize", "viz"])

# 2. Stems Sub-App
stems_app = App(
    name="stems",
    help="Manage forward digital twin wet stem synthesis and orphan stem pruning.",
)
stems_app.command(generate_stems, name="generate")
stems_app.command(prune_stems, name="prune")
app.command(stems_app)

# 3. Pack Command
app.command(pack, name="pack")

# 4. Train Command
app.command(train, name="train")

# 5. Audit Command
app.command(audit, name="audit")

# 6. List Sub-App
app.command(list_app)


def main(argv: Sequence[str] | None = None) -> int:
    """Main CLI entrypoint for the 'allomorph' command."""
    tokens = list(argv if argv is not None else sys.argv[1:])

    # Dynamic Engine Options Help Filtering
    if "train" in tokens and any(h in tokens for h in ("--help", "-h")):
        if "--engine" in tokens:
            idx = tokens.index("--engine")
            if idx + 1 < len(tokens):
                eng = tokens[idx + 1]
                if eng == "stock":
                    object.__setattr__(native_group, "_show", False)
                    object.__setattr__(stock_group, "_show", True)
                elif eng == "native":
                    object.__setattr__(stock_group, "_show", False)
                    object.__setattr__(native_group, "_show", True)
                else:
                    object.__setattr__(native_group, "_show", True)
                    object.__setattr__(stock_group, "_show", True)
        else:
            object.__setattr__(native_group, "_show", True)
            object.__setattr__(stock_group, "_show", True)

    old_argv = sys.argv
    if argv is not None:
        sys.argv = [sys.argv[0]] + list(argv)
    try:
        app(tokens, exit_on_error=True)
        return 0
    except CycloptsError as e:
        log_error(str(e))
        return 2
    except ValueError as e:
        log_error(str(e))
        return 1
    except KeyboardInterrupt:
        print("\nAborted by user.")
        return 130
    except SystemExit as e:
        return int(e.code) if isinstance(e.code, int) else (0 if e.code is None else 1)
    finally:
        sys.argv = old_argv


if __name__ == "__main__":
    sys.exit(main())
