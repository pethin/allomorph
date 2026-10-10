"""
Allomorph module execution entrypoint: python -m allomorph
"""

import sys

from allomorph.cli import main

if __name__ == "__main__":
    sys.exit(main() or 0)
