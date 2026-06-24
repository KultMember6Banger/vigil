"""Enable `python -m vigil ...` (used by the installed git pre-commit hook)."""

from __future__ import annotations

from vigil.cli import main

if __name__ == '__main__':
    main()
