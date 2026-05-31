"""Path helper used by the test suite.

Adds ``week3/`` (for the ``control`` package) and ``week 1/`` (for ``Simulator``)
to ``sys.path``. The space in ``week 1`` rules out a normal package import.
"""

from __future__ import annotations

import sys
from pathlib import Path

_TESTS_DIR = Path(__file__).resolve().parent
_WEEK3 = _TESTS_DIR.parent
_ROOT = _WEEK3.parent

for p in (str(_WEEK3), str(_ROOT / "week 1")):
    if p not in sys.path:
        sys.path.insert(0, p)
