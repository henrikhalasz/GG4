"""Shared plotting style and comparison-figure helpers.

``style`` owns the per-method colour map, axis style, and primitive panel
helpers (desired-vs-achieved, holdable region). ``plots`` will own the
multi-panel comparison figures themselves (filled in at M4).
"""

from . import style  # noqa: F401
