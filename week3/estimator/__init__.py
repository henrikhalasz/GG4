"""Week-3 estimator package.

The deployed identifier — N4SID warm-start then affine EM with known inputs.

Spec rule #2: this package is imported only by ``control.control_interface``.
The Week-2 ``estimator.py`` is never touched.
"""

from . import identify  # noqa: F401
