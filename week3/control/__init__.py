"""Week-3 control package.

Single responsibility per module — controllers, observer, plant, reachability,
readouts, closed-loop driver. Only :mod:`control.control_interface` imports the
:mod:`estimator` package; controllers / observer / reachability consume only
``(A, B, C, Q, R, a, c)``.
"""

from . import (  # noqa: F401
    closed_loop,
    control_interface,
    controllers,
    observer,
    plant,
    probes,
    reachability,
    readouts,
)
