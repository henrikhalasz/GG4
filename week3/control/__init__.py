"""Week-3 control package.

Modular closed-loop controller stack: only ``control_interface`` imports the Week-2
estimator. Controllers and the observer consume ``(A, B, C, Q, R)`` plus a plant
interface — they never see ``estimator.py`` directly.
"""

from . import (  # noqa: F401
    closed_loop,
    control_interface,
    controllers,
    metrics,
    observer,
    plant,
    probes,
    reachability,
)
