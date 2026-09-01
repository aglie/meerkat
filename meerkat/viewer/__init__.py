"""The Ewald sphere viewer -- a Qt/OpenGL window showing where the measured
spots sit in reciprocal space, with the instrument geometry live-editable.

Needs PyQt5 and PyOpenGL, which are NOT installed with meerkat:

    pip install meerkat[viewer]

Importing this package does not import Qt. `meerkat.viewer.experiment` is
numpy-only and is where the geometry lives; the Qt names below resolve on first
access, so `python -c "import meerkat.viewer"` works on a headless machine and
CI can test the geometry without a display.
"""

from __future__ import annotations

from .experiment import (
    Experiment,
    filter_spots,
    load_experiment,
    locate_experiment,
    ordered_parameters,
    read_parameters,
    spot_q_vectors,
    top_n_reflections,
)

__all__ = [
    "Experiment",
    "Window",
    "filter_spots",
    "load_experiment",
    "locate_experiment",
    "main",
    "ordered_parameters",
    "read_parameters",
    "spot_q_vectors",
    "top_n_reflections",
]

_LAZY = {"Window": "app", "main": "app"}


def __getattr__(name):
    """PEP 562: resolve the Qt-dependent names only when they are touched."""
    if name in _LAZY:
        import importlib

        module = importlib.import_module(f".{_LAZY[name]}", __name__)
        value = getattr(module, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(__all__)
