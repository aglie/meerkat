"""`meerkat view` -- open the Ewald sphere viewer.

The one subcommand that needs a GUI stack, so it is also the one that has to
explain itself when that stack is not there.
"""

from __future__ import annotations

import os

__all__ = ["MISSING_DEPENDENCIES", "add_arguments", "run"]

MISSING_DEPENDENCIES = (
    "the viewer needs PyQt5 and PyOpenGL, which meerkat does not install by "
    "default.\nInstall them with `pip install meerkat[viewer]`, or, under conda, "
    "`conda install pyqt pyopengl`."
)


def add_arguments(parser):
    parser.add_argument(
        "experiment",
        nargs="?",
        help="XPARM.XDS, GXPARM.XDS or XDS.INP to open at startup, or a directory "
        "holding one. SPOT.XDS is read from the same directory. Omit it and use "
        "File > Open.",
    )
    return parser


def run(args) -> int:
    if args.experiment is not None and not os.path.exists(args.experiment):
        raise SystemExit(f"error: {args.experiment} not found")

    try:
        from ..viewer.app import main as run_viewer
    except ImportError as exc:
        raise SystemExit(f"error: {MISSING_DEPENDENCIES}\n({exc})") from None

    return run_viewer([args.experiment] if args.experiment else [])
