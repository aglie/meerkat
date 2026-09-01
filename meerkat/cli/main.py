"""The `meerkat` command.

One binary with subcommands rather than several console scripts: `meerkat --help`
lists the whole toolkit, which is worth more than putting
`meerkat-improve-orientation` and friends on everyone's PATH with no discoverability.

argparse, not click: no new dependency, nargs=3 vector options are natural, and the
arguments are generated from PARAMETER_SPEC anyway, so decorator ergonomics would buy
nothing.
"""

from __future__ import annotations

import argparse
import sys

from .._version import __version__

__all__ = ["build_parser", "main"]


def build_parser():
    parser = argparse.ArgumentParser(
        prog="meerkat",
        description="Reciprocal space reconstruction from single-crystal x-ray data.",
    )
    parser.add_argument("--version", action="version", version=f"meerkat {__version__}")

    subparsers = parser.add_subparsers(dest="command", metavar="COMMAND")

    from . import dials_to_xds, dump_config, info, reconstruct, transform_xparm, view

    p = subparsers.add_parser(
        "reconstruct",
        help="reconstruct reciprocal space from a set of frames",
        description="Reconstruct reciprocal space. Parameters may come from a .mrk "
        "config file, from flags, or both -- flags win.",
        # Not ArgumentDefaultsHelpFormatter: every flag uses default=SUPPRESS (that is
        # what makes config-vs-flag precedence work), so it would print
        # "(default: ==SUPPRESS==)". Defaults are rendered by _help() instead.
    )
    reconstruct.add_arguments(p)
    p.set_defaults(func=reconstruct.run)

    p = subparsers.add_parser(
        "transform-xparm",
        help="apply a 3x3 transformation to the cell vectors in an XPARM",
    )
    transform_xparm.add_arguments(p)
    p.set_defaults(func=transform_xparm.run)

    p = subparsers.add_parser(
        "dump-config",
        help="recover the config that made a reconstruction",
        description="Print the .mrk that would reproduce a reconstruction, read back "
        "from the provenance recorded inside it.",
    )
    dump_config.add_arguments(p)
    p.set_defaults(func=dump_config.run)

    p = subparsers.add_parser(
        "info",
        help="summarize an XPARM or a reconstruction",
    )
    info.add_arguments(p)
    p.set_defaults(func=info.run)

    p = subparsers.add_parser(
        "dials-to-xds",
        help="convert a DIALS experiment to XDS.INP/XPARM.XDS/SPOT.XDS",
        description="Bug-free replacement for `dials.export format=xds`. Needs "
        "dials and dxtbx (not required for any other meerkat command). See "
        "meerkat.dials.to_xds for the bugs it works around.",
    )
    dials_to_xds.add_arguments(p)
    p.set_defaults(func=dials_to_xds.run)

    # Registered unconditionally, unlike improve-orientation below: PyQt5 and
    # PyOpenGL are heavy enough that most installs will not have them, and a
    # subcommand that silently does not exist is a worse answer to `meerkat view`
    # than one that says what to install. view.run() does the explaining.
    p = subparsers.add_parser(
        "view",
        help="open the Ewald sphere viewer (needs meerkat[viewer])",
        description="Show the measured spots in reciprocal space, with the "
        "instrument geometry live-editable. Needs PyQt5 and PyOpenGL, which are "
        "not installed with meerkat: `pip install meerkat[viewer]`.",
    )
    view.add_arguments(p)
    p.set_defaults(func=view.run)

    try:
        from . import improve_orientation
    except ImportError:  # scipy is an optional dependency
        pass
    else:
        p = subparsers.add_parser(
            "improve-orientation",
            help="refine experimental geometry against indexed spots",
        )
        improve_orientation.add_arguments(p)
        p.set_defaults(func=improve_orientation.run)

    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if getattr(args, "func", None) is None:
        parser.print_help()
        return 1

    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
