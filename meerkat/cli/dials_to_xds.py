"""`meerkat dials-to-xds` -- convert a DIALS experiment to XDS.INP/XPARM.XDS/SPOT.XDS.

The bug-free replacement for `dials.export format=xds`; see `meerkat.dials.to_xds`
for why that's needed.
"""

from __future__ import annotations

import os

__all__ = ["add_arguments", "run"]


def add_arguments(parser):
    parser.add_argument("expt_file", help="DIALS .expt file")
    parser.add_argument("refl_file", help="DIALS .refl file")
    parser.add_argument(
        "-o", "--output-dir", required=True,
        help="directory to write XDS.INP, XPARM.XDS and SPOT.XDS into",
    )
    parser.add_argument(
        "--experiment-index", type=int, default=0,
        help="which experiment to use, for a multi-experiment .expt/.refl (default: 0)",
    )
    return parser


def run(args) -> int:
    for path, what in ((args.expt_file, ".expt file"), (args.refl_file, ".refl file")):
        if not os.path.exists(path):
            raise SystemExit(f"error: {what} not found: {path}")

    from ..dials.to_xds import convert

    try:
        convert(
            args.expt_file, args.refl_file, args.output_dir,
            experiment_index=args.experiment_index,
        )
    except ImportError as exc:
        raise SystemExit(f"error: {exc} (dials-to-xds needs dials and dxtbx)") from None
    except ValueError as exc:
        raise SystemExit(f"error: {exc}") from None

    for name in ("XDS.INP", "XPARM.XDS", "SPOT.XDS"):
        print(f"wrote {os.path.join(args.output_dir, name)}")
    return 0
