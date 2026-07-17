"""`meerkat improve-orientation` -- refine geometry against indexed spots.

Ported from improve_orientation_v0.82.py.
"""

from __future__ import annotations

import os

import numpy as np

from ..refine.orientation import (
    REFINABLE,
    XDS_SPOT_OFFSET,
    apply_xds_corrections,
    det2hkl,
    import_instrument_parameters,
    parse_cell_restraints,
    refine_orientation,
)
from ..xds import read_spot_xds, read_xparm, write_xparm

__all__ = ["add_arguments", "run"]


def add_arguments(parser):
    parser.add_argument(
        "--xds-folder", help="directory containing SPOT.XDS and XPARM.XDS"
    )
    parser.add_argument("--spot-file", help="SPOT.XDS (default: <xds-folder>/SPOT.XDS)")
    parser.add_argument(
        "--input-xparm", help="starting XPARM (default: <xds-folder>/XPARM.XDS)"
    )
    parser.add_argument(
        "-o", "--output-file", help="refined XPARM (default: <xds-folder>/MXPARM.XDS)"
    )

    selection = parser.add_argument_group("spot selection")
    selection.add_argument("--imin", type=float, default=0.0, help="minimum spot intensity")
    selection.add_argument("--imax", type=float, default=np.inf, help="maximum spot intensity")
    selection.add_argument("--frame-min", type=float, default=0.0, help="first frame to use")
    selection.add_argument("--frame-max", type=float, default=np.inf, help="last frame to use")
    selection.add_argument(
        "--dr",
        type=float,
        default=0.08,
        help="keep spots within this distance (in hkl) of an integer reflection",
    )
    selection.add_argument(
        "--dr-schedule",
        help="comma-separated dr values for sequential refinement, tightening as it "
        "goes, e.g. '0.25,0.15,0.08,0.04'. Each pass re-selects from the full spot "
        "list using the geometry refined so far, so a better fit brings more spots in. "
        "Overrides --dr.",
    )
    selection.add_argument(
        "--filter-fcc",
        action="store_true",
        help="keep only reflections allowed by an F-centred lattice",
    )
    selection.add_argument(
        "--no-xds-corrections",
        action="store_true",
        help="ignore X/Y-CORRECTIONS.cbf even if present",
    )

    what = parser.add_argument_group("what to refine")
    what.add_argument(
        "--refine",
        default="xycenter beam cell axis",
        help="space-separated subset of: " + " ".join(REFINABLE) + ". Note 'distance' "
        "is excluded by default: it is nearly degenerate with the overall cell scale.",
    )
    what.add_argument(
        "--instrument-xparm",
        help="XPARM from a standard sample whose instrument geometry was already "
        "refined. Its detector position, beam centre, pixel size and rotation axis "
        "replace this experiment's. Crystal and scan parameters are never imported.",
    )
    what.add_argument(
        "--allow-wavelength-mismatch",
        action="store_true",
        help="permit --instrument-xparm from a different wavelength (usually a mistake)",
    )
    what.add_argument(
        "--cell-restraints",
        help="restrain the cell by symmetry: six comma-separated slots for "
        "a,b,c,alpha,beta,gamma, where a number pins a value, a repeated label ties "
        "parameters together, and '*' leaves one free. Hexagonal is 'a,a,c,90,90,120'; "
        "cubic is 'a,a,a,90,90,90'.",
    )
    what.add_argument(
        "--restraint-weight",
        type=float,
        default=1.0,
        help="strength of --cell-restraints, normalized against the number of spots: "
        "1 trusts the symmetry about as much as the data, 10 mostly imposes it, 0.1 "
        "is a gentle nudge.",
    )

    parser.add_argument("--print-hkl", action="store_true", help="print the indexed hkl")
    return parser


def _resolve_paths(args):
    if args.xds_folder is None and (args.spot_file is None or args.input_xparm is None):
        raise SystemExit(
            "error: give --xds-folder, or both --spot-file and --input-xparm.\n"
            "  (improve_orientation_v0.82.py silently built the path 'None/XPARM.XDS' "
            "here.)"
        )

    spot_file = args.spot_file or os.path.join(args.xds_folder, "SPOT.XDS")
    xparm = args.input_xparm or os.path.join(args.xds_folder, "XPARM.XDS")
    output = args.output_file or os.path.join(args.xds_folder or ".", "MXPARM.XDS")

    for path, what in ((spot_file, "spot file"), (xparm, "XPARM")):
        if not os.path.exists(path):
            raise SystemExit(f"error: {what} not found: {path}")

    return spot_file, xparm, output


def run(args) -> int:
    spot_file, xparm_path, output_file = _resolve_paths(args)

    params = read_xparm(xparm_path)

    if args.instrument_xparm:
        try:
            params = import_instrument_parameters(
                params,
                args.instrument_xparm,
                check_wavelength=not args.allow_wavelength_mismatch,
            )
        except ValueError as exc:
            raise SystemExit(f"error: {exc}") from None
        print(f"imported instrument geometry from {args.instrument_xparm}")

    refine = set(args.refine.split())
    unknown = refine - set(REFINABLE)
    if unknown:
        raise SystemExit(
            f"error: unknown refinable parameter(s): {' '.join(sorted(unknown))}\n"
            f"  choose from: {' '.join(REFINABLE)}"
        )

    if args.instrument_xparm:
        # Refining what you just imported throws away the standard sample's work.
        conflicting = refine & {"distance", "xycenter", "axis"}
        if conflicting:
            print(
                f"warning: --refine names {' '.join(sorted(conflicting))}, which "
                f"--instrument-xparm just imported; dropping them from the refinement."
            )
            refine -= conflicting

    restraints = None
    if args.cell_restraints:
        try:
            restraints = parse_cell_restraints(args.cell_restraints)
        except ValueError as exc:
            raise SystemExit(f"error: {exc}") from None
        if "cell" not in refine:
            print("warning: --cell-restraints given but 'cell' is not being refined; ignoring")
            restraints = None
        else:
            print(f"cell restraints: {restraints.describe()}")

    spots = read_spot_xds(spot_file)
    intensity, frame = spots[:, 3], spots[:, 2]
    keep = (
        (intensity > args.imin)
        & (intensity < args.imax)
        & (frame >= args.frame_min)
        & (frame <= args.frame_max)
    )
    spots = spots[keep]
    if len(spots) == 0:
        raise SystemExit(
            "error: no spots left after the intensity and frame cuts. "
            "Check --imin/--imax/--frame-min/--frame-max."
        )

    # Convert to the det2lab_xds convention BEFORE anything selects on hkl. The
    # original applied this after filtering, so the cut was made on hkl that were
    # wrong by a pixel and half an oscillation.
    xyf = spots[:, :3] + XDS_SPOT_OFFSET

    if not args.no_xds_corrections:
        try:
            xyf, applied = apply_xds_corrections(xyf, os.path.dirname(os.path.abspath(spot_file)))
        except ImportError as exc:
            raise SystemExit(f"error: {exc}") from None
        if applied:
            print("applied X/Y-CORRECTIONS.cbf")

    if args.filter_fcc:
        h, k, l = np.around(det2hkl(xyf, params))
        xyf = xyf[(np.mod(h, 2) == np.mod(l, 2)) & (np.mod(k, 2) == np.mod(l, 2))]

    print(f"{len(xyf)} spots after selection")

    if args.print_hkl:
        print(np.around(det2hkl(xyf, params)).T)

    schedule = None
    if args.dr_schedule:
        try:
            schedule = [float(v) for v in args.dr_schedule.split(",")]
        except ValueError:
            raise SystemExit(
                f"error: --dr-schedule must be comma-separated numbers, got "
                f"{args.dr_schedule!r}"
            ) from None
        if sorted(schedule, reverse=True) != schedule:
            print("warning: --dr-schedule is not decreasing; it is meant to tighten")

    print(f"refining: {' '.join(sorted(refine))}")
    try:
        result = refine_orientation(
            params,
            xyf,
            refine=refine,
            dr=args.dr,
            dr_schedule=schedule,
            restraints=restraints,
            restraint_weight=args.restraint_weight,
        )
    except ValueError as exc:
        raise SystemExit(f"error: {exc}") from None

    if not result.success:
        print(f"warning: least_squares did not converge: {result.message}")

    print(f"cell: {' '.join(f'{v:.4f}' for v in result.params['cell'])}")
    write_xparm(output_file, result.params)
    print(f"wrote {output_file}")
    return 0
