"""`meerkat transform-xparm` -- re-index a crystal to a different cell setting.

Ported from xparm_transform_0.21.py, which printed "THIS SCRIPT IS BROKEN / PIXEL
SIZE IS WRONG" on every run because its private copy of write_xparm hardcoded a
Pilatus 6M detector. It now uses the canonical meerkat.xds.write_xparm, so that is
fixed by construction rather than by remembering to fix it.
"""

from __future__ import annotations

import numpy as np

from ..xds import read_xparm, vecs2cell, write_xparm

__all__ = ["add_arguments", "run"]


def add_arguments(parser):
    parser.add_argument("-i", "--input", required=True, help="input XPARM.XDS")
    parser.add_argument("-o", "--output", required=True, help="output XPARM.XDS")
    parser.add_argument(
        "-t",
        "--transform",
        type=float,
        nargs=9,
        required=True,
        metavar="T",
        help="3x3 transformation matrix, row-major. The new cell vectors are "
        "T . (a, b, c), so '0 1 0 1 0 0 0 0 -1' swaps a and b and flips c.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="proceed even if the transform does not preserve cell volume",
    )
    return parser


def run(args) -> int:
    transform = np.asarray(args.transform, dtype=float).reshape(3, 3)
    determinant = np.linalg.det(transform)

    if abs(determinant) < 1e-12:
        raise SystemExit(
            f"error: the transformation matrix is singular (det = {determinant:.3g}); "
            "it would collapse the cell."
        )

    if not args.force and not np.isclose(abs(determinant), 1.0, atol=1e-6):
        raise SystemExit(
            f"error: |det(T)| = {abs(determinant):.6g}, so this transform changes the "
            f"cell volume by that factor.\n"
            "  That is legitimate for going to a super- or sub-cell, but it is more "
            "often a typo.\n"
            "  Pass --force if you meant it."
        )

    params = read_xparm(args.input)
    params["unit_cell_vectors"] = np.dot(transform, params["unit_cell_vectors"])
    params["cell"] = vecs2cell(params["unit_cell_vectors"])
    write_xparm(args.output, params)

    print(f"wrote {args.output}")
    print(f"  cell: {' '.join(f'{v:.4f}' for v in params['cell'])}")
    if determinant < 0:
        print(
            "  note: det(T) < 0, so this transform inverts handedness "
            "(it includes a reflection)."
        )
    print(
        "  note: space group number was copied unchanged. A cell transformation "
        "generally changes the space-group setting -- check it."
    )
    return 0
