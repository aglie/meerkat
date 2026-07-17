"""`meerkat info` -- summarize an XPARM or a reconstruction, for humans."""

from __future__ import annotations

import os

import numpy as np

__all__ = ["add_arguments", "run"]


def add_arguments(parser):
    parser.add_argument("file", help="XPARM.XDS / GXPARM.XDS, a directory holding one, or a .h5")
    return parser


def run(args) -> int:
    path = args.file
    if os.path.isdir(path) or _looks_like_xparm(path):
        return _xparm_info(path)
    return _h5_info(path)


def _looks_like_xparm(path):
    if path.lower().endswith((".h5", ".hdf5", ".nxs")):
        return False
    try:
        with open(path, "rb") as f:
            return b"XPARM" in f.readline()
    except OSError:
        return False


def _xparm_info(path) -> int:
    from ..xds import read_xparm
    from ..xds.xparm import _scalar

    p = read_xparm(path)
    cell = p["cell"]
    print(f"XPARM: {path}")
    print(f"  cell            : {' '.join(f'{v:.4f}' for v in cell[:3])}  "
          f"{' '.join(f'{v:.3f}' for v in cell[3:])}")
    print(f"  space group     : {int(_scalar(p['space_group_nr']))}")
    print(f"  wavelength      : {_scalar(p['wavelength']):.6f} A")
    print(f"  detector        : {int(_scalar(p['NX']))} x {int(_scalar(p['NY']))} px "
          f"@ {_scalar(p['pixelsize_x'])} x {_scalar(p['pixelsize_y'])} mm")
    print(f"  distance        : {_scalar(p['distance_to_detector']):.3f} mm")
    print(f"  beam centre     : {_scalar(p['x_center']):.2f}, {_scalar(p['y_center']):.2f} px")
    print(f"  oscillation     : {_scalar(p['oscillation_angle'])} deg/frame, "
          f"starting at {_scalar(p['starting_angle'])} deg on frame "
          f"{int(_scalar(p['starting_frame']))}")
    print(f"  rotation axis   : {' '.join(f'{v:.6f}' for v in p['rotation_axis'])}")
    print("  cell vectors (rows are a, b, c):")
    for name, row in zip("abc", p["unit_cell_vectors"]):
        print(f"    {name} : {' '.join(f'{v:10.6f}' for v in row)}")
    return 0


def _h5_info(path) -> int:
    import h5py

    with h5py.File(path, "r") as f:
        print(f"reconstruction: {path}")
        fmt = f["format"][()] if "format" in f else None
        if isinstance(fmt, bytes):
            fmt = fmt.decode()
        print(f"  format          : {fmt}")

        for name in ("data", "rebinned_data", "number_of_pixels_rebinned"):
            if name in f:
                d = f[name]
                print(f"  {name:16s}: {d.shape} {d.dtype}")

        for key in ("unit_cell", "lower_limits", "step_sizes", "space_group_nr", "is_direct"):
            if key in f:
                value = f[key][()]
                print(f"  {key:16s}: {np.ravel(value) if np.ndim(value) else value}")

        if "data" in f:
            grid = f["data"]
            lower = np.ravel(f["lower_limits"][()]) if "lower_limits" in f else None
            step = np.ravel(f["step_sizes"][()]) if "step_sizes" in f else None
            if lower is not None and step is not None:
                upper = lower + step * (np.array(grid.shape) - 1)
                print("  hkl range       : "
                      + ", ".join(f"[{lo:g}, {hi:g}]" for lo, hi in zip(lower, upper)))

        if "meerkat_provenance" in f:
            print("  provenance      : present (meerkat dump-config to extract)")
        else:
            print("  provenance      : none (written by meerkat < 0.4 or another tool)")
    return 0
