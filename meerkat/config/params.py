"""One definition per reconstruction parameter.

PARAMETER_SPEC is the single source of truth. It drives the dataclass, the argparse
parser, the .mrk parser, and the .mrk dumper. That matters more than it looks: with
two driving interfaces plus a provenance record, a parameter defined in one place and
forgotten in another means the CLI and the config file drift AND the provenance lies
about what was run. test_spec_matches_dataclass is what keeps them honest.

Grid handling is the fiddly part; see resolve_grid.
"""

from __future__ import annotations

import difflib
from dataclasses import dataclass, field, fields
from typing import Any, Callable

import numpy as np

__all__ = [
    "PARAMETER_SPEC",
    "ConfigError",
    "Grid",
    "ReconstructionParameters",
    "resolve_grid",
    "spec_by_keyword",
    "spec_by_name",
]


class ConfigError(Exception):
    """A configuration problem, reported with file/line context where available."""

    def __init__(self, message, filename=None, lineno=None, line_text=None, column=None):
        self.message = message
        self.filename = filename
        self.lineno = lineno
        self.line_text = line_text
        self.column = column
        super().__init__(str(self))

    def __str__(self):
        if self.filename is None or self.lineno is None:
            return self.message
        out = [f"{self.filename}:{self.lineno}: {self.message}"]
        if self.line_text is not None:
            out.append(f"  {self.line_text}")
            if self.column is not None:
                out.append("  " + " " * self.column + "^")
        return "\n".join(out)


@dataclass(frozen=True)
class ParamSpec:
    name: str  # python / dataclass field name
    keyword: str  # .mrk keyword
    type: Callable[[str], Any]
    nargs: int | None  # None = single value, N = exactly N, 0 = flag
    default: Any
    help: str

    @property
    def flag(self) -> str:
        return "--" + self.name.replace("_", "-")


def _path(s):
    return str(s)


PARAMETER_SPEC: tuple[ParamSpec, ...] = (
    # --- input ---------------------------------------------------------------
    ParamSpec(
        "data_file_template", "DATA_FILE_TEMPLATE", _path, None, None,
        "printf-style template for the frames, e.g. 'frames/img_%05i.cbf'",
    ),
    ParamSpec(
        "xparm_file", "XPARM_FILE", _path, None, ".",
        "XPARM.XDS or GXPARM.XDS file, or a directory containing one",
    ),
    ParamSpec("first_frame", "FIRST_FRAME", int, None, None, "first frame number to reconstruct"),
    ParamSpec("last_frame", "LAST_FRAME", int, None, None, "last frame number to reconstruct"),
    ParamSpec(
        "mask", "MASK", _path, None, None,
        "image whose negative pixels mark untrusted detector areas "
        "(default: derive from the first frame)",
    ),
    ParamSpec(
        "scales", "SCALES", _path, None, None,
        "text file with one scale factor per frame",
    ),
    # --- grid ----------------------------------------------------------------
    ParamSpec(
        "number_of_pixels", "NUMBER_OF_PIXELS", int, 3, None,
        "output grid dimensions, e.g. 501 501 501",
    ),
    ParamSpec("lower_limits", "LOWER_LIMITS", float, 3, None, "hkl of voxel [0,0,0]"),
    ParamSpec("upper_limits", "UPPER_LIMITS", float, 3, None, "hkl of the last voxel"),
    ParamSpec("step_sizes", "STEP_SIZES", float, 3, None, "grid step in r.l.u."),
    ParamSpec(
        "symmetric_limits", "SYMMETRIC_LIMITS", bool, 0, False,
        "mirror the given limits about the origin (upper = -lower)",
    ),
    # --- physics -------------------------------------------------------------
    ParamSpec(
        "polarization_factor", "POLARIZATION_FACTOR", float, None, 1.0,
        "1 for synchrotron, 0.5 for an unpolarized laboratory source",
    ),
    ParamSpec(
        "polarization_plane_normal", "POLARIZATION_PLANE_NORMAL", float, 3, (0.0, 1.0, 0.0),
        "normal of the polarization plane",
    ),
    ParamSpec(
        "medium", "MEDIUM", str, None, "Air",
        "medium between crystal and detector: Air or Helium",
    ),
    ParamSpec(
        "unit_cell_transform", "UNIT_CELL_TRANSFORM", float, 9, None,
        "3x3 matrix (row-major) applied to the cell vectors before reconstruction",
    ),
    ParamSpec(
        "reconstruct_in_orthonormal_basis", "RECONSTRUCT_IN_ORTHONORMAL_BASIS", bool, 0, False,
        "reconstruct on an orthonormal basis instead of the crystal's",
    ),
    # --- output --------------------------------------------------------------
    ParamSpec(
        "output_filename", "OUTPUT_FILENAME", _path, None, "reconstruction.h5",
        "output HDF5 file",
    ),
    ParamSpec(
        "output_format", "OUTPUT_FORMAT", str, None, "YELL_1.0",
        "YELL_1.0 writes a single averaged 'data' array; YELL_0.9 keeps "
        "'rebinned_data' and 'number_of_pixels_rebinned' separately",
    ),
    ParamSpec("overwrite", "OVERWRITE", bool, 0, False, "overwrite the output file if it exists"),
    ParamSpec(
        "all_in_memory", "ALL_IN_MEMORY", bool, 0, False,
        "hold the whole grid in RAM. Faster, but a 801^3 grid needs ~4.1 GB "
        "(2.06 GB data + 2.06 GB counts); without this the accumulation is done "
        "out-of-core through HDF5",
    ),
    ParamSpec("size_of_cache", "SIZE_OF_CACHE", int, None, 100, "HDF5 chunk cache size in MB"),
    # --- not implemented -----------------------------------------------------
    ParamSpec(
        "microstep_frames", "MICROSTEP_FRAMES", int, None, None,
        "[not implemented] subdivide each frame's rotation into N microsteps",
    ),
)

_BY_NAME = {s.name: s for s in PARAMETER_SPEC}
_BY_KEYWORD = {s.keyword: s for s in PARAMETER_SPEC}


def spec_by_name(name):
    return _BY_NAME[name]


def spec_by_keyword(keyword, filename=None, lineno=None, line_text=None):
    try:
        return _BY_KEYWORD[keyword]
    except KeyError:
        suggestions = difflib.get_close_matches(keyword, _BY_KEYWORD, n=3, cutoff=0.6)
        hint = f" Did you mean {' or '.join(suggestions)}?" if suggestions else ""
        raise ConfigError(
            f"unknown keyword {keyword!r}.{hint}",
            filename=filename,
            lineno=lineno,
            line_text=line_text,
        ) from None


@dataclass(frozen=True)
class Grid:
    """A resolved reciprocal-space grid. All four quantities are known and consistent."""

    lower_limits: np.ndarray
    upper_limits: np.ndarray
    step_sizes: np.ndarray
    number_of_pixels: np.ndarray

    @property
    def is_symmetric(self) -> bool:
        return bool(np.allclose(self.lower_limits, -self.upper_limits))

    @property
    def maxind(self) -> np.ndarray:
        """The legacy symmetric half-width. Only meaningful when is_symmetric."""
        return np.asarray(self.upper_limits, dtype=float)


def resolve_grid(lower=None, upper=None, step=None, n=None, symmetric=False):
    """Work out the grid from whichever of {lower, upper, step, n} were given.

    Deliberately unlike Meerkat2's parser, which has two bugs here:

      * Its "was this keyword given?" guards read `par.lower_limits[0] == NAN`
        (ReconstructionParameters.cpp:286, :299). NaN never compares equal to
        anything, so those branches are dead code and a missing LOWER_LIMITS
        propagates NaN into to_index instead of erroring. We use None and `is None`,
        which makes that bug unrepresentable.
      * SYMMETRIC_LIMITS (:288) and UPPER_LIMITS (:293) each unconditionally
        overwrite step_sizes, so an explicit STEP_SIZES is silently ignored. Here an
        over-determined grid is cross-checked and disagreement is an error: in a tool
        whose selling point is a provenance trace, a config that means something other
        than it says is the worst possible failure.
    """
    given = {
        "LOWER_LIMITS": lower,
        "UPPER_LIMITS": upper,
        "STEP_SIZES": step,
        "NUMBER_OF_PIXELS": n,
    }

    def arr(x, dtype=float):
        return None if x is None else np.asarray(x, dtype=dtype)

    lower, upper, step = arr(lower), arr(upper), arr(step)
    n = arr(n, dtype=int)

    if symmetric:
        if lower is None and upper is None:
            raise ConfigError(
                "SYMMETRIC_LIMITS needs LOWER_LIMITS or UPPER_LIMITS to mirror"
            )
        if lower is not None and upper is not None:
            if not np.allclose(lower, -upper):
                raise ConfigError(
                    f"SYMMETRIC_LIMITS given, but LOWER_LIMITS {_fmt(lower)} is not "
                    f"the negative of UPPER_LIMITS {_fmt(upper)}"
                )
        elif upper is None:
            upper = -lower
        else:
            lower = -upper

    have = {"lower": lower, "upper": upper, "step": step, "n": n}
    known = [k for k, v in have.items() if v is not None]
    if len(known) < 3:
        missing = [k for k, v in given.items() if v is None]
        raise ConfigError(
            "the grid is under-determined: give at least three of LOWER_LIMITS, "
            "UPPER_LIMITS, STEP_SIZES, NUMBER_OF_PIXELS (or use SYMMETRIC_LIMITS). "
            f"Missing: {', '.join(missing)}"
        )

    # Derive whichever one is absent, then cross-check everything that was given.
    if n is None:
        span = (upper - lower) / step + 1
        n_rounded = np.around(span).astype(int)
        if not np.allclose(span, n_rounded, atol=1e-9):
            raise ConfigError(
                f"LOWER_LIMITS, UPPER_LIMITS and STEP_SIZES imply a non-integer grid "
                f"size {_fmt(span)}; adjust STEP_SIZES so the range divides evenly"
            )
        n = n_rounded
    elif step is None:
        step = (upper - lower) / (n - 1)
    elif upper is None:
        upper = lower + step * (n - 1)
    elif lower is None:
        lower = upper - step * (n - 1)

    if np.any(n < 2):
        raise ConfigError(
            f"NUMBER_OF_PIXELS must be at least 2 in each direction, got {_fmt(n)}"
        )

    derived_step = (upper - lower) / (n - 1)
    if not np.allclose(derived_step, step, rtol=1e-9, atol=0):
        raise ConfigError(
            f"the grid is over-determined and inconsistent: LOWER_LIMITS "
            f"{_fmt(lower)}, UPPER_LIMITS {_fmt(upper)} and NUMBER_OF_PIXELS "
            f"{_fmt(n)} imply STEP_SIZES {_fmt(derived_step)}, but STEP_SIZES "
            f"{_fmt(step)} was given. Remove one of them."
        )

    return Grid(
        lower_limits=lower,
        upper_limits=upper,
        step_sizes=derived_step,
        number_of_pixels=n,
    )


def _fmt(a):
    """Render a vector for an error message.

    list(ndarray) yields numpy scalar objects, and numpy 2 reprs those as
    'np.float64(-1.5)' -- which turns an otherwise clear message into noise.
    .tolist() gives plain Python numbers.
    """
    return np.asarray(a).tolist()


def _spec_default(name):
    return _BY_NAME[name].default


@dataclass
class ReconstructionParameters:
    """Fully resolved reconstruction parameters.

    Field names and defaults mirror PARAMETER_SPEC exactly; test_spec_matches_dataclass
    asserts it.
    """

    data_file_template: str = _spec_default("data_file_template")
    xparm_file: str = _spec_default("xparm_file")
    first_frame: int | None = _spec_default("first_frame")
    last_frame: int | None = _spec_default("last_frame")
    mask: str | None = _spec_default("mask")
    scales: str | None = _spec_default("scales")

    number_of_pixels: Any = _spec_default("number_of_pixels")
    lower_limits: Any = _spec_default("lower_limits")
    upper_limits: Any = _spec_default("upper_limits")
    step_sizes: Any = _spec_default("step_sizes")
    symmetric_limits: bool = _spec_default("symmetric_limits")

    polarization_factor: float = _spec_default("polarization_factor")
    polarization_plane_normal: Any = field(
        default_factory=lambda: _spec_default("polarization_plane_normal")
    )
    medium: str = _spec_default("medium")
    unit_cell_transform: Any = _spec_default("unit_cell_transform")
    reconstruct_in_orthonormal_basis: bool = _spec_default("reconstruct_in_orthonormal_basis")

    output_filename: str = _spec_default("output_filename")
    output_format: str = _spec_default("output_format")
    overwrite: bool = _spec_default("overwrite")
    all_in_memory: bool = _spec_default("all_in_memory")
    size_of_cache: int = _spec_default("size_of_cache")

    microstep_frames: int | None = _spec_default("microstep_frames")

    def grid(self) -> Grid:
        return resolve_grid(
            lower=self.lower_limits,
            upper=self.upper_limits,
            step=self.step_sizes,
            n=self.number_of_pixels,
            symmetric=self.symmetric_limits,
        )

    def validated(self) -> ReconstructionParameters:
        """Check everything that can be checked without touching the filesystem."""
        for name in ("data_file_template", "first_frame", "last_frame"):
            if getattr(self, name) is None:
                raise ConfigError(f"{_BY_NAME[name].keyword} is required")

        if self.last_frame < self.first_frame:
            raise ConfigError(
                f"LAST_FRAME ({self.last_frame}) is before FIRST_FRAME ({self.first_frame})"
            )

        try:
            self.data_file_template % self.first_frame
        except TypeError:
            raise ConfigError(
                f"DATA_FILE_TEMPLATE {self.data_file_template!r} is not a printf-style "
                f"template; it needs an integer conversion such as %05i"
            ) from None

        if self.medium not in ("Air", "Helium"):
            raise ConfigError(f"MEDIUM must be Air or Helium, got {self.medium!r}")

        if self.output_format not in ("YELL_1.0", "YELL_0.9"):
            raise ConfigError(
                f"OUTPUT_FORMAT must be YELL_1.0 or YELL_0.9, got {self.output_format!r}"
            )

        if self.microstep_frames is not None:
            raise ConfigError(
                "MICROSTEP_FRAMES is not implemented. The microstepping code in "
                "meerkat 0.3.x never ran (an assert blocked it, and its image loader "
                "returned None), so it was removed rather than ported untested."
            )

        self.grid()  # raises if under-determined or inconsistent
        return self


def _dataclass_field_names():
    return {f.name for f in fields(ReconstructionParameters)}
