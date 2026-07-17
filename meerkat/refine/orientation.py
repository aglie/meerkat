"""Refine experimental geometry so indexed spots land on integer hkl.

Ported from improve_orientation_v0.82.py. That script was top-level straight-line code
with no `if __name__` guard, and its objective function closed over module globals --
which is why it could not be looped, tested, or reused. The refinement maths is
unchanged; the structure around it is new.

Bugs fixed in the port (see the tests for each):
  * The spot filter ran on UNCORRECTED coordinates. Spots were selected using hkl
    computed before the XDS [-1, -1, +0.5] convention offset and before the
    X/Y-CORRECTIONS tables were applied, then refined using corrected coordinates --
    so the selection was biased against the objective. This had to be fixed before
    the dr schedule could work at all, since a tightening cut on biased hkl rejects
    good spots.
  * `npdeepcopy` referenced an unimported `np` and had no return statement. Dead;
    deleted.
  * `minfunc1` was dead code carrying the same slip as minfunc.

Deliberately preserved: the objective weights with the UNREFINED cell metric. See
_Refinement.residuals.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np

from ..xds import det2lab_xds, read_xparm

__all__ = [
    "REFINABLE",
    "CellRestraints",
    "RefinementResult",
    "apply_xds_corrections",
    "import_instrument_parameters",
    "parse_cell_restraints",
    "refine_orientation",
]

# name -> number of free values in the parameter vector
REFINABLE = {
    "distance": 1,
    "xycenter": 2,
    "beam": 3,
    "cell": 9,
    "axis": 3,
}

# XDS reports spot positions with 1-based pixel indices and the frame number offset by
# half an oscillation relative to what det2lab_xds expects.
XDS_SPOT_OFFSET = np.array([-1.0, -1.0, 0.5])


def dround(a):
    """Deviation from the nearest integer."""
    return a - np.around(a)


def det2hkl(xyf, params):
    """Fractional hkl for spot positions (x, y, frame)."""
    r_star = det2lab_xds(xyf[:, :2], xyf[:, 2], **params)[0]
    return np.dot(params["unit_cell_vectors"], r_star)


def apply_xds_corrections(xyf, xds_folder):
    """Apply XDS's X/Y-CORRECTIONS.cbf detector distortion tables, if present.

    The /4 and /10 are XDS's own conventions: the correction tables are stored on a
    4x-binned grid and in units of 1/10 pixel.
    """
    x_path = os.path.join(xds_folder, "X-CORRECTIONS.cbf")
    y_path = os.path.join(xds_folder, "Y-CORRECTIONS.cbf")
    if not (os.path.exists(x_path) and os.path.exists(y_path)):
        return xyf, False

    try:
        import cbf
    except ImportError:
        raise ImportError(
            "X/Y-CORRECTIONS.cbf are present but the 'cbf' package is not installed. "
            "Install it with `pip install meerkat[cbf]`, or pass --no-xds-corrections "
            "to skip them."
        ) from None

    xcorr = cbf.read(x_path).data.T
    ycorr = cbf.read(y_path).data.T

    xyf = xyf.copy()
    index = (xyf[:, :2] / 4).astype(int)  # tables are on a 4x-binned grid
    xyf[:, 0] -= xcorr[index[:, 0], index[:, 1]] / 10  # stored in 1/10 pixel
    xyf[:, 1] -= ycorr[index[:, 0], index[:, 1]] / 10
    return xyf, True


# --------------------------------------------------------------------------- #
# Instrument parameter import
# --------------------------------------------------------------------------- #

def import_instrument_parameters(params, instrument_xparm, keys=None, check_wavelength=True):
    """Overwrite instrument geometry with values refined on another experiment.

    The intended use is a standard sample: refine the detector position, beam centre
    and rotation axis on something well behaved, then hold them while refining only
    the crystal here.

    Only INSTRUMENT_KEYS are taken. Crystal- and scan-specific values
    (unit_cell_vectors, cell, space_group_nr, starting_frame/angle, oscillation_angle)
    are never imported -- importing those from a different crystal is always wrong.
    """
    from ..xds import INSTRUMENT_KEYS

    keys = tuple(INSTRUMENT_KEYS) if keys is None else tuple(keys)
    if isinstance(instrument_xparm, (str, os.PathLike)):
        source = read_xparm(instrument_xparm)
    else:
        source = instrument_xparm

    if check_wavelength:
        here = float(np.ravel(params["wavelength"])[0])
        there = float(np.ravel(source["wavelength"])[0])
        if not np.isclose(here, there, rtol=1e-3):
            raise ValueError(
                f"the standard-sample XPARM was measured at wavelength {there:.6f} A "
                f"but this experiment is at {here:.6f} A. Importing detector geometry "
                f"across an energy change is almost certainly wrong. Pass "
                f"check_wavelength=False (--allow-wavelength-mismatch) if you are sure."
            )

    merged = dict(params)
    for key in keys:
        merged[key] = source[key]
    return merged


# --------------------------------------------------------------------------- #
# Cell restraints
# --------------------------------------------------------------------------- #

_CELL_SLOTS = ("a", "b", "c", "alpha", "beta", "gamma")


@dataclass(frozen=True)
class CellRestraints:
    """Which cell parameters are tied together and which are pinned to a value.

    `ties` maps a label to the slot indices sharing it; `fixed` maps a slot index to
    its target value. Slot order is a, b, c, alpha, beta, gamma.
    """

    ties: dict = field(default_factory=dict)
    fixed: dict = field(default_factory=dict)

    def residuals(self, cell, weight=1.0):
        """Penalty terms appended to the least-squares residual vector.

        Dimensionless by construction, so lengths and angles are commensurable:
          * lengths -> relative deviation, (a - target) / target
          * angles  -> deviation in radians, deg2rad(alpha - target)

        `weight` is applied by the caller after normalizing for the number of terms;
        see _Refinement.residuals for why that normalization is necessary.
        """
        cell = np.asarray(cell, dtype=float)
        out = []

        for slot, target in self.fixed.items():
            if slot < 3:
                out.append((cell[slot] - target) / target)
            else:
                out.append(np.deg2rad(cell[slot] - target))

        for slots in self.ties.values():
            if len(slots) < 2:
                continue
            mean = cell[list(slots)].mean()
            for slot in slots:
                if slot < 3:
                    out.append((cell[slot] - mean) / mean)
                else:
                    out.append(np.deg2rad(cell[slot] - mean))

        return weight * np.array(out) if out else np.zeros(0)

    def describe(self):
        parts = []
        for _label, slots in sorted(self.ties.items()):
            if len(slots) > 1:
                parts.append(" = ".join(_CELL_SLOTS[s] for s in slots))
        for slot, target in sorted(self.fixed.items()):
            parts.append(f"{_CELL_SLOTS[slot]} = {target:g}")
        return "; ".join(parts) if parts else "none"


def parse_cell_restraints(spec):
    """Parse a symmetry restraint spec such as 'a,a,c,90,90,120'.

    Six comma-separated slots, in the order a, b, c, alpha, beta, gamma. Each is:
      * a number  -> pin that parameter to it            (e.g. 90)
      * a label   -> tie all slots sharing the label      (e.g. a,a,c ties a and b)
      * '*'       -> leave free

    So a hexagonal cell is 'a,a,c,90,90,120'; cubic is 'a,a,a,90,90,90'; monoclinic
    with unique b is '*,*,*,90,*,90'.

    These are RESTRAINTS, not constraints: the parameters stay free and are pulled
    toward the target by a penalty term. That keeps the change small and local -- hard
    constraints would mean reparametrizing the cell from 9 free numbers into
    orientation x free-cell-parameters.
    """
    tokens = [t.strip() for t in str(spec).split(",")]
    if len(tokens) != 6:
        raise ValueError(
            f"cell restraints need 6 comma-separated values (a,b,c,alpha,beta,gamma), "
            f"got {len(tokens)}: {spec!r}"
        )

    ties: dict = {}
    fixed: dict = {}
    for slot, token in enumerate(tokens):
        if token in ("*", "?", ""):
            continue
        try:
            fixed[slot] = float(token)
            continue
        except ValueError:
            pass
        ties.setdefault(token, []).append(slot)

    for label, slots in ties.items():
        if any(s < 3 for s in slots) and any(s >= 3 for s in slots):
            raise ValueError(
                f"restraint label {label!r} ties a cell length to a cell angle "
                f"({', '.join(_CELL_SLOTS[s] for s in slots)}), which is meaningless"
            )

    return CellRestraints(ties={k: tuple(v) for k, v in ties.items()}, fixed=fixed)


# --------------------------------------------------------------------------- #
# Refinement
# --------------------------------------------------------------------------- #

@dataclass
class RefinementResult:
    params: dict
    initial_rms: float
    final_rms: float
    n_spots: int
    success: bool
    message: str
    dr: float
    warnings: tuple = ()


def _parameter_labels(refine):
    """Human labels for each column of the parameter vector, for degeneracy reports."""
    labels = []
    for name in refine:
        if name == "xycenter":
            labels += ["x_center", "y_center"]
        elif name == "beam":
            labels += ["beam_x", "beam_y", "beam_z"]
        elif name == "cell":
            labels += [f"cell[{i}{j}]" for i in range(3) for j in range(3)]
        elif name == "axis":
            labels += ["axis_x", "axis_y", "axis_z"]
        else:
            labels.append(name)
    return labels


def _expected_null_directions(refine):
    """Null directions that are structural rather than a data problem.

    update() normalizes both the wavevector (beam / |beam| / wavelength) and the
    rotation axis (axis / |axis|), so each contributes 3 parameters carrying only 2
    degrees of freedom -- their overall scale cannot affect the residual, by
    construction. Confirmed empirically: perturbing the beam vector by a scale factor
    leaves the wavevector bit-identical.

    These are harmless (least_squares copes fine), so they must not be reported as
    degeneracies or the warning becomes noise on every healthy run.
    """
    return int("beam" in refine) + int("axis" in refine)


def _check_degeneracy(jacobian, refine, tol=1e-7):
    """Report parameters the DATA cannot constrain, beyond the structural nulls.

    This is not hypothetical. The 22 Bragg peaks in tests/data all have h = 0, so a*
    is entirely unconstrained by them -- and an unguarded refinement drove the a axis
    to LENGTH ZERO while reporting an improved rms. A silently nonsensical cell is far
    worse than a refusal.

    Refining 'cell' together with 'distance' is degenerate in a subtler way: the
    overall cell scale trades against the detector distance. That is why 'distance' is
    not refined by default.
    """
    if jacobian is None or jacobian.size == 0:
        return ()

    singular_values = np.linalg.svd(jacobian, compute_uv=False)
    if singular_values.size == 0 or singular_values[0] == 0:
        return ("the refinement has no sensitivity to any parameter",)

    labels = _parameter_labels(refine)
    relative = singular_values / singular_values[0]
    rank = int(np.sum(relative > tol))

    allowed = _expected_null_directions(refine)
    if rank >= len(labels) - allowed:
        return ()

    # The null space says which combinations are unconstrained; report the parameters
    # that dominate it.
    _, _, vt = np.linalg.svd(jacobian)
    null_space = vt[rank:]
    involvement = np.sqrt((null_space**2).sum(axis=0))
    worst = np.argsort(involvement)[::-1][: len(labels) - rank]
    named = ", ".join(labels[i] for i in sorted(worst) if involvement[i] > 0.1)

    return (
        f"the data constrain only {rank} of {len(labels)} refined parameters; "
        f"{len(labels) - rank} direction(s) are undetermined"
        + (f", dominated by {named}" if named else "")
        + ". The refined values along those directions are arbitrary -- refine fewer "
        "parameters, add restraints, or use spots covering more of reciprocal space.",
    )


def _check_cell_is_physical(before, after):
    lengths_before = np.asarray(before)[:3]
    lengths_after = np.asarray(after)[:3]
    problems = []
    if np.any(lengths_after <= 0) or not np.all(np.isfinite(lengths_after)):
        problems.append(
            f"the refined cell is not physical: {np.round(np.asarray(after), 4).tolist()}"
        )
    elif np.any(lengths_after < 0.5 * lengths_before) or np.any(
        lengths_after > 2.0 * lengths_before
    ):
        problems.append(
            f"the refined cell lengths {np.round(lengths_after, 4).tolist()} differ "
            f"from the input {np.round(lengths_before, 4).tolist()} by more than a "
            f"factor of two -- the refinement has probably run away"
        )
    return tuple(problems)


class _Refinement:
    """The least-squares problem for one dr cut.

    Exists so the objective is not a closure over module globals, which is what stopped
    the original script from being looped or tested.
    """

    def __init__(self, params, xyf, refine, restraints=None, restraint_weight=1.0):
        unknown = set(refine) - set(REFINABLE)
        if unknown:
            raise ValueError(
                f"unknown refinable parameter(s): {' '.join(sorted(unknown))}. "
                f"Choose from: {' '.join(REFINABLE)}"
            )
        self.params = params
        self.xyf = xyf
        self.refine = tuple(k for k in REFINABLE if k in refine)  # stable order
        self.restraints = restraints
        self.restraint_weight = restraint_weight

    def extract(self):
        """Pack the refined parameters into a flat vector."""
        out = []
        if "distance" in self.refine:
            out.append(np.ravel(self.params["distance_to_detector"]))
        if "xycenter" in self.refine:
            out.append(np.ravel(self.params["x_center"]))
            out.append(np.ravel(self.params["y_center"]))
        if "beam" in self.refine:
            out.append(np.ravel(self.params["wavevector"]))
        if "cell" in self.refine:
            out.append(np.ravel(self.params["unit_cell_vectors"]))
        if "axis" in self.refine:
            out.append(np.ravel(self.params["rotation_axis"]))
        return np.hstack(out) if out else np.zeros(0)

    def update(self, x):
        """Unpack a parameter vector back into a params dict."""
        from ..xds import vecs2cell

        unpacked = {}
        for name in self.refine:
            width = REFINABLE[name]
            unpacked[name] = x[:width]
            x = x[width:]

        updated = {
            k: (v.copy() if isinstance(v, np.ndarray) else v)
            for k, v in self.params.items()
        }

        if "cell" in unpacked:
            vectors = unpacked["cell"].reshape(3, 3)
            updated["unit_cell_vectors"] = vectors
            updated["cell"] = vecs2cell(vectors)

        if "axis" in unpacked:
            axis = unpacked["axis"]
            updated["rotation_axis"] = axis / np.linalg.norm(axis)

        if "xycenter" in unpacked:
            updated["x_center"], updated["y_center"] = unpacked["xycenter"]

        if "distance" in unpacked:
            updated["distance_to_detector"] = unpacked["distance"]

        if "beam" in unpacked:
            # The v0.82 fix: this used to test `if 'wavevector' in ...`, which is not a
            # legal keyword, so it could never fire -- the three beam values were
            # packed, consumed and dropped. Refining the beam did nothing, silently.
            beam = unpacked["beam"]
            wavelength = float(np.ravel(updated["wavelength"])[0])
            updated["wavevector"] = beam / np.linalg.norm(beam) / wavelength

        return updated

    def residuals(self, x):
        """Deviation of every spot from its nearest integer hkl, in A^-1.

        The metric deliberately uses the UNREFINED cell (self.params), not the trial
        one. That keeps the residual a fixed linear map of the hkl deviation; letting
        it float would let the optimizer shrink the residual by rescaling the cell
        rather than by improving the fit. It reads like a copy-paste slip and is not
        one -- do not "fix" it without re-reading this and running the tests.
        """
        trial = self.update(x)
        reciprocal_cell = np.linalg.inv(self.params["unit_cell_vectors"]).T
        deviation = dround(det2hkl(self.xyf, trial))
        out = np.ravel(np.dot(reciprocal_cell, deviation))

        if self.restraints is not None:
            penalty = self.restraints.residuals(trial["cell"])
            if penalty.size:
                out = np.hstack([out, self._restraint_scale(out.size, penalty.size) * penalty])
        return out

    def _restraint_scale(self, n_data, n_restraints):
        """Make --restraint-weight mean something independent of the spot count.

        least_squares minimizes the sum of squares, so a handful of restraint terms
        competing with 3*N spot terms is simply outvoted: measured on real data with
        1888 spots, weight=1 moved the cell by ~1e-4 A -- nothing. Scaling by
        sqrt(n_data / n_restraints) equalizes the two blocks' total contribution, so
        weight=1 means "trust the symmetry about as much as the spots", weight=10
        means "mostly impose it", and 0.1 means "a gentle nudge".

        Without this the useful weight would depend on how many spots happened to
        survive the dr cut, which is not something anyone should have to think about.
        """
        return self.restraint_weight * np.sqrt(n_data / n_restraints)

    def rms(self, x):
        """RMS deviation per spot, in A^-1.

        The residual holds three components per spot, so the mean square is multiplied
        by 3 to give a per-spot figure. (The original called this "A-1 or thereabouts
        (TODO: fix this later)".) Restraint terms are excluded so the number stays
        comparable across runs with and without restraints.
        """
        trial = self.update(x)
        reciprocal_cell = np.linalg.inv(self.params["unit_cell_vectors"]).T
        deviation = dround(det2hkl(self.xyf, trial))
        residual = np.ravel(np.dot(reciprocal_cell, deviation))
        return float(np.sqrt(np.mean(residual**2) * 3))


def select_spots(xyf, params, dr):
    """Keep spots whose predicted hkl is within dr of an integer.

    Must be given ALREADY-CORRECTED coordinates. The original filtered raw ones and
    corrected them afterwards, so the cut was biased by a pixel and half an
    oscillation relative to the objective it was feeding.
    """
    deviation = dround(det2hkl(xyf, params))
    distance = np.sqrt(np.sum(deviation**2, axis=0))
    return xyf[distance < dr]


def refine_orientation(
    params,
    xyf,
    refine=("xycenter", "beam", "cell", "axis"),
    dr=0.08,
    dr_schedule=None,
    restraints=None,
    restraint_weight=1.0,
    ftol=1e-6,
    verbose=True,
):
    """Refine geometry against spots, optionally through a tightening dr schedule.

    xyf must already carry the XDS convention offset and any detector corrections.

    With dr_schedule, each pass re-selects from the FULL spot list using the current
    refined geometry. That is the whole point: as the fit improves, more spots fall
    inside a tighter cut, so the selection and the fit reinforce each other. Filtering
    down a shrinking list instead -- which is what the original's destructive
    `spots = spots[...]` would give you -- can only ever lose spots.
    """
    from scipy.optimize import least_squares

    schedule = list(dr_schedule) if dr_schedule else [dr]

    result = None
    current = params
    for step_dr in schedule:
        selected = select_spots(xyf, current, step_dr)
        if len(selected) < 4:
            raise ValueError(
                f"only {len(selected)} spot(s) are within dr={step_dr} of an integer "
                f"hkl -- too few to refine. Loosen --dr, or check that the input "
                f"orientation is roughly right to begin with."
            )

        problem = _Refinement(current, selected, refine, restraints, restraint_weight)
        start = problem.extract()
        if start.size == 0:
            raise ValueError("nothing to refine: --refine selected no parameters")

        initial_rms = problem.rms(start)
        fit = least_squares(problem.residuals, start, ftol=ftol)
        final_rms = problem.rms(fit.x)

        refined = problem.update(fit.x)
        messages = _check_degeneracy(getattr(fit, "jac", None), problem.refine)
        messages += _check_cell_is_physical(current["cell"], refined["cell"])

        current = refined
        result = RefinementResult(
            params=current,
            initial_rms=initial_rms,
            final_rms=final_rms,
            n_spots=len(selected),
            success=bool(fit.success),
            message=str(fit.message),
            dr=step_dr,
            warnings=messages,
        )

        if verbose:
            print(
                f"  dr={step_dr:<6g} {len(selected):5d} spots   "
                f"rms {initial_rms:.6f} -> {final_rms:.6f} A^-1"
                + ("" if fit.success else f"   [least_squares: {fit.message}]")
            )
            for message in messages:
                print(f"    warning: {message}")

    return result
