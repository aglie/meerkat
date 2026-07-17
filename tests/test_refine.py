"""Orientation refinement.

Uses the committed XPARM + SPOT.XDS, whose spot positions and hkl come from XDS.
"""

import numpy as np
import pytest

pytest.importorskip("scipy")

from meerkat.refine.orientation import (  # noqa: E402
    CellRestraints,
    _Refinement,
    det2hkl,
    dround,
    import_instrument_parameters,
    parse_cell_restraints,
    refine_orientation,
    select_spots,
)
from meerkat.xds import read_xparm  # noqa: E402


@pytest.fixture
def spots(refine_spots):
    """Real spots, already offset into the det2lab_xds convention.

    Deliberately NOT the 22 Bragg peaks used by the geometry tests: every one of them
    has h = 0, so a* is unconstrained and refining the cell against them drives the a
    axis to length zero. See conftest.refine_spots.
    """
    return refine_spots


@pytest.fixture
def xparm(refine_xparm):
    """The XPARM matching `spots`, overriding the session-wide 22-peak one."""
    return refine_xparm


class TestCellRestraintParsing:
    def test_hexagonal(self):
        r = parse_cell_restraints("a,a,c,90,90,120")
        assert r.ties["a"] == (0, 1)
        assert r.fixed == {3: 90.0, 4: 90.0, 5: 120.0}

    def test_cubic(self):
        r = parse_cell_restraints("a,a,a,90,90,90")
        assert r.ties["a"] == (0, 1, 2)

    def test_free_slots(self):
        r = parse_cell_restraints("*,*,*,90,*,90")
        assert r.ties == {}
        assert r.fixed == {3: 90.0, 5: 90.0}

    def test_wrong_number_of_slots(self):
        with pytest.raises(ValueError, match="6 comma-separated"):
            parse_cell_restraints("a,a,c,90,90")

    def test_tying_a_length_to_an_angle_is_rejected(self):
        with pytest.raises(ValueError, match="meaningless"):
            parse_cell_restraints("x,b,c,x,90,90")

    def test_describe(self):
        assert "a = b" in parse_cell_restraints("a,a,c,90,90,120").describe()


class TestCellRestraintResiduals:
    def test_satisfied_restraints_cost_nothing(self):
        r = parse_cell_restraints("a,a,c,90,90,120")
        np.testing.assert_allclose(
            r.residuals([5.0, 5.0, 7.0, 90.0, 90.0, 120.0]), 0.0, atol=1e-12
        )

    def test_violated_length_tie_is_penalized(self):
        r = parse_cell_restraints("a,a,c,90,90,120")
        assert np.abs(r.residuals([5.0, 5.5, 7.0, 90.0, 90.0, 120.0])).max() > 0.01

    def test_violated_fixed_angle_is_penalized(self):
        r = parse_cell_restraints("a,a,c,90,90,120")
        assert np.abs(r.residuals([5.0, 5.0, 7.0, 90.0, 90.0, 118.0])).max() > 0.01

    def test_lengths_and_angles_are_commensurable(self):
        """A 1% length error and ~0.57 deg angle error must cost about the same,
        otherwise one silently dominates."""
        r = parse_cell_restraints("*,*,*,90,90,90")
        angle = np.abs(r.residuals([5.0, 5.0, 7.0, 90.0, 90.0, 91.0])).max()
        assert angle == pytest.approx(np.deg2rad(1.0), rel=1e-6)

    def test_no_restraints_gives_empty(self):
        assert CellRestraints().residuals([5, 5, 7, 90, 90, 120]).size == 0


class TestSelectSpots:
    def test_selects_close_spots(self, xparm, spots):
        loose = select_spots(spots, xparm, 0.3)
        tight = select_spots(spots, xparm, 0.02)
        assert len(tight) < len(loose) <= len(spots)

    def test_selected_spots_really_are_within_dr(self, xparm, spots):
        selected = select_spots(spots, xparm, 0.05)
        distance = np.sqrt(np.sum(dround(det2hkl(selected, xparm)) ** 2, axis=0))
        assert np.all(distance < 0.05)


class TestRefinement:
    def test_refinement_improves_the_fit(self, xparm, spots):
        result = refine_orientation(xparm, spots, dr=0.15, verbose=False)
        assert result.final_rms < result.initial_rms
        assert result.n_spots > 4

    def test_refined_cell_stays_physical(self, xparm, spots):
        result = refine_orientation(xparm, spots, dr=0.15, verbose=False)
        cell = np.asarray(result.params["cell"])
        np.testing.assert_allclose(cell[:3], np.asarray(xparm["cell"])[:3], rtol=0.05)
        assert np.all(cell[3:] > 0) and np.all(cell[3:] < 180)

    def test_rotation_axis_stays_normalized(self, xparm, spots):
        result = refine_orientation(xparm, spots, dr=0.15, refine=("axis", "cell"), verbose=False)
        assert np.linalg.norm(result.params["rotation_axis"]) == pytest.approx(1.0)

    def test_dr_schedule_recovers_from_a_bad_starting_geometry(self, xparm, spots):
        """What the schedule is actually for.

        From an already-good start it buys nothing (measured: 0.000459 scheduled vs
        0.000388 single -- marginally worse, because the last pass selects spots using
        the geometry refined so far rather than the original). Its value is that a
        tight dr cannot even START when the input orientation is off: with the beam
        centre displaced by 6 px, dr=0.04 selects ZERO spots and the refinement fails
        outright, while the schedule opens loose and walks in.
        """
        bad = dict(xparm)
        bad["x_center"] = np.ravel(xparm["x_center"])[0] + 6.0
        bad["y_center"] = np.ravel(xparm["y_center"])[0] - 6.0

        with pytest.raises(ValueError, match="too few to refine"):
            refine_orientation(bad, spots, dr=0.04, verbose=False)

        scheduled = refine_orientation(
            bad, spots, dr_schedule=[0.4, 0.25, 0.15, 0.08, 0.04], verbose=False
        )
        assert scheduled.final_rms < 0.001
        # and it found its way back to the true beam centre
        assert np.ravel(scheduled.params["x_center"])[0] == pytest.approx(
            np.ravel(xparm["x_center"])[0], abs=1.5
        )

    def test_schedule_reselects_from_the_full_spot_list(self, xparm, spots):
        """Each pass must re-filter the master list, not a shrinking one.

        As the geometry improves more spots come inside a given cut; filtering a
        progressively shrunken list can only ever lose them.
        """
        result = refine_orientation(xparm, spots, dr_schedule=[0.02, 0.25], verbose=False)
        # The final (loose) pass must see far more spots than the tight one before it.
        assert result.n_spots > len(select_spots(spots, xparm, 0.02))

    def test_unknown_refinable_parameter_is_rejected(self, xparm, spots):
        with pytest.raises(ValueError, match="unknown refinable"):
            refine_orientation(xparm, spots, refine=("banana",), verbose=False)

    def test_too_few_spots_is_a_clear_error(self, xparm, spots):
        with pytest.raises(ValueError, match="too few to refine"):
            refine_orientation(xparm, spots, dr=1e-9, verbose=False)

    def test_refining_nothing_is_an_error(self, xparm, spots):
        with pytest.raises(ValueError, match="nothing to refine"):
            refine_orientation(xparm, spots, refine=(), dr=0.15, verbose=False)


class TestBeamRefinementActuallyDoesSomething:
    """Regression for the v0.82 bug: `--refine beam` was a silent no-op.

    The check tested 'wavevector', which is not a legal keyword, so the three beam
    values were packed into the parameter vector, consumed by the unpacker, and
    dropped -- free parameters with an identically zero Jacobian.
    """

    def test_beam_values_reach_the_output(self, xparm, spots):
        problem = _Refinement(xparm, spots[:50], refine=("beam",))
        x = problem.extract()
        assert x.size == 3

        tilted = x.copy()
        tilted[0] += 0.01  # a DIRECTION change -- see test_beam_scaling_is_a_null_direction
        assert not np.allclose(problem.update(tilted)["wavevector"], xparm["wavevector"]), (
            "perturbing the beam direction did not change the wavevector"
        )

    def test_beam_perturbation_changes_the_residual(self, xparm, spots):
        problem = _Refinement(xparm, spots[:50], refine=("beam",))
        x = problem.extract()
        tilted = x.copy()
        tilted[0] += 0.01
        assert not np.allclose(problem.residuals(x), problem.residuals(tilted)), (
            "the beam parameters have a zero Jacobian -- refining them does nothing"
        )

    def test_beam_scaling_is_a_null_direction(self, xparm, spots):
        """Documents a structural degeneracy, so it is not rediscovered as a bug.

        update() sets wavevector = beam / |beam| / wavelength, so the overall scale of
        the beam vector cannot affect anything. 'beam' therefore carries 3 parameters
        but only 2 degrees of freedom. Same for 'axis'. _expected_null_directions
        accounts for both, so healthy runs do not emit a degeneracy warning.
        """
        problem = _Refinement(xparm, spots[:50], refine=("beam",))
        x = problem.extract()
        np.testing.assert_allclose(
            problem.update(x * 1.05)["wavevector"], problem.update(x)["wavevector"]
        )

    def test_wavevector_keeps_its_length(self, xparm, spots):
        """|k| must stay 1/lambda; only the direction is free."""
        problem = _Refinement(xparm, spots[:50], refine=("beam",))
        tilted = problem.extract()
        tilted[0] += 0.01
        updated = problem.update(tilted)
        wavelength = float(np.ravel(xparm["wavelength"])[0])
        assert np.linalg.norm(updated["wavevector"]) == pytest.approx(1 / wavelength)


class TestMetricIsHeldConstant:
    """The objective weights with the UNREFINED cell. Deliberate; pin it.

    A constant metric keeps the residual a fixed linear map of the hkl deviation. If
    it floated with the trial cell, the optimizer could shrink the residual by
    rescaling the cell instead of by fitting better.
    """

    def test_metric_comes_from_the_reference_cell_not_the_trial(self, xparm, spots):
        """Evaluate the SAME trial vector under two different reference cells.

        If the weighting metric were taken from the trial parameters, both problems
        would weight identically and the residuals would match. They must not.
        """
        stretched = dict(xparm)
        stretched["unit_cell_vectors"] = np.asarray(xparm["unit_cell_vectors"]) * 1.10

        a = _Refinement(xparm, spots[:50], refine=("cell",))
        b = _Refinement(stretched, spots[:50], refine=("cell",))

        trial = a.extract()  # the same trial cell in both cases
        assert not np.allclose(a.residuals(trial), b.residuals(trial)), (
            "the metric follows the trial cell -- the optimizer could shrink the "
            "residual by rescaling the cell instead of by fitting better"
        )

    def test_metric_is_constant_across_trials(self, xparm, spots):
        """The weighting must not change as the optimizer moves."""
        problem = _Refinement(xparm, spots[:50], refine=("cell",))
        reference = np.linalg.inv(problem.params["unit_cell_vectors"]).T
        problem.residuals(problem.extract() * 1.05)
        np.testing.assert_array_equal(
            reference, np.linalg.inv(problem.params["unit_cell_vectors"]).T
        )


class TestInstrumentImport:
    def test_imports_instrument_but_not_crystal(self, xparm_path, tmp_path):
        from meerkat.xds import CRYSTAL_KEYS, write_xparm

        standard = read_xparm(str(xparm_path))
        standard["distance_to_detector"] = np.array([250.0])
        standard["x_center"] = np.array([1000.0])
        standard["unit_cell_vectors"] = standard["unit_cell_vectors"] * 2  # nonsense cell
        path = tmp_path / "STANDARD.XDS"
        write_xparm(str(path), standard)

        params = read_xparm(str(xparm_path))
        merged = import_instrument_parameters(params, str(path))

        assert float(np.ravel(merged["distance_to_detector"])[0]) == pytest.approx(250.0)
        assert float(np.ravel(merged["x_center"])[0]) == pytest.approx(1000.0)
        for key in CRYSTAL_KEYS:
            np.testing.assert_allclose(
                np.ravel(merged[key]), np.ravel(params[key]),
                err_msg=f"{key} must never be imported from another experiment",
            )

    def test_wavelength_mismatch_is_refused(self, xparm_path, tmp_path):
        from meerkat.xds import write_xparm

        standard = read_xparm(str(xparm_path))
        standard["wavelength"] = np.array([1.5418])  # Cu Ka vs the file's 0.7749
        path = tmp_path / "OTHER.XDS"
        write_xparm(str(path), standard)

        with pytest.raises(ValueError, match="wavelength"):
            import_instrument_parameters(read_xparm(str(xparm_path)), str(path))

    def test_wavelength_mismatch_can_be_overridden(self, xparm_path, tmp_path):
        from meerkat.xds import write_xparm

        standard = read_xparm(str(xparm_path))
        standard["wavelength"] = np.array([1.5418])
        path = tmp_path / "OTHER.XDS"
        write_xparm(str(path), standard)

        merged = import_instrument_parameters(
            read_xparm(str(xparm_path)), str(path), check_wavelength=False
        )
        assert merged is not None


class TestRestrainedRefinement:
    def test_restraints_pull_the_cell_toward_symmetry(self, xparm, spots):
        """xparm's cell is 8.2287 8.2299 11.0122 90.013 90.025 59.974 -- nearly
        a = b, alpha = beta = 90, gamma = 60. Restraining should tighten that."""
        free = refine_orientation(xparm, spots, dr=0.15, verbose=False)
        tied = refine_orientation(
            xparm,
            spots,
            dr=0.15,
            restraints=parse_cell_restraints("a,a,c,90,90,60"),
            restraint_weight=10.0,
            verbose=False,
        )
        free_cell = np.asarray(free.params["cell"])
        tied_cell = np.asarray(tied.params["cell"])

        assert abs(tied_cell[0] - tied_cell[1]) < abs(free_cell[0] - free_cell[1])
        assert abs(tied_cell[3] - 90) <= abs(free_cell[3] - 90) + 1e-6
        assert abs(tied_cell[5] - 60) <= abs(free_cell[5] - 60) + 1e-6

    def test_restraint_weight_is_normalized_against_spot_count(self, xparm, spots):
        """weight must mean the same thing whatever the spot count.

        Without the sqrt(n_data / n_restraints) scaling, a few restraint terms are
        outvoted by 3*N spot terms: measured on 1888 real spots, weight=1 moved the
        cell by ~1e-4 A, i.e. nothing.
        """
        restraints = parse_cell_restraints("a,a,c,90,90,60")
        few = _Refinement(xparm, spots[:20], ("cell",), restraints, 1.0)
        many = _Refinement(xparm, spots, ("cell",), restraints, 1.0)

        n_restraints = restraints.residuals(xparm["cell"]).size
        ratio_few = few._restraint_scale(3 * 20, n_restraints)
        ratio_many = many._restraint_scale(3 * len(spots), n_restraints)
        assert ratio_many > ratio_few  # scales with the data block

        # And the normalized influence is the same in both cases.
        assert ratio_few / np.sqrt(3 * 20) == pytest.approx(ratio_many / np.sqrt(3 * len(spots)))
