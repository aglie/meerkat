"""The Ewald viewer.

Split the way the package is: the geometry (meerkat.viewer.experiment) is
numpy-only and tested everywhere, the widgets are tested only where PyQt5 and a
display exist. CI has neither, which is exactly why the split is worth having --
without it the viewer would be entirely untested rather than mostly tested.
"""

import subprocess
import sys
import textwrap

import numpy as np
import pytest

from meerkat.viewer import (
    Experiment,
    filter_spots,
    load_experiment,
    locate_experiment,
    ordered_parameters,
    read_parameters,
    spot_q_vectors,
    top_n_reflections,
)
from meerkat.xds import XDS_SPOT_OFFSET, det2lab_xds, read_spot_xds, read_xparm

# A minimal XDS.INP carrying every keyword params_from_xds_inp needs.
XDS_INP = """\
!  A synthetic XDS.INP, only the keywords the viewer reads.
 DETECTOR=PILATUS
 NX= 2463  NY= 2527  QX= 0.172000  QY= 0.172000
 DIRECTION_OF_DETECTOR_X-AXIS= 1.0 0.0 0.0
 DIRECTION_OF_DETECTOR_Y-AXIS= 0.0 1.0 0.0
 DETECTOR_DISTANCE= 200.058
 ORGX= 1214.77  ORGY= 1261.04
 ROTATION_AXIS= 1.0 0.0 0.0
 OSCILLATION_RANGE= 0.1
 X-RAY_WAVELENGTH= 0.774899
 INCIDENT_BEAM_DIRECTION= 0.0 0.0 1.0
"""


@pytest.fixture
def xds_folder(tmp_path, xparm_path, spot_path):
    """A directory that looks like an XDS output folder."""
    (tmp_path / "XPARM.XDS").write_text(xparm_path.read_text())
    (tmp_path / "SPOT.XDS").write_text(spot_path.read_text())
    (tmp_path / "XDS.INP").write_text(XDS_INP)
    return tmp_path


# --------------------------------------------------------------------------- #
# Locating and reading an experiment
# --------------------------------------------------------------------------- #


def test_a_folder_resolves_to_its_xparm_and_spots(xds_folder):
    parameter_file, spot_file = locate_experiment(xds_folder)
    assert parameter_file.endswith("XPARM.XDS")
    assert spot_file.endswith("SPOT.XDS")


def test_a_folder_prefers_xparm_over_xds_inp(xds_folder):
    """XPARM records the phi origin and XDS.INP does not, so opening the folder
    through XDS.INP would silently place every spot half an oscillation away."""
    assert locate_experiment(xds_folder)[0].endswith("XPARM.XDS")


def test_xds_inp_is_used_when_that_is_all_there_is(xds_folder):
    (xds_folder / "XPARM.XDS").unlink()
    assert locate_experiment(xds_folder)[0].endswith("XDS.INP")


def test_a_named_parameter_file_is_honoured(xds_folder):
    parameter_file, _ = locate_experiment(xds_folder / "XDS.INP")
    assert parameter_file.endswith("XDS.INP")


def test_a_folder_with_no_parameter_file_says_so(tmp_path):
    with pytest.raises(FileNotFoundError, match="XPARM.XDS"):
        locate_experiment(tmp_path)


def test_a_missing_spot_file_says_so(tmp_path, xparm_path):
    (tmp_path / "XPARM.XDS").write_text(xparm_path.read_text())
    with pytest.raises(FileNotFoundError, match="SPOT.XDS"):
        locate_experiment(tmp_path)


def test_an_unrecognised_parameter_file_is_refused(tmp_path):
    path = tmp_path / "NOTAFILE.TXT"
    path.write_text("")
    with pytest.raises(ValueError, match="do not know how to read"):
        read_parameters(str(path))


def test_loading_gives_parameters_det2lab_can_use(xds_folder):
    experiment = load_experiment(xds_folder)
    assert experiment.has_orientation  # it came from an XPARM
    assert len(experiment.spots) > 0
    assert experiment.spots.shape[1] == 4
    assert experiment.q_vectors().shape == (len(experiment.spots), 3)


def test_xds_inp_gives_no_orientation(xds_folder):
    """The viewer says so in the status bar; the flag is what it reads."""
    (xds_folder / "XPARM.XDS").unlink()
    assert not load_experiment(xds_folder).has_orientation


# --------------------------------------------------------------------------- #
# Geometry
# --------------------------------------------------------------------------- #


def test_q_vectors_apply_the_xds_convention_offset(xparm, spot_path):
    """The bug this port fixes: the 2021--2025 viewer passed SPOT.XDS positions
    to det2lab_xds untouched, so the whole cloud sat one pixel and half an
    oscillation from where the data says it is."""
    spots = read_spot_xds(str(spot_path))

    q = spot_q_vectors(spots, xparm)

    corrected = spots[:, :3] + XDS_SPOT_OFFSET
    expected = det2lab_xds(corrected[:, :2], corrected[:, 2], **xparm)[0].T
    np.testing.assert_allclose(q, expected)

    uncorrected = det2lab_xds(spots[:, :2], spots[:, 2], **xparm)[0].T
    assert not np.allclose(q, uncorrected), "the offset made no difference -- check the fixture"


def test_q_vectors_agree_with_what_improve_orientation_computes(xparm, spot_path):
    """The reason the offset is applied here at all.

    `meerkat improve-orientation` shifts SPOT.XDS coordinates by XDS_SPOT_OFFSET
    before it does any geometry -- it did not until 0.4.0, and that was recorded
    as a bug. A viewer used to judge whether a geometry is good enough for the
    refinement has to place a spot exactly where the refinement places it, or it
    is answering a different question than the one being asked of it.

    Not asserted by comparing deviation from integer hkl: on a geometry this
    imperfect the residual is ~0.03 rlu and a one-pixel shift does not move it
    measurably. Agreement with the refinement is the property that matters and
    the one that is actually testable.
    """
    from meerkat.refine.orientation import det2hkl

    spots = read_spot_xds(str(spot_path))
    q = spot_q_vectors(spots, xparm)

    refinement_hkl = det2hkl(spots[:, :3] + XDS_SPOT_OFFSET, xparm)
    np.testing.assert_allclose(np.dot(xparm["unit_cell_vectors"], q.T), refinement_hkl)


def test_q_vectors_of_no_spots_is_an_empty_cloud(xparm):
    assert spot_q_vectors(np.zeros((0, 4)), xparm).shape == (0, 3)


def test_a_zero_rotation_axis_raises_rather_than_returning_nonsense(xparm, spot_path):
    """Dragging the rotation axis through zero in the parameter panel used to
    take the whole window down. It must raise so the window can catch it."""
    spots = read_spot_xds(str(spot_path))
    broken = dict(xparm, rotation_axis=np.zeros(3))
    with pytest.raises(Exception, match="rotation vector"):
        spot_q_vectors(spots, broken)


# --------------------------------------------------------------------------- #
# Filtering
# --------------------------------------------------------------------------- #


SPOTS = np.array(
    [
        [10.0, 20.0, 1.0, 100.0],
        [11.0, 21.0, 5.0, 500.0],
        [12.0, 22.0, 9.0, 900.0],
    ]
)


def test_filter_bounds_are_inclusive():
    assert len(filter_spots(SPOTS, imin=100, imax=900)) == 3


def test_filter_cuts_on_intensity():
    kept = filter_spots(SPOTS, imin=200)
    assert kept[:, 3].tolist() == [500.0, 900.0]


def test_filter_cuts_on_frame():
    kept = filter_spots(SPOTS, frame_min=2, frame_max=8)
    assert kept[:, 2].tolist() == [5.0]


def test_filter_with_no_bounds_keeps_everything():
    np.testing.assert_array_equal(filter_spots(SPOTS), SPOTS)


def test_filter_of_nothing_is_nothing():
    assert len(filter_spots(np.zeros((0, 4)), imin=1)) == 0


def test_top_n_reflections_takes_the_brightest():
    assert top_n_reflections(SPOTS, 2)[:, 3].tolist() == [500.0, 900.0]


def test_top_n_reflections_of_a_short_list_is_the_whole_list():
    np.testing.assert_array_equal(top_n_reflections(SPOTS, 10), SPOTS)


# --------------------------------------------------------------------------- #
# The parameter panel's ordering
# --------------------------------------------------------------------------- #


def test_parameter_order_hides_the_non_geometric_keys(xparm_path):
    shown = ordered_parameters(read_xparm(str(xparm_path)))
    assert "unit_cell_vectors" not in shown
    assert "space_group_nr" not in shown
    assert "detector_segment_geometry" not in shown


def test_parameter_order_groups_the_detector_vectors(xparm_path):
    shown = ordered_parameters(read_xparm(str(xparm_path)))
    assert shown.index("detector_x") + 1 == shown.index("detector_y")
    assert shown.index("detector_y") + 1 == shown.index("detector_normal")


def test_an_unknown_parameter_still_appears():
    assert "surprise" in ordered_parameters({"wavelength": 1.0, "surprise": 2.0})


# --------------------------------------------------------------------------- #
# The dependency contract
# --------------------------------------------------------------------------- #


def test_importing_the_viewer_does_not_import_qt():
    """`import meerkat.viewer` must work on a headless machine with no Qt.

    Subprocess, because pytest may already have imported PyQt5 for the widget
    tests below and an in-process check would pass vacuously.
    """
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            textwrap.dedent(
                """
                import sys
                import meerkat.viewer
                meerkat.viewer.load_experiment      # the numpy-only half works
                offenders = sorted(
                    m for m in ("PyQt5", "PyQt6", "OpenGL", "fabio", "h5py", "scipy")
                    if m in sys.modules
                )
                assert not offenders, "importing meerkat.viewer pulled in " + repr(offenders)
                print("ok")
                """
            ),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"{result.stdout}\n{result.stderr}"


def test_view_subcommand_exists_even_without_qt():
    """A subcommand that vanishes when its dependency is missing cannot tell the
    user what to install. `meerkat view` always exists; only running it fails."""
    from meerkat.cli.main import build_parser

    subparsers = [
        a for a in build_parser()._actions if isinstance(getattr(a, "choices", None), dict)
    ][0]
    assert "view" in subparsers.choices


def test_view_reports_a_missing_experiment(tmp_path):
    from meerkat.cli.main import main

    with pytest.raises(SystemExit, match="not found"):
        main(["view", str(tmp_path / "nope")])


def test_experiment_can_be_built_from_parts(xds_folder):
    """The window rebuilds the cloud from a live-edited parameter dict, not from
    the file, so Experiment must work when handed parameters directly."""
    experiment = load_experiment(xds_folder)
    rebuilt = Experiment(experiment.params, experiment.spots)
    np.testing.assert_allclose(rebuilt.q_vectors(), experiment.q_vectors())
