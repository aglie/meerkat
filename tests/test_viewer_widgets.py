"""The viewer's Qt widgets.

A separate file from test_viewer.py so that the whole module can be skipped when
PyQt5 is not installed, which is the normal case: `pip install meerkat` does not
install it and CI does not either. test_viewer.py holds everything that can be
checked without Qt, and that is deliberately most of it.
"""

import numpy as np
import pytest

from meerkat.viewer import load_experiment
from meerkat.xds import read_xparm

pytest.importorskip("PyQt5", reason="the viewer's widgets need PyQt5")
pytest.importorskip("OpenGL", reason="the viewer's widgets need PyOpenGL")

SPOTS = np.array(
    [
        [10.0, 20.0, 1.0, 100.0],
        [11.0, 21.0, 5.0, 500.0],
        [12.0, 22.0, 9.0, 900.0],
    ]
)

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


@pytest.fixture(scope="module")
def qapp():
    """One QApplication for the module. Qt allows exactly one per process."""
    from PyQt5 import QtWidgets

    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture
def xds_folder(tmp_path, xparm_path, spot_path):
    (tmp_path / "XPARM.XDS").write_text(xparm_path.read_text())
    (tmp_path / "SPOT.XDS").write_text(spot_path.read_text())
    return tmp_path


def test_spot_filter_reports_the_full_range_on_load(qapp):
    from meerkat.viewer.widgets import SpotsFilterWidget

    widget = SpotsFilterWidget()
    widget.spots = SPOTS

    assert widget.min.value() == 100
    assert widget.max.value() == 900
    assert len(widget.filteredSpots) == 3, "loading a spot list must not filter any of it out"


def test_spot_filter_clamps_intensities_qt_cannot_hold(qapp):
    """A long scan overflows the C++ int behind QSpinBox; Qt silently clamps
    setValue, which used to leave the max bound below the real maximum and so cut
    off the brightest spots."""
    from meerkat.viewer.widgets import SpotsFilterWidget

    widget = SpotsFilterWidget()
    widget.spots = np.array([[1.0, 1.0, 1.0, 5e9]])

    assert len(widget.filteredSpots) == 1


def test_spot_filter_keeps_spots_on_a_fractional_frame(qapp):
    """SPOT.XDS frame numbers and intensities are not integers. Truncating the
    upper bound to an int put the last frame outside it, so simply opening a
    dataset hid part of it."""
    from meerkat.viewer.widgets import SpotsFilterWidget

    fractional = np.array([[1.0, 1.0, 1399.5, 100.5], [2.0, 2.0, 1.5, 10.5]])
    widget = SpotsFilterWidget()
    widget.spots = fractional

    assert len(widget.filteredSpots) == 2


def test_spot_filter_narrows_on_the_intensity_bound(qapp):
    from meerkat.viewer.widgets import SpotsFilterWidget

    widget = SpotsFilterWidget()
    widget.spots = SPOTS
    widget.min.setValue(200)

    assert widget.filteredSpots[:, 3].tolist() == [500.0, 900.0]


def test_property_widget_keeps_six_decimals(qapp):
    """Qt's default of two decimals rounded a direction cosine to 0.01 on load,
    silently changing the geometry the window was drawing."""
    from meerkat.viewer.widgets import PropertyWidget

    widget = PropertyWidget("detector_x", np.array([0.999999, -0.000949, 0.000863]))

    assert widget.values[1] == pytest.approx(-0.000949, abs=1e-9)
    assert len(widget.valueWidgets) == 3


def test_property_widget_emits_what_the_user_typed(qapp):
    from meerkat.viewer.widgets import PropertyWidget

    widget = PropertyWidget("distance_to_detector", np.array([200.058]))
    seen = []
    widget.valueChanged.connect(lambda name, values: seen.append((name, values.copy())))

    widget.valueWidgets[0].setValue(210.0)

    assert seen[-1][0] == "distance_to_detector"
    assert seen[-1][1][0] == pytest.approx(210.0)


def test_hashmap_widget_rebuilds_when_the_map_is_replaced(qapp, xparm_path):
    from meerkat.viewer.widgets import HashMapWidget

    widget = HashMapWidget()
    assert widget.hashmap == {}

    widget.hashmap = read_xparm(str(xparm_path))
    assert "wavelength" in widget.hashmap


def test_window_opens_an_experiment(qapp, xds_folder):
    """The end-to-end check: a folder on disk becomes points on the widget."""
    from meerkat.viewer.app import Window

    window = Window()
    window.openExperiment(str(xds_folder))

    experiment = load_experiment(xds_folder)
    assert len(window.spots) == len(experiment.spots)
    assert window.ewaldSphereWidget.points.shape == (len(experiment.spots), 3)


def test_window_survives_a_geometry_it_cannot_compute(qapp, xds_folder):
    """Dragging the rotation axis through zero used to take the window down."""
    from meerkat.viewer.app import Window

    window = Window()
    window.openExperiment(str(xds_folder))
    window.changeInstrumentalParameters(dict(window.hashMapWidget.hashmap,
                                             rotation_axis=np.zeros(3)))

    assert "cannot compute" in window.statusBar().currentMessage()
