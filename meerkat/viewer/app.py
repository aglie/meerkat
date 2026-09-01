"""The Ewald sphere viewer's main window.

Ported from evaldpy/evald_qt5_v2d.py. Imports PyQt5 at module scope.
"""

from __future__ import annotations

import os
import sys

import numpy as np
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtWidgets import (
    QAction,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QRadioButton,
    QVBoxLayout,
)

from .experiment import load_experiment, spot_q_vectors
from .widgets import EwaldSphereWidget, HashMapWidget, SpotsFilterWidget

__all__ = ["Window", "main"]

_OPEN_FILTER = "XDS experiment (XPARM.XDS GXPARM.XDS XDS.INP);;All files (*)"


class Window(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()

        self.experiment = None
        self.spots = np.zeros((0, 4))

        self.ewaldSphereWidget = EwaldSphereWidget()

        mainLayout = QHBoxLayout()
        mainLayout.addWidget(self.ewaldSphereWidget)

        controls = QVBoxLayout()

        self.hashMapWidget = HashMapWidget()
        self.hashMapWidget.valueChanged.connect(self.changeInstrumentalParameters)
        controls.addWidget(self.hashMapWidget)

        zoom = QHBoxLayout()
        zoom.addWidget(QLabel("zoom: "))
        for text, slot in (
            ("+", self.ewaldSphereWidget.zoomIn),
            ("-", self.ewaldSphereWidget.zoomOut),
        ):
            button = QPushButton(text)
            button.setFixedWidth(30)
            button.pressed.connect(slot)
            zoom.addWidget(button)
        reset = QPushButton("reset view")
        reset.pressed.connect(self.ewaldSphereWidget.resetView)
        zoom.addWidget(reset)
        zoom.addStretch()
        controls.addLayout(zoom)

        selection = QHBoxLayout()
        rotateMode = QRadioButton("rotation mode")
        rotateMode.setChecked(True)
        rotateMode.toggled.connect(self.ewaldSphereWidget.setRotationMode)
        selection.addWidget(rotateMode)
        selection.addWidget(QRadioButton("selection mode"))
        for text, slot in (
            ("select rect", self.ewaldSphereWidget.selectPointsInRectangle),
            ("unselect rect", self.ewaldSphereWidget.unselectPointsInRectangle),
        ):
            button = QPushButton(text)
            button.pressed.connect(slot)
            selection.addWidget(button)
        controls.addLayout(selection)

        selectAll = QHBoxLayout()
        selectAll.addStretch()
        for text, slot in (
            ("select all", self.ewaldSphereWidget.selectAllPoints),
            ("unselect all", self.ewaldSphereWidget.unselectAllPoints),
        ):
            button = QPushButton(text)
            button.pressed.connect(slot)
            selectAll.addWidget(button)
        controls.addLayout(selectAll)

        self.spotsFilter = SpotsFilterWidget()
        self.spotsFilter.filteredSpotsChanged.connect(self.updateSpots)
        controls.addWidget(self.spotsFilter)

        mainLayout.addLayout(controls)

        central = QtWidgets.QWidget()
        central.setLayout(mainLayout)
        self.setCentralWidget(central)

        fileMenu = self.menuBar().addMenu("&File")

        openAction = QAction("&Open", self)
        openAction.setShortcut("Ctrl+O")
        openAction.triggered.connect(self.openExperimentDialog)
        fileMenu.addAction(openAction)

        saveAction = QAction("&Save selected", self)
        saveAction.setShortcut("Ctrl+S")
        saveAction.triggered.connect(self.saveSelectedPointsDialog)
        fileMenu.addAction(saveAction)

        exitAction = QAction("&Exit", self)
        exitAction.setShortcut("Ctrl+Q")
        exitAction.setStatusTip("Exit application")
        exitAction.triggered.connect(QtWidgets.QApplication.instance().quit)
        fileMenu.addAction(exitAction)

        self.statusBar()
        self.setWindowTitle("Ewald sphere")

    # -- data ------------------------------------------------------------- #

    def calculateEwaldSphere(self, params):
        """Recompute the point cloud. A geometry the parameters make nonsensical
        is reported in the status bar rather than killing the window -- the panel
        exists precisely so parameters can be dragged around, and a rotation axis
        passing through zero on its way from one value to another used to crash
        the whole viewer."""
        try:
            self.ewaldSphereWidget.points = spot_q_vectors(self.spots, params)
        except Exception as exc:
            self.statusBar().showMessage(f"cannot compute the Ewald sphere: {exc}")
        else:
            self.statusBar().showMessage(f"{len(self.spots)} spots")

    @QtCore.pyqtSlot(dict)
    def changeInstrumentalParameters(self, newParams):
        self.calculateEwaldSphere(newParams)

    def openExperimentDialog(self):
        filename, _ = QFileDialog.getOpenFileName(
            self, caption="Open XPARM.XDS, GXPARM.XDS or XDS.INP", filter=_OPEN_FILTER
        )
        if not filename:  # the user cancelled
            return
        self.openExperiment(filename)

    def openExperiment(self, path):
        """Open a parameter file, or a folder holding one, plus its SPOT.XDS."""
        try:
            experiment = load_experiment(str(path))
        except (OSError, ValueError) as exc:
            QtWidgets.QMessageBox.warning(self, "Cannot open experiment", str(exc))
            return

        self.experiment = experiment
        self.spots = experiment.spots
        self.hashMapWidget.hashmap = experiment.params
        self.spotsFilter.spots = experiment.spots
        self.calculateEwaldSphere(experiment.params)

        source = os.path.basename(experiment.parameter_file)
        self.setWindowTitle(f"Ewald sphere -- {experiment.parameter_file}")
        if not experiment.has_orientation:
            self.statusBar().showMessage(
                f"{len(self.spots)} spots from {source}, which records no phi origin "
                "-- open XPARM.XDS instead for the true rotation"
            )

    @QtCore.pyqtSlot(np.ndarray)
    def updateSpots(self, newSpots):
        self.spots = newSpots
        self.calculateEwaldSphere(self.hashMapWidget.hashmap)

    def saveSelectedPointsDialog(self):
        selected = self.ewaldSphereWidget.selectedPoints
        if not np.any(selected):
            QtWidgets.QMessageBox.information(self, "Save selected", "No spots are selected.")
            return

        filename, _ = QFileDialog.getSaveFileName(self, caption="Save selected spots")
        if not filename:  # the user cancelled -- the original crashed here
            return

        np.savetxt(str(filename), self.spots[selected, :4], fmt=" %g")
        self.statusBar().showMessage(f"wrote {int(np.count_nonzero(selected))} spots to {filename}")


def main(argv=None):
    """Run the viewer. `argv[0]`, if given, is an experiment to open at startup."""
    argv = list(sys.argv if argv is None else [sys.argv[0], *argv])

    app = QtWidgets.QApplication(argv)
    window = Window()
    if len(argv) > 1:
        window.openExperiment(argv[1])
    window.show()
    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())
