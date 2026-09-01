"""The Qt widgets. Imports PyQt5 and PyOpenGL at module scope.

Ported from evaldpy/evald_qt5_v2d.py -- the version that has been in use on the
beamline. The behaviour is deliberately unchanged apart from the fixes noted
inline; this is a port, not a redesign.

PyQt5 rather than PyQt6: v2d is what is tested, and it is built on QGLWidget,
which Qt6 removed in favour of QOpenGLWidget. Porting the GL widget is a real
change with no way to verify it short of a beamline laptop, so it is left for
when someone can sit in front of one. evaldpy/evald_qt6.py is a machine
translation that has never been confirmed to run.
"""

from __future__ import annotations

import numpy as np
from OpenGL import GL
from PyQt5 import QtCore, QtGui, QtOpenGL, QtWidgets
from PyQt5.QtWidgets import (
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QSpinBox,
    QVBoxLayout,
)

from .gl import ShaderProgram

__all__ = ["EwaldSphereWidget", "HashMapWidget", "PropertyWidget", "SpotsFilterWidget"]

# QSpinBox is backed by a C++ int. Intensities in SPOT.XDS routinely exceed it on
# a long scan, and the original set the maximum to 10_000_000 while calling
# setValue with the raw maximum intensity -- which Qt then silently clamped, so
# the "max" filter cut off real spots. Clamp explicitly, at the actual limit.
_INT_MAX = 2147483647


class SpotsFilterWidget(QtWidgets.QWidget):
    """Intensity and frame-number bounds on the spot list."""

    filteredSpotsChanged = QtCore.pyqtSignal(np.ndarray)

    def __init__(self, parent=None):
        super().__init__(parent)

        self.mainLayout = QVBoxLayout()
        self.setLayout(self.mainLayout)
        self.mainLayout.setContentsMargins(0, 0, 0, 0)

        self._spots = np.zeros((0, 4))
        self._filteredSpots = self._spots
        self._updating = False

        self.mainLayout.addWidget(QLabel("Bragg peak filter"))

        self.max = self._spinBox("max:")
        self.min = self._spinBox("min:")

        frames = QHBoxLayout()
        frames.addWidget(QLabel("Frames min/max:"))
        self.minframe = QSpinBox()
        self.minframe.setMaximum(_INT_MAX)
        self.minframe.valueChanged.connect(self.recalculateFilteredSpots)
        frames.addWidget(self.minframe)
        self.maxframe = QSpinBox()
        self.maxframe.setMaximum(_INT_MAX)
        self.maxframe.valueChanged.connect(self.recalculateFilteredSpots)
        frames.addWidget(self.maxframe)
        self.mainLayout.addLayout(frames)

    def _spinBox(self, label):
        row = QHBoxLayout()
        row.addWidget(QLabel(label))
        box = QSpinBox()
        box.setMaximum(_INT_MAX)
        box.valueChanged.connect(self.recalculateFilteredSpots)
        row.addWidget(box)
        self.mainLayout.addLayout(row)
        return box

    @property
    def spots(self):
        return self._spots

    @spots.setter
    def spots(self, inp):
        inp = np.asarray(inp, dtype=float)
        self._spots = inp
        self._filteredSpots = inp
        if len(inp) == 0:
            return

        # Setting four boxes fires four valueChanged signals, each of which used
        # to refilter and repaint against a half-updated set of bounds -- and the
        # first of them, with min still at 0 and max not yet raised, emitted an
        # empty spot list. Set them all, then filter once.
        self._updating = True
        try:
            # Floor the lower bounds and ceil the upper ones. Intensities and
            # frame numbers in SPOT.XDS are not integers -- a spot at frame
            # 1399.5 fell outside a bound truncated to 1399, so simply opening a
            # dataset hid its last frame.
            self.min.setValue(_floor(np.min(inp[:, 3])))
            self.max.setValue(_ceil(np.max(inp[:, 3])))
            self.minframe.setValue(_floor(np.min(inp[:, 2])))
            self.maxframe.setValue(_ceil(np.max(inp[:, 2])))
        finally:
            self._updating = False
        self.recalculateFilteredSpots()

    @property
    def filteredSpots(self):
        return self._filteredSpots

    def recalculateFilteredSpots(self, _=None):
        if self._updating or len(self._spots) == 0:
            return

        from .experiment import filter_spots

        self._filteredSpots = filter_spots(
            self._spots,
            imin=self.min.value(),
            imax=_upper(self.max),
            frame_min=self.minframe.value(),
            frame_max=_upper(self.maxframe),
        )
        self.filteredSpotsChanged.emit(self._filteredSpots)


class PropertyWidget(QtWidgets.QWidget):
    """One instrument parameter: its name and a spin box per component."""

    valueChanged = QtCore.pyqtSignal(str, np.ndarray)

    def __init__(self, name, values, parent=None):
        super().__init__(parent)

        self.mainLayout = QHBoxLayout()
        self.setLayout(self.mainLayout)
        self.mainLayout.setContentsMargins(0, 0, 0, 0)

        self.name = name
        self.values = np.atleast_1d(np.asarray(values, dtype=float)).copy()
        self.valueWidgets = []

        label = QLabel(name)
        label.setMinimumWidth(140)
        self.mainLayout.addWidget(label)

        for value in self.values:
            box = QDoubleSpinBox()
            box.setRange(-1e6, 1e6)
            # Six decimals, not Qt's default two: a detector direction cosine and
            # a wavelength both live well below the third decimal, and rounding
            # them to 0.01 on load silently changed the geometry being displayed.
            box.setDecimals(6)
            box.setSingleStep(0.001)
            box.setValue(float(value))
            box.valueChanged.connect(self.changeValues)
            self.mainLayout.addWidget(box)
            self.valueWidgets.append(box)

    @QtCore.pyqtSlot()
    def changeValues(self):
        for i, box in enumerate(self.valueWidgets):
            self.values[i] = box.value()
        self.valueChanged.emit(self.name, self.values)


class HashMapWidget(QtWidgets.QWidget):
    """The instrument-parameter panel: one PropertyWidget per parameter."""

    valueChanged = QtCore.pyqtSignal(dict)

    def __init__(self, hashmap=None, parent=None):
        super().__init__(parent)

        self.mainLayout = QVBoxLayout()
        self.setLayout(self.mainLayout)
        self._hashmap = {}
        self.hashmap = {} if hashmap is None else hashmap

    def setup(self):
        from .experiment import ordered_parameters

        self.mainLayout.addWidget(QLabel("Instrumental parameters"))
        for name in ordered_parameters(self._hashmap):
            widget = PropertyWidget(name, self._hashmap[name])
            widget.valueChanged.connect(self.childValueChanged)
            self.mainLayout.addWidget(widget)
        self.mainLayout.addStretch()

    @QtCore.pyqtSlot(str, np.ndarray)
    def childValueChanged(self, name, newval):
        self._hashmap[str(name)] = newval
        self.valueChanged.emit(self._hashmap)

    def swipe(self):
        # see http://stackoverflow.com/questions/9374063/pyqt4-remove-widgets-and-layout-as-well
        for i in reversed(range(self.mainLayout.count())):
            item = self.mainLayout.itemAt(i)
            if isinstance(item, QtWidgets.QWidgetItem):
                item.widget().deleteLater()
            self.mainLayout.removeItem(item)

    @property
    def hashmap(self):
        return self._hashmap

    @hashmap.setter
    def hashmap(self, inp):
        assert isinstance(inp, dict)
        self._hashmap = inp
        self.swipe()
        self.setup()


class EwaldSphereWidget(QtOpenGL.QGLWidget):
    """Draws the reciprocal-space point cloud.

    Usage:

        w = EwaldSphereWidget()
        w.points = [[0, 0, 0], [1, 0, 0]]
        layout.addWidget(w)

    Drag rotates and the wheel zooms. In selection mode, dragging draws a
    rectangle whose contents the select/unselect buttons act on.
    """

    def __init__(self, parent=None):
        super().__init__(parent)

        self.lastPos = QtCore.QPoint()
        self.rectStartPos = QtCore.QPoint()
        self.projectionMatrix = np.eye(4)
        self._points = np.zeros((0, 3), dtype=np.float32)
        self._selectedPoints = np.zeros(0, dtype=np.bool_)
        self.selectedRectangle = None
        self.openGlIsInitialized = False
        self.mouseInteractionMode = "rotate"

    @QtCore.pyqtSlot(bool)
    def setRotationMode(self, doSet):
        self.mouseInteractionMode = "rotate" if doSet else "select"

    def setMouseInteractionMode(self, mode):
        assert mode in ("rotate", "select")
        self.mouseInteractionMode = mode

    @QtCore.pyqtSlot()
    def selectPointsInRectangle(self):
        self.setSelectionToThePointsInRectangle(True)

    @QtCore.pyqtSlot()
    def unselectPointsInRectangle(self):
        self.setSelectionToThePointsInRectangle(False)

    @QtCore.pyqtSlot()
    def selectAllPoints(self):
        self.setSelectionOfAllPoints(True)

    @QtCore.pyqtSlot()
    def unselectAllPoints(self):
        self.setSelectionOfAllPoints(False)

    def setSelectionOfAllPoints(self, val):
        self._selectedPoints[:] = val
        self.loadPointsToGPU()
        self.updateGL()

    def setSelectionToThePointsInRectangle(self, val):
        if self.selectedRectangle is None or self._points.size == 0:
            return

        r = np.array(self.selectedRectangle)
        left, right = np.sort(r[:, 0])
        up, down = np.sort(r[:, 1])
        pointsInGL = self.pointsInGLSpace()
        inRectangle = (
            (left <= pointsInGL[:, 0])
            & (pointsInGL[:, 0] <= right)
            & (up <= pointsInGL[:, 1])
            & (pointsInGL[:, 1] <= down)
        )

        self._selectedPoints[inRectangle] = val
        self.loadPointsToGPU()
        self.updateGL()

    @property
    def selectedPoints(self):
        return self._selectedPoints

    @property
    def points(self):
        return self._points

    @points.setter
    def points(self, inp):
        inp = np.atleast_2d(np.asarray(inp, dtype=np.float32))
        if inp.size == 0:
            inp = np.zeros((0, 3), dtype=np.float32)
        self._points = inp
        self._selectedPoints = np.zeros(len(inp), dtype=np.bool_)
        self.loadPointsToGPU()
        self.updateGL()

    def loadPointsToGPU(self):
        if not self.openGlIsInitialized:
            return
        self._pointsProgram.set_attribute("position", self._points)
        color = np.ones((len(self._points), 4), dtype=np.float32)
        color[self._selectedPoints] = np.array([1, 0, 0, 1], dtype=np.float32)
        self._pointsProgram.set_attribute("color", color)

    def minimumSizeHint(self):
        return QtCore.QSize(50, 50)

    def sizeHint(self):
        return QtCore.QSize(1000, 1000)

    def pointsInGLSpace(self):
        return np.dot(self._points, self.projectionAspectMatrix())

    def projectionAspectMatrix(self):
        # The first factor squares up x against y; the last keeps points from
        # being depth-clipped.
        aspect = [1.0 * self.height() / max(self.width(), 1), 1, 0.001]
        return self.projectionMatrix[:3, :3] * aspect

    def updateProjectionMatrix(self):
        self._pointsProgram.set_uniform("M", self.projectionAspectMatrix())

    def setupPointsProgram(self):
        self._pointsProgram = ShaderProgram(
            """
            #version 120
            uniform mat3x3 M;
            attribute vec3 position;
            attribute vec4 color;
            void main() {
                gl_Position = vec4(M*position, 1.0);
                gl_FrontColor = color;
            }
            """,
            """
            #version 120
            void main() {
               gl_FragColor = gl_Color;
            }
            """,
            {"M": ("mat3x3",)},
            {"position": (3, "f"), "color": (4, "f")},
            GL.GL_POINTS,
        )

    def _setupFlatProgram(self, rgba, drawAs):
        """A program that draws its geometry in one flat colour, in clip space."""
        colour = ", ".join(f"{c:.1f}" for c in rgba)
        return ShaderProgram(
            """
            #version 120
            attribute vec2 position;
            void main() {
                gl_Position = vec4(position, 0, 1.0);
            }
            """,
            f"""
            #version 120
            void main() {{
               gl_FragColor = vec4({colour});
            }}
            """,
            {},
            {"position": (2, "f")},
            drawAs,
        )

    def initializeGL(self):
        self.qglClearColor(QtGui.QColor.fromCmykF(0.39, 0.39, 0.0, 0.0).darker())
        GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glEnable(GL.GL_CULL_FACE)
        GL.glPointSize(2)

        self.setupPointsProgram()
        self._rectangleProgram = self._setupFlatProgram((1, 1, 1, 1), GL.GL_LINE_LOOP)
        self._centerMarkerProgram = self._setupFlatProgram((1, 0, 1, 1), GL.GL_POINTS)
        self._centerMarkerProgram.set_attribute("position", np.array([[0.0, 0.0]]))
        self._rectangleProgram.set_attribute("position", np.zeros((0, 2)))

        self.updateProjectionMatrix()

        self.openGlIsInitialized = True
        self.loadPointsToGPU()

    def drawSelectionRectangle(self):
        r = self.selectedRectangle
        if r is None:
            self._rectangleProgram.set_attribute("position", np.zeros((0, 2)))
            return
        corners = [
            [r[0][0], r[0][1]],
            [r[0][0], r[1][1]],
            [r[1][0], r[1][1]],
            [r[1][0], r[0][1]],
        ]
        self._rectangleProgram.set_attribute("position", np.array(corners, dtype=np.float32))

    def paintGL(self):
        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
        self._pointsProgram.run()
        self.drawSelectionRectangle()
        self._rectangleProgram.run()
        self._centerMarkerProgram.run()

    def resizeGL(self, width, height):
        if min(width, height) < 0:
            return
        GL.glViewport(0, 0, width, height)
        if self.openGlIsInitialized:
            self.updateProjectionMatrix()

    def rotate(self, rx, ry):
        about_x = np.array(
            [
                [1, 0, 0, 0],
                [0, np.cos(rx), np.sin(rx), 0],
                [0, -np.sin(rx), np.cos(rx), 0],
                [0, 0, 0, 1],
            ]
        )
        about_y = np.array(
            [
                [np.cos(ry), 0, np.sin(ry), 0],
                [0, 1, 0, 0],
                [-np.sin(ry), 0, np.cos(ry), 0],
                [0, 0, 0, 1],
            ]
        )
        self.projectionMatrix = self.projectionMatrix @ about_x @ about_y
        self.updateProjectionMatrix()
        self.updateGL()

    def zoom(self, factor):
        self.projectionMatrix = self.projectionMatrix @ np.diag([factor, factor, factor, 1.0])
        self.updateProjectionMatrix()
        self.updateGL()

    def zoomIn(self):
        self.zoom(1.1)

    def zoomOut(self):
        self.zoom(0.9)

    def resetView(self):
        self.projectionMatrix = np.eye(4)
        self.updateProjectionMatrix()
        self.updateGL()

    def pix2GL(self, qpos):
        # OpenGL clip coordinates run -1..1 in both directions.
        return [
            2.0 * qpos.x() / max(self.width(), 1) - 1,
            -2.0 * qpos.y() / max(self.height(), 1) + 1,
        ]

    def mousePressEvent(self, event):
        if self.mouseInteractionMode == "select":
            self.selectedRectangle = None
            self.rectStartPos = event.pos()
            self.updateGL()
        else:
            self.lastPos = event.pos()

    def mouseMoveEvent(self, event):
        if self.mouseInteractionMode == "select":
            self.selectedRectangle = [self.pix2GL(self.rectStartPos), self.pix2GL(event.pos())]
            self.updateGL()
            return

        dx = event.x() - self.lastPos.x()
        dy = event.y() - self.lastPos.y()
        if event.buttons() & QtCore.Qt.LeftButton:
            self.rotate(-dy * 0.01, dx * 0.01)
        self.lastPos = event.pos()

    def wheelEvent(self, event):
        self.zoomIn() if event.angleDelta().y() > 0 else self.zoomOut()


def _floor(value):
    """A lower bound from SPOT.XDS as an int QSpinBox will accept."""
    return int(np.clip(np.floor(value), -_INT_MAX, _INT_MAX))


def _ceil(value):
    """An upper bound from SPOT.XDS as an int QSpinBox will accept."""
    return int(np.clip(np.ceil(value), -_INT_MAX, _INT_MAX))


def _upper(box):
    """The upper bound a spin box expresses, or None for "no bound".

    A box pinned at its own maximum cannot say anything about data beyond it, and
    on a long scan the intensities do run past a C++ int. Reading it as a literal
    cut discarded every spot brighter than 2147483647 -- the brightest ones,
    which are the ones worth looking at.
    """
    value = box.value()
    return None if value >= _INT_MAX else value
