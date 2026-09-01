"""A thin wrapper over the OpenGL calls the viewer makes.

Imports PyOpenGL at module scope, so import this only from the widget layer.

Ported verbatim in behaviour from evaldpy/evald_qt5_v2d.py, whose own header
notes the boilerplate came from https://open.gl/content/code/c2_color_triangle.txt.
The shaders are GLSL 120 -- the fixed-function-adjacent profile QGLWidget gives
by default. That is old, and it is also what has been working on the beamline
laptops for five years; a core-profile rewrite is a separate change.
"""

from __future__ import annotations

import numpy as np
from OpenGL import GL

__all__ = ["ShaderProgram"]

_GL_TYPE = {"f": GL.GL_FLOAT, "i": GL.GL_INT}


class ShaderProgram:
    """A compiled shader program plus the vertex buffers feeding it.

    attributes is {name: (size, type)}, type being 'f' or 'i'.
    uniforms is {name: (type_string,)}; only 'mat3x3' is supported.
    """

    def __init__(self, vertex_source, fragment_source, uniforms, attributes, draw_as):
        self._draw_as = draw_as
        self._program = self._compile(vertex_source, fragment_source)

        self._attributes = {}
        for name, (size, type_char) in attributes.items():
            self._attributes[name] = {
                "buffer": GL.glGenBuffers(1),
                "location": GL.glGetAttribLocation(self._program, name),
                "size": size,
                "type": type_char,
                "length": 0,
            }
        self.bind_attribute_buffers()

        self._uniforms = {
            name: {
                "type_string": spec[0],
                "location": GL.glGetUniformLocation(self._program, name),
            }
            for name, spec in uniforms.items()
        }

    @staticmethod
    def _compile(vertex_source, fragment_source):
        program = GL.glCreateProgram()
        for source, kind in (
            (vertex_source, GL.GL_VERTEX_SHADER),
            (fragment_source, GL.GL_FRAGMENT_SHADER),
        ):
            shader = GL.glCreateShader(kind)
            GL.glShaderSource(shader, source)
            GL.glCompileShader(shader)
            if not GL.glGetShaderiv(shader, GL.GL_COMPILE_STATUS):
                log = GL.glGetShaderInfoLog(shader)
                raise RuntimeError(f"shader failed to compile: {_text(log)}")
            GL.glAttachShader(program, shader)

        GL.glLinkProgram(program)
        # The original printed the info logs unconditionally and never checked the
        # link status, so a failed link showed up as an empty window with a
        # cheerful blank log line rather than an error.
        if not GL.glGetProgramiv(program, GL.GL_LINK_STATUS):
            log = _text(GL.glGetProgramInfoLog(program))
            raise RuntimeError(f"shader program failed to link: {log}")

        GL.glUseProgram(program)
        return program

    def bind_attribute_buffers(self):
        for attribute in self._attributes.values():
            GL.glEnableVertexAttribArray(attribute["location"])
            GL.glBindBuffer(GL.GL_ARRAY_BUFFER, attribute["buffer"])
            GL.glVertexAttribPointer(
                attribute["location"],
                attribute["size"],
                _GL_TYPE[attribute["type"]],
                GL.GL_FALSE,
                0,
                None,
            )

    def set_uniform(self, name, value):
        uniform = self._uniforms[name]
        assert uniform["type_string"] == "mat3x3"
        assert value.size == 9

        GL.glUseProgram(self._program)
        GL.glUniformMatrix3fv(
            uniform["location"], 1, False, np.ascontiguousarray(value, dtype=np.float32)
        )

    def set_attribute(self, name, value):
        attribute = self._attributes[name]
        value = np.ascontiguousarray(value, dtype=np.float32)
        assert value.shape == (0,) or value.shape[1] == attribute["size"]

        attribute["length"] = value.shape[0]
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, attribute["buffer"])
        GL.glBufferData(GL.GL_ARRAY_BUFFER, value, GL.GL_STATIC_DRAW)

    def run(self):
        lengths = {attribute["length"] for attribute in self._attributes.values()}
        assert len(lengths) == 1, f"attribute buffers disagree on length: {sorted(lengths)}"
        (length,) = lengths
        if length == 0:
            return

        GL.glUseProgram(self._program)
        self.bind_attribute_buffers()
        GL.glDrawArrays(self._draw_as, 0, length)


def _text(log):
    return log.decode("utf-8", "replace").strip() if isinstance(log, bytes) else str(log).strip()
