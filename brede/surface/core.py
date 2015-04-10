"""Handle surfaces."""


from __future__ import absolute_import, division, print_function

import numpy as np

from ..core.matrix import Matrix


class Surface(object):

    """Representation of a surface with faces and vertices."""

    def __init__(self, vertices=None, faces=None, vertex_values=None):
        """Setup vertices, faces and optionally vertex values."""
        self._vertices = vertices
        self._faces = faces
        self._vertex_values = vertex_values


class TriSurface(Surface):

    """Representation of a triangularized surface with faces and vertices.

    Examples
    --------
    >>> vertices = [[0, 1, 0], [1, 0, 0], [0, 0, -1], [-1, 0, 0],
    ...             [0, 1, 0], [0, -1, 0]]
    >>> faces = [[0, 2, 1], [0, 3, 2], [0, 4, 3], [0, 1, 4],
    ...          [5, 1, 2], [5, 2, 3], [5, 3, 4], [5, 4, 1]]
    >>> surface = TriSurface(vertices, faces)

    """

    def __init__(self, vertices=None, faces=None, vertex_values=None):
        """Setup vertices, faces and optionally vertex values."""
        self._vertices = np.array(vertices)
        self._faces = np.array(faces)
        self.vertex_values = vertex_values

    def __repr__(self):
        """Return string representation."""
        return "TriSurface(n_vertices={}, n_faces={})".format(
            self._vertices.shape[0], self._faces.shape[0])

    def __str__(self):
        """Return string representation."""
        return self.__repr__()

    @classmethod
    def read_obj(cls, filename):
        """Read Wavefront obj file.

        Only faces and vertices are read from the Wavefront file.

        Arguments
        ---------
        filename : str
            Filename for Wavefront file.

        Returns
        -------
        surface : Surface
            Surface object with the read surface.

        """
        vertices = []
        faces = []
        with open(filename) as fid:
            for line in fid:
                elements = line.split()
                if not elements:
                    # Empty line
                    continue

                if elements[0] == 'v':
                    vertices.append([float(element)
                                     for element in elements[1:4]])
                elif elements[0] == 'vn':
                    # TODO
                    pass
                elif elements[0] == 'f':
                    faces.append([int(element.split('//')[0])
                                  for element in elements[1:]])
                else:
                    # TODO
                    pass
        return cls(np.array(vertices), np.array(faces) - 1)

    @property
    def vertex_values(self):
        return self._vertex_values

    @vertex_values.setter
    def vertex_values(self, values):
        if values is None:
            self._vertex_values = None
        elif len(values) == self._vertices.shape[0]:
            self._vertex_values = np.asarray(values)
        else:
            raise ValueError('values should be None or length of vertices')

    def faces_as_matrix(self):
        """Return faces as matrix."""
        return Matrix(self._faces)

    def plot(self, *args, **kwargs):
        """Plot surface."""
        self._plot_mayavi(*args, **kwargs)

    def _plot_mayavi(self, *args, **kwargs):
        """Plot surface with Mayavi.

        The x-axis is switched to account for the Mayavi's right-handed
        coordinate system and Talairach's left-handed coordinate system.

        """
        # Delayed import of Mayavi
        from mayavi.mlab import triangular_mesh

        if self._vertex_values is None:
            handle = triangular_mesh(
                self._vertices[:, 0],
                self._vertices[:, 1],
                self._vertices[:, 2],
                self._faces,
                scalars=self._vertex_values,
                *args, **kwargs)
        else:
            handle = triangular_mesh(
                self._vertices[:, 0],
                self._vertices[:, 1],
                self._vertices[:, 2],
                self._faces, 
                scalars=self._vertex_values,
                *args, **kwargs)
        return handle

    def show(self):
        """Show the plotted surface."""
        self._show_mayavi()

    def _show_mayavi(self):
        from mayavi.mlab import show
        show()


read_obj = TriSurface.read_obj
