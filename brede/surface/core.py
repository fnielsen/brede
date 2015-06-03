"""Handle surfaces."""


from __future__ import absolute_import, division, print_function

import numpy as np


class Surface(object):

    """Representation of a surface with faces and vertices."""

    def __init__(self, vertices=None, faces=None, vertex_values=None):
        """Setup vertices, faces and optionally vertex values."""
        self._vertices = vertices
        self._faces = faces
        self._vertex_values = vertex_values

    def __repr__(self):
        """Return string representation."""
        return "Surface(n_vertices={}, n_faces={})".format(
            len(self._vertices), len(self._faces))

    def __str__(self):
        """Return string representation."""
        return self.__repr__()

    @property
    def vertices(self):
        """Return vertices."""
        return self._vertices

    @property
    def faces(self):
        """Return faces."""
        return self._faces

    @property
    def vertex_values(self):
        """Return vertex values."""
        return self._vertex_values

    @vertex_values.setter
    def vertex_values(self, values):
        """Check and set vertex values."""
        if values is None:
            self._vertex_values = None
        elif len(values) == self._vertices.shape[0]:
            self._vertex_values = np.asarray(values)
        else:
            raise ValueError('values should be None or length of vertices')

    def find_closest_vertex(self, coordinate):
        """Return the index of the vertex closest to a given point.

        The distance is computed as the Euclidean distance.

        Parameters
        ----------
        coordinate : tuple of int or float

        Returns
        -------
        index : int
            Index of the vertex that is closest to the coordinate

        Examples
        --------
        >>> vertices = [[0, 1, 0], [1, 0, 0], [0, 0, -1], [-1, 0, 0],
        ...             [0, 1, 0], [0, -1, 0]]
        >>> faces = [[0, 2, 1], [0, 3, 2], [0, 4, 3], [0, 1, 4],
        ...          [5, 1, 2], [5, 2, 3], [5, 3, 4], [5, 4, 1]]
        >>> surface = Surface(vertices, faces)
        >>> surface.find_closest_vertex((2, 0, 0))
        1

        """
        if self._vertices is None:
            return None

        distances = np.sum((np.asarray(self.vertices) - coordinate) ** 2,
                           axis=1)
        index = np.argmin(distances)
        return index


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

    def plot(self, *args, **kwargs):
        """Plot surface.

        Presently Mayavi plots the surface.

        Parameters
        ----------
        title : str
            String to use as title in the plot

        """
        return self._plot_mayavi(*args, **kwargs)

    def _plot_mayavi(self, *args, **kwargs):
        """Plot surface with Mayavi.

        The x-axis is switched to account for the Mayavi's right-handed
        coordinate system and Talairach's left-handed coordinate system.

        Parameters
        ----------
        title : str
            String to use as title in the plot

        """
        # Delayed import of Mayavi
        from mayavi.mlab import title as mlab_title, triangular_mesh

        title = kwargs.pop('title', None)

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

        if title is not None:
            mlab_title(title)

        return handle

    def colorbar(self, *args, **kwargs):
        """Show colorbar for rendered surface."""
        return self._colorbar_mayavi(*args, **kwargs)

    def _colorbar_mayavi(self, *args, **kwargs):
        """Show colorbar in Mayavi."""
        # Delayed import of Mayavi
        from mayavi.mlab import colorbar

        colorbar()

    def show(self):
        """Show the plotted surface."""
        self._show_mayavi()

    def _show_mayavi(self):
        from mayavi.mlab import show
        show()


read_obj = TriSurface.read_obj
