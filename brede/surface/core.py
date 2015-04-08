"""Handle surfaces."""


from __future__ import division, print_function

from brede.core.matrix import Matrix

from mayavi.mlab import triangular_mesh

import numpy as np

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
                    [0, 1, 0], [0, -1, 0]]
    >>> faces = [[0, 2, 1], [0, 3, 2], [0, 4, 3], [0, 1, 4], 
                 [5, 1, 2], [5, 2, 3], [5, 3, 4], [5, 4, 1]]
    >>> surface = TriSurface(vertices, faces)

    """

    def __init__(self, vertices=None, faces=None, vertex_values=None):
        """Setup vertices, faces and optionally vertex values."""
        self._vertices = np.array(vertices)
        self._faces = np.array(faces)
        self._vertex_values = np.array(vertex_values)

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
                if elements[0] == 'v':
                    vertices.append([float(element) 
                                     for element in elements[1:]])
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

    def faces_as_array(self):
        pass

    def plot(self, *args, **kwargs):
        """Plot surface."""
        self._plot_mayavi(*args, **kwargs)

    def _plot_mayavi(self):
        """Plot surface with Mayavi.

        The x-axis is switched to account the Mayavi's right-handed coordinate 
        system and Talairach's left-handed coordinate system.

        """
        if self._vertex_values is None:
            handle = triangular_mesh(
                -self._vertices[:, 0],
                self._vertices[:, 1],
                self._vertices[:, 2],
                self._faces,
                scalars=self._vertex_values)
        else:
            handle = triangular_mesh(
                -self._vertices[:, 0],
                self._vertices[:, 1],
                self._vertices[:, 2],
                self._faces)
        return handle
