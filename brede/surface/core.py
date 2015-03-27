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


    def plot(self, *args, **kwargs):
        """Plot surface."""
        self._plot_mayavi(*args, **kwargs)

    def _plot_mayavi(self):
        """Plot surface with Mayavi."""
        if self._vertex_values is None:
            handle = triangular_mesh(
                self._vertices[:, 0],
                self._vertices[:, 1],
                self._vertices[:, 2],
                self._faces,
                scalars=self._vertex_values)
        else:
            handle = triangular_mesh(
                self._vertices[:, 0],
                self._vertices[:, 1],
                self._vertices[:, 2],
                self._faces)
        return handle
