"""Handle surfaces."""


from mayavi.mlab import triangular_mesh


class Surface(object):

    """Representation of a surface with faces and vertices."""

    def __init__(self, vertices=None, faces=None, vertex_values=None):
        """Setup vertices, faces and optionally vertex values."""
        self._vertices = vertices
        self._faces = faces
        self._vertex_values = vertex_values

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
