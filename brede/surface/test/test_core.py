"""Test of core module in surface.

Description
-----------
These test functions can be run with

    python -m py.test brede/surface/test/test_core.py

"""


from __future__ import absolute_import

from .. import core


def test_vertex_values_to_colors():
    """Test vertex values transformation."""
    vertex_values = [0.03, 0, 4, 0.3]
    colors = core.vertex_values_to_colors(vertex_values)
    assert colors.max() == 1.0
    assert colors.min() == 0.0


def test_surface():
    """Test instancing of Surface class."""
    vertices = [[0, 1, 0], [1, 0, 0], [0, 0, -1], [-1, 0, 0],
                [0, 1, 0], [0, -1, 0]]
    faces = [[0, 2, 1], [0, 3, 2], [0, 4, 3], [0, 1, 4],
             [5, 1, 2], [5, 2, 3], [5, 3, 4], [5, 4, 1]]
    surface = core.Surface(vertices, faces)
    assert surface.find_closest_vertex((2, 0, 0)) == 1


def test_trisurface():
    """Test instancing of TriSurface class."""
    vertices = [[0, 1, 0], [1, 0, 0], [0, 0, -1], [-1, 0, 0],
                [0, 1, 0], [0, -1, 0]]
    faces = [[0, 2, 1], [0, 3, 2], [0, 4, 3], [0, 1, 4],
             [5, 1, 2], [5, 2, 3], [5, 3, 4], [5, 4, 1]]
    surface = core.TriSurface(vertices, faces)
    assert surface.find_closest_vertex((2, 0, 0)) == 1
