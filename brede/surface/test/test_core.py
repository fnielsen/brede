"""Test of core module in surface."""


from .. import core


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
