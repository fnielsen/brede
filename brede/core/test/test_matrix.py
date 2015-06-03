"""Test brede.core.matrix."""


from __future__ import absolute_import, division, print_function

import numpy as np

import pytest

from .. import matrix


@pytest.fixture
def matrix_a():
    """Make simple 2-by-2 matrix."""
    return matrix.Matrix([[1, 2], [3, 4]])


def test_nans(matrix_a):
    """Test nans method."""
    nans = matrix_a.nans()
    assert np.isnan(nans.ix[0, 0])
    assert not np.isnan(matrix_a.ix[0, 0])


def test_collapse_to_two_by_two():
    """Test collapse_to_two_by_two method of Matrix."""
    A = matrix.Matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                      index=['a', 'b', 'c'],
                      columns=['x', 'y', 'z'])

    B = A.collapse_to_two_by_two(first_rows=['a'],
                                 first_columns=['y'])
    assert B.iloc[0, 0] == 2.0
    assert B.iloc[0, 1] == 4.0
    assert B.iloc[1, 0] == 13.0
    assert B.iloc[1, 1] == 26.0


def test_matrix_accuracy():
    """Test accuracy method of Matrix."""
    A = matrix.Matrix([[33, 10], [15, 42]])
    assert A.accuracy() == 0.75
