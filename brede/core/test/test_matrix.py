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
