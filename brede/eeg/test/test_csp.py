"""Test of csp."""

import numpy as np
import numpy.random as npr

import pytest

from ..csp import CSP


@pytest.fixture
def data():
    """Setup a simulated data set."""
    class Data():
        N = 100

        y = np.zeros(N)
        indices0 = np.arange(N // 2)
        indices1 = np.arange(N // 2, N)
        y[indices1] = 1

        A0 = np.array([[1, 4], [1, 1]])
        A1 = np.array([[4, 1], [1, 1]])
        X = npr.randn(N, 2)
        X[indices0, :] = X[indices0, :].dot(A0)
        X[indices1, :] = X[indices1, :].dot(A1)

    return Data()


def test_csp(data):
    """Test common spatial patterns."""
    csp = CSP()
    assert csp

    csp.fit(data.X, data.y)
    assert csp.weights_.all()

    Z = csp.transform(data.X)
    assert Z.shape == (100, 2)

    assert np.std(Z[data.indices0, 0]) > np.std(Z[data.indices1, 0])
    assert np.std(Z[data.indices0, 1]) < np.std(Z[data.indices1, 1])
