"""Test of plotting."""


import matplotlib.pyplot as plt

import pandas as pd

from .. import plotting


def test_topoplot_empty():
    """Test topoplot function for empty input."""
    plotting.topoplot()
    plt.ion()
    plt.show()
    plt.close()


def test_topoplot_series():
    """Test topoplot function for Series-like input."""
    data = {'C3': 4, 'Fz': 3, 'C4': 5}
    plotting.topoplot(data)
    plt.ion()
    plt.show()
    plt.close()


def test_topoplot_dataframe():
    """Test topoplot function for dataframe-like input."""
    data = pd.DataFrame({'C3': [4, 2], 'Fz': [3, 3], 'C4': [5, 3]})
    plotting.topoplot(data)
    plt.ion()
    plt.show()
    plt.close()
