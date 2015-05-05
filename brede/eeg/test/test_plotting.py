"""Test of plotting."""


import matplotlib.pyplot as plt
import pandas as pd

from .. import plotting


def test_topoplot_empty():
    plotting.topoplot()
    plt.ion()
    plt.show()
    plt.close()


def test_topoplot_series():
    data = {'C3': 4, 'Fz': 3, 'C4': 5}
    plotting.topoplot(data)
    plt.ion()
    plt.show()
    plt.close()


def test_topoplot_dataframe():
    data = pd.DataFrame({'C3': [4, 2], 'Fz': [3, 3], 'C4': [5, 3]})
    plotting.topoplot(data)
    plt.ion()
    plt.show()
    plt.close()
