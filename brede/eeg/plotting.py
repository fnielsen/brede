#!/usr/bin/env python
"""
Topographic plot.

References
----------
The five percent electrode system for high-resolution EEG and ERP
measurement, Robert Oostenveld, Peter Praamstra.

"""

from __future__ import division

from math import cos, pi, sin

import matplotlib.pyplot as plt

import numpy as np


ELECTRODES = {
    'AF3': (-0.25, 0.62),
    'AF4': (0.25, 0.62),
    'AFz': (0, 0.6),
    'C3': (-0.4, 0),
    'C4': (0.4, 0),
    'Cz': (0, 0),
    'F3': (-0.35, 0.4),
    'F4': (0.35, 0.4),
    'F7': (0.8 * cos(0.8 * pi), 0.8 * sin(0.8 * pi)),
    'F8': (0.8 * cos(0.2 * pi), 0.8 * sin(0.2 * pi)),
    'FC5': (-0.57, 0.25),
    'FC6': (0.57, 0.25),
    'FCz': (0, 0.2),
    'Fz': (0, 0.4),
    'FP1': (0.8 * cos(0.6 * pi), 0.8 * sin(0.6 * pi)),
    'FP2': (0.8 * cos(0.4 * pi), 0.8 * sin(0.4 * pi)),
    'Fpz': (0, 0.8),
    'Iz': (0, -1),
    'Nz': (0, 1),
    'P3': (-0.35, -0.42),
    'P4': (0.35, -0.42),
    'P7': (0.8 * cos(1.2 * pi), 0.8 * sin(1.2 * pi)),
    'P8': (0.8 * cos(1.8 * pi), 0.8 * sin(1.8 * pi)),
    'Pz': (0, -0.4),
    'O1': (0.8 * cos(1.4 * pi), 0.8 * sin(1.4 * pi)),
    'O2': (0.8 * cos(1.6 * pi), 0.8 * sin(1.6 * pi)),
    'Oz': (0, -0.8),
    'T7': (-0.8, 0),
    'T8': (0.8, 0),
    'T9': (-1, 0),
    'T10': (1, 0),
    'TP9': (cos(1.1 * pi), sin(1.1 * pi)),
    'TP10': (cos(1.9 * pi), sin(1.9 * pi)),
}


class TopoPlot(object):

    """Topographic plot."""

    def __init__(self, data=None, axes=None):
        """Setup defaults."""
        if axes is None:
            axes = plt.axes()
        self.axes = axes
        self.center = np.array((0, 0))

    def electrodes(self):
        """Draw electrodes."""
        for electrode, position in ELECTRODES.items():
            circle = plt.Circle(self.center + position,
                                radius=0.04, fill=False)
            self.axes.add_patch(circle)
            position = self.center + position
            self.axes.text(position[0], position[1], electrode,
                           verticalalignment='center',
                           horizontalalignment='center',
                           size=6)

    def head(self):
        """Draw outer head."""
        circle = plt.Circle(self.center, radius=1, fill=False)
        self.axes.add_patch(circle)

    def inner_head(self):
        """Draw inner head."""
        circle = plt.Circle(self.center, radius=0.8, fill=False)
        self.axes.add_patch(circle)

    def nose(self):
        """Draw nose."""
        pass

    def draw(self):
        """Draw all."""
        self.head()
        self.inner_head()
        self.electrodes()
        self.axes.axis((-1.2, 1.2, -1.2, 1.2))


def topoplot(data=None, axes=None):
    """Plot topogrampic map of the scalp in 2-D circular view.

    References
    ----------
    https://github.com/compmem/ptsa/blob/master/ptsa/plotting/topo.py

    http://sccn.ucsd.edu/~jung/tutorial/topoplot.htm

    Examples
    --------
    >>> import pandas as pd
    >>> data = {'C3': 1, 'C4': 0.5}
    >>> topoplot(data)

    """
    topo_plot = TopoPlot(data=data, axes=axes)
    topo_plot.draw()
    plt.show()


def main():
    """Topographic plot."""
    topoplot()


if __name__ == '__main__':
    main()
