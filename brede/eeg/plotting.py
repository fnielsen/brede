#!/usr/bin/env python
"""
Plot EEG data.

Usage:
  plotting.py topoplot [options] [<file>]

Options:
  -h --help         Show this screen.
  --version         Show version.
  --sample-index=N  Row index (indexed from one).
  --transpose       Transpose data.


References
----------
The five percent electrode system for high-resolution EEG and ERP
measurement, Robert Oostenveld, Peter Praamstra.

"""

from __future__ import absolute_import, division, print_function

from math import cos, pi, sin

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

from scipy.interpolate import griddata


ELECTRODES = {
    'AF3': (-0.25, 0.62),
    'AF4': (0.25, 0.62),
    'AF7': (0.8 * cos(0.7 * pi), 0.8 * sin(0.7 * pi)), 
    'AF8': (0.8 * cos(0.3 * pi), 0.8 * sin(0.3 * pi)), 
    'AFz': (0, 0.6),
    'C1': (-0.2, 0),
    'C2': (0.2, 0),
    'C3': (-0.4, 0),
    'C4': (0.4, 0),
    'C5': (-0.6, 0),
    'C6': (0.6, 0),
    'CP1': (0.2 * cos(1.25 * pi), 0.2 * sin(1.25 * pi)), 
    'CP2': (0.2 * cos(1.75 * pi), 0.2 * sin(1.75 * pi)), 
    'CP3': (0.4 * cos(1.17 * pi), 0.4 * sin(1.17 * pi)), 
    'CP4': (0.4 * cos(1.83 * pi), 0.4 * sin(1.83 * pi)), 
    'CP5': (0.6 * cos(1.15 * pi), 0.6 * sin(1.15 * pi)), 
    'CP6': (0.6 * cos(1.85 * pi), 0.6 * sin(1.85 * pi)), 
    'CPz': (0, -0.2), 
    'Cz': (0, 0),
    'F1': (-0.18, 0.41),
    'F2': (0.18, 0.41),
    'F3': (-0.35, 0.4),
    'F4': (0.35, 0.4),
    'F5': (-0.5, 0.43),
    'F6': (0.5, 0.43),
    'F7': (0.8 * cos(0.8 * pi), 0.8 * sin(0.8 * pi)),
    'F8': (0.8 * cos(0.2 * pi), 0.8 * sin(0.2 * pi)),
    'FC1': (-0.2, 0.21),
    'FC2': (0.2, 0.21),
    'FC3': (-0.39, 0.23),
    'FC4': (0.39, 0.23),
    'FC5': (-0.57, 0.25),
    'FC6': (0.57, 0.25),
    'FCz': (0, 0.2),
    'TP7': (0.8 * cos(1.1 * pi), 0.8 * sin(1.1 * pi)),
    'TP8': (0.8 * cos(1.9 * pi), 0.8 * sin(1.9 * pi)),
    'Fpz': (0, 0.8),
    'FT7': (0.8 * cos(0.9 * pi), 0.8 * sin(0.9 * pi)),
    'FT8': (0.8 * cos(0.1 * pi), 0.8 * sin(0.1 * pi)),
    'Fz': (0, 0.4),
    'FP1': (0.8 * cos(0.6 * pi), 0.8 * sin(0.6 * pi)),
    'FP2': (0.8 * cos(0.4 * pi), 0.8 * sin(0.4 * pi)),
    'Fpz': (0, 0.8),
    'Iz': (0, -1),
    'Nz': (0, 1),
    'P1': (-0.18, -0.41),
    'P2': (0.18, -0.41),
    'P3': (-0.35, -0.42),
    'P4': (0.35, -0.42),
    'P5': (-0.5, -0.43),
    'P6': (0.5, -0.43),
    'P7': (0.8 * cos(1.2 * pi), 0.8 * sin(1.2 * pi)),
    'P8': (0.8 * cos(1.8 * pi), 0.8 * sin(1.8 * pi)),
    'PO3': (-0.24, -0.62),
    'PO4': (0.24, -0.62),
    'PO7': (0.8 * cos(1.3 * pi), 0.8 * sin(1.3 * pi)),
    'PO8': (0.8 * cos(1.7 * pi), 0.8 * sin(1.7 * pi)),
    'POz': (0, -0.6),
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
        """Setup defaults.

        Parameters
        ----------
        data : Pandas.Series or dict
            Pandas Series with values indexed by electrodes.
        axes : matplotlib.axes.AxesSubplot object
            Axis object to render on.

        """
        if axes is None:
            fig = plt.figure()
            axes = fig.gca()
        self.axes = axes
        self.center = np.array((0, 0))
        if data is not None:
            self.data = pd.Series(data)
        else:
            self.data = None

    def draw_electrodes(self):
        """Draw electrodes."""
        for electrode, position in ELECTRODES.items():
            circle = plt.Circle(self.center + position,
                                radius=0.04, fill=True,
                                facecolor=(1, 1, 1))
            self.axes.add_patch(circle)
            position = self.center + position
            self.axes.text(position[0], position[1], electrode,
                           verticalalignment='center',
                           horizontalalignment='center',
                           size=6)

    def draw_head(self):
        """Draw outer head."""
        circle = plt.Circle(self.center, radius=1, fill=False)
        self.axes.add_patch(circle)

    def draw_inner_head(self):
        """Draw inner head."""
        circle = plt.Circle(self.center, radius=0.8, fill=False)
        self.axes.add_patch(circle)

    def draw_nose(self):
        """Draw nose."""
        nose = plt.Line2D([sin(-0.1), 0, sin(0.1)],
                          [cos(-0.1), 1.1, cos(0.1)],
                          color=(0, 0, 0))
        self.axes.add_line(nose)

    def draw_data(self, method='linear', number_of_contours=10):
        """Draw countours from provided data. """
        if self.data is not None:
            # Coordinates for points to interpolate to
            xi, yi = np.mgrid[-1:1:100j, -1:1:100j]

            # Electrode positions for data to interpolate from
            points = [(ELECTRODES[electrode.upper().replace('FPZ', 'Fpz').replace('Z', 'z')]) 
                      for electrode in self.data.index]

            # Interpolate
            zi = griddata(points, self.data.values, (xi, yi), method=method)

            # Defaults
            if number_of_contours is None:
                number_of_contours = 10

            # Draw
            plt.contourf(xi, yi, zi, number_of_contours)

            # TODO: center

    def draw(self, method='linear', number_of_contours=None):
        """Draw all components in topoplot including the data.

        Parameters
        ----------
        data : pandas.Series, optional
            Series with values and indexed by electrode names.
        methods : str, optional
            Interpolation method
        number_of_contours : int
            Number of contours in the colored plot.

        Examples
        --------
        >>> import matplotlib.pyplot as plt
        >>> data = {'O1': 1, 'O2': 2, 'P3': -2, 'P4': -4}
        >>> plt.ion()
        >>> topo_plot = TopoPlot(data)
        >>> topo_plot.draw()

        """
        self.draw_head()
        self.draw_inner_head()
        self.draw_electrodes()
        self.draw_nose()
        self.draw_data(method=method, number_of_contours=number_of_contours)
        self.axes.axis((-1.2, 1.2, -1.2, 1.2))


def topoplot(data=None, axes=None, method='linear', number_of_contours=10):
    """Plot topographic map of the scalp in 2-D circular view.

    Draw the colored scalp map based on data in a Pandas Series where
    the values are indexed according to electrode name.

    Parameters
    ----------
    data : pandas.Series, optional
        Series with values and indexed by electrode names.
    methods : str, optional
        Interpolation method
    number_of_contours : int
        Number of contours in the colored plot.

    References
    ----------
    https://github.com/compmem/ptsa/blob/master/ptsa/plotting/topo.py

    http://sccn.ucsd.edu/~jung/tutorial/topoplot.htm

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> data = {'O1': 1, 'O2': 2, 'P3': -2, 'P4': -4}
    >>> plt.ion()
    >>> topoplot(data)

    """
    topo_plot = TopoPlot(data=data, axes=axes)
    topo_plot.draw(method=method, number_of_contours=number_of_contours)


def main():
    """Hande command-line interface to topographic plot."""
    from docopt import docopt

    args = docopt(__doc__)
    if args['topoplot']:
        if args['<file>'] is None:
            topoplot()
        else:
            filename = args['<file>']
            if filename.lower().endswith('.csv'):
                df = pd.read_csv(filename, index_col=0)
                if args['--transpose']:
                    df = df.T
                if args['--sample-index'] is None:
                    series = (df ** 2).mean()
                else:
                    sample_index = int(args['--sample-index'])
                    series = df.iloc[sample_index - 1, :]
            else:
                exit('Only csv files handled')
            topoplot(series)
    plt.show()


if __name__ == '__main__':
    main()
