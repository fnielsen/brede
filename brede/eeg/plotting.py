#!/usr/bin/env python
"""
Plot EEG data.

Usage:
  plotting.py [options] [<file>]

Options:
  -h --help         Show this screen.
  --version         Show version.
  --center          Center the data before plotting
  --sample-index=N  Row index (indexed from one).
  --transpose       Transpose data.
  --xlim=lim        X-axis limits.

Data
----
ELECTRODES : dict
    Dictionary indexed by electrode name with 2D positions as values

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


__all__ = ('ELECTRODES', 'MultiPlot', 'TopoPlot', 'topoplot')


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
    'CP1': (-0.18, -0.2),
    'CP2': (0.18, -0.2),
    'CP3': (-0.36, 0.4 * sin(1.17 * pi)),
    'CP4': (0.36, 0.4 * sin(1.83 * pi)),
    'CP5': (0.6 * cos(1.12 * pi), 0.6 * sin(1.12 * pi)),
    'CP6': (0.6 * cos(1.88 * pi), 0.6 * sin(1.88 * pi)),
    'CPz': (0, -0.2),
    'Cz': (0, 0),
    'F1': (-0.18, 0.4),
    'F2': (0.18, 0.4),
    'F3': (-0.35, 0.41),
    'F4': (0.35, 0.41),
    'F5': (-0.5, 0.43),
    'F6': (0.5, 0.43),
    'F7': (0.8 * cos(0.8 * pi), 0.8 * sin(0.8 * pi)),
    'F8': (0.8 * cos(0.2 * pi), 0.8 * sin(0.2 * pi)),
    'FC1': (-0.2, 0.21),
    'FC2': (0.2, 0.21),
    'FC3': (-0.39, 0.22),
    'FC4': (0.39, 0.22),
    'FC5': (-0.57, 0.23),
    'FC6': (0.57, 0.23),
    'FCz': (0, 0.2),
    'FP1': (0.8 * cos(0.6 * pi), 0.8 * sin(0.6 * pi)),
    'FP2': (0.8 * cos(0.4 * pi), 0.8 * sin(0.4 * pi)),
    'Fpz': (0, 0.8),
    'FT7': (0.8 * cos(0.9 * pi), 0.8 * sin(0.9 * pi)),
    'FT8': (0.8 * cos(0.1 * pi), 0.8 * sin(0.1 * pi)),
    'Fz': (0, 0.4),
    'Iz': (0, -1),
    'Nz': (0, 1),
    'P1': (-0.18, -0.41),
    'P2': (0.18, -0.41),
    'P3': (-0.35, -0.42),
    'P4': (0.35, -0.42),
    'P5': (-0.5, -0.44),
    'P6': (0.5, -0.44),
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
    'TP7': (0.8 * cos(1.1 * pi), 0.8 * sin(1.1 * pi)),
    'TP8': (0.8 * cos(1.9 * pi), 0.8 * sin(1.9 * pi)),
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
            self.figure = plt.figure()
            axes = self.figure.gca()
        else:
            self.figure = axes.get_figure()
        self.axes = axes
        self.center = np.array((0, 0))
        if isinstance(data, dict):
            self.data = pd.Series(data)
        elif isinstance(data, pd.Series):
            self.data = data
        elif data is None:
            self.data = None
        else:
            raise ValueError("Wrong type of value for 'data': {}".format(
                type(data)))

    @staticmethod
    def normalize_electrode_name(name):
        """Normalize electrode name.

        Parameters
        ----------
        name : str
            Name of electrode to be normalized

        Examples
        --------
        >>> TopoPlot.normalize_electrode_name('fpz')
        'Fpz'

        >>> TopoPlot.normalize_electrode_name('AFZ')
        'AFz'

        """
        return name.upper().replace('FPZ', 'Fpz').replace('Z', 'z')

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
        """Draw countours from provided data."""
        if self.data is not None:
            # Coordinates for points to interpolate to
            xi, yi = np.mgrid[-1:1:100j, -1:1:100j]

            # Electrode positions for data to interpolate from
            points = []
            for electrode in self.data.index:
                name = TopoPlot.normalize_electrode_name(electrode)
                points.append(ELECTRODES[name])

            # Interpolate
            # TODO: Will not work with 2 electrodes.
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
        self.axes.axis('equal')


class MultiPlot(TopoPlot):

    """Multiple plots orginize topographically.

    References
    ----------
    http://www.fieldtriptoolbox.org/reference/ft_multiploter

    """

    def __init__(self, data=None, axes=None, xlim=None, ylim=None):
        """Setup defaults.

        Parameters
        ----------
        data : Pandas.DataFrame
            Pandas DataFrame with values indexed by electrodes.
        axes : matplotlib.axes.AxesSubplot object
            Axis object to render on.

        """
        if axes is None:
            self.figure = plt.figure()
            axes = self.figure.gca()
        else:
            self.figure = axes.get_figure()
        self.axes = axes

        # Contains a list of axes used to plot data data from individual
        # electrodes
        self._subaxes = []

        self.xlim = xlim
        self.ylim = ylim

        self.center = np.array((0, 0))

        if isinstance(data, pd.DataFrame):
            self.data = data
        elif data is None:
            self.data = None
        else:
            raise ValueError("Wrong type of value for 'data': {}".format(
                type(data)))

    def add_subplot_axes(self, ax, rect, axis_bgcolor=None):
        """Add subaxes to currect specified axes.

        References
        ----------
        Pablo https://stackoverflow.com/users/2309442/pablo

        Pablo's answer to "Embedding small plots inside subplots in matplotlib"
        https://stackoverflow.com/questions/17458580/

        """
        # Modified from
        # https://stackoverflow.com/questions/17458580/
        box = ax.get_position()
        width, height = box.width, box.height
        subaxes_box = [(rect[0], rect[1]), (rect[0]+rect[2], rect[1]+rect[3])]
        subaxes_display_coords = ax.transData.transform(subaxes_box)
        trans_figure = self.figure.transFigure.inverted()
        subaxes_figure_coords = trans_figure.transform(subaxes_display_coords)
        x, y = subaxes_figure_coords[0, :]
        width, height = (subaxes_figure_coords[1, :] -
                         subaxes_figure_coords[0, :])
        subaxes = self.figure.add_axes(
            [x, y, width, height], axis_bgcolor=axis_bgcolor)
        x_labelsize = subaxes.get_xticklabels()[0].get_size()
        y_labelsize = subaxes.get_yticklabels()[0].get_size()
        x_labelsize *= rect[2]**0.5
        y_labelsize *= rect[3]**0.5
        subaxes.xaxis.set_tick_params(labelsize=x_labelsize)
        subaxes.yaxis.set_tick_params(labelsize=y_labelsize)
        return subaxes

    def draw_data(self, width=0.25, height=0.25, xlim=None, ylim=None):
        """Draw data."""
        if self.data is not None:

            if ylim is None:
                if self.ylim is None:
                    ylim = self.auto_ylim(xlim)
                else:
                    ylim = self.ylim

            if xlim is None:
                xlim = self.xlim

            for electrode in self.data.columns:
                if electrode in ELECTRODES:

                    # Axes and position
                    x, y = ELECTRODES[electrode]
                    subaxes = self.add_subplot_axes(
                        self.axes, [x - width/2, y - height/2, width, height],
                        axis_bgcolor='w')

                    # Actual data plot
                    self.data.ix[:, electrode].plot(
                        ax=subaxes, xlim=xlim, ylim=ylim)

                    # Annotation
                    # http://matplotlib.org/users/transforms_tutorial.html
                    subaxes.text(0.5, 0.95, electrode,
                                 transform=subaxes.transAxes,
                                 fontweight='bold', va='top', ha='center')
                    subaxes.set_yticklabels([])
                    subaxes.set_xticklabels([])

                    self._subaxes.append(subaxes)

    @property
    def xlim(self):
        """Return xlim for subplots."""
        lim = [ax.get_xlim() for ax in self._subaxes]
        if lim == []:
            lim = None
        return lim

    @xlim.setter
    def xlim(self, left=None, right=None):
        """Set x-axis limits on all subplots."""
        for ax in self._subaxes:
            ax.set_xlim(left, right)
        self.figure.canvas.draw()

    @property
    def ylim(self):
        """Return ylim for subplots."""
        lim = [ax.get_ylim() for ax in self._subaxes]
        if lim == []:
            lim = None
        return lim

    @ylim.setter
    def ylim(self, bottom=None, top=None):
        """Set y-axis limits on all subplots."""
        for ax in self._subaxes:
            ax.set_ylim(bottom, top)
        self.figure.canvas.draw()

    def auto_ylim(self, xlim=None):
        """Return an estimate for a good ylim."""
        electrodes = [col for col in self.data.columns
                      if col in ELECTRODES]
        if xlim is None:
            data = self.data.ix[:, electrodes]
        else:
            indices = ((self.data.index >= xlim[0]) &
                       (self.data.index <= xlim[1]))
            data = self.data.ix[indices, electrodes]
        min_data = data.min().min()
        max_data = data.max().max()
        if min_data >= 0:
            ylim = 0, max_data
        else:
            abs_max = max(abs(min_data), max_data)
            ylim = -abs_max, abs_max
        return ylim

    def draw(self, xlim=None, ylim=None):
        """Draw all components in multiplot including the data.

        Parameters
        ----------
        xlim : tuple of floats, optional
            X-axis limits used for each individual plots
        ylim : tuple of floats, optional
            Y-axis limits used for each individual plots

        """
        self.axes.axis((-1.2, 1.2, -1.2, 1.2))
        self.draw_head()
        self.draw_inner_head()
        self.draw_nose()
        self.draw_data(xlim=xlim, ylim=ylim)


def topoplot(data=None, axes=None, method='linear', number_of_contours=10,
             xlim=None, ylim=None):
    """Plot topographic map of the scalp in 2-D circular view.

    Draw the colored scalp map based on data in a Pandas Series where
    the values are indexed according to electrode name.

    Parameters
    ----------
    data : pandas.Series or pandas.DataFrame, optional
        Series with values and indexed by electrode names.
    methods : str, optional
        Interpolation method
    number_of_contours : int
        Number of contours in the colored plot.
    xlim : 2-tuple of floats, optional
        Limits of x-axis in multiplot
    ylim : 2-tuple of floats, optional
        Limits of y-axis in multiplot

    References
    ----------
    https://github.com/compmem/ptsa/blob/master/ptsa/plotting/topo.py

    http://sccn.ucsd.edu/~jung/tutorial/topoplot.htm

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> data = {'O1': 1, 'O2': 2, 'P3': -2, 'P4': -4}
    >>> plt.ion()
    >>> topo_plot = topoplot(data)

    """
    if isinstance(data, pd.Series) or isinstance(data, dict) or data is None:
        topo_plot = TopoPlot(data=data, axes=axes)
        topo_plot.draw(method=method, number_of_contours=number_of_contours)
        return topo_plot
    elif isinstance(data, pd.DataFrame):
        multi_plot = MultiPlot(data=data, axes=axes)
        multi_plot.draw(xlim=xlim, ylim=ylim)
        return multi_plot


def show():
    """Show plot."""
    plt.show()


def main(args):
    """Handle command-line interface to topographic plot."""
    xlim = args['--xlim']
    if args['--xlim'] is not None:
        xlim = [float(lim) for lim in xlim.split(',')]

    if args['<file>'] is None:
        topoplot()
    else:
        filename = args['<file>']
        if filename.lower().endswith('.csv'):
            from .core import read_csv

            df = read_csv(filename, index_col=0)
            if args['--transpose']:
                df = df.T
            if args['--sample-index'] is None:
                if args['--center'] is not None:
                    df = df.center()
                topoplot(df, xlim=xlim)
            else:
                sample_index = int(args['--sample-index'])
                series = df.iloc[sample_index - 1, :]
                topoplot(series)
        else:
            exit('Only csv files handled')
    plt.show()


if __name__ == '__main__':
    from docopt import docopt

    main(docopt(__doc__))
