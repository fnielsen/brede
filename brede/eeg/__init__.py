"""EEG modules, classes and functions."""

from __future__ import absolute_import


from . import plotting
from .core import EegRun
from .plotting import TopoPlot, topoplot


__all__ = ('EegRun', 'TopoPlot', 'plotting', 'topoplot')
