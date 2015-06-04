"""EEG modules, classes and functions."""


from __future__ import absolute_import

from . import plotting
from .core import EEGAuxRun, EEGRun
from .plotting import TopoPlot, topoplot


__all__ = ('EEGAuxRun', 'EEGRun', 'TopoPlot', 'plotting', 'topoplot')
