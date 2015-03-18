"""Brede."""


from __future__ import absolute_import

from . import eeg
from . import io
from .eeg import EegRun, topoplot
from .io import read_edf
from brede.core.matrix import Matrix


__all__ = ('eeg', 'io', 'EegRun', 'Matrix', 'read_edf', 'topoplot')
