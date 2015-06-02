"""Vertex EEG data."""


from __future__ import absolute_import, division, print_function

from .core import EEGAuxRun


class EEGAuxVertexRun(EEGAuxRun):

    """Represent EEG data on the vertex level.

    The data should be a time series with consecutive time samples equally
    spaced. The primary data is EEG data where each vertex has an associated
    vertex. The data frame may contain auxiliary data in other columns.

    """

    @property
    def _constructor(self):
        return type(self)

    def __init__(self, data=None, index=None, columns=None, dtype=None,
                 copy=False, sampling_rate=None,
                 eeg_columns=None, surface=None):
        """Construct dataframe-like object."""
        super(EEGAuxVertexRun, self).__init__(
            data=data, index=index, columns=columns,
            dtype=dtype, copy=copy,
            sampling_rate=None, eeg_columns=None)

        self._surface = surface

    @property
    def surface(self):
        """Surface associated with data set."""
        return self._surface
