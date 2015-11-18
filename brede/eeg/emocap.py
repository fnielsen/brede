"""Emocap data containers.

Usage:
  brede.eeg.emocap [options]

Options:
  -h --help  Help

"""

from __future__ import absolute_import, division, print_function

from brede.data.sbs2 import SBS2Data

from pandas import concat, read_csv as pandas_read_csv

from .electrode import EEGAuxElectrodeRun
from .vertex import EEGAuxVertexRun


ELECTRODES = [
    'P4',
    'Fz',
    'TP10',
    'C3',
    'O1',
    'F4',
    'P3',
    'Cz',
    'Fpz',
    'F3',
    'O2',
    'C4',
    'TP9',
    'Pz',
]


class EmocapElectrodeRun(EEGAuxElectrodeRun):
    """Represent a EEG data set recorded from Emocap."""

    _metadata = ['_eeg_columns', '_sampling_rate']

    @property
    def _constructor(self):
        return type(self)

    def __init__(self, data=None, index=None, columns=None, dtype=None,
                 copy=False, sampling_rate=128.0, eeg_columns=None):
        """Construct dataframe-like object."""
        super(EmocapElectrodeRun, self).__init__(
            data=data, index=index, columns=columns,
            dtype=dtype, copy=copy,
            sampling_rate=sampling_rate)

        if eeg_columns is None:
            self._eeg_columns = [column for column in self.columns
                                 if column in ELECTRODES]
        else:
            self._eeg_columns = eeg_columns

    @classmethod
    def read_csv(cls, filename, sampling_rate=128.0, *args, **kwargs):
        """Read comma-separated file.

        Parameters
        ----------
        filename : str
            Filename for the csv file

        Returns
        -------
        eeg_run : EEGRun
            EEGRun dataframe with read data.

        """
        # TODO: This will not instance in derived class.
        return EmocapElectrodeRun(
            pandas_read_csv(filename, *args, **kwargs),
            sampling_rate=sampling_rate)

    def invert(self, method='LORETA'):
        """Return inverted EEG electrode data (sources).

        Parameters
        ----------
        method : 'LORETA' or 'minimumnorm', optional
            Inversion method.

        Returns
        -------
        sources : EmocapVertexRun
            Result of inversion.

        """
        sbs2_data = SBS2Data()

        inverse_model = sbs2_data.inverse_model(
            hardware='emocap', method=method)
        surface = sbs2_data.surface(model='small')

        electrodes = self.eeg_columns[:]
        electrodes.sort()

        X = self.ix[:, electrodes].values
        B = inverse_model.ix[electrodes, :].values
        vertex_names = ['Vertex {}'.format(n + 1) for n in range(B.shape[1])]
        product = EEGAuxVertexRun(
            X.dot(B), index=self.index, columns=vertex_names,
            eeg_columns=vertex_names)

        aux = self.ix[:, self.not_eeg_columns]
        total = concat([product, aux], axis=1)

        all_columns = vertex_names + self.not_eeg_columns

        return EmocapVertexRun(
            total, index=self.index, columns=all_columns,
            eeg_columns=vertex_names, surface=surface)


read_csv = EmocapElectrodeRun.read_csv


class EmocapVertexRun(EEGAuxVertexRun):
    """Represent a Emocap data set at the vertex level."""

    _metadata = ['_eeg_columns', '_sampling_rate', '_surface']

    @property
    def _constructor(self):
        return type(self)

    def __init__(self, data=None, index=None, columns=None, dtype=None,
                 copy=False, sampling_rate=128.0, eeg_columns=None,
                 surface=None):
        """Construct dataframe-like object."""
        super(EmocapVertexRun, self).__init__(
            data=data, index=index, columns=columns,
            dtype=dtype, copy=copy,
            sampling_rate=sampling_rate, surface=surface)

        if eeg_columns is None:
            self._eeg_columns = [column for column in self.columns
                                 if column.startswith('Vertex ')]
        else:
            self._eeg_columns = eeg_columns


def main(args):
    """Handle command-line interface."""
    pass

if __name__ == '__main__':
    import docopt

    main(docopt.docopt(__doc__))
