"""Core data structures for EEG data."""

from __future__ import absolute_import, division, print_function

# Absolute path here because of circular dependency
import brede.io

import numpy as np

from pandas import DataFrame
from pandas import read_csv as pandas_read_csv


class UnevenSamplingRate(Exception):

    """Exception for uneven sampling intervals."""

    pass


class EEGRun(DataFrame):

    """Represent a EEG data set.

    The Pandas DataFrame class is reused and extended with, e.g., Fourier
    transformation.

    """

    @property
    def _constructor(self):
        return EEGRun

    def __init__(self, data=None, index=None, columns=None, dtype=None,
                 copy=False, sampling_rate=1.0):
        """Construct dataframe-like object."""
        DataFrame.__init__(self, data=data, index=index, columns=columns,
                           dtype=dtype, copy=copy)
        if sampling_rate is not None:
            self.index = np.arange(0, len(self) * sampling_rate, sampling_rate)

    def sampling_rate(self):
        """Return sampling rate."""
        intervals = np.diff(self.index.values)
        interval = intervals.mean()
        interval_variation = max(intervals - interval) / interval
        if interval_variation > 10 ** -10:
            raise UnevenSamplingRate
        return 1 / interval

    @classmethod
    def read_csv(cls, filename, sampling_rate=1.0, *args, **kwargs):
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
        return cls(pandas_read_csv(filename, *args, **kwargs),
                   sampling_rate=sampling_rate)

    @classmethod
    def read_edf(cls, filename):
        """Read EDF file.

        Parameters
        ----------
        filename : str
            Filename for EDF file.

        Returns
        -------
        eeg_run : EEGRun
            EEGRun dataframe with read data.

        """
        return cls(brede.io.read_edf(filename))

    def fft(self):
        """Fourier transform of data.

        Returns
        -------
        df : pandas.DataFrame
            Dataframe with frequencies in rows and electrodes in columns.

        Examples
        --------
        >>> eeg_run = EEGRun({'Cz': [1, -1, 1, -1]})
        >>> fourier = eeg_run.fft()
        >>> fourier.Cz.real
        array([ 0.,  0.,  4.,  0.])

        """
        fourier = np.fft.fft(self, axis=0)
        frequencies = np.fft.fftfreq(self.shape[0], 1 / self.sampling_rate())
        return DataFrame(fourier, index=frequencies, columns=self.columns)


read_csv = EEGRun.read_csv
