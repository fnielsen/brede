"""Core data structures for EEG data.

Representing EEG data on the vertex level.

"""


from __future__ import absolute_import, division, print_function

# Absolute path here because of circular dependency
import brede.io

import matplotlib.pyplot as plt

import numpy as np

from pandas import DataFrame, Series
from pandas import read_csv as pandas_read_csv
from pandas.core.internals import BlockManager

from scipy.signal import lfilter, welch

from .csp import CSP
from .filter import bandpass_filter_coefficients, lowpass_filter_coefficients
from .plotting import MultiPlot
from ..core.matrix import Matrix
from ..core.tensor import Tensor
from ..core.tensor4d import Tensor4D


ELECTRODES = {
    'AF3',
    'AF4',
    'AF7',
    'AF8',
    'AFz',
    'C1',
    'C2',
    'C3',
    'C4',
    'C5',
    'C6',
    'CP1',
    'CP2',
    'CP3',
    'CP4',
    'CP5',
    'CP6',
    'CPz',
    'Cz',
    'F1',
    'F2',
    'F3',
    'F4',
    'F5',
    'F6',
    'F7',
    'F8',
    'FC1',
    'FC2',
    'FC3',
    'FC4',
    'FC5',
    'FC6',
    'FCz',
    'FP1',
    'FP2',
    'Fpz',
    'FT7',
    'FT8',
    'Fz',
    'Iz',
    'Nz',
    'P1',
    'P2',
    'P3',
    'P4',
    'P5',
    'P6',
    'P7',
    'P8',
    'PO3',
    'PO4',
    'PO7',
    'PO8',
    'POz',
    'Pz',
    'O1',
    'O2',
    'Oz',
    'T7',
    'T8',
    'T9',
    'T10',
    'TP7',
    'TP8',
    'TP9',
    'TP10'
}

EMOTIV_TO_EMOCAP_MAP = {
    'F3': 'P4',
    'FC6': 'Fz',
    'P7': 'TP10',
    'T8': 'C3',
    'F7': 'O1',
    'F8': 'F4',
    'T7': 'P3',
    'P8': 'Cz',
    'AF4': 'Fpz',
    'F4': 'F3',
    'AF3': 'O2',
    'O2': 'C4',
    'O1': 'TP9',
    'FC5': 'Pz',
}


def fix_electrode_name(electrode):
    """Make electrode name canonical.

    Parameters
    ----------
    electrode : str
        Name of electrode in the 10-20 system

    Returns
    -------
    name : str
        New canonical name for electrode

    Examples
    --------
    >>> fix_electrode_name('Fc5')
    'FC5'

    >>> fix_electrode_name('Poz')
    'POz'

    """
    return electrode.upper().replace('Z', 'z')


class UnevenSamplingRateError(Exception):

    """Exception for uneven sampling intervals."""

    pass


class ElectrodeNameError(Exception):

    """Exception for wrong electrode name."""

    pass


class EEGRun(Matrix):

    """Represent a EEG data set.

    The 'run' should be a temporally contiguous data set with a fixed sampling
    rate.

    The Pandas DataFrame class is reused and extended with, e.g., Fourier
    transformation.

    """

    _metadata = ['_sampling_rate']

    @property
    def _constructor(self):
        return type(self)

    def __init__(self, data=None, index=None, columns=None, dtype=None,
                 copy=False, sampling_rate=None):
        """Construct dataframe-like object."""
        super(EEGRun, self).__init__(
            data=data, index=index, columns=columns,
            dtype=dtype, copy=copy)

        if sampling_rate is not None:
            if index is None and not isinstance(data, BlockManager):
                self.index = np.arange(
                    0, len(self) / sampling_rate, 1 / sampling_rate)
            self._sampling_rate = float(sampling_rate)
        else:
            if self.shape[0] == 1:
                self._sampling_rate = 1.0
            else:
                try:
                    self._sampling_rate = self.sampling_rate_from_index()
                except UnevenSamplingRateError:
                    self._sampling_rate = 1.0

    def sampling_rate_from_index(self):
        """Return sampling rate.

        Raises
        ------
        UnevenSamplingRateError

        """
        intervals = np.diff(self.index.values)
        interval = intervals.mean()
        interval_variation = max(intervals - interval) / interval
        if interval_variation > 10 ** -10:
            raise UnevenSamplingRateError
        return 1 / interval

    @property
    def sampling_rate(self):
        """Return sampling rate.

        Raises
        ------
        UnevenSamplingRateError

        """
        return self._sampling_rate

    @sampling_rate.setter
    def sampling_rate(self, value):
        """Set the index from the sampling rate.

        Parameters
        ----------
        value : float
            Sampling rate

        """
        self._sampling_rate = value
        # self.index = np.arange(0, len(self) / value, 1 / value)

    @classmethod
    def read_csv(cls, filename, sampling_rate=None, *args, **kwargs):
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
        return EEGRun(pandas_read_csv(filename, *args, **kwargs),
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

    def emotiv_to_emocap(self, check_all=True, change_qualities=True,
                         inplace=False):
        """Change column names for Emotiv electrodes to Emocap.

        Specialized method to change the electrode names in the dataframe
        column from Emotiv electrode names to Emocap electrode names.

        Returns
        -------
        eeg_run : EEGRun
            EEGRun data

        """
        emotiv_to_emocap_map = EMOTIV_TO_EMOCAP_MAP

        if check_all:
            for column in emotiv_to_emocap_map:
                if column not in self.columns:
                    message = "Electrode '{}' not in columns".format(column)
                    raise ElectrodeNameError(message)

        columns = []
        for column in self.columns:
            if column in emotiv_to_emocap_map:
                columns.append(emotiv_to_emocap_map[column])
            elif (change_qualities and column.endswith(' quality') and
                  column[:-8] in emotiv_to_emocap_map):
                columns.append(emotiv_to_emocap_map[column[:-8]] + ' quality')
            else:
                columns.append(column)

        if inplace:
            self.columns = columns
            return self
        else:
            return self._constructor(
                self.values, index=self.index, copy=True, columns=columns,
                sampling_rate=self.sampling_rate)

    def fix_electrode_names(self, inplace=False):
        """Make electrode names canonical.

        Parameters
        ----------
        inplace : bool, optional
            Determines if the object should be copied or modified

        Returns
        -------
        eeg_run : EEGRun
            Copy or modified data.

        """
        new_columns = []
        for column in self.columns:
            new_name = fix_electrode_name(column)
            if new_name in ELECTRODES:
                new_columns.append(new_name)
            else:
                new_columns.append(column)
        if inplace:
            self.columns = new_columns
            return self
        else:
            return self._constructor(self, columns=new_columns, copy=True)

    def find_transitions(self, columns):
        """Return integer-location row indices with transisitions.

        The first index is not returned.

        Parameters
        ----------
        columns : str
            Column or columns to look for transitions

        Returns
        -------
        indices : list of int
             List of integer-location indices

        Examples
        --------
        >>> eeg_run = EEGRun([[1, 2], [1, 3], [2, 3]], columns=['C3', 'C4'])
        >>> eeg_run.find_transitions('C3')
        [2]

        >>> eeg_run.find_transitions('C4')
        [1]

        >>> eeg_run.find_transitions(['C3', 'C4'])
        [1, 2]

        """
        assert not np.any(self.ix[:, columns].isnull())
        # >>> np.nan == np.nan
        # False
        # >>> (np.nan, 2) == (np.nan, 2)
        # True

        previous = None
        indices = []
        series = self.ix[:, columns]
        if isinstance(series, DataFrame):
            series = (value for key, value in self.ix[:, columns].iterrows())
        for index, elements in enumerate(series):
            if index == 0:
                previous = elements
                continue

            if np.any(elements != previous):
                indices.append(index)
                previous = elements
        return indices

    def merge_events(self, events, left_on=None, right_on=None,
                     fill_method='pad'):
        """Merge event data with eeg data.

        Parameters
        ----------
        events : brede.core.matrix.Matrix
            Data frame with events
        left_on : str, optional
            Column to match on. If None uses the index
        right_on : str, optional
            Column to match on. If None uses the index
        fill_method : 'pad' or None
            Parameter forwarded to fillna's method parameter

        Returns
        -------
        new : EEGAuxRun

        See also
        --------
        pandas.DataFrame.fillna

        """
        if left_on is None:
            left = self.index.values
        else:
            left = self.ix[:, left_on].values
        if right_on is None:
            right = events.index.values
        else:
            right = events.ix[:, right_on].values

        matrix_events = Matrix(
            np.zeros((self.shape[0], events.shape[1])) * np.nan,
            index=self.index,
            columns=events.columns)

        for event_index in right:
            nearest_index = np.argmin(abs(left - event_index))
            matrix_events.iloc[nearest_index, :] = events.ix[event_index, :]

        if fill_method is not None:
            matrix_events = matrix_events.fillna(method='pad')

        new = self._constructor(self, copy=True)
        new = new.combine_first(matrix_events)
        return new

    def rereference(self, mode='mean', electrode=None, inplace=False):
        """Rereference electrode values.

        Parameters
        ----------
        mode : mean, median or electrode, optional
            Type of rereferencing
        electrode : str, optional
            Electrode name corresponding to column
        inplace : bool, optional
            Whether to copy or inplace modify

        Returns
        -------
        new : EEGRun
            New EEG data rereferenced.

        Examples
        --------
        >>> eeg_run = EEGRun([[1, 2, 6]], columns=['C3', 'Cz', 'C4'])
        >>> float(eeg_run.rereference().ix[0, 'C3'])
        -2.0

        >>> float(eeg_run.rereference(mode='median').ix[0, 'C3'])
        -1.0

        """
        if mode == 'mean':
            reference = self.mean(axis=1)
        elif mode == 'median':
            reference = self.median(axis=1)
        elif mode == 'electrode':
            if electrode is None:
                raise ValueError('electrode parameter should be defined')
            reference = self.ix[:, electrode]
        else:
            raise ValueError('Wrong mode parameter: {}'.format(mode))

        if inplace:
            self -= np.tile(reference, (self.shape[1], 1)).T
            return self
        else:
            new = self._constructor(self, copy=True)
            new -= np.tile(reference, (self.shape[1], 1)).T
            return new

    def bandpass_filter(self, low_cutoff_frequency=1.0,
                        high_cutoff_frequency=45.0, order=4, inplace=False):
        """Filter electrode data temporally with bandpass filter.

        Parameters
        ----------
        low_cutoff_frequency : float
            Frequency in Hertz
        high_cutoff_frequency : float
            Frequency in Hertz
        order : int, optional
            Order of filter [default: 4].

        """
        b, a = bandpass_filter_coefficients(
            low_cutoff_frequency, high_cutoff_frequency,
            sampling_rate=self.sampling_rate, order=order)
        Y = lfilter(b, a, self, axis=0)

        if inplace:
            self.ix[:, :] = Y
            return self
        else:
            new = self._constructor(Y, columns=self.columns, copy=True,
                                    sampling_rate=self.sampling_rate)
            return new

    def lowpass_filter(self, cutoff_frequency=1.0,
                       order=4, inplace=False):
        """Filter electrode data temporally with lowpass filter.

        Parameters
        ----------
        cutoff_frequency : float
            Frequency in Hertz
        order : int, optional
            Order of filter [default: 4].
        inplace : bool, optional
            Whether to make a new object or an overwrite

        """
        b, a = lowpass_filter_coefficients(
            cutoff_frequency,
            sampling_rate=self.sampling_rate, order=order)
        Y = lfilter(b, a, self, axis=0)

        if inplace:
            self.ix[:, :] = Y
            return self
        else:
            new = self._constructor(Y, columns=self.columns, copy=True,
                                    sampling_rate=self.sampling_rate)
            return new

    def fft(self):
        """Fourier transform of data.

        Returns
        -------
        spectrum : Spectra
            Dataframe-like with frequencies in rows and electrodes in columns.

        Examples
        --------
        >>> eeg_run = EEGRun({'Cz': [1, -1, 1, -1]})
        >>> fourier = eeg_run.fft()
        >>> fourier.Cz.real
        array([ 0.,  0.,  4.,  0.])

        """
        fourier = np.fft.fft(self, axis=0)
        frequencies = np.fft.fftfreq(self.shape[0], 1 / self.sampling_rate)
        return Spectra(fourier, index=frequencies, columns=self.columns)

    def plot_electrode_spectrogram(self, electrode, nfft=None, noverlap=128):
        """Plot the spectrogram for specified electrode.

        Parameters
        ----------
        electrode : str
            Electrode name corresponding to column name.
        NFFT : int, optional
            The number of data points used in the FFT.
        noverlap : int, optional
            The number of overlap between time blocks.

        """
        plt.specgram(self.ix[:, electrode], NFFT=nfft, Fs=self.sampling_rate,
                     noverlap=noverlap)
        plt.title('Spectrogram for {}'.format(electrode))
        plt.xlabel('Time [seconds]')
        plt.ylabel('Frequency [Hz]')

    def welch(self, window='hanning', nperseg=256):
        """Return Welch periodogram.

        Parameters
        ----------
        window : str or tuple or array_like, optional
            Desired window
        nperseg : int, optional
            Length of each segment.  Defaults to 256.

        Returns
        -------
        periodogram : Spectra

        """
        frequencies, Pxx = welch(self.T, fs=self.sampling_rate, window=window,
                                 nperseg=nperseg)
        periodogram = Spectra(Pxx.T, index=frequencies, columns=self.columns)
        return periodogram


class EEGRuns(Tensor):

    """Multiple EEGRuns of the same length."""

    _metadata = ['sampling_rate']

    @property
    def _constructor(self):
        return type(self)

    def __init__(self, data=None, items=None, major_axis=None, minor_axis=None,
                 dtype=None, copy=False, sampling_rate=1.0):
        """Construct Panel-like object."""
        super(EEGRuns, self).__init__(
            data=data, items=items, major_axis=major_axis,
            minor_axis=minor_axis,
            dtype=dtype, copy=copy)
        if sampling_rate is not None:
            self.major_axis = np.arange(
                0, self.shape[1] / sampling_rate, 1 / sampling_rate)

    @property
    def sampling_rate(self):
        """Return sampling rate.

        Raises
        ------
        UnevenSamplingRateError

        """
        intervals = np.diff(self.major_axis.values)
        interval = intervals.mean()
        interval_variation = max(intervals - interval) / interval
        if interval_variation > 10 ** -10:
            raise UnevenSamplingRateError
        return 1 / interval

    @sampling_rate.setter
    def sampling_rate(self, value):
        """Set the major axis from the sampling rate.

        Parameters
        ----------
        value : float
            Sampling rate

        """
        self.major_axis = np.arange(0, self.shape[1] / value, 1 / value)

    def fft(self):
        """Fourier transform of data.

        Returns
        -------
        spectrum : Tensor
            Panel-like with frequencies in major_axis

        Examples
        --------
        >>> eeg_runs = EEGRuns({'Trial 1': {'Cz': [1, -1, 1, -1]}})
        >>> fourier = eeg_runs.fft()
        >>> fourier['Trial 1', :, 'Cz'].real
        array([ 0.,  0.,  4.,  0.])

        """
        fourier = np.fft.fft(self, axis=1)
        frequencies = np.fft.fftfreq(self.shape[1], 1 / self.sampling_rate)
        return Spectra3D(fourier, items=self.items,
                         major_axis=frequencies, minor_axis=self.minor_axis)


class EEGRuns4D(Tensor4D):

    """Multiple EEGRuns of the same length."""

    _metadata = ['sampling_rate']

    @property
    def _constructor(self):
        return type(self)

    def __init__(self, data=None, labels=None, items=None,
                 major_axis=None, minor_axis=None,
                 dtype=None, copy=False, sampling_rate=1.0):
        """Construct Panel-like object."""
        super(EEGRuns4D, self).__init__(
            data=data, labels=labels, items=items,
            major_axis=major_axis, minor_axis=minor_axis,
            dtype=dtype, copy=copy)
        if sampling_rate is not None:
            self.major_axis = np.arange(
                0, self.shape[2] / sampling_rate, 1 / sampling_rate)

    @property
    def sampling_rate(self):
        """Return sampling rate.

        Raises
        ------
        UnevenSamplingRateError

        """
        intervals = np.diff(self.major_axis.values)
        interval = intervals.mean()
        interval_variation = max(intervals - interval) / interval
        if interval_variation > 10 ** -10:
            raise UnevenSamplingRateError
        return 1 / interval

    @sampling_rate.setter
    def sampling_rate(self, value):
        """Set the major axis from the sampling rate.

        Parameters
        ----------
        value : float
            Sampling rate

        """
        self.major_axis = np.arange(0, self.shape[2] / value, 1 / value)

    def fft(self):
        """Fourier transform of data.

        Returns
        -------
        spectra : Spectra4D
            Panel4D-like with frequencies in major_axis

        Examples
        --------
        >>> eeg_runs = EEGRuns4D([[[[1], [-1], [1], [-1]]]],
        ...     labels=['baseline'], items=['Trial 1'],
        ...     minor_axis=['Cz'])
        >>> fourier = eeg_runs.fft()
        >>> fourier['baseline', 'Trial 1', :, 'Cz'].real
        array([ 0.,  0.,  4.,  0.])

        """
        fourier = np.fft.fft(self, axis=2)
        frequencies = np.fft.fftfreq(self.shape[2], 1 / self.sampling_rate)
        return Spectra4D(fourier, labels=self.labels, items=self.items,
                         major_axis=frequencies, minor_axis=self.minor_axis)


class EEGAuxRun(EEGRun):

    """Represent a EEG data set with auxilliary data.

    The Pandas DataFrame class is reused and extended with, e.g., Fourier
    transformation.

    Attributes
    ----------
    sampling_rate : float
        Sampling rate in Hertz
    eeg_columns : list of str
        Names for columns that contain EEG data (e.g., electrodes)

    """

    _metadata = ['_eeg_columns', '_sampling_rate']

    @property
    def _constructor(self):
        return type(self)

    def __init__(self, data=None, index=None, columns=None, dtype=None,
                 copy=False, sampling_rate=None, eeg_columns=None):
        """Construct dataframe-like object."""
        super(EEGAuxRun, self).__init__(
            data=data, index=index, columns=columns,
            dtype=dtype, copy=copy, sampling_rate=sampling_rate)

        if eeg_columns is None:
            if hasattr(data, '_eeg_columns'):
                self._eeg_columns = data._eeg_columns
            else:
                self._eeg_columns = []
        else:
            self._eeg_columns = eeg_columns

    def __getitem__(self, key):
        """Get column or columns."""
        value = super(EEGAuxRun, self).__getitem__(key)

        if isinstance(value, EEGAuxRun):
            new_columns = [column for column in key
                           if column in self._eeg_columns]
            value._eeg_columns = new_columns
        return value

    @property
    def eeg_columns(self):
        """Return columns that contain EEG data."""
        return self._eeg_columns

    @property
    def not_eeg_columns(self):
        """Return columns that are not electrodes.

        Examples
        --------
        >>> from numpy.random import randn
        >>> data = EEGAuxRun(randn(20, 3), columns=['C3', 'C4', 'State'],
        ...     eeg_columns=['C3', 'C4'])
        >>> data.not_eeg_columns
        ['State']

        """
        not_eeg_columns = [column for column in self.columns
                           if column not in set(self._eeg_columns)]
        return not_eeg_columns

    @classmethod
    def read_csv(cls, filename, sampling_rate=None, *args, **kwargs):
        """Read comma-separated file.

        Parameters
        ----------
        filename : str
            Filename for the csv file

        Returns
        -------
        eeg_run : EEGAuxRun
            EEGAuxRun dataframe with read data.

        """
        # TODO: This will not instance in derived class.
        return EEGAuxRun(pandas_read_csv(filename, *args, **kwargs),
                         sampling_rate=sampling_rate)

    def global_field_power(self):
        """Compute global field power.

        Returns
        -------
        gfp : pandas.Series
            Global field power

        References
        ----------
        Reference-free identification of components of checkerboard-evoked
        multichannel potential fields

        """
        reference = self.ix[:, self._eeg_columns].mean(axis=1)
        centered = self.ix[:, self._eeg_columns] - np.tile(
            reference, (len(self._eeg_columns), 1)).T
        gfp = (centered ** 2).mean(axis=1)
        return gfp

    def abser(self, inplace=False):
        """Compute the absolute value for the electrode data.

        Returns
        -------
        new : EEGAuxRun
            New DataFrame-like object with absolute electrode data.

        """
        if inplace:
            self.ix[:, self._eeg_columns] = self.ix[:, self.eeg_columns].abs()
            return self
        else:
            new = self._constructor(self, copy=True)
            new.ix[:, self._eeg_columns] = new.ix[:, self._eeg_columns].abs()
            return new

    def center(self, inplace=False):
        """Center the EEG data.

        The mean for each EEG column is computed and subtracted from the
        relevant columns.

        Returns
        -------
        new : EEGAuxRun
            New DataFrame-like object with centered EEG data.

        """
        means = self.ix[:, self._eeg_columns].mean(axis=0)
        if inplace:
            self.ix[:, self._eeg_columns] -= means
            return self
        else:
            new = self._constructor(self, copy=True)
            new.ix[:, self._eeg_columns] -= means
            return new

    def rereference(self, mode='mean', column=None, inplace=False):
        """Rereference EEG values.

        Parameters
        ----------
        mode : mean, median or electrode, optional
            Type of rereferencing
        column : str, optional
            Electrode name corresponding to column
        inplace : bool, optional
            Whether to copy or inplace modify

        Returns
        -------
        new : EEGAuxRun
            New EEG data rereferenced.

        Examples
        --------
        >>> eeg_run = EEGAuxRun([[1, 2, 6]], columns=['C3', 'Cz', 'C4'],
        ...     eeg_columns=['C3', 'Cz', 'C4'])
        >>> float(eeg_run.rereference().ix[0, 'C3'])
        -2.0

        >>> float(eeg_run.rereference(mode='median').ix[0, 'C3'])
        -1.0

        """
        if mode == 'mean':
            reference = self.ix[:, self._eeg_columns].mean(axis=1)
        elif mode == 'median':
            reference = self.ix[:, self._eeg_columns].median(axis=1)
        elif mode == 'electrode':
            if column is None:
                raise ValueError('column parameter should be defined')
            reference = self.ix[:, column]
        else:
            raise ValueError('Wrong mode parameter: {}'.format(mode))

        if inplace:
            self.ix[:, self._eeg_columns] -= np.tile(
                reference, (len(self._eeg_columns), 1)).T
            return self
        else:
            new = self._constructor(self, copy=True)
            new.ix[:, self._eeg_columns] -= np.tile(
                reference, (len(self._eeg_columns), 1)).T
            return new

    def power(self, inplace=False):
        """Compute the power for the EEG data.

        Returns
        -------
        new : EEGAuxRun
            New DataFrame-like object with standardized EEG data.

        """
        if inplace:
            self.ix[:, self._eeg_columns] **= 2
            return self
        else:
            new = self._constructor(self, copy=True)
            new.ix[:, self._eeg_columns] **= 2
            return new

    def standardize(self, inplace=False):
        """Standardize the EEG data.

        Standardize means to devide with the standard deviation of the time
        series of each electrode.

        Returns
        -------
        new : EEGAuxRun
            New DataFrame-like object with standardized EEG data.

        """
        stds = self.ix[:, self._eeg_columns].std(axis=0)
        if inplace:
            self.ix[:, self._eeg_columns] /= stds
            return self
        else:
            new = self._constructor(self, copy=True)
            new.ix[:, self._eeg_columns] /= stds
            return new

    def bandpass_filter(self, low_cutoff_frequency=1.0,
                        high_cutoff_frequency=45.0, order=4, inplace=False):
        """Filter EEG data with a temporal bandpass filter.

        Parameters
        ----------
        low_cutoff_frequency : float
            Frequency in Hertz
        high_cutoff_frequency : float
            Frequency in Hertz
        order : int, optional
            Order of filter [default: 4].

        Returns
        -------
        new : EEGAuxRun
            New DataFrame-like object with bandpass filtered EEG data.

        """
        b, a = bandpass_filter_coefficients(
            low_cutoff_frequency, high_cutoff_frequency,
            sampling_rate=self.sampling_rate, order=order)
        Y = lfilter(b, a, self.ix[:, self._eeg_columns], axis=0)

        if inplace:
            self.ix[:, self._eeg_columns] = Y
            return self
        else:
            new = self._constructor(self, copy=True)
            new.ix[:, self._eeg_columns] = Y
            return new

    def lowpass_filter(self, cutoff_frequency=1.0,
                       order=4, inplace=False):
        """Filter EEG data temporally with lowpass filter.

        Parameters
        ----------
        cutoff_frequency : float
            Frequency in Hertz
        order : int, optional
            Order of filter [default: 4].
        inplace : bool, optional
            Whether to make a new object or an overwrite

        Returns
        -------
        new : EEGAuxRun
            New DataFrame-like object with bandpass filtered EEG data.

        """
        b, a = lowpass_filter_coefficients(
            cutoff_frequency,
            sampling_rate=self.sampling_rate, order=order)
        Y = lfilter(b, a, self.ix[:, self._eeg_columns], axis=0)

        if inplace:
            self.ix[:, self._eeg_columns] = Y
            return self
        else:
            new = self._constructor(self, copy=True)
            new.ix[:, self._eeg_columns] = Y
            return new

    def csp(self, group_by, n_components=None):
        """Common spatial patterns.

        Parameters
        ----------
        group_by : str or it
            Column to group samples

        Returns
        -------
        Z : brede.eeg.core.EEGAuxRun
            Projected data
        W : brede.core.matrix.Matrix
            Weights for projection

        """
        states = self[group_by].unique()
        state_to_dummy = {state: n for n, state in enumerate(states)}
        y = self[group_by].apply(lambda state: state_to_dummy[state]).values
        X = self.ix[:, self._eeg_columns].values

        csp = CSP(n_components=n_components)

        Z = csp.fit_transform(X, y)
        csp_names = ['CSP {}'.format(n + 1) for n in range(Z.shape[1])]

        W = Matrix(csp.weights_, columns=csp_names, index=self._eeg_columns)
        return Z, W

    def fft(self):
        """Fourier transform of EEG data.

        Returns
        -------
        spectrum : Spectra
            Dataframe-like with frequencies in rows and electrodes/vertices
            in columns.

        Examples
        --------
        >>> eeg_run = EEGRun({'Cz': [1, -1, 1, -1]})
        >>> fourier = eeg_run.fft()
        >>> fourier.Cz.real
        array([ 0.,  0.,  4.,  0.])

        """
        fourier = np.fft.fft(self.ix[:, self._eeg_columns], axis=0)
        frequencies = np.fft.fftfreq(self.shape[0], 1 / self.sampling_rate)
        return Spectra(fourier, index=frequencies, columns=self._eeg_columns)

    def welch(self, window='hanning', nperseg=256):
        """Return Welch periodogram of electrode data.

        Parameters
        ----------
        window : str or tuple or array_like, optional
            Desired window
        nperseg : int, optional
            Length of each segment.  Defaults to 256.

        Returns
        -------
        periodogram : Spectra

        """
        frequencies, Pxx = welch(
            self.ix[:, self._eeg_columns].T,
            fs=self.sampling_rate, window=window, nperseg=nperseg)
        periodogram = Spectra(Pxx.T, index=frequencies,
                              columns=self._eeg_columns)
        return periodogram

    def peak_frequency(self, min_frequency=0.0, max_frequency=None,
                       column=None):
        """Return frequency with peak magnitude.

        Parameters
        ----------
        min_frequency : float
            Minimum frequency to search for peak from
        max_frequency : float
            Maximum frequency to search for peak from
        column: str, optional
            Electrode or other column to use for finding peak. If None then the
            mean of the magnitude across all electrodes is used.

        Returns
        -------
        freq : float
            Peak frequency

        """
        frequency = self.fft().peak_frequency(
            min_frequency=min_frequency, max_frequency=max_frequency,
            column=column)
        return frequency

    def plot_column_spectrum(self, column):
        """Plot the spectrum of an electrode.

        Parameter
        ---------
        column : str
            Column in the data frame containing the spectrum

        """
        self.fft().plot_column_spectrum(column)

    def plot_mean_spectrum(self):
        """Plot mean spectrum across electrodes."""
        self.fft().plot_mean_spectrum()


class Spectra(DataFrame):

    """Represent spectra for an EEG signal as a dataframe-like object.

    Each frequency is in each row, electrodes in columns

    """

    def plot_column_spectrum(self, column):
        """Plot the spectrum of a columns (e.g., electrode).

        Parameters
        ----------
        column : str
            Electrode name associated with column

        """
        positive_frequencies = self.index >= 0
        plt.plot(self.index[positive_frequencies],
                 np.abs(self.ix[positive_frequencies, column]))
        plt.xlabel('Frequency')
        plt.ylabel('Magnitude')
        plt.title('Spectrum of {}'.format(column))

    def plot_mean_spectrum(self):
        """Plot the spectrum of the mean across all electrodes.

        Only the positive part of the spectrum is shown.

        """
        positive_frequencies = self.index >= 0
        plt.plot(self.index[positive_frequencies],
                 np.mean(np.abs(self.ix[positive_frequencies, :]), axis=1))
        plt.xlabel('Frequency')
        plt.ylabel('Magnitude')
        plt.title('Mean spectrum across {} electrodes'.format(self.shape[1]))

    def plot_spectra(self, title=None, xlim=None, ylim=None):
        """Plot multiple spectra."""
        positive_frequencies = self.index >= 0
        multi_plot = MultiPlot(self.ix[positive_frequencies, :].abs())
        multi_plot.draw(title=title, xlim=xlim, ylim=ylim)

    def peak_frequency(self, min_frequency=0.0, max_frequency=None,
                       column=None):
        """Return frequency with peak magnitude.

        Parameters
        ----------
        min_frequency : float
            Minimum frequency to search for peak from
        max_frequency : float
            Maximum frequency to search for peak from
        column : str, optional
            Electrode to use for finding peak. If None then the mean of the
            magnitude across all electrodes is used.

        Returns
        -------
        freq : float
            Peak frequency

        Examples
        --------
        >>> spectrum = Spectra({'Cz': [1, 2, 3, 1, 2, 10, 3, 4, 1]})
        >>> spectrum.peak_frequency()
        5

        """
        min_frequency = max(0.0, min_frequency)
        if max_frequency is None:
            max_frequency = max(self.index)

        indices = (self.index >= min_frequency) & (self.index <= max_frequency)
        if column is None:
            magnitudes = np.mean(np.abs(self.ix[indices, :]), axis=1)
        else:
            magnitudes = np.abs(self.ix[indices, column])

        peak_frequency = magnitudes.argmax()
        return peak_frequency

    def show(self):
        """Show Matplotlib plot."""
        plt.show()


class Spectra3D(Tensor):

    """Represent spectra for an EEG signal as a Panel-like object.

    Each run is in items, each frequency is in each major_axis, electrodes in
    minor_axis.

    """

    def plot_column_run_spectrum(self, column, run):
        """Plot the spectrum of an electrode.

        Parameters
        ----------
        electrode : str
            Electrode name associated with column
        run : str or int
            Index for run.

        """
        positive_frequencies = self.major_axis >= 0
        plt.plot(self.major_axis[positive_frequencies],
                 np.abs(self[run, positive_frequencies, column]))
        plt.xlabel('Frequency')
        plt.ylabel('Magnitude')
        plt.title('Spectrum of {} for run {}'.format(column, run))

    def plot_mean_spectrum(self):
        """Plot the spectrum of the mean across electrodes and runs.

        Only the positive part of the spectrum is shown.

        """
        positive_frequencies = self.major_axis >= 0
        magnitudes = np.abs(self[:, positive_frequencies, :])
        plt.plot(self.major_axis[positive_frequencies],
                 magnitudes.values.mean(axis=(0, 2)))
        plt.xlabel('Frequency')
        plt.ylabel('Magnitude')
        plt.title('Mean spectrum across {} runs and {} electrodes'.format(
            self.shape[0], self.shape[2]))

    def peak_frequency(self, min_frequency=0.0, max_frequency=None,
                       column=None, run=None):
        """Return frequency with peak magnitude.

        Parameters
        ----------
        min_frequency : float
            Minimum frequency to search for peak from
        max_frequency : float
            Maximum frequency to search for peak from
        column : str, optional
            Electrode to use for finding peak. If None then the mean of the
            magnitude across all electrodes is used.

        Returns
        -------
        freq : float
            Peak frequency

        Examples
        --------
        >>> spectrum = Spectra3D({'Trial 1':
        ...                      {'Cz': [1, 2, 3, 1, 2, 10, 3, 4, 1]}})
        >>> spectrum.peak_frequency()
        5

        """
        min_frequency = max(0.0, min_frequency)
        if max_frequency is None:
            max_frequency = max(self.major_axis)

        indices = ((self.major_axis >= min_frequency) &
                   (self.major_axis <= max_frequency))
        if column is None and run is None:
            magnitudes = np.abs(self[:, indices, :]).mean(axis=2).mean(axis=1)
        elif column is None:
            magnitudes = np.abs(self[run, indices, :]).mean(axis=2)
        elif run is None:
            magnitudes = np.abs(self[:, indices, column]).mean(axis=0)
        else:
            magnitudes = np.abs(self[run, indices, column])

        peak_frequency = magnitudes.argmax()
        return peak_frequency

    def show(self):
        """Show Matplotlib plot."""
        plt.show()


class Spectra4D(Tensor4D):

    """Represent spectra for an EEG signal as a Panel4-like object.

    Each run is in items, each frequency is in each major_axis, electrodes in
    minor_axis.

    """

    def plot_column_label_run_spectrum(self, column, run, label):
        """Plot the spectrum of a column (e.g., an electrode).

        Parameters
        ----------
        column : str
            Electrode name associated with column
        run : str or int
            Index for run.

        """
        positive_frequencies = self.major_axis >= 0
        plt.plot(self.major_axis[positive_frequencies],
                 np.abs(self[label, run, positive_frequencies, column]))
        plt.xlabel('Frequency')
        plt.ylabel('Magnitude')
        plt.title('Spectrum of {} for run {}'.format(column, run))

    def plot_mean_spectrum(self):
        """Plot the spectrum of the mean across electrodes and runs.

        Only the positive part of the spectrum is shown.

        """
        positive_frequencies = self.major_axis >= 0
        magnitudes = np.abs(self[:, :, positive_frequencies, :])
        plt.plot(self.major_axis[positive_frequencies],
                 magnitudes.values.mean(axis=(0, 1, 3)))
        plt.xlabel('Frequency')
        plt.ylabel('Magnitude')
        plt.title(('Mean spectrum across {} labels, {} runs '
                   'and {} electrodes').format(
            self.shape[0], self.shape[1], self.shape[3]))

    def peak_frequency(self, min_frequency=0.0, max_frequency=None,
                       column=None, run=None, label=None):
        """Return frequency with peak magnitude.

        Parameters
        ----------
        min_frequency : float
            Minimum frequency to search for peak from
        max_frequency : float
            Maximum frequency to search for peak from
        column : str, optional
            Electrode to use for finding peak. If None then the mean of the
            magnitude across all electrodes is used.
        run : Index, optional
            Index for the run
        label : Index, optional
            Index for the label

        Returns
        -------
        freq : float
            Peak frequency

        Examples
        --------
        >>> spectrum = Spectra4D({'baseline': {'Trial 1':
        ...                      {'Cz': [1, 2, 3, 1, 2, 10, 3, 4, 1]}}})
        >>> spectrum.peak_frequency()
        5

        """
        min_frequency = max(0.0, min_frequency)
        if max_frequency is None:
            max_frequency = max(self.major_axis)

        indices = ((self.major_axis >= min_frequency) &
                   (self.major_axis <= max_frequency))
        if column is None and run is None and label is None:
            magnitudes = np.abs(self[:, :, indices, :]).values.mean(
                axis=(0, 1, 3))
            magnitudes = Series(magnitudes, index=self.major_axis[indices])
        else:
            magnitudes = np.abs(self[label, run, indices, column])
        # TODO: Combination when some of the indices are not parameters.

        peak_frequency = magnitudes.argmax()
        return peak_frequency

    def show(self):
        """Show Matplotlib plot."""
        plt.show()


read_csv = EEGAuxRun.read_csv
