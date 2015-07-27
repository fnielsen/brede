"""Core data structures for EEG data.

Representing EEG data on the electrode level.

"""


from __future__ import absolute_import, division, print_function


import matplotlib.pyplot as plt

import numpy as np

from pandas import read_csv as pandas_read_csv

from .core import EEGAuxRun, EEGRun, EEGRuns, EEGRuns4D, Spectra3D, Spectra4D
from .csp import CSP
from .vertex import EEGAuxVertexRun
from ..core import Matrix
from ..io import read_edf


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


class EEGElectrodeRun(EEGRun):

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

    @classmethod
    def read_csv(cls, filename, sampling_rate=None, *args, **kwargs):
        """Read comma-separated file.

        Parameters
        ----------
        filename : str
            Filename for the csv file

        Returns
        -------
        eeg_run : EEGElectrodeRun
            EEGElectrodeRun dataframe with read data.

        """
        # TODO: This will not instance in derived class.
        return EEGElectrodeRun(pandas_read_csv(filename, *args, **kwargs),
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
        eeg_run : EEGElectrodeRun
            EEGElectrodeRun dataframe with read data.

        """
        return cls(read_edf(filename))

    def emotiv_to_emocap(self, check_all=True, change_qualities=True,
                         inplace=False):
        """Change column names for Emotiv electrodes to Emocap.

        Specialized method to change the electrode names in the dataframe
        column from Emotiv electrode names to Emocap electrode names.

        Returns
        -------
        eeg_run : EEGElectrodeRun
            EEGElectrodeRun data

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

    def inverse_modeling(self, inverse_model):
        """Inverse modeling.

        Parameters
        ----------
        inverse_model : pandas.DataFrame
            Data fra with inverse model

        Returns
        -------
        vertex_run : EEGAuxVertexRun
            DataFrame-like object with eeg data.

        """
        assert set(inverse_model.index) == set(self.columns)
        return EEGAuxVertexRun(
            self.dot(inverse_model.ix[self.columns, :]),
            index=self.index,
            columns=inverse_model.columns,
            eeg_columns=inverse_model.columns,
            sampling_rate=self.sampling_rate)

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


class EEGElectrodeRuns(EEGRuns):

    """Multiple EEGRuns of the same length."""

    _metadata = ['_sampling_rate']

    @property
    def _constructor(self):
        return type(self)

    def __init__(self, data=None, items=None, major_axis=None, minor_axis=None,
                 dtype=None, copy=False, sampling_rate=1.0):
        """Construct Panel-like object."""
        super(EEGElectrodeRuns, self).__init__(
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
        >>> eeg_runs = EEGElectrodeRuns({'Trial 1': {'Cz': [1, -1, 1, -1]}})
        >>> fourier = eeg_runs.fft()
        >>> fourier['Trial 1', :, 'Cz'].real
        array([ 0.,  0.,  4.,  0.])

        """
        fourier = np.fft.fft(self, axis=1)
        frequencies = np.fft.fftfreq(self.shape[1], 1 / self.sampling_rate)
        return Spectra3D(fourier, items=self.items,
                         major_axis=frequencies, minor_axis=self.minor_axis)


class EEGElectrodeRuns4D(EEGRuns4D):

    """Multiple EEGRuns of the same length."""

    _metadata = ['_sampling_rate']

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


class EEGAuxElectrodeRun(EEGAuxRun):

    """Represent a EEG data set with auxilliary data.

    The Pandas DataFrame class is reused and extended with, e.g., Fourier
    transformation.

    Attributes
    ----------
    sampling_rate : float
        Sampling rate in Hertz
    eeg_columns : list of str
        Names for columns that are electrodes

    """

    _metadata = ['_eeg_columns', '_sampling_rate']

    @property
    def _constructor(self):
        return type(self)

    def __init__(self, data=None, index=None, columns=None, dtype=None,
                 copy=False, sampling_rate=None, eeg_columns=None):
        """Construct dataframe-like object."""
        super(EEGAuxElectrodeRun, self).__init__(
            data=data, index=index, columns=columns,
            dtype=dtype, copy=copy,
            sampling_rate=sampling_rate, eeg_columns=eeg_columns)

        if eeg_columns is None:
            self._eeg_columns = [column for column in self.columns
                                 if column in ELECTRODES]
        else:
            self._eeg_columns = eeg_columns

    def emotiv_to_emocap(self, check_all=True, change_qualities=True,
                         inplace=False):
        """Change column names for Emotiv electrodes to Emocap.

        Specialized method to change the electrode names in the dataframe
        column from Emotiv electrode names to Emocap electrode names.

        """
        old_electrodes = self._eeg_columns
        new = super(EEGAuxRun, self).emotiv_to_emocap(
            check_all=check_all,
            change_qualities=change_qualities,
            inplace=inplace)
        new_electrodes = [EMOTIV_TO_EMOCAP_MAP[electrode]
                          for electrode in old_electrodes]

        if inplace:
            self._eeg_columns = new_electrodes
            return self
        else:
            new._eeg_columns = new_electrodes
            return new

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
        return EEGAuxElectrodeRun(
            pandas_read_csv(filename, *args, **kwargs),
            sampling_rate=sampling_rate)

    def only_eeg_columns(self):
        """Extract data for only EEG electrode columns.

        Discard columns that does not contain EEG electrode data.

        Returns
        -------
        electrode_run : EEGElectrodeRun
            DataFrame-like object with columns extracted.

        """
        return EEGElectrodeRun(
            self.ix[:, self.eeg_columns])

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
        X = self.ix[:, self.electrodes].values

        csp = CSP(n_components=n_components)

        Z = csp.fit_transform(X, y)
        csp_names = ['CSP {}'.format(n + 1) for n in range(Z.shape[1])]

        W = Matrix(csp.weights_, columns=csp_names, index=self.electrodes)
        return Z, W

    def plot_electrode_spectrum(self, electrode):
        """Plot the spectrum of an electrode."""
        self.fft().plot_electrode_spectrum(electrode)

    def plot_mean_spectrum(self):
        """Plot mean spectrum across electrodes."""
        self.fft().plot_mean_spectrum()


read_csv = EEGAuxElectrodeRun.read_csv
