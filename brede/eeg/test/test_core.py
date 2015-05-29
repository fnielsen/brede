"""Tests for brede.eeg.core submodule."""


from __future__ import absolute_import, division, print_function

import numpy as np

import pytest

from .. import core


@pytest.fixture
def eeg_run_1d():
    """Return an instance of EEGRun with data."""
    eeg_run = core.EEGRun(
        [[1, 2, 6]], columns=['C3', 'Cz', 'C4'],
        sampling_rate=2.0)
    return eeg_run


@pytest.fixture
def eeg_aux_run():
    """Return an instance of EEGAuxRun with data."""
    eeg_aux_run = core.EEGAuxRun(
        [[1, 2, 'yes'], [3, 4, 'no']],
        columns=['C3', 'C4', 'label'],
        sampling_rate=2.0)
    return eeg_aux_run


def test_electrodes():
    """Test a constant."""
    assert 'AF3' in core.ELECTRODES


def test_eegrun():
    """Test EEGRun."""
    eeg_run = core.EEGRun([[1, 2], [3, 4]], columns=['C3', 'C4'],
                          sampling_rate=2.0)
    assert eeg_run.index[0] == 0.0
    assert eeg_run.sampling_rate == 2.0

    assert eeg_run.ix[0.0, 'C3'] == 1
    assert eeg_run.ix[0.5, 'C4'] == 4


def test_eegrun_constructor():
    """Test result constructur."""
    eeg_run = core.EEGRun([[1, 2], [3, 4]], columns=['C3', 'C4'],
                          sampling_rate=2.0)
    assert isinstance(np.isnan(eeg_run), core.EEGRun)


def test_eegrun_rereference(eeg_run_1d):
    """Test rereference method."""
    assert eeg_run_1d.rereference().ix[0, 'C3'] == -2.0
    assert eeg_run_1d.rereference(mode='median').ix[0, 'C3'] == -1.0
    elem = eeg_run_1d.rereference(mode='electrode', electrode='C4').ix[0, 'C3']
    assert elem == -5.0


def test_eegrun_bandpass_filter():
    """Test bandpass filtering."""
    # Set up date
    sampling_rate = 100
    N = 60 * sampling_rate
    sinus15 = np.sin(2 * np.pi * 15 / sampling_rate * np.arange(N))
    sinus2 = np.sin(2 * np.pi * 2 / sampling_rate * np.arange(N))
    eeg_run = core.EEGRun({'C3': sinus15, 'C4': sinus2},
                          sampling_rate=sampling_rate)

    # Filtering
    filtered = eeg_run.bandpass_filter(10, 20)
    assert isinstance(filtered, core.EEGRun)
    assert filtered.shape == (N, 2)
    assert filtered['C3'].max() > 0.9
    assert filtered['C4'].max() < 0.1


def test_eegruns():
    """Test EEGRuns."""
    eeg_runs = core.EEGRuns([[[1, 2], [3, 4]]],
                            items=['Trial 1'],
                            minor_axis=['C3', 'C4'],
                            sampling_rate=2.0)
    assert eeg_runs.major_axis[0] == 0.0
    assert eeg_runs.major_axis[1] == 0.5
    assert eeg_runs.sampling_rate == 2.0

    assert eeg_runs['Trial 1', 0.0, 'C3'] == 1
    assert eeg_runs['Trial 1', 0.5, 'C4'] == 4


def test_eegruns_constructor():
    """Test result constructur."""
    eeg_runs = core.EEGRuns([[[1, 2], [3, 4]]],
                            items=['Trial 1'],
                            minor_axis=['C3', 'C4'],
                            sampling_rate=2.0)
    assert isinstance(np.isnan(eeg_runs), core.EEGRuns)


def test_eegruns4d():
    """Test EEGRuns4D."""
    eeg_runs = core.EEGRuns4D([[[[1, 2], [3, 4]]]],
                              labels=['baseline'],
                              items=['Trial 1'],
                              minor_axis=['C3', 'C4'],
                              sampling_rate=2.0)
    assert eeg_runs.major_axis[0] == 0.0
    assert eeg_runs.major_axis[1] == 0.5
    assert eeg_runs.sampling_rate == 2.0

    assert eeg_runs['baseline', 'Trial 1', 0.0, 'C3'] == 1
    assert eeg_runs['baseline', 'Trial 1', 0.5, 'C4'] == 4


def test_eegruns4d_constructor():
    """Test result constructur."""
    eeg_runs4d = core.EEGRuns4D(
        [[[[1, 2], [3, 4]]]],
        labels=['baseline'], items=['Trial 1'],
        minor_axis=['C3', 'C4'], sampling_rate=2.0)
    assert isinstance(np.isnan(eeg_runs4d), core.EEGRuns4D)


def test_eeg_aux_run(eeg_aux_run):
    """Test setup of EEGAuxRun."""
    assert len(eeg_aux_run.shape) == 2
    assert eeg_aux_run.shape[1] == 3
    assert eeg_aux_run.shape[0] == 2
    assert eeg_aux_run.electrodes == ['C3', 'C4']


def test_eeg_aux_run_getitem(eeg_aux_run):
    """Test indexing in EEGAuxRun."""
    new_eeg_aux_run = eeg_aux_run[['C3']]
    assert new_eeg_aux_run.electrodes == ['C3']
