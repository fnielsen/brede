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
def eeg_run_emotiv():
    """Return EEG data set with emotiv electrodes."""
    emotiv_electrodes = [
        'F3', 'FC6', 'P7', 'T8', 'F7', 'F8', 'T7',
        'P8', 'AF4', 'F4', 'AF3', 'O2', 'O1', 'FC5']

    eeg_run = core.EEGRun(
        [range(14)], columns=emotiv_electrodes,
        sampling_rate=2.0)

    return eeg_run


@pytest.fixture
def eeg_aux_run():
    """Return an instance of EEGAuxRun with data."""
    eeg_aux_run = core.EEGAuxRun(
        [[1, 2, 'yes'], [3, 4, 'no']],
        columns=['C3', 'C4', 'label'],
        eeg_columns=['C3', 'C4'],
        sampling_rate=2.0)
    return eeg_aux_run


@pytest.fixture
def eeg_aux_run_emotiv():
    """Return EEG data set with emotiv electrodes and auxillary."""
    columns = [
        'F3', 'FC6', 'P7', 'T8', 'F7', 'F8', 'T7',
        'P8', 'AF4', 'F4', 'AF3', 'O2', 'O1', 'FC5',
        'extra1', 'extra2']

    eeg_aux_run = core.EEGAuxRun(
        [range(16)], columns=columns,
        eeg_columns=['F3', 'FC6', 'P7', 'T8', 'F7', 'F8', 'T7',
                     'P8', 'AF4', 'F4', 'AF3', 'O2', 'O1', 'FC5'],
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


def test_eegrun_emotiv_to_emocap(eeg_run_emotiv):
    """Test emotiv_to_emocap method."""
    translated = eeg_run_emotiv.emotiv_to_emocap()
    assert 'C3' in translated.columns
    assert 'Cz' in translated.columns
    assert 'C4' in translated.columns
    assert translated.columns[eeg_run_emotiv.columns == 'T8'] == 'C3'
    assert translated.ix[0, 'P4'] == 0


def test_eegrun_find_transitions():
    """Test of EEGRun find_transitions method."""
    run = core.EEGRun([[1, 2], [1, 3], [2, 3]], columns=['C3', 'C4'])
    assert run.find_transitions('C3') == [2]
    assert run.find_transitions('C4') == [1]
    assert run.find_transitions(['C3', 'C4']) == [1, 2]

    run = core.EEGRun([[np.nan, 2], [np.nan, 3], [1, 3], [1, 3]],
                      columns=['C3', 'C4'])
    # TODO
    # assert run.find_transitions('C3') == [2]


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
    assert eeg_aux_run.eeg_columns == ['C3', 'C4']


def test_eeg_aux_run_getitem(eeg_aux_run):
    """Test indexing in EEGAuxRun."""
    new = eeg_aux_run[['C3']]
    assert new.eeg_columns == ['C3']

    new = eeg_aux_run.ix[:1, :]
    # TODO
    # assert new.eeg_columns == ['C3', 'C4']


def test_eegauxrun_emotiv_to_emocap(eeg_aux_run_emotiv):
    """Test emotiv_to_emocap method."""
    translated = eeg_aux_run_emotiv.emotiv_to_emocap()
    assert 'C3' in translated.columns
    assert 'Cz' in translated.columns
    assert 'C4' in translated.columns
    assert translated.columns[eeg_aux_run_emotiv.columns == 'T8'] == 'C3'
    assert 'extra1' in translated.columns
    assert 'extra2' in translated.columns
    assert translated.ix[0, 'P4'] == 0
