"""Tests for brede.eeg.core submodule."""


from __future__ import absolute_import

import numpy as np

from .. import core


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
