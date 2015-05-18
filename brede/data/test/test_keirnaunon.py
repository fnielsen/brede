"""Tests for keirnaunon submodule."""


from __future__ import absolute_import

import numpy as np

from .. import keirnaunon


def test_name():
    """Test of name property."""
    ka = keirnaunon.KeirnAunon()
    assert ka.name == "1989 Keirn and Aunon"


def test_sampling_rate():
    """Test constant for sampling rate."""
    assert keirnaunon.SAMPLING_RATE == 250


def test_trial():
    """Test of trial method."""
    ka = keirnaunon.KeirnAunon()

    eeg_run = ka.trial()

    assert eeg_run.sampling_rate == 250

    # First element
    assert eeg_run.ix[0, 0] == 1.345

    eeg_run = ka.trial(trial=2)
    assert eeg_run.ix[0, 0] == 1.378

    eeg_run = ka.trial(subject=3, state='multiplication', trial=2)
    assert eeg_run.ix[0, 0] == 11.169

    eeg_run = ka.trial(2, 'counting', 3)
    assert abs(-20.507 - eeg_run.ix[0, 0]) < 0.0000001


def test_trials_for_subject_state():
    """Test of trials_for_subject_state method."""
    ka = keirnaunon.KeirnAunon()

    eeg_runs = ka.trials_for_subject_state()

    assert eeg_runs.sampling_rate == 250

    assert eeg_runs.shape[0] == 10
    assert eeg_runs[1, 0.0, 'C3'] == 1.345
    assert eeg_runs[1, 0.0, 'O2'] == 3.006
    assert eeg_runs[5, 0.0, 'O2'] == 0.301

    eeg_runs = ka.trials_for_subject_state(subject=1, state='baseline')
    assert eeg_runs.shape[0] == 10
    assert eeg_runs[1, 0.0, 'C3'] == 1.345
    assert eeg_runs[1, 0.0, 'O2'] == 3.006
    assert eeg_runs[5, 0.0, 'O2'] == 0.301

    eeg_runs = ka.trials_for_subject_state(subject=7, state='counting')
    assert eeg_runs.shape[0] == 5
    assert eeg_runs[5, 0.0, 'C3'] == -6.204
    assert eeg_runs[5, 0.0, 'O2'] == -1.482
    assert eeg_runs[5, :, 'O2'].iloc[-1] == 13.341


def test_trials_for_subject():
    """Test of trials_for_subject method."""
    ka = keirnaunon.KeirnAunon()

    eeg_runs = ka.trials_for_subject()

    assert eeg_runs.sampling_rate == 250

    assert eeg_runs.shape[0] == 5
    assert eeg_runs['baseline', 1, 0.0, 'C3'] == 1.345


def test_peak_frequency():
    """Test peak frequency finding for dataset."""
    ka = keirnaunon.KeirnAunon()

    eeg_runs = ka.trials_for_subject()
    eeg_runs.fft().peak_frequency(min_frequency=5.0) == 60.0


def test_subject_4_letter_composing_trial_10():
    """Test the NaN in a specific trial."""
    ka = keirnaunon.KeirnAunon()
    trial = ka.trial(4, 'letter-composing', 10)
    assert np.isnan(trial).all().all()
