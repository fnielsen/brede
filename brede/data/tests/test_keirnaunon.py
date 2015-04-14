"""Tests for keirnaunon submodule."""


from __future__ import absolute_import

from .. import keirnaunon


def test_name():
    """Test of name property."""
    ka = keirnaunon.KeirnAunon()
    assert ka.name == "1989 Keirn and Aunon"


def test_trial():
    """Test of trial method."""
    ka = keirnaunon.KeirnAunon()

    # First element
    eeg_run = ka.trial()
    assert eeg_run.ix[0, 0] == 1.345

    eeg_run = ka.trial(trial=2)
    assert eeg_run.ix[0, 0] == 1.378

    eeg_run = ka.trial(subject=3, state='multiplication', trial=2)
    assert eeg_run.ix[0, 0] == 11.169
