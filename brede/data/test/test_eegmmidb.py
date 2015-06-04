"""Test of eegmmidb."""


from __future__ import absolute_import, division, print_function

from brede.eeg.core import EEGAuxRun

import pytest

from ..eegmmidb import EEGMMIDB


@pytest.fixture
def eegmmidb():
    """Return instance of primary class."""
    eegmmidb = EEGMMIDB()
    return eegmmidb


def test_run(eegmmidb):
    """Test run method."""
    run = eegmmidb.run()
    assert isinstance(run, EEGAuxRun)

    run11 = eegmmidb.run(1, 1)
    assert (run11 == run).all().all()

    run11 = eegmmidb.run(run=1, subject=1)
    assert (run11 == run).all().all()


def test_runs_for_subject(eegmmidb):
    """Test of run_for_subject method."""
    runs = eegmmidb.runs_for_subject()
    assert isinstance(runs[1], EEGAuxRun)
    assert len(runs) == 14
