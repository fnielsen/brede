"""Test of emocap."""


from __future__ import absolute_import, division, print_function

import numpy.random as npr

import pytest

from .. import emocap


@pytest.fixture
def emocap_electrode_run_aux():
    """Generate a small data set with auxialiary data."""
    eeg_run = emocap.EmocapElectrodeRun(
        [[1, 2, 3], [4, 5, 6]],
        columns=['C3', 'C4', 'Extra'])
    return eeg_run


@pytest.fixture
def emocap_electrode_run_100():
    """Generate a Emocap data set."""
    npr.seed(1729)
    run = emocap.EmocapElectrodeRun(
        npr.randn(100, 14), columns=emocap.ELECTRODES)
    return run


def test_emocap_electrode_run(emocap_electrode_run_aux):
    """Test EmocapElectrodeRun."""
    eeg_run = emocap.EmocapElectrodeRun([[1, 2], [3, 4]])
    assert isinstance(eeg_run.ix[:, :], emocap.EmocapElectrodeRun)
    assert eeg_run.sampling_rate == 128.0
    assert eeg_run.iloc[1, 1] == 4

    eeg_run = emocap.EmocapElectrodeRun([[1, 2], [3, 4]], columns=['C3', 'C4'])
    assert eeg_run.sampling_rate == 128.0
    assert eeg_run.iloc[1, 1] == 4
    assert eeg_run.ix[0.0, 'C4'] == 2
    assert eeg_run.ix[1 / 128, 'C4'] == 4

    assert emocap_electrode_run_aux.sampling_rate == 128.0
    assert emocap_electrode_run_aux.iloc[1, 1] == 5
    assert emocap_electrode_run_aux.ix[0.0, 'C4'] == 2
    assert emocap_electrode_run_aux.ix[1 / 128, 'C4'] == 5


def test_emocap_electrode_run_iloc(emocap_electrode_run_aux):
    indexed = emocap_electrode_run_aux.iloc[1:, :]
    assert isinstance(indexed, emocap.EmocapElectrodeRun)
    assert indexed.index[0] == 1 / 128


def test_emocap_electrode_run_ix(emocap_electrode_run_aux):
    indexed = emocap_electrode_run_aux.ix[(1 / 128):, :]
    assert isinstance(indexed, emocap.EmocapElectrodeRun)
    assert indexed.index[0] == 1 / 128


def test_emocap_electrode_run_rereference(emocap_electrode_run_aux):
    """Test rereference method in parent classes."""
    rereferenced = emocap_electrode_run_aux.rereference()
    assert isinstance(rereferenced, emocap.EmocapElectrodeRun)
    assert rereferenced.iloc[0, 0] == -0.5


def test_emocap_electrode_run_invert(emocap_electrode_run_100):
    """Test invert method of EmocapElectrodeRun."""
    sources = emocap_electrode_run_100.invert()
    assert isinstance(sources, emocap.EmocapVertexRun)
    assert sources.shape == (100, 1028)
