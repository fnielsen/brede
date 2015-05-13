"""Test of sbs2 module."""


import pytest

from .. import sbs2


@pytest.fixture
def sbs2_data():
    """Instance object."""
    sbs2_data = sbs2.SBS2Data()
    return sbs2_data


def test_electrode_names(sbs2_data):
    """Test electrode_name property."""
    names = sbs2_data.electrode_names()
    assert 'AF3' in names
    assert 'Fpz' not in names

    names = sbs2_data.electrode_names('emotiv')
    assert 'AF3' in names
    assert 'Fpz' not in names

    names = sbs2_data.electrode_names('emocap')
    assert 'AF3' not in names
    assert 'Fpz' in names


def test_name(sbs2_data):
    """Test name."""
    assert sbs2_data.name == "SBS2 data files"
