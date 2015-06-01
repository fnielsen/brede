"""Test of sbs2 module."""


import pytest

from .. import sbs2


@pytest.fixture
def sbs2_data():
    """Instance object."""
    sd = sbs2.SBS2Data()
    return sd


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


def test_forward_model(sbs2_data):
    """Test forward model."""
    forward = sbs2_data.forward_model()
    assert forward.shape == (1028, 14)


def test_inverse_model(sbs2_data):
    """Test inverse model."""
    inverse = sbs2_data.inverse_model()
    assert inverse.shape == (14, 1028)

    inverse = sbs2_data.inverse_model(method='minimumnorm')
    assert inverse.shape == (14, 1028)

    inverse = sbs2_data.inverse_model(method='LORETA')
    assert inverse.shape == (14, 1028)


def test_spatial_coherence(sbs2_data):
    """Test spatial coherence."""
    coherence = sbs2_data.spatial_coherence()

    # Symmetry
    assert (coherence.T == coherence).all().all()
