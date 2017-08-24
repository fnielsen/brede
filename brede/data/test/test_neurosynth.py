"""Test of neurosynth module."""


import pytest

from .. import neurosynth


@pytest.fixture
def neurosynth_database():
    """Return fixture for handle to neurosynth database."""
    nd = neurosynth.NeurosynthDatabase()
    return nd


@pytest.fixture
def neurosynth_database_frame(neurosynth_database):
    """Return fixture for neurosynth dataframe."""
    df = neurosynth_database.database()
    return df


def test_neurosynth_database_name(neurosynth_database):
    """Test name of neurosynth database object."""
    assert neurosynth_database.name == 'Neurosynth'


def test_neurosynth_database_description(neurosynth_database):
    """Test description of neurosynth database object."""
    assert len(neurosynth_database.description) > 20


def test_pmid(neurosynth_database_frame):
    """Test whether a specific PMID is in the data."""
    assert 23400116 in set(neurosynth_database_frame.id)
