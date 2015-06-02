"""Test of vertex."""


import numpy as np

from .. import vertex


def test_eeg_aux_vertex_run():
    """Test EEGAuxVertexRun."""
    run = vertex.EEGAuxVertexRun([[1, 2], [3, 4]])
    assert run.iloc[0, 0] == 1

    run = vertex.EEGAuxVertexRun(np.array([[1, 2], [3, 4]]))
    assert run.iloc[0, 0] == 1
