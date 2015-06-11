"""Handle interface to SPM EEG's data."""


from brede.core.matrix import Matrix

from scipy.io import loadmat


class SPMException(Exception):

    """General exception for this module."""

    pass


def read_electrode_positions(filename):
    """Read eletrode positions from SPM EEG file.

    Parameters
    ----------
    filename : str
        Filename for SPM with spm_eeg_* data.

    Returns
    -------
    matrix : brede.core.matrix.Matrix
        Structure with electrode names and positions.

    References
    ----------
    SPM8's spm_eeg_loaddata

    """
    struct = loadmat(filename, struct_as_record=False, squeeze_me=True)
    eeg = struct['D'].sensors.eeg
    if eeg.unit != 'mm':
        raise SPMException('Unhandled unit')
    matrix = Matrix(eeg.pnt, index=eeg.label, columns=['x', 'y', 'z'])
    return matrix
