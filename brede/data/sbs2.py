"""brede.data.sbs2 - Interface to SBS2 data.

Usage:
  brede.data.sbs2 [options]

Options:
  -h --help     Help

Smartphone brain scanner data. Presently a surface is plotted.

"""


from __future__ import absolute_import, print_function

from os import chdir, getcwd, makedirs
from os.path import exists, expanduser, join

from shutil import move, rmtree
from subprocess import call
from tempfile import mkdtemp

from numpy import eye

from pandas import read_csv

from scipy.linalg import pinv

from ..config import config
from ..core.matrix import Matrix
from ..surface import read_obj


SBS2_SVN_DIR = ('https://github.com/SmartphoneBrainScanner/'
                'smartphonebrainscanner2-core/trunk/sbs2_data')

SBS2_ELECTRODES_EMOTIV = ['P7', 'FC6', 'T7', 'P8', 'O2', 'O1', 'FC5',
                          'F3', 'F4', 'T8', 'F7', 'F8', 'AF4', 'AF3']

SBS2_ELECTRODES_EMOCAP = ['TP10', 'Fz', 'P3', 'Cz', 'C4', 'TP9',  'Pz', 'P4',
                          'F3', 'C3', 'O1', 'F4', 'Fpz', 'O2']


class SBS2Data(object):

    """Interface to SBS2 (Smartphone Brain Scanner) data.

    Examples
    --------
    >>> sbs2_data = SBS2Data()
    >>> K = sbs2_data.spatial_coherence()
    >>> Kinv = sbs2_data.inverse_spatial_coherence()
    >>> KK = K.dot(Kinv)

    >>> import numpy as np
    >>> (np.diag(KK) - 1 < 0.1).all()
    True

    """

    def __init__(self, redownload=False):
        """Setup directories and filenames.

        Parameters
        ----------
        redownload : bool
            Download the files anew.

        """
        self.data_dir = expanduser(config.get('data', 'data_dir'))
        self.sbs2_dir = join(self.data_dir, 'sbs2')
        self.unpack(redownload=redownload)

    @property
    def name(self):
        """Return short name for database."""
        return "SBS2 data files"

    @property
    def description(self):
        """Return a descriptive string about the data."""
        return ("Smartphone brain scanner datafiles.")

    def __str__(self):
        """Return descriptive string."""
        string = "<SBS2Data>"
        return string

    def make_dir(self):
        """Make SBS2 data directory."""
        if not exists(self.sbs2_dir):
            makedirs(self.sbs2_dir)

    def download(self):
        """Download SBS2 files.

        The download is done with 'svn' which should be installed, and the data
        is downloaded from Github.

        """
        self.make_dir()
        saved_dir = getcwd()
        temp_dir = mkdtemp()
        chdir(temp_dir)
        try:
            call(['svn', 'export',  SBS2_SVN_DIR])
        finally:
            chdir(saved_dir)
        move(join(temp_dir, 'sbs2_data'), self.sbs2_dir)
        rmtree(temp_dir)

    def unpack(self, redownload=False):
        """Extract the downloaded compressed Neurosynth dump file.

        It tests if the relevant database file is already downloaded.
        If not call then the download method is called.

        """
        if redownload or not exists(self.sbs2_dir):
            self.download()
            # no need for extraction

    def electrode_names(self, hardware='emotiv'):
        """Return electrode names."""
        if hardware == 'emotiv':
            electrodes = SBS2_ELECTRODES_EMOTIV
        elif hardware == 'emocap':
            electrodes = SBS2_ELECTRODES_EMOCAP
        else:
            raise ValueError('Wrong argument to model')
        return electrodes

    def forward_model(self, hardware='emotiv'):
        """Return forward model.

        Reads forward model from either
        sbs2_data/hardware/emotiv/forwardmodel_spheres_reduced.txt
        or sbs2_data/hardware/emocap/forwardmodel_spheres_reduced.txt

        1000000 is multipled on the values in the file before the values are
        returned.

        Arguments
        ---------
        hardware : 'emotiv' or 'emocap'
            Hardward type for forward model

        Returns
        -------
        matrix : brede.core.matrix.Matrix
            Matrix with forward model returned channels x vertices

        Examples
        --------
        >>> sbs2_data = SBS2Data()
        >>> forward_model = sbs2_data.forward_model()
        >>> forward_model.index[0]
        'P7'

        """
        filename = join(self.sbs2_dir, 'sbs2_data', 'hardware', hardware,
                        'forwardmodel_spheres_reduced.txt')
        matrix = Matrix(read_csv(filename, sep='\t', header=None))
        matrix.index = self.electrode_names(hardware)

        # void Sbs2SourceReconstrucionLoreta::setupModel() line 41
        matrix *= 1000000
        return matrix

    def spatial_coherence(self, hardware='emotiv'):
        """Return spatial coherence matrix.

        Reads the spatial coherence from:

        spatialCoherenceSmooth0-2_reduced.txt

        The matrix seems to be L'*L matrix, when compared with the
        notation in 'Smartphones as pocketable labs: Visions for
        mobile brain imaging and neurofeedback'.

        It returns a matrix sized vertices-by-vertices compatible with the
        small surface model.

        Arguments
        ---------
        hardware : 'emotiv' or 'emocap'
            Hardward type for forward model

        Returns
        -------
        matrix : brede.core.matrix.Matrix
            Matrix with spatial coherence size 1028 x 1028

        Examples
        --------
        >>> sbs2_data = SBS2Data()
        >>> spatial_coherence = sbs2_data.spatial_coherence()
        >>> spatial_coherence.shape
        (1028, 1018)

        """
        filename = join(self.sbs2_dir, 'sbs2_data', 'hardware', hardware,
                        'spatialCoherenceSmooth0-2_reduced.txt')
        matrix = Matrix(read_csv(filename, sep='\t', header=None))
        return matrix

    def inverse_spatial_coherence(self, hardware='emotiv'):
        """Return inverse spatial coherence matrix.

        Reads the inverse spatial coherence from:

        spatialCoherenceSmooth0-2_reduced_inverse.txt

        It returns a matrix sized vertices-by-vertices compatible with the
        small surface model.

        Arguments
        ---------
        hardware : 'emotiv' or 'emocap'
            Hardward type for forward model

        Returns
        -------
        matrix : brede.core.matrix.Matrix
            Matrix with inverse spatial coherence size 1028 x 1028

        Examples
        --------
        >>> sbs2_data = SBS2Data()
        >>> inverse_spatial_coherence = sbs2_data.inverse_spatial_coherence()
        >>> inverse_spatial_coherence.shape
        (1028, 1018)

        """
        filename = join(self.sbs2_dir, 'sbs2_data', 'hardware', hardware,
                        'spatialCoherenceSmooth0-2_reduced_inverse.txt')
        matrix = Matrix(read_csv(filename, sep='\t', header=None))
        return matrix

    def surface(self, model='small'):
        """Return surface from SBS2 data.

        Read a surface from mesh_ctx_5124_normals.obj (large) or
        vertface_brain_reduced.obj (small).

        Arguments
        ---------
        model : 'small' or 'large'
            Indicate which model should be read.

        Examples
        --------
        >>> sbs2_data = SBS2Data()
        >>> surface = sbs2_data.surface()
        >>> surface.plot()
        >>> surface.show()

        """
        if model == 'small':
            filename = 'vertface_brain_reduced.obj'
        elif model == 'large':
            filename = 'mesh_ctx_5124_normals.obj'
        else:
            raise ValueError('model should be small or large')

        full_filename = join(self.sbs2_dir, 'sbs2_data', filename)
        surface = read_obj(full_filename)
        return surface

    def backward_model(self, hardware='emotiv', method='LORETA'):
        """Compute and return backward model.

        Arguments
        ---------
        hardware : 'emotiv' or 'emocap'
            Hardward type for forward model
        method : 'LORETA' or 'minimumnorm'
            Estimation type

        Returns
        -------
        matrix : brede.core.matrix.Matrix
            Matrix with backward model size 1028 x 14

        Examples
        --------
        >>> sbs2_data = SBS2Data()
        >>> backward_model = sbs2_data.backward_model()
        >>> backward_model.shape
        (1028, 14)

        References
        ----------
        Smartphones as pocketable labs: Visions for mobile brain imaging and
        neurofeedback. Arkadiusz Stopczynski, Carsten Stahlhut,
        Michael Kai Petersen, Jakob Eg Larsen, Camilla Falk Jensen,
        Marieta Georgieva Ivanova, Tobias S. Andersen, Lars Kai Hansen.
        Affective Computing and Intelligent Interaction 6975: 317-318. 2011

        https://github.com/SmartphoneBrainScanner/smartphonebrainscanner2-core
        /src/source_reconstruction/loreta/sbs2sourcereconstruction_loreta.cpp

        """
        # https://github.com/SmartphoneBrainScanner/smartphonebrainscanner2-core
        # /src/source_reconstruction/loreta/sbs2sourcereconstruction_loreta.cpp
        # line 62-63!
        inv_alpha = 0.0100
        inv_beta = 0.3781

        # Forward model, F matrix
        forward = self.forward_model(hardware).values

        identity = eye(forward.shape[0])

        if method == 'minimumnorm':
            # The spatial coherence is the identity matrix and disappears
            backward = forward.T.dot(pinv(forward.dot(forward.T)
                                          + inv_beta/inv_alpha * identity))
        elif method == 'LORETA':
            # Spatial coherence, L matrix
            coherence = self.spatial_coherence(hardware).values

            forward_and_coherence = forward.dot(coherence)
            sigma_inv = inv_alpha * \
                forward_and_coherence.dot(forward.T) \
                + inv_beta * identity
            backward = inv_alpha * forward_and_coherence.T.dot(pinv(sigma_inv))
        else:
            raise ValueError('Wrong method')

        backward = Matrix(backward)
        backward.columns = self.electrode_names(hardware)
        return backward


def main(args):
    """Handle command-line interface."""
    sbs2_data = SBS2Data()
    surface = sbs2_data.surface(model='large')
    surface.plot()
    surface.show()


if __name__ == '__main__':
    from docopt import docopt

    main(docopt(__doc__))
