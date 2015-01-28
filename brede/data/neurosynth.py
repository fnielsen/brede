"""Interface to Neurosynth data."""


from __future__ import absolute_import

import tarfile

from os import chdir, getcwd, makedirs
from os.path import exists, expanduser, join

from urllib import urlretrieve

from brede.config import config

import pandas as pd


NEUROSYNTH_DATABASE_URL = "http://old.neurosynth.org/data/current_data.tar.gz"
NEUROSYNTH_DATABASE_URL = ("https://github.com/neurosynth/neurosynth-data/"
                           "blob/master/archive/data_0.4.September_2014.tar.gz"
                           "?raw=true")


class NeurosynthDatabase(object):

    """Interface to dump of Neurosynth.

    Data from the Neurosynth website will be downloaded to a local directory.

    Example
    -------
    >>> nd = NeurosynthDatabase()
    >>> database = nd.database()
    >>> 'MNI' in database.space.values
    True

    """

    def __init__(self):
        """Setup directories and filenames."""
        self.data_dir = expanduser(config.get('data', 'data_dir'))
        self.neurosynth_dir = join(self.data_dir, 'neurosynth')
        self.neurosynth_database_filename = join(self.neurosynth_dir,
                                                 'database.txt')
        self.neurosynth_features_filename = join(self.neurosynth_dir,
                                                 'features.txt')
        self.neurosynth_download_filename = join(self.neurosynth_dir,
                                                 'current_data.tar.gz')
        self.neurosynth_database_url = NEUROSYNTH_DATABASE_URL

    @property
    def name(self):
        """Return short name for database."""
        return "Neurosynth"

    @property
    def description(self):
        """Return a descriptive string about the data."""
        return ("Neurosynth is a database setup by Tal Yarkoni and "
                "contains stereotaxic coordinates from functional "
                "neuroimaging studies.")

    def __str__(self):
        """Return descriptive string."""
        string = "<NeurosynthDatabase({}x{})>"
        df = self.database()
        return string.format(*df.shape)

    def make_dir(self):
        """Make Neurosynth data directory."""
        if not exists(self.data_dir):
            makedirs(self.neurosynth_dir)

    def download(self):
        """Download Neurosynth database file."""
        self.make_dir()
        urlretrieve(self.neurosynth_database_url,
                    self.neurosynth_download_filename)

    def unpack(self):
        """Extract the downloaded compressed Neurosynth dump file."""
        if (not exists(self.neurosynth_database_filename) and
                not exists(self.neurosynth_features_filename)):
            if not exists(self.neurosynth_download_filename):
                self.download()
            cwd = getcwd()
            chdir(self.neurosynth_dir)
            try:
                with tarfile.open(self.neurosynth_download_filename,
                                  'r:gz') as fid:
                    fid.extractall()
            finally:
                chdir(cwd)

    def database(self):
        """Return database as dataframe.

        Returns
        -------
        database : pandas.DataFrame
            Dataframe with data from database.txt.

        """
        self.unpack()
        database = pd.read_csv(self.neurosynth_database_filename,
                               sep='\t', low_memory=False)
        return database

    def features(self):
        """Return Neurosynth features as dataframe.

        Returns
        -------
        features : pandas.DataFrame
            Dataframe with features

        Examples
        --------
        >>> nd = NeurosynthDatabase()
        >>> features = nd.features()
        >>> 9106283 in features.index
        True

        """
        self.unpack()
        features = pd.read_csv(self.neurosynth_features_filename,
                               sep='\t', low_memory=False,
                               index_col=0)
        return features
