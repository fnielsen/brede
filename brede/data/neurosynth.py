"""brede.data.neurosynth - Interface to Neurosynth data.

Usage:
  brede.data.neurosynth <command>

Options:
  -h --help     Help

Examples:
   $ python -m brede.data.neurosynth featurenames

If command=redownload the the database file is redownloaded from the Git
repository. This command will also unpack the new data and overwrite the old.

command=featurenames will return a comma-separated list of feature names, i.e.
the first row of the features.txt file.

Otherwise outputs the Neurosynth database as comma-separated values
(This is a pretty long listing).

"""


from __future__ import absolute_import, print_function

import logging

import tarfile

from os import chdir, getcwd, makedirs
from os.path import exists, expanduser, join

from urllib import urlretrieve

from nltk.tokenize.punkt import PunktSentenceTokenizer

import pandas as pd

from .core import Data
from .pubmed import Pubmed
from ..config import config


NEUROSYNTH_DATABASE_URL = "http://old.neurosynth.org/data/current_data.tar.gz"
NEUROSYNTH_DATABASE_URL = ("https://github.com/neurosynth/neurosynth-data/"
                           "blob/master/current_data.tar.gz?raw=true")


class NeurosynthDatabase(Data):
    """Interface to dump of Neurosynth.

    Data from the Neurosynth website will be downloaded to a local directory.
    Data is read from the local directory. Coordinates and 'features' (words)
    are available from the database.

    Example
    -------
    >>> nd = NeurosynthDatabase()
    >>> database = nd.database()
    >>> 'MNI' in database.space.values
    True

    """

    def __init__(self):
        """Setup directories and filenames."""
        self.logger = logging.getLogger(__name__ + '.Pubmed')
        self.logger.addHandler(logging.NullHandler())

        self.data_dir = expanduser(config.get('data', 'data_dir'))
        self.logger.info('Data directory: {}'.format(self.data_dir))

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
        if not exists(self.neurosynth_dir):
            makedirs(self.neurosynth_dir)

    def download(self):
        """Download Neurosynth database file."""
        self.make_dir()
        urlretrieve(self.neurosynth_database_url,
                    self.neurosynth_download_filename)

    def unpack(self, reunpack=False):
        """Extract the downloaded compressed Neurosynth dump file."""
        if reunpack or ((not exists(self.neurosynth_database_filename) and
                         not exists(self.neurosynth_features_filename))):
            if not exists(self.neurosynth_download_filename):
                self.download()
            cwd = getcwd()
            chdir(self.neurosynth_dir)
            try:
                with tarfile.open(self.neurosynth_download_filename,
                                  'r:gz') as fid:
                    
                    import os
                    
                    def is_within_directory(directory, target):
                        
                        abs_directory = os.path.abspath(directory)
                        abs_target = os.path.abspath(target)
                    
                        prefix = os.path.commonprefix([abs_directory, abs_target])
                        
                        return prefix == abs_directory
                    
                    def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
                    
                        for member in tar.getmembers():
                            member_path = os.path.join(path, member.name)
                            if not is_within_directory(path, member_path):
                                raise Exception("Attempted Path Traversal in Tar File")
                    
                        tar.extractall(path, members, numeric_owner=numeric_owner) 
                        
                    
                    safe_extract(fid)
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
        self.logger.info('Reading {}'.format(
            self.neurosynth_database_filename))
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
        >>> 23400116 in features.index
        True

        """
        self.unpack()
        features = pd.read_csv(self.neurosynth_features_filename,
                               sep='\t', low_memory=False,
                               index_col=0)
        return features

    def feature_names(self):
        """Return list of feature names.

        Returns
        -------
        feature_names : list of strings
            Words and phrases from first line of features.txt

        Examples
        --------
        >>> nd = NeurosynthDatabase()
        >>> 'attention' in nd.feature_names()
        True

        """
        self.unpack()
        features = pd.read_csv(self.neurosynth_features_filename,
                               sep='\t', low_memory=False, nrows=2,
                               index_col=0)
        return features.columns.tolist()

    def medlines(self):
        """Return list of Medline structures for papers in Neurosynth.

        Returns
        -------
        medlines : list of Bio.Medline.Record
            List of Medline strutures

        Examples
        --------
        >>> nd = NeurosynthDatabase()
        >>> medlines = nd.medlines()
        >>> authors = [m['FAU'] for m in medlines if m['PMID'] == '15238438']
        >>> 'Nielsen, Finn A' in authors[0]
        True

        """
        nd_database = self.database()
        pubmed = Pubmed()
        medlines = pubmed.get_medlines(set(nd_database.id))
        return medlines

    def sentences(self):
        """Yield sentences from abstracts.

        Yields
        ------
        sentences : str
            Yields sentences from abstract.

        """
        tokenizer = PunktSentenceTokenizer()

        for medline in self.medlines():
            if 'AB' not in medline:
                continue
            abstract = medline['AB']
            sentences = tokenizer.tokenize(abstract)
            for sentence in sentences:
                yield sentence


def main(args):
    """Handle command-line interface."""
    command = args['<command>']
    if command == 'redownload':
        nd = NeurosynthDatabase()
        nd.download()
        nd.unpack(reunpack=True)
    elif command == 'featurenames':
        nd = NeurosynthDatabase()
        print(",".join(nd.feature_names()))
    elif command == 'sentences':
        nd = NeurosynthDatabase()
        for sentence in nd.sentences():
            print(sentence)

    else:
        nd = NeurosynthDatabase()
        print(nd.database().to_csv())


if __name__ == '__main__':
    from docopt import docopt

    main(docopt(__doc__))
