"""Interface to Pubmed information."""


import errno

import os
from os.path import expanduser, join

try:
    import cPickle as pickle
except ImportError:
    import pickle


from Bio import Entrez, Medline

from ..config import config


class Pubmed(object):

    """Interface to Pubmed bibliographic information."""

    def __init__(self):
        """Setup directories."""
        self.data_dir = expanduser(config.get('data', 'data_dir'))
        self.pubmed_dir = join(self.data_dir, 'pubmed')
        self.setup_data_dir()
        Entrez.email = config.get('bio', 'email')

    def setup_data_dir(self):
        """Create pubmed data directory if not exists."""
        # https://stackoverflow.com/questions/273192/
        try:
            os.makedirs(self.pubmed_dir)
        except OSError as exception:
            if exception.errno != errno.EEXIST:
                raise

    def download_medline(self, pmid):
        """Download bibliographic information from PubMed.

        Parameters
        ----------
        pmid : int or str
            PMID identifier for PubMed document

        Returns
        -------
        record : Bio.Medline.Record
            Dict-like structure with bibliographic fields

        Example
        -------
        >>> pubmed = Pubmed()
        >>> paper = pubmed.download_medline(15238438)
        >>> paper['TI'][:28]
        'Right temporoparietal cortex'

        """
        handle = Entrez.efetch(db="pubmed", id=str(pmid),
                               rettype='medline', retmode='text')
        records = list(Medline.parse(handle))
        if len(records) > 0:
            record = records[0]
        else:
            record = None
        return record

    def get_pmid(self, pmid):
        """Return bibliographic information from PubMed.

        Parameters
        ----------
        pmid : int or str
            PMID identifier for PubMed document

        Returns
        -------
        record : Bio.Medline.Record
            Dict-like structure with bibliographic fields

        Example
        -------
        >>> pubmed = Pubmed()
        >>> paper = pubmed.get_pmid(15238438)
        >>> paper['TI'][:28]
        'Right temporoparietal cortex'

        """
        try:
            medline = self.load_medline(pmid)
        except IOError as exception:
            if exception.errno == errno.ENOENT:
                medline = self.download_medline(pmid)
                self.save_medline(medline)
            else:
                raise
        return medline

    def medline_filename(self, pmid):
        """Return full filename for local MEDLINE file."""
        # Sanitization pmid should only consists of numbers
        pmid = str(int(str(pmid)))
        filename = join(self.pubmed_dir, pmid + '.pck')
        return filename

    def load_medline(self, pmid):
        """Load MEDLINE file from local file database."""
        filename = self.medline_filename(pmid)
        with open(filename) as fid:
            medline = pickle.load(fid)
        return medline

    def save_medline(self, medline):
        """Save a MEDLINE record on the local file system."""
        filename = self.medline_filename(medline['PMID'])
        with open(filename, 'w') as fid:
            pickle.dump(medline, fid)
