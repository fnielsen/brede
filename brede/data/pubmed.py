"""Interface to Pubmed information."""

import errno
import os
import time
from datetime import datetime
from os.path import expanduser, join

try:
    import cPickle as pickle
except ImportError:
    import pickle

from Bio import Entrez, Medline

from brede.config import config


class Pubmed(object):

    """Interface to Pubmed bibliographic information.

    PubMed data in the form of MEDLINE records are cache on the local
    filesystem as pickle data in the data directory. If the record is not
    available locally it is fetch via the Internet on the Entrez server.
    """

    last_download_time = datetime.now()

    def __init__(self):
        """Setup directories."""
        self.data_dir = expanduser(config.get('data', 'data_dir'))
        self.pubmed_dir = join(self.data_dir, 'pubmed')
        self.setup_data_dir()
        self.pause_between_downloads = 0.334

        Entrez.email = config.get('bio', 'email')
        Entrez.tool = 'brede'

    @property
    def name(self):
        """Return short name for this database."""
        return "PubMed"

    @property
    def description(self):
        """Return descriptive string for this database."""
        return ("Bibliographic records from the PubMed online database.")

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

        Note the guideline (http://www.ncbi.nlm.nih.gov/books/NBK25497/):

        "In order not to overload the E-utility servers, NCBI recommends
        that users post no more than three URL requests per second and
        limit large jobs to either weekends or between 9:00 PM and
        5:00 AM Eastern time during weekdays."

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
        # Pause to not overload Entrez server.
        interval = (datetime.now() - Pubmed.last_download_time).total_seconds()
        pause = max(0, self.pause_between_downloads - interval)
        time.sleep(pause)

        # Actual download
        handle = Entrez.efetch(db="pubmed", id=str(pmid),
                               rettype='medline', retmode='text')

        Pubmed.last_download_time = datetime.now()

        records = list(Medline.parse(handle))
        if len(records) > 0:
            record = records[0]
        else:
            record = None
        return record

    def get_medline(self, pmid):
        """Return bibliographic information from PubMed.

        Parameters
        ----------
        pmid : int or str
            PMID identifier for PubMed document

        Returns
        -------
        medline : Bio.Medline.Record
            Dict-like structure with bibliographic fields

        Example
        -------
        >>> pubmed = Pubmed()
        >>> paper = pubmed.get_medline(15238438)
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

    def get_medlines(self, pmids):
        """Return MEDLINE information from several PMIDs.

        Parameters
        ----------
        pmids : list of int or list of str
            PMID identifiers for the PubMed database

        Returns
        -------
        medlines : list of Bio.Medline.Record
            List of dict-like objects with bibliographic data

        """
        medlines = [self.get_medline(pmid) for pmid in pmids]
        return medlines

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
