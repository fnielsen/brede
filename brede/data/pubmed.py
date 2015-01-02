"""Interface to Pubmed information."""


from os.path import expanduser, join

from Bio import Entrez, Medline

from ..config import config


class Pubmed(object):

    """Interface to Pubmed bibliographic information."""

    def __init__(self):
        """Setup directories."""
        self.data_dir = expanduser(config.get('data', 'data_dir'))
        self.pubmed_dir = join(self.data_dir, 'pubmed')
        Entrez.email = config.get('bio', 'email')

    def get_pmid(self, pmid):
        """Return bibliographic information from PubMed.

        Example
        -------
        >>> pubmed = Pubmed()
        >>> paper = pubmed.get_pmid(15238438)
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
