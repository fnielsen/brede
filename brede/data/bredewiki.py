"""brede.data.bredewiki - Interface to Brede Wiki data.

Usage:
  brede.data.bredewiki [options]

Options:
  -h --help     Help

"""


from __future__ import absolute_import, print_function

from os import makedirs
from os.path import exists, expanduser, join

from urllib import urlretrieve

from brede.config import config

import db


BREDEWIKI_TEMPLATES_URL = ("http://neuro.compute.dtu.dk/services/"
                           "bredewiki/download/bredewiki-templates.sqlite3")


class BredeWikiTemplates(db.DB):
    """Interface to dump of Brede Wiki Templates.

    The data is downloaded from the file bredewiki-templates.sqlite3 in the
    directory http://neuro.compute.dtu.dk/services/bredewiki/download/

    It is loaded with automagically via the DB class in the db.py module,
    os that tables are available from BredeWikiTemplates().tables.

    Example
    -------
    >>> bwt = BredeWikiTemplates()
    >>> papers = bwt.tables.brede_paper.all()
    >>> 11227136 in set(papers._pmid.dropna().astype(int))
    True

    >>> # Brain regions from LPBA40 brain atlas
    >>> brain_regions = bwt.tables.brede_brain_region.all()
    >>> lpba_regions = brain_regions.ix[brain_regions._lpba.notnull(),
    ...                                 ['_name', '_lpba']]
    >>> 'Brain stem' in set(lpba_regions._name)
    True

    """

    def __init__(self, redownload=False):
        """Setup directories and filenames.

        Parameters
        ----------
        redownload : bool
            Download the database file a new.

        """
        self.data_dir = expanduser(config.get('data', 'data_dir'))
        self.bredewiki_dir = join(self.data_dir, 'bredewiki')
        self.bredewiki_templates_filename = join(self.bredewiki_dir,
                                                 'bredewiki-templates.sqlite3')
        self.bredewiki_templates_url = BREDEWIKI_TEMPLATES_URL

        self.unpack(redownload=redownload)
        super(BredeWikiTemplates, self).__init__(
            filename=self.bredewiki_templates_filename,
            dbtype="sqlite")

    @property
    def name(self):
        """Return short name for database."""
        return "Brede Wiki Templates"

    @property
    def description(self):
        """Return a descriptive string about the data."""
        return ("Brede Wiki Templates is a database of structured information"
                "from the Brede Wiki.")

    def __str__(self):
        """Return descriptive string."""
        string = "<BredeWikiTemplates>"
        return string

    def make_dir(self):
        """Make Brede Wiki data directory."""
        if not exists(self.bredewiki_dir):
            makedirs(self.bredewiki_dir)

    def download(self):
        """Download Brede Wiki Templates database file."""
        self.make_dir()
        urlretrieve(self.bredewiki_templates_url,
                    self.bredewiki_templates_filename)

    def unpack(self, redownload=False):
        """Extract the downloaded file.

        It tests if the relevant database file is already downloaded.
        If not call then the download method is called.

        """
        if redownload or not exists(self.bredewiki_templates_filename):
            self.download()
            # no need for extraction


def main(args):
    """Handle command-line interface."""
    bwt = BredeWikiTemplates()
    print(bwt)


if __name__ == '__main__':
    from docopt import docopt

    main(docopt(__doc__))
