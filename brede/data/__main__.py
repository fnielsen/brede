#!/usr/bin/env python
"""brede.data.

Usage:
  brede.data

Options:
  -h --help  Help

Printing simple overview of available databases.

"""

from __future__ import print_function

from brede.data.neurosynth import NeurosynthDatabase
from brede.data.pubmed import Pubmed

from docopt import docopt


def main(args):
    """Handle command-line interface."""
    databases = [NeurosynthDatabase, Pubmed]
    for database in databases:
        db = database()
        print("{}: {}".format(db.name, db.description))


if __name__ == '__main__':
    main(docopt(__doc__))
