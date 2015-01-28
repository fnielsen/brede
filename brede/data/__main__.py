#!/usr/bin/env python
"""
Usage:
  brede.data 

Options:
  -h --help  Help

Printing simple overview of available databases.

"""

from __future__ import print_function

from docopt import docopt

from brede.data.neurosynth import NeurosynthDatabase
from brede.data.pubmed import Pubmed


def main(args):
    databases = [NeurosynthDatabase, Pubmed]
    for database in databases:
        db = database()
        print("{}: {}".format(db.name, db.description))


main(docopt(__doc__))
