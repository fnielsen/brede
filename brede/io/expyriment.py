#!/usr/bin/env python
"""Interface to Expyriment files.

Usage:
  breed.io.expyriment [options] <file>

Options:
  -h --help  Help

"""

from __future__ import absolute_import, division, print_function

import pandas as pd


def read_xpe(filename, *args, **kwargs):
    """Read 'xpe' Expyriment file.

    Expyriment's XPE files are data files with comma-separated values and a
    header

    The reading function will use the 'Time' column as the index.

    Parameters
    ----------
    filename : str
        Filename with extension for the xpe file
    index_col : str, optional
        Column to use as index [default: 'Time']

    Returns
    -------
    df : pandas.DataFrame
        Dataframe with data from xpe file.

    """
    df = pd.read_csv(filename, *args, index_col='Time', comment="#", **kwargs)
    return df


def main(args):
    """Handle command-line arguments."""
    df = read_xpe(args['<file>'])
    print(df)


if __name__ == '__main__':
    import docopt

    main(docopt.docopt(__doc__))
