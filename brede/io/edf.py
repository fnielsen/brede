#!/usr/bin/env python
"""edf.py.

Usage:
  edf.py [options] <file>

Options:
  -h --help  Help

"""

from __future__ import absolute_import, division, print_function

# Absolute path here because of circular dependency
import brede.eeg

from eegtools.io import edfplus

import pandas as pd


def edf_to_df(edf):
    """Convert read edf data structure to pandas data frame.

    Eletrodes are distributed over columns while each sample has its own row.

    """
    labels = [label.replace('.', '') for label in edf.chan_lab]
    df = pd.DataFrame(edf.X.T, columns=labels, index=edf.time)
    return df


def read_edf(filename):
    """Read EEG from European Data Format (EDF) files into dataframe.

    Eletrodes are distributed over columns while each sample has its own row.

    Parameters
    ----------
    filename : str
        Filename of a EDF file.

    Returns
    -------
    df : pandas.DataFrame

    """
    edf_data = edfplus.load_edf(filename)
    df = edf_to_df(edf_data)
    return df


def main(args):
    """Run script."""
    if args['<file>'] is not None:
        filename = args['<file>']
        df = read_edf(filename)
        sample = df.ix[0, :]
        brede.eeg.topoplot(sample)


if __name__ == '__main__':
    import docopt

    main(docopt.docopt(__doc__))
