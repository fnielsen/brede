#!/usr/bin/env python
"""
Usage:
  edf.py [options] [<file>]

Options:
  -h --help  Help

"""

from __future__ import absolute_import 

from eegtools.io import edfplus

# import pandas as pd

# import eeg


def edf_to_df(edf):
    """Convert read edf data structure to pandas data frame.

    Eletrodes are distributed over columns while each sample has its own row.

    """
    labels = [label.replace('.', '') for label in edf.chan_lab]
    df = pd.DataFrame(edf.X.T, columns=labels, index=edf.time)
    return df


def read_edf(file):
    """Read EEG from European Data Format (EDF) files into dataframe.

    Eletrodes are distributed over columns while each sample has its own row.

    Parameters
    ----------
    file : str
        Filename of a EDF file.

    Returns
    -------
    df : pandas.DataFrame

    """
    edf_data = edfplus.load_edf(full_filename)
    df = edf_to_df(edf_data)
    return df


def main(args):
    """Run script."""
    if args['<file>'] is not None:
        filename = args['<file>']
        df = read_edf(filename)
        sample = df.ix[0, :]
#        eeg.topoplot(sample)


if __name__ == '__main__':
    import docopt

    main(docopt.docopt(__doc__))


