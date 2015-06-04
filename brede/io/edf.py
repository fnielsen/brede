#!/usr/bin/env python
"""Interface to EDF files.

Usage:
  edf.py [options] <file>

Options:
  -h --help       Help
  -i --index=<i>  Time index to plot spatially

Description:
  This module acts both as a module and as a script. As a script it will read
  a EDF file and plot it.

Examples:
  $ python -m brede.io.edf S001R01.edf

"""

from __future__ import absolute_import, division, print_function

from eegtools.io import edfplus

from pandas import DataFrame


def edf_to_df(edf):
    """Convert read edf data structure to pandas data frame.

    Eletrodes are distributed over columns while each sample has its own row.

    """
    labels = [label.replace('.', '') for label in edf.chan_lab]
    df = DataFrame(edf.X.T, columns=labels, index=edf.time)
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
    """Run script.

    Parameters
    ----------
    args : dict
        Command-line interface arguments in Docopt format.

    """
    import matplotlib.pyplot as plt

    from ..eeg.plotting import topoplot

    if args['<file>'] is not None:
        filename = args['<file>']
        df = read_edf(filename)
        if args['--index']:
            index = int(args['--index'])
            sample = df.ix[index, :]
            topoplot(sample)
        else:
            topoplot(df)
        plt.show()


if __name__ == '__main__':
    from docopt import docopt

    main(docopt(__doc__))
