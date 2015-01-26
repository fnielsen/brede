#!/usr/bin/env python
"""brede.

Usage:
  brede [options] [<inputs>]...

Options:
  -h --help           Help

Try to parse input arguments and do something.

Examples
--------
    $ python -m brede

    $ python -m brede topoplot ~/data/eegmmidb/S001/S001R01.edf

"""

from __future__ import absolute_import, division, print_function

import sys

from os.path import exists, splitext

import matplotlib.pyplot as plt

from .eeg import plotting
from .io import read_edf


def looks_like_file(filename):
    """Try to determine if a name looks like a filename."""
    _, ext = splitext(filename)
    if ext != '':
        return True
    return False


def main():
    """Try to parse input arguments and do something."""
    import docopt

    args = docopt.docopt(__doc__)

    inputs = args['<inputs>']
    if len(inputs) == 0:
        # No input arguments: Lets just show something
        plotting.topoplot()
        plt.show()

    else:
        filenames = [input_ for input_ in inputs if looks_like_file(input_)]
        command = 'topoplot'
        if not looks_like_file(inputs[0]):
            command = inputs[0]

        if command == 'topoplot':
            if len(filenames) == 0:
                plotting.topoplot()
                plt.show()
            elif len(filenames) == 1:
                filename = filenames[0]
                if not exists(filename):
                    sys.exit('File does not exists: {}'.format(filename))
                if filename.endswith('.edf'):
                    dataframe = read_edf(filename)
                    sample = dataframe.ix[0, :]
                    plotting.topoplot(sample)
                    plt.title('First sample of {}'.format(filename))
                    plt.show()
                else:
                    sys.exit("Could not handle file type")
            else:
                # TODO
                pass
        else:
            sys.exit('Did not recognize command: {}'.format(command))


if __name__ == '__main__':
    main()
