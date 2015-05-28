"""eegmmidb examples.

Usage:
  brede.data.examples.eegmmidb [options]

Options:
  -h --help       Help
  --subject=<n>   Subject id [default: 1]

"""

from __future__ import absolute_import, division, print_function

from brede.data.eegmmidb import EEGMMIDB
from brede.eeg.plotting import topoplot

import matplotlib.pyplot as plt

import pandas as pd


def csp_of_two_runs(subject=1, bandpass_filtering=False, low=7, high=30):
    """Common spatial patterns on two runs."""
    # Load data
    eegmmidb = EEGMMIDB()
    move_fist = eegmmidb.run(subject=subject, run=3)
    move_fist['state'] = 'Move fist'
    eyes_open = eegmmidb.run(subject=subject, run=1)
    eyes_open['state'] = 'Eyes open'

    if bandpass_filtering:
        move_fist.bandpass_filter(low, high, inplace=True)
        eyes_open.bandpass_filter(low, high, inplace=True)

    # Concatenate data
    offset = move_fist.index[-1] + 1 / move_fist.sampling_rate
    eyes_open.index += offset
    eeg_data = pd.concat((move_fist, eyes_open))

    Z, W = eeg_data.csp(group_by='state', n_components=1)
    topoplot(W.ix[:, 'CSP 1'], title='Subject {}'.format(subject))
    plt.show()


def main(args):
    """Handle command-line interface."""
    subject = int(args['--subject'])

    csp_of_two_runs(subject=subject)


if __name__ == '__main__':
    import docopt

    main(docopt.docopt(__doc__))
