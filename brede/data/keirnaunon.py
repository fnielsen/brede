"""brede.data.keirnaunon - Interface to 1989 Keirn and Aunon data.

Usage:
  keirnaunon [options]

Options:
  -h --help     Help message


EEG data from a study by Keirn and Aunon.

References
----------
Zachary A. Keirn, Jorge I. Aunon, A new mode of communication between
man and his surroundings, IEEE Transactions on Biomedical Engineering
37 (12): 1209-12-14. 1990.

1989 Keirn and Aunon
http://www.cs.colostate.edu/eeg/main/data/1989_Keirn_and_Aunon

"""


from __future__ import absolute_import, division, print_function

import errno

import gzip

from os import makedirs
from os.path import exists, expanduser, join

from urllib import urlretrieve

from .core import Data
from ..config import config
from ..eeg.core import EEGRun, EEGRuns


URL = "http://www.cs.colostate.edu/eeg/data/alleegdata.ascii.gz"

ELECTRODES = ['C3', 'C4', 'P3', 'P4', 'O1', 'O2']

SAMPLING_RATE = 1./250

# This constant is setup via the _number_of_trials function
# The key here is the subject identifier.
NUMBER_OF_TRIALS = {
    1: 10,
    2: 5,
    3: 10,
    4: 10,
    5: 15,
    6: 10,
    7: 5
}


class KeirnAunon(Data):

    """EEG data from a study by Keirn and Aunon.

    Examples
    --------
    >>> ka = KeirnAunon()
    >>> eeg_run = ka.trial()
    >>> fourier = eeg_run.fft()
    >>> fourier.plot_mean_spectrum()
    >>> fourier.show()

    """

    def __init__(self, subject=1, state='baseline', trial=1):
        """Setup metadata."""
        self.data_dir = join(expanduser(config.get('data', 'data_dir')),
                             'keirnaunon')
        self.gzip_filename = join(self.data_dir, 'alleegdata.ascii.gz')
        self.filename = join(self.data_dir, 'alleegdata.ascii')

    @property
    def _constructor(self):
        return type(self)

    def __str__(self):
        """Return descriptive string."""
        string = "<KeirnAunon(filename={})>"
        return string.format(self.filename)

    @property
    def name(self):
        """Return name of data set."""
        return """1989 Keirn and Aunon"""

    @property
    def description(self):
        """Return readme text from data file."""
        self.unpack()
        with open(self.filename) as fid:
            text = fid.readline().strip()
        return text

    def make_dirs(self):
        """Make data directories."""
        try:
            makedirs(self.data_dir)
        except OSError as exception:
            if exception.errno != errno.EEXIST:
                raise

    def download(self):
        """Download data set from web site."""
        self.make_dirs()
        urlretrieve(URL, self.filename)

    def unpack(self, redownload=False):
        """Unpack data set from conditional download."""
        if not exists(self.filename) or redownload:
            if not exists(self.gzip_filename) or redownload:
                self.download()
            with gzip.open(self.gzip_filename) as infile:
                content = infile.read()
            with open(self.filename, 'w') as outfile:
                outfile.write(content)

    @staticmethod
    def _match_line(subject, state, trial):
        return 'subject {}, {}, trial {}'.format(subject, state, trial)

    def trial(self, subject=1, state='baseline', trial=1):
        """Read data from the 6 electrodes from specified trial.

        The possible states are: baseline, multiplication, letter-composing,
        rotation and counting.

        Arguments
        ---------
        subject : int, optional
            Subject identifier from 1 to 7.
        state : 'baseline' or 'multiplication', optional
            State for trial.
        trial : int, optional
            Trial identifier from 1 to 5 or 15.

        Returns
        -------
        eeg_run : brede.eeg.core.EEGRun
            Data in a EEGRun DataFrame-like object.

        Examples
        --------
        >>> ka = KeirnAunon()
        >>> eeg_run = ka.trial(subject=3, state='multiplication', trial=2)
        >>> eeg_run.ix[0, 0]
        11.169

        >>> eeg_run.sampling_rate
        250.0

        See also
        --------
        brede.eeg.core

        """
        self.unpack()
        match_line = self._match_line(subject, state, trial)
        with open(self.filename) as fid:
            while fid.readline().strip() != match_line:
                # Skipping lines until relevant data
                pass
            data = [[float(elem) for elem in fid.readline().split()]
                    for n in range(6)]
            # The line with EOG is skipped here.
        eeg_run = EEGRun(zip(*data),
                         columns=ELECTRODES,
                         sampling_rate=SAMPLING_RATE)
        return eeg_run

    def trials_for_subject_state(self, subject=1, state='baseline'):
        """Return data from all trials for a subject/state combination.

        Parameters
        ----------
        subject : int, optional
            Subject identifier
        state : 'baseline' or other, optional
            State of trial

        Returns
        -------
        eeg_runs : brede.eeg.core.EEGRuns
            3D Panel-like structure with (trial x time x electrode).

        """
        self.unpack()
        number_of_trials = NUMBER_OF_TRIALS[subject]
        trials = range(1, number_of_trials + 1)
        data = []
        with open(self.filename) as fid:
            for trial in trials:
                match_line = self._match_line(subject, state, trial)
                while fid.readline().strip() != match_line:
                    # Skipping lines until relevant data
                    pass
                data.append(zip(*[[float(elem)
                                   for elem in fid.readline().split()]
                                  for n in range(6)]))

        # trial x time x electrode
        return EEGRuns(data, items=trials, minor_axis=ELECTRODES)


def _number_of_trials():
    """Find the number of trials for each subject/state combination.

    Note that within subject all states have the same number of trials.

    Examples
    --------
    >>> number_of_trials = _number_of_trials()
    >>> number_of_trials[7]
    5

    """
    ka = KeirnAunon()
    number_of_trials = {}
    with open(ka.filename) as fid:
        for line in fid:
            if line.startswith('subject'):
                subject = int(line[8])
                state = line.split()[2]
                trial = int(line.split()[4])
                number_of_trials[(subject, state)] = trial
                number_of_trials[subject] = trial
    return number_of_trials


def main(args):
    """Handle command-line interface."""
    ka = KeirnAunon()
    print(ka.name)


if __name__ == '__main__':
    from docopt import docopt

    main(docopt(__doc__))
