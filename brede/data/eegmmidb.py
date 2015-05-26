"""brede.data.eegmmidb - Interface to eegmmidb data.

Usage:
  brede.data.eegmmidb [options]

Options:
  -h --help      Help message
  --run=<n>      Run identifier [default: 1]
  --subject=<n>  Subject identifier [default: 1]


EEG Motor Movement/Imagery Dataset.

References
----------
EEG Motor Movement/Imagery Dataset
http://physionet.nlm.nih.gov/pn4/eegmmidb/

http://neuro.compute.dtu.dk/wiki/EEG_Motor_Movement/Imagery_Dataset

Schalk, G., McFarland, D.J., Hinterberger, T., Birbaumer, N., Wolpaw,
J.R. BCI2000: A General-Purpose Brain-Computer Interface (BCI)
System. IEEE Transactions on Biomedical Engineering 51(6):1034-1043,
2004.

Examples
--------
>>> from brede.data.eegmmidb import EEGMMIDB
>>> eegmmidb = EEGMMIDB()

"""


from __future__ import absolute_import, division, print_function

from os.path import exists, expanduser, join

from urllib import urlretrieve

import pandas as pd

from .core import Data
from ..config import config
from ..eeg.core import EEGRun
from ..io.edf import read_edf


URL_RECORDS = "http://physionet.nlm.nih.gov/pn4/eegmmidb/RECORDS"
URL_BASE = "http://physionet.nlm.nih.gov/pn4/eegmmidb/"

# Stated on http://physionet.nlm.nih.gov/pn4/eegmmidb/
SAMPLING_RATE = 160

STATES = ['Baseline, eyes open',
          'Baseline, eyes closed',
          'Task 1 (open and close left or right fist)',
          'Task 2 (imagine opening and closing left or right fist)',
          'Task 3 (open and close both fists or both feet) ',
          'Task 4 (imagine opening and closing both fists or both feet)']


class EEGMMIDB(Data):

    """EEG Motor Movement/Imagery Dataset.

    This class can be viewed as a singleton class. There are no instance
    variables.

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> eegmmidb = EEGMMIDB()

    """

    def __init__(self):
        """Setup metadata."""
        self.data_dir = join(expanduser(config.get('data', 'data_dir')),
                             'eegmmidb')
        self.records_filename = join(self.data_dir, 'RECORDS')

    @property
    def _constructor(self):
        """Return class for instancing a new object."""
        return type(self)

    def __str__(self):
        """Return descriptive string."""
        string = "<EEGMMIDB()>"
        return string

    name = "EEG Motor Movement/Imagery Dataset"

    description = "EEG data set"

    def make_dirs(self):
        """Make data directories."""
        self.make_dir()

        for subject in range(1, 110):
            self.make_dir(join(self.data_dir, 'S{:03}'.format(subject)))

    def download_records(self):
        """Download index for data set from web site.

        References
        ----------
        http://physionet.nlm.nih.gov/pn4/eegmmidb/RECORDS

        """
        self.make_dirs()
        urlretrieve(URL_RECORDS, self.records_filename)

    @staticmethod
    def data_filename(subject, run):
        """Return partial filename.

        Examples
        --------
        >>> EEGMMIDB.data_filename(2, 3)
        'S002/S002R03.edf'

        """
        return "S{:03}/S{:03}R{:02}.edf".format(subject, subject, run)

    def local_data_filename(self, filename=None, subject=None, run=None):
        """Return filename for data on the local drive."""
        if filename is not None:
            filename = join(self.data_dir, *filename.split('/'))
        elif subject is not None and run is not None:
            filename = join(self.data_dir, "S{:03}".format(subject),
                            "S{:03}R{:2}.edf")
        else:
            raise ValueError('Missing input argument')
        return filename

    def download_data_file(self, filename):
        """Download data file from web site."""
        # EEG data file.
        url = URL_BASE + filename
        urlretrieve(url, self.local_data_filename(filename))

        # Event file
        url += '.event'
        local_filename = filename + '.event'
        urlretrieve(url, self.local_data_filename(local_filename))

    def download_data_files(self):
        """Download data files."""
        data_filenames = [filename.strip()
                          for filename in open(self.records_filename)]
        for filename in data_filenames:
            self.download_data_file(filename)

    def download(self):
        """Download RECORDS and data files."""
        self.download_records()
        self.download_data_files()

    def unpack_records(self, redownload=False):
        """Unpack RECORDS index file."""
        if not exists(self.records_filename) or redownload:
            self.make_dirs()
            self.download_records()

    def unpack_data_file(self, filename, redownload=False):
        """Unpack data file.

        Examples
        --------
        >>> eegmmidb = EEGMMIDB()
        >>> eegmmidb.unpack_data_file('S001/S001R01.edf')

        """
        self.unpack_records()

        if not exists(self.local_data_filename(filename)) or redownload:
            self.download_data_file(filename)

        event_filename = filename + '.event'
        if not exists(self.local_data_filename(event_filename)) or redownload:
            self.download_data_file(event_filename)

    def unpack(self, redownload=False):
        """Unpack data set from conditional download."""
        self.unpack_records()

        data_filenames = [filename.strip()
                          for filename in open(self.records_filename)]
        for filename in data_filenames:
            self.unpack_data_file(filename, redownload=redownload)

    def run(self, subject=1, run=1):
        """Read and return EEG data.

        Parameters
        ----------
        subject : int, optional
            Subject identifier from 1 to 109.
        run : int, optional
            Run identifier from 1 to 14.

        Returns
        -------
        eeg_run : brede.eeg.core.EEGRun
            Data in a EEGRun DataFrame-like object.

        Examples
        --------
        >>> eegmmidb = EEGMMIDB()
        >>> run = eegmmidb.run()
        >>> run.sampling_rate
        160.0

        >>> run.shape
        (9760, 64)

        See also
        --------
        brede.eeg.core

        """
        data_filename = EEGMMIDB.data_filename(subject=subject, run=run)

        # Make sure it is downloaded
        self.unpack_data_file(data_filename)

        filename = self.local_data_filename(data_filename)
        data = read_edf(filename)
        eeg_run = EEGRun(data, sampling_rate=SAMPLING_RATE)
        eeg_run.fix_electrode_names(inplace=True)
        return eeg_run

    def runs_for_subject(self, subject=1):
        """Return all runs for a subject.

        This will return a dictionary of (time x electrode) runs.

        The two first runs have different number of time points compared to the
        rest of the runs.

        Parameters
        ----------
        subject : int, optional
            Identifier for subject from 1 to 109

        Returns
        -------
        eeg_runs : dict of brede.eeg.core.EEGRun
            Dictionary indexed by run identifier.

        """
        eeg_runs = {run: pd.DataFrame(self.run(subject=subject, run=run))
                    for run in range(1, 15)}

        return eeg_runs

        # This cannot work as there are different number of time points
        # in the runs.
        # run x time x electrode
        # return EEGRuns(eeg_runs, minor_axis=self.run(subject=1).columns,
        #                sampling_rate=SAMPLING_RATE)


def main(args):
    """Handle command-line interface."""
    from brede.eeg.plotting import topoplot
    import matplotlib.pyplot as plt

    subject = int(args['--subject'])
    run = int(args['--run'])

    eegmmidb = EEGMMIDB()
    eeg_run = eegmmidb.run(subject=subject, run=run)
    topoplot(eeg_run, title="Subject={}, Run={}".format(subject, run))
    plt.show()


if __name__ == '__main__':
    from docopt import docopt

    main(docopt(__doc__))
