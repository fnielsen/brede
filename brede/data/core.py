"""General data set."""

import errno

from os import makedirs


class Data(object):

    """Abstract class for data."""

    def __str__(self):
        """Return descriptive string."""
        string = "<" + self.__class__ + ">"
        return string

    @property
    def name(self):
        """Return short name for data."""
        raise NotImplementedError

    @property
    def description(self):
        """Return a descriptive string about the data."""
        raise NotImplementedError

    def make_dir(self, dir_name=None):
        """Make data directory."""
        if dir_name is None:
            dir_name = self.data_dir
        try:
            makedirs(dir_name)
        except OSError as exception:
            if exception.errno != errno.EEXIST:
                raise

    def download(self):
        """Download Neurosynth database file."""
        raise NotImplementedError

    def unpack(self):
        """Extract the downloaded compressed Neurosynth dump file."""
        raise NotImplementedError
