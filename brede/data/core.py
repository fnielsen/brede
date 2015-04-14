"""General data set."""


from os import makedirs
from os.path import exists


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

    def make_dir(self):
        """Make data directory."""
        if not exists(self.data_dir):
            makedirs(self.data_dir)

    def download(self):
        """Download Neurosynth database file."""
        raise NotImplementedError

    def unpack(self):
        """Extract the downloaded compressed Neurosynth dump file."""
        raise NotImplementedError
