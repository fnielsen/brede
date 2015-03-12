"""Sets of words."""


from os.path import join, split


class Words(set):

    """Abstract set to load words."""

    def data_dir(self):
        """Return directory where the text files are."""
        return join(split(__file__)[0], 'words_data')

    def full_filename(self, filename):
        """Return filename with full with data directory."""
        return join(self.data_dir(), filename)

    def read_words(self, filename):
        """Return words from a file.

        Parameters
        ----------
        filename : str
            Filename without path

        Returns
        -------
        words : list of str
            List of strings with words

        Examples
        --------
        >>> words = Words()
        >>> neuroanatomy_words = words.read_words('neuroanatomy_words.txt')
        >>> 'thalamus' in neuroanatomy_words
        True

        """
        full_filename = self.full_filename(filename)
        with open(full_filename) as f:
            words = f.read().splitlines()
        return words


class NeuroanatomyWords(Words):

    """Set of neuroanatomical words and phrases.

    Examples
    --------
    >>> words = NeuroanatomyWords()
    >>> 'amygdala' in words
    True

    >>> 'inferior temporal gyrus' in words
    True

    """

    def __init__(self):
        """Read neuroanatomy_words.txt file and setup set."""
        words = self.read_words('neuroanatomy_words.txt')
        super(NeuroanatomyWords, self).__init__(words)
