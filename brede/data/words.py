r"""Sets of words.

Usage:
  brede.data.words [options] <set>

Options:
  -h --help             Help
  -s=<sep> --sep=<sep>  Separator for output [default: \n]

The set can be 'neuroanatomy', 'cognitive' or 'neuroimagingmethod'.

"""


from __future__ import print_function

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
        words = [word.strip() for word in words]
        return words

    def join(self, sep=u'\n'):
        """Join words together to one string."""
        return sep.join(self)


class CognitiveWords(Words):

    """Set of cognitive words and phrases.

    Examples
    --------
    >>> words = CognitiveWords()
    >>> 'memory' in words
    True

    >>> 'mri' in words
    False

    """

    def __init__(self):
        """Read cognitive_words.txt file and setup set."""
        words = self.read_words('cognitive_words.txt')
        super(CognitiveWords, self).__init__(words)


class NeuroimagingMethodWords(Words):

    """Set of neuroimaging method words and phrases.

    Examples
    --------
    >>> words = NeuroimagingMethodWords()
    >>> 'positron' in words
    True

    """

    def __init__(self):
        """Read neuroimaging_method_words.txt file and setup set."""
        words = self.read_words('neuroimaging_method_words.txt')
        super(NeuroimagingMethodWords, self).__init__(words)


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


def main(args):
    """Handle command-line interface."""
    word_set = args['<set>']
    sep = args['--sep']
    if word_set == 'cognitive':
        words = CognitiveWords()
    elif word_set == 'neuroanatomy':
        words = NeuroanatomyWords()
    elif word_set == 'neuroimagingmethod':
        words = NeuroimagingMethodWords()

    if sep in (r'\n', r'\t'):
        sep = sep.decode('string_escape')
    print(words.join(sep=sep))


if __name__ == "__main__":
    from docopt import docopt

    main(docopt(__doc__))
