r"""Sets of words.

Usage:
  brede.data.words [options] <set>

Options:
  -h --help             Help
  -s=<sep> --sep=<sep>  Separator for output [default: \n]

The set can be 'neuroanatomy', 'cognitive' or 'neuroimagingmethod'.


Examples
--------
>>> neuroanatomy_words = NeuroanatomyWords()
>>> text = 'Lesions were found in the left amygdala and anterior insula.'
>>> neuroanatomy_words.find_all(text)
['left amygdala', 'anterior insula']

"""


from __future__ import print_function

import re

from os.path import join, split


class Words(set):

    """Abstract set to load words."""

    def __init__(self, *args, **kwargs):
        """Set up set and pattern."""
        super(Words, self).__init__(*args, **kwargs)
        self._pattern = None

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

        # Strip tailing whitespace - if any
        words = [word.strip() for word in words]

        # Drop empty lines
        words = [word for word in words if not word == '']

        return words

    def join(self, sep=u'\n'):
        """Join words together to one string."""
        return sep.join(self)

    def pattern(self):
        """Return compiled pattern for regular expression match.

        Returns
        -------
        pattern : _sre.SRE_Pattern
            Compiled pattern to match to a string.

        """
        if self._pattern is None:
            words = list(self)

            # The longest words are first in the list
            words.sort(key=lambda word: len(word), reverse=True)

            # Some words might contain parentheses
            words = [re.escape(word) for word in words]

            # Setup compiled pattern
            self._pattern = re.compile(r"\b(" + "|".join(words) + r")\b")

        return self._pattern

    def find_all(self, text, clean_whitespace=True):
        """Find all words in a text.

        The text is automatically lower-cased.

        A simple regular expression match is used.

        Parameters
        ----------
        text : str
            String with text where words are to be found.

        Returns
        -------
        words : list of str
            List of words

        """
        if clean_whitespace:
            text = re.sub(r"\s+", " ", text)
        words = self.pattern().findall(text.lower())
        return words


class CognitiveWords(Words):

    """Set of cognitive words and phrases.

    The cognitive words contain both dashes and parentheses.

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


class NeuroanatomyWords(Words):

    r"""Set of neuroanatomical words and phrases.

    Examples
    --------
    >>> words = NeuroanatomyWords()
    >>> 'amygdala' in words
    True

    >>> 'inferior temporal gyrus' in words
    True

    >>> text = 'Lesions were found in the left\namygdala and insula.'
    >>> words.find_all(text)
    ['left amygdala', 'insula']

    >>> words.find_all(text, clean_whitespace=False)
    ['left', 'amygdala', 'insula']

    """

    def __init__(self):
        """Read neuroanatomy_words.txt file and setup set."""
        words = self.read_words('neuroanatomy_words.txt')
        super(NeuroanatomyWords, self).__init__(words)


class NeurodisorderWords(Words):

    """Set of neuro- and psychiatry disorder and condition words and phrases.

    Examples
    --------
    >>> words = NeurodisorderWords()
    >>> 'alzheimer disease' in words
    True

    >>> 'inferior temporal gyrus' in words
    False

    """

    def __init__(self):
        """Read neurodisorder_words.txt file and setup set."""
        words = self.read_words('neurodisorder_words.txt')
        super(NeurodisorderWords, self).__init__(words)


class NeuroimagingMethodWords(Words):

    """Set of neuroimaging method words and phrases.

    Examples
    --------
    >>> words = NeuroimagingMethodWords()
    >>> 'positron' in words
    True

    >>> 'amygdala' in words
    False

    """

    def __init__(self):
        """Read neuroimaging_method_words.txt file and setup set."""
        words = self.read_words('neuroimaging_method_words.txt')
        super(NeuroimagingMethodWords, self).__init__(words)


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
