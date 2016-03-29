"""Interface to Wikipedia text.

Usage:
  wikipedia.py <page>

"""

from __future__ import print_function

import re

from brede.config import config

import requests


URL_WIKIPEDIA_INDEX = "https://en.wikipedia.org/w/index.php"
URL_WIKIPEDIA_API = "https://en.wikipedia.org/w/api.php"


class WikiPage(object):
    """Represent a Wikipedia page.

    Examples
    --------
    >>> wp = WikiPage('The Beatles')
    >>> html = wp.to_html()

    """

    def __init__(self, title):
        """Setup title and empty content."""
        self.title = title
        self._raw = None
        self.url_base = URL_WIKIPEDIA_INDEX
        self.setup_patterns()

    def __str__(self):
        """String representation of object."""
        return "<WikiPage(title={})>".format(self.title)

    @property
    def raw(self):
        """Raw wikitext from Wikipedia page."""
        if self._raw is None:
            self._raw = self.download()
        return self._raw

    def download(self):
        """Download page from Wikipedia.

        Returns
        -------
        text: str
            Content of page

        Examples
        --------
        >>> wp = WikiPage('The Beatles')
        >>> text = wp.download()
        >>> 'John Lennon' in text
        True

        """
        user_agent = config.get('requests', 'user_agent')
        response = requests.get(self.url_base,
                                params={'action': 'raw',
                                        'title': self.title.encode('utf-8')},
                                headers={'user-agent': user_agent})
        return response.text

    def setup_patterns(self):
        """Setup regular expression patterns for wiki markup handling."""
        self.pattern_template = re.compile(r'{{[^}]+?}}',
                                           flags=re.DOTALL | re.UNICODE)
        self.pattern_category = re.compile(r'\[\[Category:[^\]]+?\]\]',
                                           flags=re.UNICODE)
        self.pattern_quote = re.compile(r"''+", flags=re.UNICODE)
        self.pattern_ref = re.compile(r'<ref>|</ref>|<ref/>',
                                      flags=re.DOTALL | re.UNICODE)
        self.pattern_link = re.compile(r'\[\[([^\]\|]+?)\]\]',
                                       flags=re.UNICODE)
        self.pattern_link2 = re.compile(r'\[\[[^\|]+?\|([^\]]+?)\]\]',
                                        flags=re.UNICODE)
        self.pattern_h2 = re.compile(r'^==([^=]+)==\s*$',
                                     flags=re.UNICODE | re.MULTILINE)
        self.pattern_h3 = re.compile(r'^===([^=]+)===\s*$',
                                     flags=re.UNICODE | re.MULTILINE)
        self.pattern_h4 = re.compile(r'^====([^=]+)====\s*$',
                                     flags=re.UNICODE | re.MULTILINE)
        self.pattern_newlines = re.compile(r"\n\n+")
        self.pattern_table = re.compile(r"{\|.+?\|}",
                                        flags=re.DOTALL | re.UNICODE)

    def strip_markup(self):
        """Return text stripped for some wiki markup.

        Returns
        -------
        text : str
            String stripped for some of the wiki markup

        """
        text = self.pattern_template.sub('', self.raw)
        text = self.pattern_category.sub('', text)
        text = self.pattern_quote.sub('', text)
        text = self.pattern_ref.sub('', text)
        text = self.pattern_link.sub(r'\1', text)
        text = self.pattern_link2.sub(r'\1', text)
        text = self.pattern_table.sub('', text)

        # Strip more than two newlines.
        text = self.pattern_newlines.sub("\n\n", text)
        return text

    def to_html(self):
        """Convert text to HTML.

        Attempts to convert parts of the page to an HTML representation.

        Note all cases are handled and the text is not escaped.

        Returns
        -------
        html : str
            String formatted in HTML

        Examples
        --------
        >>> wiki_page = WikiPage('Love Me Do')
        >>> html = wiki_page.to_html()
        >>> '1962' in html
        True

        """
        text = self.strip_markup()

        # Handle headings
        text = self.pattern_h2.sub(r'<h2>\1</h2>', text)
        text = self.pattern_h3.sub(r'<h3>\1</h3>', text)
        text = self.pattern_h4.sub(r'<h4>\1</h4>', text)

        # TODO: entities decoded, sanitized output

        html = "<html><head><title>{}</title><head><body>".format(self.title)
        html = text
        html += "</body></html>"

        return html

    def to_text(self):
        """Convert raw wikitext to human-readable text.

        Returns
        -------
        text : str
            Text of wikitext.

        Examples
        --------
        >>> wiki_page = WikiPage('Love Me Do')
        >>> text = wiki_page.to_text()

        """
        text = self.strip_markup()

        # Handle headings
        text = self.pattern_h2.sub(r'\n\1.\n', text)
        text = self.pattern_h3.sub(r'\n\1.\n', text)
        text = self.pattern_h4.sub(r'\n\1.\n', text)

        return text


def main(args):
    """Handle command-line interface.

    Parameters
    ----------
    args : dict
        Dict in docopt format with parse input arguments

    """
    wiki_page = WikiPage(args['<page>'])
    print(wiki_page.to_html())


if __name__ == '__main__':
    from docopt import docopt

    main(docopt(__doc__))
