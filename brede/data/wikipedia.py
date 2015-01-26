"""Interface to Wikipedia text."""

import re

import requests

from brede.config import config


WIKIPEDIA_INDEX = "https://en.wikipedia.org/w/index.php"


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
        self.url_base = WIKIPEDIA_INDEX

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

        Parameters
        ----------
        title : str
            Page title on English Wikipedia

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

    def to_html(self):
        """Convert text to HTML.

        Attempts to convert parts of the page to a HTML representation.

        Note all cases are handled and the text is not escaped.

        """
        pattern_template = re.compile(r'{{[^}]+}}',
                                      flags=re.DOTALL | re.UNICODE)
        pattern_category = re.compile(r'\[\[Category:[^\]]+\]\]',
                                      flags=re.UNICODE)
        pattern_quote = re.compile(r"''+", flags=re.UNICODE)
        pattern_ref = re.compile(r'<ref>|</ref>|<ref/>',
                                 flags=re.DOTALL | re.UNICODE)
        pattern_link = re.compile(r'\[\[([^\]\|]+)\]\]',
                                  flags=re.UNICODE)
        pattern_link2 = re.compile(r'\[\[[^\|]+\|([^\]]+)\]\]',
                                   flags=re.UNICODE)

        text = pattern_template.sub('', self.raw)
        text = pattern_category.sub('', text)
        text = pattern_quote.sub('', text)
        text = pattern_ref.sub('', text)
        text = pattern_link.sub(r'\1', text)
        text = pattern_link2.sub(r'\1', text)

        pattern_h2 = re.compile(r'^==([^=]+)==\s*$', flags=re.UNICODE)
        pattern_h3 = re.compile(r'^===([^=]+)===\s*$', flags=re.UNICODE)
        pattern_h4 = re.compile(r'^====([^=]+)====\s*$', flags=re.UNICODE)
        text = pattern_h2.sub(r'<h2>\1</h2>', text)
        text = pattern_h3.sub(r'<h3>\1</h3>', text)
        text = pattern_h4.sub(r'<h4>\1</h4>', text)

        html = "<html><head><title>{}</title><head><body>"
        html = text
        html += "</body></html>"

        return html
