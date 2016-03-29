#!/usr/bin/env python
"""Interface to a MediaWiki instance.

Usage:
  mediawiki.py [options] <page>

Options:
  --depth=<depth>     Depth for category search [default: 1]
  -h --help           Print help
  -q --query=<query>  Query [default: raw]
  -w --wiki=<wiki>    Wiki to query [default: brede]

"""

from __future__ import print_function, unicode_literals

import sys

from brede.config import config

import requests

reload(sys)
sys.setdefaultencoding('utf-8')


URL_BASE_BREDE = "http://neuro.compute.dtu.dk/w/"
URL_BASE_WIKIPEDIA = "https://en.wikipedia.org/w/"


class WikiError(Exception):
    """Exception for misspecified wiki."""

    pass


class Wiki(object):
    """Represents an interface to a MediaWiki instance."""

    def __init__(self, wiki=None, url=None):
        """Setup URL etc. to MediaWiki API."""
        if wiki is None and url is None:
            self.wiki = 'Brede'
            self.url_base = URL_BASE_BREDE
        elif url is None:
            if wiki.lower() == 'brede':
                self.wiki = 'Brede'
                self.url_base = URL_BASE_BREDE
            elif wiki.lower() == 'wikipedia':
                self.wiki = 'Wikipedia'
                self.url_base = URL_BASE_WIKIPEDIA
            else:
                raise WikiError(("Wrong 'wiki' specified. ",
                                 "should be 'brede' or 'wikipedia'"))
        else:
            self.url = url

        self.user_agent = config.get('requests', 'user_agent')

    def page(self, title):
        """Download page from Wikipedia.

        Presently this calls download_page.

        Parameters
        ----------
        title : str
            Page title on the MediaWiki instance.

        Returns
        -------
        text: str
            Content of page

        """
        return self.download_page(title)

    def download_page(self, title):
        """Download page from Wikipedia.

        Parameters
        ----------
        title : str
            Page title on the MediaWiki instance.

        Returns
        -------
        text: str
            Content of page

        Examples
        --------
        >>> wiki = Wiki()
        >>> text = wiki.page('Amygdala')
        >>> 'basal ganglia' in text.lower()
        True

        """
        response = requests.get(self.url_base + 'index.php',
                                params={'action': 'raw',
                                        'title': title.encode('utf-8')},
                                headers={'user-agent': self.user_agent})
        return response.text

    def api(self, params):
        """Call the MediaWiki API and return structured content."""
        params['format'] = 'json'
        response = requests.get(self.url_base + 'api.php',
                                params=params,
                                headers={'user-agent': self.user_agent})
        return response.json()

    @staticmethod
    def _fix_category_title(category):
        if category.startswith('Category:'):
            return category
        return 'Category:' + category

    def category_members(self, category, category_pages=False, depth=1):
        """Return page titles for category.

        Parameters
        ----------
        category : str
            Page title of category page.
        category_pages : bool
            Determines if category pages should be returned.
        depth : int
            Depth to search in category graph.

        Returns
        -------
        members : generator
            Generator yielding strings of page titles.

        """
        if depth <= 0:
            return

        params = {'action': 'query',
                  'list': 'categorymembers',
                  'cmlimit': 500}

        subcategories = {category: depth}
        visited_categories = set()
        visited_pages = set()

        run = True
        while len(subcategories) > 0 and run:
            subcategory, depth = subcategories.popitem()
            while subcategory in visited_categories:
                if len(subcategories) < 1:
                    run = False
                    break
                subcategory, depth = subcategories.popitem()
            if not run:
                break

            params['cmtitle'] = self._fix_category_title(subcategory).\
                encode('utf-8')

            # First page
            response = self.api(params)
            for member in response['query']['categorymembers']:
                title = member['title']
                if not category_pages and not title.startswith('Category:'):
                    if title not in visited_pages:
                        yield title
                        visited_pages.add(title)
                if depth > 1 and title.startswith('Category:'):
                    subcategories[title] = depth - 1

            # Iterate if multiple pages
            while "query-continue" in response:
                params['cmcontinue'] = response['query-continue'][
                    'categorymembers']['cmcontinue']
                response = self.api(params)
                for member in response['query']['categorymembers']:
                    title = member['title']
                    if not category_pages and \
                            not title.startswith('Category:'):
                        if title not in visited_pages:
                            yield title
                            visited_pages.add(title)
                    if depth > 1 and title.startswith('Category:'):
                        subcategories[title] = depth - 1

            visited_categories.add(subcategory)


def main(args):
    """Handle command-line interface."""
    wiki = Wiki(wiki=args['--wiki'])

    if args['--query'] == 'raw':
        print(wiki.page(args['<page>']))
    elif args['--query'] == 'categorymembers':
        members = wiki.category_members(args['<page>'],
                                        depth=int(args['--depth']))
        for member in members:
            print(member)


if __name__ == '__main__':
    from docopt import docopt

    main(docopt(__doc__))
