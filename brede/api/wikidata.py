#!/usr/bin/env python
"""Interface to Wikidata.

Usage:
  wikidata.py [options] [<query>]

Options:
  -h --help    Help
  --id         Return id if relevant
  --limit=<n>  Limit for the number of results [default: 1]

Example:
  $ python -m brede.api.wikidata --id "Helle Thorning"

"""

from __future__ import absolute_import, division, print_function

import json

import requests

from ..config import config


API_URL = "https://www.wikidata.org/w/api.php"


class Wikidata(object):

    """Interface to wikidata.org."""

    def __init__(self):
        """Setup credentials for an IBM Watson instance."""
        self.user_agent = config.get('requests', 'user_agent')
        self.language = 'en'

    def headers(self):
        """Return dict with header information for request."""
        return {'user-agent': self.user_agent}

    def find_entities(self, query, limit=7):
        """Return entities from a Wikidata search.

        Parameters
        ----------
        query : str
            String with query
        limit : int, optional
            Maximum number of results to return

        Returns
        -------
        entities : generator
            Generator with entities

        """
        if limit < 1:
            raise StopIteration

        params = {'action': 'wbsearchentities',
                  'language': self.language,
                  'format': 'json',
                  'limit': limit,
                  'search': query}

        index, running = 1, True
        while running:
            response = requests.get(
                API_URL, params=params,
                headers=self.headers()).json()
            entities = response['search']
            for entity in entities:
                yield entity
                if index >= limit:
                    running = False
                    break
                index += 1
            if 'search-continue' in response:
                params['continue'] = response['search-continue']
            else:
                running = False

    def find_entity(self, query):
        """Return first entity from a Wikidata search.

        Parameters
        ----------
        query : str
            Query string.

        Returns
        -------
        entity : dict
           Dictionary with entity information. An empty dict is returned if no
           entity is identified.

        """
        try:
            entity = next(self.find_entities(query, limit=1))
        except StopIteration:
            entity = {}
        return entity

    def find_entity_id(self, query):
        """Return first id of entity from a Wikidata search.

        Parameters
        ----------
        query : str
            Query for item to find.

        Returns
        -------
        id_ : str
            String with entity identifier

        Examples
        --------
        >>> wikidata = Wikidata()
        >>> id_ = wikidata.find_entity_id('Barack Obama')
        >>> str(id_)     # Python 2 returns Unicode
        'Q76'

        """
        entity = self.find_entity(query)
        id_ = entity['id']
        return id_


def main(args):
    """Handle command-line arguments."""
    limit = int(args['--limit'])

    wikidata = Wikidata()

    if limit == 1:
        entity = wikidata.find_entity(args['<query>'])
        if args['--id']:
            print(str(entity['id']))
        else:
            print(json.dumps(entity))
    else:
        for entity in wikidata.find_entities(args['<query>'], limit=limit):
            if args['--id']:
                print(str(entity['id']))
            else:
                print(json.dumps(entity))


if __name__ == '__main__':
    import docopt

    main(docopt.docopt(__doc__))
