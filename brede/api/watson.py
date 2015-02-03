"""Interface to IBM Watson."""


from __future__ import print_function

from ConfigParser import SafeConfigParser

from itertools import islice

import json

from os.path import expanduser

import requests

import yaml

from brede.config import config


class WatsonException(Exception):
    """Any kind of exception from Watson."""
    pass


class WatsonFailedError(WatsonException):
    """Status 'Failed' from API."""
    pass


class WatsonMissingConfig(WatsonException):
    """Exception for missing watson section in config."""
    pass


class WatsonResponse(dict):
    
    """Represent a response from the IBM Watson API."""

    def to_yaml(self):
        """Convert to YAML representation."""
        yaml.safe_dump(dict(self))

    def show(self, n=5, show_text=False):
        """Print evidence list."""
        for evidence in islice(self['question']['evidencelist'], n):
            if show_text:
                text = ' - ' + evidence['text'][:40]
            else:
                text = ''
            print('{:5} : {}{}'.format(int(float(evidence['value'])*1000),
                                     evidence['title'], 
                                     text))


class Watson(object):

    """Interface to IBM Watson.

    The use of this class requires credentials to an IBM Watson instance.

    Example
    -------
    >>> try: 
    ...     api = Watson()
    ... except WatsonMissingConfig: 
    ...     # Watson not available, so we don't test for queries.
    ...     print(True)
    ... else:
    ...     answer = api.ask('Who was called John?')
    ...     yaml = answer.to_yaml()
    ...     print(yaml)
    True

    """

    def __init__(self, user=None, password=None, url=None):
        """Setup credientials for an IBM Watson instance."""
        if user is None and password is None and url is None:
            self.check_config()
        self.user = user if user else config.get('watson', 'user')
        self.password = password if password else config.get('watson', 'password')
        self.url = url if url else config.get('watson', 'url')
        self.headers = {'Content-type': 'application/json', 
                        'Accept': 'application/json'}

    def check_config(self):
        """Check the configuration file for credentials to Watson.

        The section in the configuration file should contain user, password 
        and url:

        [watson]
        user = <username>
        password = <password>
        url = <URL to API>

        """
       
        if not config.has_section('watson'):
            raise WatsonMissingConfig("Missing [watson] section in the config file.")

    def ask(self, question, items=5):
        """Query the IBM Watson with a question.

        Parameters
        ----------
        question : str
            String with question.

        Returns
        -------
        response : WatsonResponse
            Dict-like structure.
        
        """
        data = {"question": {"questionText": question,
                             "items": items}}
        response = requests.post(self.url, 
                         headers=self.headers, 
                         data=json.dumps(data), 
                         auth=(self.user, self.password)).json()
        if response['question']['status'] == 'Failed':
            raise WatsonFailedError("'Failed' returned from IBM Watson API.")
        return WatsonResponse(response)
