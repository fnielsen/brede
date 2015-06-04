"""Watson interface.

Usage:
  brede.api.watson [options] [--json|--yaml] <question>

Options:
  -h --help                    Show this screen.
  -i <items>, --items <items>  Items to return [default: 5].
  -j, --json                   Output in JSON.
  --yaml                       Output in YAML

Description:
  The use of this program requires credentials to an IBM Watson instance which
  should be included in a configuration file (read by brede.config).

Example:
  $ python -m brede.api.watson "Do you have any documents in your corpus?"

Watson is a trademark of IBM.

"""


from __future__ import print_function

import json

from itertools import islice

from brede.config import config

import requests

import yaml


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

    @property
    def evidencelist(self):
        """Return the evidencelist field from the JSON response."""
        return self['question']['evidencelist']

    def retrieval_rank(self, title):
        """Return retrieval rank for a given document title.

        If the title is not found 'inf' (floating point) is returned.

        Parameters
        ----------
        title : str
            Ground truth title of the document.

        Returns
        -------
        rank : float
            Rank of document.

        Examples
        --------
        >>> resp = {'question': {'evidencelist': [{'title': 'a'},
        ...                                       {'title': 'b'}]}}
        >>> watson_response = WatsonResponse(resp)
        >>> watson_response.retrieval_rank('b')
        2.0

        """
        rank = float('inf')
        for n, evidence in enumerate(self.evidencelist, start=1):
            # With Watson 2.24 the evidence list may have empty dictionary
            # https://developer.ibm.com/answers/questions/182196/ ...
            # missing-data-in-item-in-the-evidencelist-returned/
            if 'title' in evidence and title == evidence['title']:
                rank = float(n)
                break
        return rank

    def to_json(self):
        """Convert data to JSON representation.

        Returns
        -------
        json : str
            String in JSON representation.

        """
        return json.dumps(dict(self))

    def to_yaml(self):
        """Convert to YAML representation."""
        return yaml.safe_dump(dict(self))

    def show(self, n=5, show_text=False):
        """Print evidence list."""
        for evidence in islice(self['question']['evidencelist'], n):
            if not evidence:
                # The API (2.24 probably) can apparent return an empty evidence
                # in the first returned list item of the evidencelist.
                print('MISSING')
                continue
            if show_text:
                text = ' - ' + evidence['text'][:40]
            else:
                text = ''
            print('{:5} : {}{}'.format(int(float(evidence['value']) * 1000),
                                       evidence['title'],
                                       text))


class Watson(object):

    """Interface to IBM Watson.

    The use of this class requires credentials to an IBM Watson instance.

    User, password and API URL are read from the brede.config if it is not
    specified.

    Example
    -------
    >>> try:
    ...     api = Watson()
    ... except WatsonMissingConfig:
    ...     # Watson not available, so we don't test for queries.
    ...     print(True)
    ... else:
    ...     answer = api.ask('Who was called John?')
    ...     'question' in answer
    True

    """

    def __init__(self, user=None, password=None, url=None):
        """Setup credentials for an IBM Watson instance."""
        if user is None and password is None and url is None:
            self.check_config()
        self.user = user or config.get('watson', 'user')
        self.password = password or config.get('watson', 'password')
        self.url = url or config.get('watson', 'url')
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
            message = "Missing [watson] section in the config file."
            raise WatsonMissingConfig(message)

    def ask(self, question, items=None):
        """Query the IBM Watson with a question.

        Communicates with the IBM Watson API by sending a query formed in JSON
        and parsing the returned JSON answer.

        Items should be between 1 and 10 according to the documentation,
        but if items are not send to the API the response may contain more than
        10 items!

        Parameters
        ----------
        question : str
            String with question.
        items : int
            Number of items (answers) that the API should return,.

        Returns
        -------
        response : WatsonResponse
            Dict-like structure.

        Raises
        ------
        err : WatsonException
            May be raise, e.g., with word in question too long.

        References
        ----------
        http://www.ibm.com/smarterplanet/us/en/ibmwatson/developercloud/apis/

        """
        # Items should be between 1 and 10, but if not send to the API
        # the response may contain more than 10 items!?
        if items is None or items > 10:
            # Only 'questionText' seems to be required
            data = {"question": {"questionText": question}}
        else:
            data = {"question": {"questionText": question,
                                 "items": items,
                                 "evidenceRequest": {"items": items,
                                                     "profile": "no"}}}
        response = requests.post(self.url + '/question',
                                 headers=self.headers,
                                 data=json.dumps(data),
                                 auth=(self.user, self.password))
        try:
            response.raise_for_status()
        except requests.exceptions.HTTPError:
            response_data = response.json()
            raise WatsonException(response_data['message'])
        except:
            raise
        response_data = response.json()
        if response_data['question']['status'] == 'Failed':
            raise WatsonFailedError("'Failed' returned from IBM Watson API.")
        return WatsonResponse(response_data)

    def _ping(self):
        """Ping the Watson service.

        This apparently does not work.

        References
        ----------
        http://www.ibm.com/smarterplanet/us/en/ibmwatson/developercloud/apis/

        """
        response = requests.get(self.url + '/ping',
                                headers=self.headers,
                                auth=(self.user, self.password))
        if response.status_code == 200:
            return True
        return False

    def _services(self):
        """Return services.

        This apparently does not work.

        References
        ----------
        http://www.ibm.com/smarterplanet/us/en/ibmwatson/developercloud/apis/

        """
        response = requests.get(self.url + '/services',
                                headers=self.headers,
                                auth=(self.user, self.password))
        return response.json()


def main(args):
    """Handle command-line interface."""
    watson = Watson()
    answer = watson.ask(args['<question>'], items=int(args['--items']))
    if args['--json']:
        print(answer.to_json())
    elif args['--yaml']:
        print(answer.to_yaml())
    else:
        answer.show(n=int(args['--items']))


if __name__ == '__main__':
    from docopt import docopt

    main(docopt(__doc__))
