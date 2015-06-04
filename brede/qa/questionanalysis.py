#!/usr/bin/env python
"""Question analysis.

Usage:
  questionanalysis.py [options] <question>

Options:
  -h --help  Help

Examples:
  python -m brede.qa.questionanalysis "Where did Uta Frith get her degree?"

"""

from __future__ import absolute_import, division, print_function

import json

from lazy import lazy

from nltk import ne_chunk, pos_tag, word_tokenize

from ..api.wikidata import Wikidata


class Question(object):

    """Represent a question and its analysis.

    Examples
    --------
    >>> question = Question("Where did Uta Frith get her degree?")

    """

    def __init__(self, question):
        """Setup question and API connections."""
        self.question = question
        self.wikidata = Wikidata()

    @lazy
    def named_entities(self):
        """Return list of named entities."""
        tree = ne_chunk(self.pos_tags)
        entities = [" ".join([text for text, label in subtree.leaves()])
                    for subtree in tree.subtrees(lambda t: t.label() == 'NE')]
        return entities

    @lazy
    def pos_tags(self):
        """Return list of part-of-speech tags in question."""
        tags = pos_tag(self.token_list)
        return tags

    @lazy
    def proper_nouns(self):
        """Return list of proper nouns in the question.

        Examples
        --------
        >>> question = Question("Where did Uta Frith get her degree?")
        >>> question.proper_nouns
        'Uta Frith'

        """
        if self.pos_tags == []:
            return []

        phrases = []
        current_phrase = []
        for word, tag in self.pos_tags:

            if tag == 'NNP':
                current_phrase.append(word)
            elif current_phrase == []:
                pass
            else:
                phrases.append(" ".join(current_phrase))
                current_phrase = []

        if self.pos_tags[-1][1] == 'NNS':
            phrases.append(current_phrase[:])
        return phrases

    @lazy
    def proper_nouns_wikidata_entities(self):
        """Return Wikidata entities for proper nouns in question.

        Examples
        --------
        >>> question = Question("What time is it?")
        >>> question.proper_nouns_wikidata_entities
        []

        >>> question = Question("Where did Uta Frith get her degree?")
        >>> entities = question.proper_nouns_wikidata_entities
        >>> str(entities[0]['id'])
        'Q8219'

        """
        entities = []
        for proper_noun in self.proper_nouns:
            entity = self.wikidata.find_entity(proper_noun)
            entities.append(entity)
        return entities

    @lazy
    def token_list(self):
        """Return list of tokens in the question."""
        tokens = word_tokenize(self.question)
        return tokens

    def to_json(self):
        """Convert data in object to a JSON string."""
        data = {
            'question': self.question,
            'named_entities': self.named_entities,
            'token_list': self.token_list,
            'pos_tags': self.pos_tags,
            'proper_nouns': self.proper_nouns,
            'proper_nouns_wikidata_entities':
                self.proper_nouns_wikidata_entities}
        return json.dumps(data)


def main(args):
    """Handle command-line interface."""
    question = Question(args['<question>'])
    print(question.to_json())


if __name__ == '__main__':
    import docopt

    main(docopt.docopt(__doc__))
