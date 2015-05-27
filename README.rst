Brede
=====

Question analysis
-----------------

    >>> from brede.qa.questionanalysis import Question
    >>> question = Question("Which articles has Karl Friston written about amygdala?")
    >>> url_base = 'https://wikidata.org/wiki/'
    >>> str(url_base + question.proper_nouns_wikidata_entities[0]['id'])
    http://wikidata.org/wiki/Q6371926

