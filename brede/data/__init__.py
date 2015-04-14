"""Interface to data.

An interface to various data sets is provided. Some of these data sets may be
downloaded to subdirectories in the directory ~/brede_data/. Other data sets
are read from included files in brede.

Examples
--------
>>> from brede.data import words
>>> cognitive_words = words.CognitiveWords()
>>> 'memory' in cognitive_words
True

>>> from brede.data.pubmed import Pubmed
>>> pubmed = Pubmed()
>>> medline = pubmed.get_medline(19668704)
>>> medline['TI']
'Visualizing data mining results with the brede tools.'

"""


from __future__ import absolute_import

from . import keirnaunon, neurosynth, pubmed, sbs2, wikipedia
from .bredewiki import BredeWikiTemplates
from .neurosynth import NeurosynthDatabase
from .pubmed import Pubmed
from .sbs2 import SBS2Data
from .wikipedia import WikiPage


__all__ = ('BredeWikiTemplates', 'keirnaunon',
           'neurosynth', 'NeurosynthDatabase',
           'pubmed', 'Pubmed', 'sbs2', 'SBS2Data', 'wikipedia', 'WikiPage')
