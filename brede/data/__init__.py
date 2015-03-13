"""Interface to data."""


from brede.data import eeg, neurosynth, pubmed, wikipedia
from brede.data.bredewiki import BredeWikiTemplates
from brede.data.neurosynth import NeurosynthDatabase
from brede.data.pubmed import Pubmed
from brede.data.wikipedia import WikiPage

__all__ = ('BredeWikiTemplates', 'eeg',
           'neurosynth', 'NeurosynthDatabase',
           'pubmed', 'Pubmed', 'wikipedia', 'WikiPage')
