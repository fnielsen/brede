Brede
=====

Question analysis
-----------------

    >>> from brede.qa.questionanalysis import Question
    >>> question = Question("Which articles has Karl Friston written about amygdala?")
    >>> url_base = 'https://wikidata.org/wiki/'
    >>> str(url_base + question.proper_nouns_wikidata_entities[0]['id'])
    http://wikidata.org/wiki/Q6371926


EEG data
--------

    >>> from brede.data.eegmmidb import EEGMMIDB
    >>> eegmmidb = EEGMMIDB()
    >>> eegmmidb.name 
    'EEG Motor Movement/Imagery Dataset'

Load some data from the first subject:

    >>> eyes_open = eegmmidb.run(run=1)
    >>> eyes_open['state'] = 'Eyes open'

Plot the EEG time series in a topoplot:

    >>> import matplotlib.pyplot as plt
    >>> from brede.eeg.plotting import topoplot
    >>> topoplot(eyes_open)
    >>> plt.show()

Load more data:

    >>> move_fist = eegmmidb.run(run=3)
    >>> move_fist['state'] = 'Move fist'

Installation
------------
Brede has a dependency on Mayavi2 which may be difficult to install on Python3.
As of 2017, Brede has only been running on Python2.7
