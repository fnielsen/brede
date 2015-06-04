"""Test brede.io.expyriment."""


from __future__ import absolute_import, division, print_function

from StringIO import StringIO

from .. import expyriment


DATA = """#Expyriment  (Revision ; Python 2.7.3), .xpd-file, coding: UTF-8
#date: Fri May 23 2014 17:14:24
#--EXPERIMENT INFO
#e mainfile: visualinstructions.py
#e sha1: 33664b
#e modules:
#e Experiment: Left right
#e no between subject factors
#e Block 0: block
#e     block factors:
#e     n trials: 40
#e     trial factors: Instruction = [Left, Relax]
#e
#--SUBJECT INFO
#s id: 1
subject_id,Instruction,Time
1,Left,1400858077.117449
1,Relax,1400858087.145365
1,Left,1400858097.154673
1,Relax,1400858107.173556
"""


def test_read_xpe():
    """Test read_xpe function."""
    df = expyriment.read_xpe(StringIO(DATA))

    assert 'subject_id' in df.columns
    assert df.ix[1400858107.173556, 'Instruction'] == 'Relax'
