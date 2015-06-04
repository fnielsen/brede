"""Test of electrode submodule."""


from __future__ import absolute_import, division, print_function

from StringIO import StringIO

from .. import electrode


FILE = """
Time,Gyro X,Gyro Y,F7,F8,AF3,AF4,FC5,FC6,F3,F4,O1,O2,P7,P8
1400858061.78,1649,1719,9414,8155,8112,8610,8700,9168,9088,9237,8202,9755,8102
1400858061.78,1650,1719,9421,8159,8129,8636,8699,9176,9099,9235,8225,9765,8091
1400858061.79,1650,1719,9412,8161,8117,8642,8693,9173,9080,9228,8213,9750,8071
"""


def test_read_csv():
    """Test read_csv."""
    run = electrode.read_csv(StringIO(FILE))
    assert isinstance(run, electrode.EEGAuxElectrodeRun)
