"""Panel-like object."""


from __future__ import absolute_import, division

from pandas import Panel


class Tensor(Panel):

    """Extended panel object."""

    @property
    def _constructor(self):
        return Tensor
