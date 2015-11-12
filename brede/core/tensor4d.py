"""Panel4-like object."""


from __future__ import absolute_import, division

from pandas import Panel4D


class Tensor4D(Panel4D):
    """Extended panel4D object."""

    @property
    def _constructor(self):
        return type(self)
