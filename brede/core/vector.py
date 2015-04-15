"""Series-like object."""


from __future__ import absolute_import, division

from pandas import Series


class Vector(Series):

    """Extended series object."""

    @property
    def _constructor(self):
        return Vector
