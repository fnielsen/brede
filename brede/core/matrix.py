"""Dataframe object."""


from __future__ import division

import numpy as np

from pandas import DataFrame

from scipy.sparse import csr_matrix


class Matrix(DataFrame):

    """Extended dataframe object.

    This object corresponds to a Pandas dataframe object.

    """

    @property
    def _constructor(self):
        return Matrix

    def idf(self):
        """Return scaled version with inverse document frequency.

        Returns
        -------
        scaled : Matrix
            Matrix with IDF scaling

        Examples
        --------
        >>> matrix = Matrix([[2, 0, 1], [1, 0, 0]])
        >>> scaled = matrix.idf()
        >>> scaled.iloc[0, 0]
        0.0

        >>> scaled.iloc[0, 1]
        0.0

        >>> from math import log
        >>> scaled.iloc[0, 2] == log(2)
        True

        References
        ----------
        https://en.wikipedia.org/wiki/Tfidf

        """
        # Total number of documents
        total = self.shape[0]

        # Number of documents for term t
        nt = (self != 0).sum()

        # Ordinary idf-scaling
        term_weights = np.log(total / nt)

        # If a term is not present in any document
        term_weights[np.isinf(term_weights)] = 0.0

        scaled = self * term_weights
        return scaled

    def nmf(self, n_components=None, tol=None, max_iter=200,
            random_state=None):
        """Return non-negative factorization matrices.

        Parameters
        ----------
        n_components : int, optional
            Number of components.

        Returns
        -------
        w : Matrix
            Left factorized matrix
        h : Matrix
            Right factorized matrix

        Examples
        --------
        >>> matrix = Matrix([[1, 1, 0], [0, 0, 1]])
        >>> w, h = matrix.nmf(n_components=2)
        >>> round(h.ix[0, 2], 2)
        0.0
        >>> round(h.ix[1, 2], 2)
        1.0

        >>> round(w.ix[0, 0], 2)
        1.41
        >>> round(h.ix[0, 0], 2)
        0.71

        """
        if n_components is None:
            n_components = int(np.ceil(np.sqrt(float(min(self.shape)) / 2)))

        small = 1000 * self.max().max() * np.finfo(float).eps * \
            np.sqrt(np.max(self.shape))

        if tol is None:
            tol = small

        # Initialize factorized matrices
        if random_state is not None:
            np.random.seed(random_state)
        w = np.mat(np.random.rand(self.shape[0], n_components))
        h = np.mat(np.random.rand(n_components, self.shape[1]))

        # Make sparse
        x = csr_matrix(self)

        x2 = x.multiply(x).sum()
        cost_old = np.finfo(float).max
        for n in range(0, max_iter):
            h = np.multiply(h, (w.T * x) / (w.T * w * h + small))
            h = np.mat(np.where(h < 10**-100, 0, h))
            whht = w * (h * h.T)
            xht = x * h.T

            cost_new = x2 - 2 * np.trace(w.T * xht) + np.trace(w.T * whht)
            cost_difference = np.abs(cost_old - cost_new)
            if cost_difference > small:
                cost_old = cost_new
            else:
                break

            w = np.multiply(w, xht / (whht + small))
            w = np.mat(np.where(w < 10**-100, 0, w))

        w, h = self._adjust_wh(w, h)

        # Convert to dataframes
        component_names = ['Component %d' % (n+1) for n in range(n_components)]
        w = Matrix(w, index=self.index, columns=component_names)
        h = Matrix(h, index=component_names, columns=self.columns)

        return w, h

    def _adjust_wh(self, w, h):
        """Change scaling of W and H and reorder them according to scaling.

        A helper function to nmf.

        """
        wsum = np.asarray(w.sum(axis=0)).flatten()
        hsum = np.asarray(h.sum(axis=1)).flatten()
        whsum = wsum * hsum
        whsumsqrt = np.sqrt(whsum)
        indices = np.argsort(-whsum)
        d_w = np.mat(np.diag(whsumsqrt[indices] / wsum[indices]))
        d_h = np.mat(np.diag(whsumsqrt[indices] / hsum[indices]))
        return w[:, indices] * d_w, d_h * h[indices, :]

    def wta(self, axis=1):
        """Return winner-take-all version.

        With axis=1 each row will be examined independently to find the maximum
        value. That element will be retained. The other elements in the row
        will be zeroed.

        Examples
        --------
        >>> matrix = Matrix([[2, 1], [4, 3]])
        >>> wta = matrix.wta()
        >>> wta.ix[0, 0]
        2
        >>> wta.ix[0, 1]
        0
        >>> wta.ix[1, 0]
        4
        >>> wta.ix[1, 1]
        0

        >>> wta = matrix.wta(axis=0)
        >>> wta.ix[0, 0]
        0
        >>> wta.ix[1, 0]
        4
        >>> wta.ix[1, 1]
        3

        """
        if axis == 1:
            wta = np.where(self == np.tile(self.max(axis=1).values,
                                           (self.shape[1], 1)).T,
                           self, 0)
        elif axis == 0:
            wta = np.where(self == np.tile(self.max(axis=0).values,
                                           (self.shape[0], 1)),
                           self, 0)
        return Matrix(wta, index=self.index, columns=self.columns)