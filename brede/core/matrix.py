"""Dataframe object."""


from __future__ import absolute_import, division, print_function

import numpy as np

from matplotlib.pyplot import matshow, pause

from pandas import DataFrame

from scipy.sparse import csr_matrix

from sklearn.decomposition import FastICA

from .vector import Vector


class Matrix(DataFrame):
    """Extended dataframe object.

    This object corresponds to a Pandas DataFrame object.

    """

    @property
    def _constructor(self):
        return Matrix

    def nans(self):
        """Return matrix with NaN of same size as original."""
        return np.nan + self._constructor(self)

    def collapse_to_two_by_two(self, first_rows, first_columns,
                               second_rows=None, second_columns=None):
        """Collapse a matrix to a two-by-two matrix.

        Elements that are merged are added together.

        Examples
        --------
        >>> A = Matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]],
        ...            index=['a', 'b', 'c'], columns=['x', 'y', 'z'])
        >>> B = A.collapse_to_two_by_two(first_rows=['a'], first_columns=['y'])
        >>> B.values
        array([[  2.,   4.],
               [ 13.,  26.]])

        """
        two_by_two = np.zeros((2, 2))

        for n, column in enumerate(self):
            if column in first_columns:
                j = 0
            elif second_columns is None:
                j = 1
            elif column in second_columns:
                j = 1
            else:
                j = -1
            for m, row in enumerate(self.index):
                if row in first_rows:
                    i = 0
                elif second_rows is None:
                    i = 1
                elif row in second_rows:
                    i = 1
                else:
                    i = -1
                if i != -1 and j != -1:
                    two_by_two[i, j] += self.iloc[m, n]
        return Matrix(two_by_two)

    def accuracy(self):
        """Compute accuracy.

        The accuracy is computed as the sum of the diagonal divided
        by the sum of the total matrix.

        Examples
        --------
        >>> matrix = Matrix([[33, 10], [15, 42]])
        >>> matrix.accuracy()
        0.75

        """
        return np.sum(np.diag(self)) / np.sum(np.sum(self))

    def ica(self, n_components=None):
        """Return result from independent component analysis.

        X = SA + m

        Sklearn's FastICA implementation is used.

        Parameters
        ----------
        n_components : int, optional
            Number of ICA components.

        Returns
        -------
        source : Matrix
            Estimated source matrix (S)
        mixing_matrix : Matrix
            Estimated mixing matrix (A)
        mean_vector : brede.core.vector.Vector
            Estimated mean vector

        References
        ----------
        http://scikit-learn.org/stable/modules/decomposition.html#ica

        """
        if n_components is None:
            n_components = int(np.ceil(np.sqrt(float(min(self.shape)) / 2)))

        ica = FastICA(n_components=n_components)
        sources = Matrix(ica.fit_transform(self.values), index=self.index)
        mixing_matrix = Matrix(ica.mixing_.T, columns=self.columns)
        mean_vector = Vector(ica.mean_, index=self.columns)

        return sources, mixing_matrix, mean_vector

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

    def matshow(self, fignum=None, **kw):
        """Plot matrix as a in a matrix-like color plot.
        
        """
        axes_image = matshow(self)
        ax = axes_image.get_axes()
        if len(self.columns) < 25:
            ax.set_xticks(range(len(self.columns)))
        ax.set_xticklabels(self.columns)

        # Necessary for update of drawing
        pause(0.001)

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
            h = np.mat(np.where(h < 10 ** -100, 0, h))
            whht = w * (h * h.T)
            xht = x * h.T

            cost_new = x2 - 2 * np.trace(w.T * xht) + np.trace(w.T * whht)
            cost_difference = np.abs(cost_old - cost_new)
            if cost_difference > small:
                cost_old = cost_new
            else:
                break

            w = np.multiply(w, xht / (whht + small))
            w = np.mat(np.where(w < 10 ** -100, 0, w))

        w, h = self._adjust_wh(w, h)

        # Convert to dataframes
        component_names = ['Component %d' % (n + 1)
                           for n in range(n_components)]
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
