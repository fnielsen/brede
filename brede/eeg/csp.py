"""brede.eeg.csp - common spatial patterns.

Usage:
  brede.eeg.csp [options] [<file>]

Options:
  -h --help  Help

"""

from __future__ import absolute_import, division, print_function

import numpy as np

from scipy.linalg import eig

from sklearn import base


class CSP(base.BaseEstimator, base.TransformerMixin):

    """Common spatial patterns.

    References
    ----------
    EEGTools
    https://github.com/breuderink/eegtools/
       blob/master/examples/ex_csp_motor_imagery.py

    """

    def class_correlations(self, X, y):
        """Return list of class correlations."""
        class_correlations = [
            np.corrcoef(X[y == class_indicator, :], rowvar=0)
            for class_indicator in np.unique(y)]
        return class_correlations

    def class_covariances(self, X, y):
        """Return list of class covariances."""
        class_covariances = [
            np.cov(X[y == class_indicator, :], rowvar=0)
            for class_indicator in np.unique(y)]
        return class_covariances

    def fit(self, X, y):
        """Fit common spatial patterns.

        Projection with the generalized eigenvalue problem

        eig(class_covariance0, sum(covariances))

        The weights are ordered so the eigenvectors associated with
        the largest eigenvalue is first.

        Parameters
        ----------
        X : array_like
            Data matrix (time points x channels)
        y : 1D array_like
            Vector with class indicator variables

        Returns
        -------
        self : CSP
            The self object

        References
        ----------
        Christian Andreas Kothe,  Lecture 7.3 Common Spatial Patterns
        https://www.youtube.com/watch?v=zsOULC16USU

        EEGTools-like
        https://github.com/breuderink/eegtools/
        blob/master/examples/ex_csp_motor_imagery.py

        Common spatial pattern
        https://en.wikipedia.org/wiki/Common_spatial_pattern

        """
        # Generalized eigenvalue problem on the class covariances
        class_covariances = self.class_covariances(X, y)
        total_covariance = sum(class_covariances)
        eigenvalues, eigenvectors = eig(class_covariances[0],
                                        total_covariance)

        # Reorder data
        eigenvalues = np.real(eigenvalues)
        indices = np.argsort(-eigenvalues)
        eigenvalues = eigenvalues[indices]
        eigenvectors = eigenvectors[:, indices]

        # The model parameters
        self.weights_ = eigenvectors

        return self

    def transform(self, X):
        """Project data matrix with CSP.

        Parameters
        ----------
        X : array_like
            Data matrix

        Returns
        -------
        Z : array_like
            Projected data matrix (time points x latent variables)

        """
        return X.dot(self.weights_)


def main(args):
    """Handle command-line script."""
    csp = CSP()
    print(csp)


if __name__ == '__main__':
    import docopt

    main(docopt.docopt(__doc__))
