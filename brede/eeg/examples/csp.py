"""brede.eeg.examples.csp - Example on common spatial patterns.

Usage:
  brede.eeg.examples.csp [options]

Options:
  -h --help  Help

"""

from __future__ import absolute_import, division, print_function

import matplotlib.pyplot as plt

import numpy as np
import numpy.random as npr

from ..csp import CSP


def csp_with_two_electrodes():
    """Example for common spatial patterns."""
    # Generate data
    N = 100
    A0 = np.array([[1, 4], [1, 1]])
    A1 = np.array([[4, 1], [1, 1]])
    y = np.zeros(N)
    indices0 = np.arange(N // 2)
    indices1 = np.arange(N // 2, N)
    y[indices1] = 1
    X = npr.randn(N, 2)
    X[indices0, :] = X[indices0, :].dot(A0)
    X[indices1, :] = X[indices1, :].dot(A1)

    plt.figure(figsize=(10, 5))

    # Plot data
    plt.subplot(1, 2, 1)
    plt.plot(X[indices0, 0], X[indices0, 1], 'or')
    plt.hold(True)
    plt.plot(X[indices1, 0], X[indices1, 1], 'xb')
    plt.axis('equal')
    plt.legend(['Class 0', 'Class 1'])
    plt.xlabel('C3')
    plt.ylabel('C4')
    plt.title('Simulated EEG data')

    # Estimate common spatial patterns
    csp = CSP()
    csp.fit(X, y)
    Z = csp.transform(X)

    # Plot estimated result
    plt.subplot(1, 2, 2)
    plt.plot(Z[indices0, 0], Z[indices0, 1], 'or')
    plt.hold(True)
    plt.plot(Z[indices1, 0], Z[indices1, 1], 'xb')
    plt.axis('equal')
    plt.legend(['Class 0', 'Class 1'])
    plt.xlabel('First latent variable')
    plt.ylabel('Second latent variable')
    plt.title('Projected EEG data')

    plt.show()


def main(args):
    """Handle command-line interface."""
    csp_with_two_electrodes()


if __name__ == '__main__':
    import docopt

    main(docopt.docopt(__doc__))
