"""Filters.

Usage:
  brede.eeg.filter [options]

Options:
  -h --help  Help

"""

from __future__ import absolute_import, division, print_function

from scipy.signal import butter


def lowpass_filter_coefficients(cutoff_frequency, sampling_rate, order=5):
    """Return lowpass filter coefficients.

    Parameters
    ----------
    cutoff_frequency : float
        Frequency in Hertz
    sampling_rate : float
        Frequency in Hertz
    order : int, optional
        Order of filter

    Returns
    -------
    b : numpy.ndarray
        Filter coefficients
    a : numpy.ndarray
        Filter coefficients

    """
    nyqvist = 0.5 * sampling_rate
    Wn = cutoff_frequency / nyqvist
    b, a = butter(order, Wn, btype='low')
    return b, a


def bandpass_filter_coefficients(low_cutoff_frequency, high_cutoff_frequency,
                                 sampling_rate, order=5):
    """Return bandpass filter coefficients.

    Parameters
    ----------
    low_cutoff_frequency : float
        Frequency in Hertz
    high_cutoff_frequency : float
        Frequency in Hertz
    sampling_rate : float
        Frequency in Hertz
    order : int, optional
        Order of filter

    Returns
    -------
    b : numpy.ndarray
        Filter coefficients
    a : numpy.ndarray
        Filter coefficients

    References
    ----------
    http://wiki.scipy.org/Cookbook/ButterworthBandpass

    """
    nyqvist = 0.5 * sampling_rate
    low = low_cutoff_frequency / nyqvist
    high = high_cutoff_frequency / nyqvist
    b, a = butter(order, [low, high], btype='band')
    return b, a


def main(args):
    """Handle command-line interface."""
    b, a = lowpass_filter_coefficients(2.0, 160)
    print(b)


if __name__ == "__main__":
    from docopt import docopt

    main(docopt(__doc__))
