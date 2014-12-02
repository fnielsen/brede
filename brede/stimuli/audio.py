#!/usr/bin/env python
"""
audio.py.

Usage:
  audio.py makefile [options] --output <outfile>
  audio.py -h | --help
  audio.py --version

Options:
  -o <outfile>, --output <outfile>   Output filename.

"""

from __future__ import division, print_function

import math

import sys

import numpy as np
import numpy.random as npr

from scikits.audiolab import Format, Sndfile


class WaveWriter(object):

    """Writer for WAV files."""

    def __init__(self, filename, channels=1, samplerate=44100):
        """Set defaults."""
        self._format = Format('wav')
        self._filename = filename
        self._channels = channels
        self._samplerate = samplerate
        self._sampwidth = 2
        self._fid = None

    def __enter__(self):
        """Open file."""
        self._fid = Sndfile(self._filename, 'w', self._format,
                            self._channels, self._samplerate)
        return self

    def __exit__(self, exception_type, value, traceback):
        """Close file."""
        self._fid.close()

    def amplitude_modulated_noise(self, length=10.0, frequency=40.0):
        """Return amplitude modulated noise."""
        number_of_samples = int(length * self._samplerate)
        sample_points = np.linspace(0, length, number_of_samples,
                                    endpoint=False)
        amplitudes = np.sin((2 * math.pi * frequency) * sample_points)
        data = amplitudes * npr.randn(number_of_samples)
        return data

    def sinusoide(self, length=10.0, frequency=40.0):
        """Return sinusoide tone.

        Parameters
        ----------
        length : float
            Length in seconds of the sample
        frequency : float
            Frequency in Hertz of the sinusoide

        Return
        ------
        data : numpy.array
            Array with audio signal.
        """
        number_of_samples = int(length * self._samplerate)
        sample_points = np.linspace(0, length, number_of_samples,
                                    endpoint=False)
        amplitudes = np.sin((2 * math.pi * frequency) * sample_points)
        return amplitudes

    def write_amplitude_modulated_noise(self, length=10.0, frequency=40.0):
        """Write amplitude modulated noise to a file.

        Parameters
        ----------
        length : float
            Length in seconds of the sample
        frequency : float
            Frequency in Hertz of the modulation
        """
        if self._fid is not None:
            data = self.amplitude_modulated_noise(length=length,
                                                  frequency=frequency)
            self._fid.write_frames(data)

    def write_sinusoide(self, length=10.0, frequency=40.0):
        """Write sinusoide tone to file."""
        if self._fid is not None:
            data = self.sinusoide(length=length, frequency=frequency)
            self._fid.write_frames(data)


def main():
    """Read and dispatch command line arguments."""
    from docopt import docopt

    args = docopt(__doc__)

    if args['makefile']:
        filename = args['--output']
        with WaveWriter(filename) as wave_writer:
            wave_writer.write_amplitude_modulated_noise(10)
    else:
        sys.exit(1)


if __name__ == '__main__':
    main()
