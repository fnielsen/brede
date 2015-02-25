#!/usr/bin/env python
"""
audio.py.

Usage:
  audio.py makefile [options] --output <outfile>
  audio.py -h | --help
  audio.py --version

Options:
  -o <outfile>, --output <outfile>   Output filename.
  --length <seconds>                 Length of sample in seconds [default: 10]
  --frequency <frequency>            Frequency [default: 40]
  --noise-type <noise-type>          Noise type [default: normalwhite]

"""

from __future__ import division, print_function

import math

import sys

import numpy as np
import numpy.random as npr

from scikits.audiolab import Format, Sndfile


class WaveWriterException(Exception):

    """General exception in the WaveWriter."""

    pass


class WaveWriter(object):

    """Writer for WAV files."""

    def __init__(self, filename, channels=1, samplerate=44100):
        """Set defaults."""
        self._format = Format('wav', encoding='pcm16')
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

    def amplitude_modulated_noise(self, length=10.0, frequency=40.0,
                                  noisetype='normalwhite'):
        """Return amplitude modulated noise.

        Parameters
        ----------
        length : float
            Length of the sample in seconds
        frequency : float
           Frequency in Hertz of the modulation
        noisetype : 'normalwhite' or 'uniformwhite'
            Type of random generator for the noise.

        """
        number_of_samples = int(length * self._samplerate)
        sample_points = np.linspace(0, length, number_of_samples,
                                    endpoint=False)
        amplitudes = 0.5 - 0.5 * np.cos((2 * math.pi * frequency) *
                                        sample_points)
        if noisetype == 'normalwhite':
            data = amplitudes * npr.randn(number_of_samples)
        elif noisetype == 'uniformwhite':
            data = amplitudes * (2 * npr.rand(number_of_samples) - 1)
        else:
            raise WaveWriterException('Wrong noise type')
        return data

    def silence(self, length=10.0):
        """Return silence.

        Parameters
        ----------
        length : float
            Length in seconds of the sample

        Returns
        -------
        data : numpy.array
            Array with audio signal.
        """
        number_of_samples = int(length * self._samplerate)
        return np.zeros(number_of_samples)

    def sinusoide(self, length=10.0, frequency=40.0):
        """Return sinusoide tone.

        Parameters
        ----------
        length : float
            Length in seconds of the sample
        frequency : float
            Frequency in Hertz of the sinusoide

        Returns
        -------
        data : numpy.array
            Array with audio signal.
        """
        number_of_samples = int(length * self._samplerate)
        sample_points = np.linspace(0, length, number_of_samples,
                                    endpoint=False)
        amplitudes = np.sin((2 * math.pi * frequency) * sample_points)
        return amplitudes

    def write_amplitude_modulated_noise(self, length=10.0, frequency=40.0,
                                        noisetype='normalwhite'):
        """Write amplitude modulated noise to a file.

        Parameters
        ----------
        length : float
            Length in seconds of the sample.
        frequency : float
            Frequency in Hertz of the modulation.
        noisetype : 'normalwhite' or 'uniformwhite'
            Type of random generator for the noise.

        """
        if self._fid is not None:
            data = self.amplitude_modulated_noise(length=length,
                                                  frequency=frequency,
                                                  noisetype=noisetype)
            absmax = np.max(np.abs(data))
            data /= absmax
            self._fid.write_frames(data)

    def write_sinusoide(self, length=10.0, frequency=40.0):
        """Write sinusoide tone to file.

        Parameters
        ----------
        length : float
            Length in seconds of the sample.
        frequency : float
            Frequency in Hertz of the modulation.

        """
        if self._fid is not None:
            data = self.sinusoide(length=length, frequency=frequency)
            self._fid.write_frames(data)

    def write_silence(self, length=10.0):
        """Write silence to file.

        Parameters
        ----------
        length : float
            Length in seconds of the sample.

        """
        if self._fid is not None:
            data = self.silence(length=length)
            self._fid.write_frames(data)


def main(args):
    """Read and dispatch command line arguments."""
    if args['makefile']:
        filename = args['--output']
        length = float(args['--length'])
        frequency = float(args['--frequency'])
        noisetype = args['--noise-type']
        with WaveWriter(filename) as wave_writer:
            wave_writer.write_amplitude_modulated_noise(length, frequency,
                                                        noisetype)
    else:
        sys.exit(1)


if __name__ == '__main__':
    from docopt import docopt

    main(docopt(__doc__))
