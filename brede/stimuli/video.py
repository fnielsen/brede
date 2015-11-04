#!/usr/bin/env python
r"""
video.py.

Usage:
  video.py makefile [options] --output <outfile>
  video.py -h | --help
  video.py --version

Options:
  -o <outfile>, --output <outfile>  Output filename.
  --length <seconds>                Length of sample in seconds [default: 10.0]
  --fps <fps>                       Frames per second [default: 30]
  --stimulus-type <type>            Stimulus type [default: checkerboard]
  --change-frequency <freq>         Stimulus frequency in Hz [default: 6.0]

Example:
  python -m brede.stimuli.video makefile --output \
    checkerboard_6Hz_30fps_10secs.mp4 --length=10

"""

from __future__ import division, print_function

import sys

import numpy as np

from skvideo.io import VideoWriter


class Video(object):
    """General video class."""

    pass


class CheckerboardVideo(Video):
    """Checkerboard stimulus video."""

    def __init__(self, filename, frame_size=(768, 480), fps=30):
        """Setup variables.

        Parameters
        ----------
        filename : str
            Filename for the output video file
        frame_size : (int, int)
            Pixel width and height of the video
        fps : float
            Frames per seconds

        """
        self._filename = filename
        self._frame_size = frame_size
        self._fps = fps

    def __enter__(self):
        """Open file."""
        self._video_writer = VideoWriter(
            self._filename, frameSize=self._frame_size, fps=self._fps)
        self._video_writer.open()
        return self

    def __exit__(self, exception_type, value, traceback):
        """Close file."""
        self._video_writer.release()

    def write_frames(self, length=10, change_frequency=6.0, checker_size=32):
        """Write video frames to file.

        Parameters
        ----------
        length : float
            Length in seconds of the written frames
        change_frequency : float
            Frequency of change in the stimulus
        checker_size : int
            Number of pixels for each checker field

        """
        # Prepare image
        checkerboard = np.tile(
            np.kron(np.array([[0, 1], [1, 0]]),
                    np.ones((checker_size, checker_size))),
            (checker_size, checker_size))
        checkerboard = checkerboard[:self._frame_size[1], :self._frame_size[0]]
        image = np.tile(checkerboard[:, :, np.newaxis] * 255, (1, 1, 3))

        frame_change = self._fps // change_frequency
        assert frame_change == int(frame_change)

        # Write frames
        for frame_num in range(int(length * self._fps)):
            if frame_num % frame_change == 0:
                image = 255 - image
            self._video_writer.write(image)


def main(args):
    """Read and dispatch command line arguments."""
    if args['makefile']:
        filename = args['--output']
        length = float(args['--length'])
        frequency = float(args['--change-frequency'])
        stimulus_type = args['--stimulus-type']

        if stimulus_type == 'checkerboard':
            with CheckerboardVideo(filename) as video:
                video.write_frames(length, frequency)
        else:
            sys.exit(1)

    else:
        sys.exit(1)


if __name__ == '__main__':
    from docopt import docopt

    main(docopt(__doc__))
