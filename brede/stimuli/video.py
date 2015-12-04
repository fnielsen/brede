#!/usr/bin/env python
r"""
video.py.

Usage:
  video.py makefile [options] --output <outfile>
  video.py -h | --help
  video.py --version

Options:
  -o <outfile>, --output <outfile>  Output filename.
  -i <infile>, --input <infile>     Input filename(s).
  --change-frequency <freq>         Stimulus frequency in Hz [default: 6.0]
  --fps <fps>                       Frames per second [default: 30]
  --frame-height <height>           Height of frame in pixels [default: 720]
  --frame-width <width>             Width of frame in pixels [default: 480]
  --length <seconds>                Length of sample in seconds [default: 10.0]
  --stimulus-type <type>            Stimulus type [default: checkerboard]

Example:
  python -m brede.stimuli.video makefile --output \
    checkerboard_6Hz_30fps_10secs.mp4 --length=10

"""

from __future__ import division, print_function

import sys

from glob import glob

from itertools import cycle

from PIL import Image

import numpy as np

from skvideo.io import VideoWriter


class WriteError(Exception):
    """Writing error."""

    pass


class VideoFile(object):
    """General video class."""

    def __init__(self, filename, frame_size=(720, 480), fps=30):
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

    @property
    def frame_size(self):
        """Return width and height of video frame.

        Returns
        -------
        size : 2-tuple
            Width and height of frame.

        """
        return self._frame_size


class CheckerboardVideoFile(VideoFile):
    """Checkerboard stimulus video."""

    def write_frames(self, length=10, change_frequency=6.0, checker_size=48):
        """Write video frames to file.

        Parameters
        ----------
        length : float
            Length in seconds of the written frames
        change_frequency : float
            Frequency of change in the stimulus in Hz
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


class ImagesVideoFile(VideoFile):
    """Writer for video film consistenting of a series of images."""

    def absolute_scale_image(self, image, new_width=100):
        """Scale image to a fixed width within the frame.

        The image is centered in the frame.

        Parameters
        ----------
        image : numpy.ndarray
            Numpy array with color image
        new_width : int
            New width of image in pixels within the frame.

        Returns
        -------
        frame : numpy.ndarray
            Numpy array representing a frame with the scaled image at the
            center.

        """
        new_image = np.zeros((self.frame_size[1], self.frame_size[0], 3))
        new_height = int(image.shape[0] * (new_width / image.shape[1]))
        x_offset = int((new_image.shape[1] - new_width) / 2)
        y_offset = int((new_image.shape[0] - new_height) / 2)
        x_indices = np.linspace(0, image.shape[1] - 1, new_width).astype(int)
        y_indices = np.linspace(0, image.shape[0] - 1, new_height).astype(int)
        new_image[y_offset:y_offset + new_height,
                  x_offset:x_offset + new_width,
                  :] = image[y_indices, :, :][:, x_indices, :]
        return new_image

    def write_frames(self, images, length=10, change_frequency=6.0,
                     new_width=100):
        """Write video frames to file.

        Parameters
        ----------
        images : iterable
            Iterable of images (filenames).
        length : float
            Length in seconds of the written frames.
        change_frequency : float
            Frequency of change in the stimulus.

        """
        frame_change = self._fps // change_frequency
        assert frame_change == int(frame_change)

        infinite_images = cycle(images)

        # Write frames
        for frame_num in range(int(length * self._fps)):
            if frame_num % frame_change == 0:
                item = next(infinite_images)
                if type(item) == str:
                    # Assume filename
                    image = np.array(Image.open(item))
                else:
                    raise ValueError('images should be a list of strings')
                if image.ndim == 2:
                    # Convert black and white image to color
                    image = np.tile(np.reshape(
                        image, (image.shape[0], image.shape[1], 1)),
                        (1, 1, 3))
                resized_image = self.absolute_scale_image(
                    image, new_width=new_width)
            try:
                # Write frame to file
                self._video_writer.write(resized_image)
            except IndexError:
                raise WriteError('Error with file %s' % str(item))


def main(args):
    """Read and dispatch command line arguments."""
    if args['makefile']:
        filename = args['--output']
        length = float(args['--length'])
        frequency = float(args['--change-frequency'])
        stimulus_type = args['--stimulus-type']
        frame_size = (int(args['--frame-width']),
                      int(args['--frame-height']))

        if stimulus_type == 'checkerboard':
            with CheckerboardVideoFile(
                    filename, frame_size=frame_size) as video:
                video.write_frames(length, frequency)
        elif stimulus_type == 'images':
            images = glob(args['--input'])
            with ImagesVideoFile(
                    filename, frame_size=frame_size) as video:
                    video.write_frames(images, length, frequency)
        else:
            sys.exit(1)

    else:
        sys.exit(1)


if __name__ == '__main__':
    from docopt import docopt

    main(docopt(__doc__))
