"""Configuration for Brede."""


from __future__ import absolute_import

try:
    import ConfigParser as configparser
except ImportError:
    import configparser

import logging

from os.path import exists, expanduser


CONFIG_FILENAMES = [
    'brede.cfg',
    '~/etc/brede.cfg',
    'brede.cfg'
    ]


logger = logging.getLogger(__name__)
logging.getLogger(__name__).addHandler(logging.NullHandler())


config = configparser.ConfigParser()
for filename in CONFIG_FILENAMES:
    full_filename = expanduser(filename)
    if exists(full_filename):
        logger.warn('Reading configuration file from {}'.format(full_filename))
        config.read(full_filename)
        break
else:
    logger.warn('No configuration file found')
