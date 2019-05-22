# -*- coding: utf-8 -*-

"""Top-level package for MLPrimitives."""

__author__ = 'MIT Data To AI Lab'
__email__ = 'dailabmit@gmail.com'
__version__ = '0.1.10'

import os

from mlblocks import MLBlock

MLPRIMITIVES_JSONS_PATH = os.path.join(os.path.dirname(__file__), 'jsons')


def load_primitive(primitive, arguments=None):
    arguments = arguments or dict()

    return MLBlock(primitive, **arguments)
