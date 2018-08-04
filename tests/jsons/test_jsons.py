# -*- coding: utf-8 -*-

import json
import os
from unittest.mock import patch

import mlblocks
from mlblocks.mlblock import MLBlock


PRIMITIVES_PATH = os.path.normpath(
    os.path.join(
        os.path.dirname(__file__),
        os.pardir,
        os.pardir,
        'jsons'
    )
)


@patch('mlblocks.PRIMITIVES_PATHS', new=[PRIMITIVES_PATH])
def test_jsons():
    """Validate jsons"""

    for primitive_filename in os.listdir(PRIMITIVES_PATH):
        primitive_path = os.path.join(PRIMITIVES_PATH, primitive_filename)
        with open(primitive_path, 'r') as f:
            primitive = json.load(f)

        primitive_name = primitive['name']
        # primitive_name = primitive.replace('.json', '')
        MLBlock(primitive_name)
