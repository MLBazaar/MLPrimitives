# -*- coding: utf-8 -*-

import json
import os
from unittest.mock import patch

from mlblocks import MLBlock

PRIMITIVES_PATH = os.path.normpath(
    os.path.join(
        os.path.dirname(__file__),
        os.pardir,
        os.pardir,
        'jsons'
    )
)


HYPERPARAMETER_DEFAULTS = {
    'int': 1,
    'float': 1.,
    'bool': True,
    'list': [],
    'dict': dict(),
}


@patch('mlblocks.PRIMITIVES_PATHS', new=[PRIMITIVES_PATH])
def test_jsons():
    """Validate jsons"""

    for primitive_filename in os.listdir(PRIMITIVES_PATH):
        try:
            primitive_path = os.path.join(PRIMITIVES_PATH, primitive_filename)
            with open(primitive_path, 'r') as f:
                primitive = json.load(f)

            primitive_name = primitive['name']
            fixed_hyperparameters = primitive.get('hyperparameters', dict()).get('fixed', dict())

            init_hyperparameters = dict()
            for name, hyperparameter in fixed_hyperparameters.items():
                if 'default' not in hyperparameter:
                    type_ = hyperparameter.get('type')
                    init_hyperparameters[name] = HYPERPARAMETER_DEFAULTS.get(type_)

            MLBlock(primitive_name, **init_hyperparameters)

        except Exception:
            raise ValueError("Invalid JSON primitive: {}".format(primitive_filename))
