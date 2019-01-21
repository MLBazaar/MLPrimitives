# -*- coding: utf-8 -*-

import json
import os

from mlblocks import MLPipeline

from mlprimitives import MLPRIMITIVES_JSONS_PATH

HYPERPARAMETER_DEFAULTS = {
    'int': 1,
    'float': 1.,
    'bool': True,
    'list': [],
    'dict': dict(),
}


def test_jsons():
    """Validate MLBlocks primitive jsons"""

    primitives = (f for f in os.listdir(MLPRIMITIVES_JSONS_PATH) if f.endswith('.json'))
    for primitive_filename in primitives:
        try:
            primitive_path = os.path.join(MLPRIMITIVES_JSONS_PATH, primitive_filename)
            with open(primitive_path, 'r') as f:
                primitive = json.load(f)

            primitive_name = primitive['name']
            fixed_hyperparameters = primitive.get('hyperparameters', dict()).get('fixed', dict())

            init_hyperparameters = dict()
            for name, hyperparameter in fixed_hyperparameters.items():
                if 'default' not in hyperparameter:
                    type_ = hyperparameter.get('type')
                    init_hyperparameters[name] = HYPERPARAMETER_DEFAULTS.get(type_)

            block_name = primitive_name + '#1'
            mlpipeline = MLPipeline([primitive_name], {block_name: init_hyperparameters})

            # Validate methods
            mlblock = mlpipeline.blocks[block_name]
            if mlblock._class:
                fit = primitive.get('fit')
                if fit:
                    assert hasattr(mlblock.instance, fit['method'])

                produce = primitive['produce']
                assert hasattr(mlblock.instance, produce['method'])

        except Exception:
            raise ValueError("Invalid JSON primitive: {}".format(primitive_filename))
