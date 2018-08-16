# -*- coding: utf-8 -*-

import json
import os
from unittest.mock import patch

import pandas as pd
from mlblocks import MLPipeline
from sklearn import datasets

PRIMITIVES_PATH = os.path.normpath(
    os.path.join(
        os.path.dirname(__file__),
        os.pardir,
        'mlblocks_primitives'
    )
)


HYPERPARAMETER_DEFAULTS = {
    'int': 1,
    'float': 1.,
    'bool': True,
    'list': [],
    'dict': dict(),
}

BOSTON = datasets.load_boston()
DIGITS = datasets.load_digits()
WINE = datasets.load_wine()
DATASETS = {
    'boston': (pd.DataFrame(BOSTON.data), pd.Series(BOSTON.target)),
    'digits': (pd.DataFrame(DIGITS.data), pd.Series(DIGITS.target)),
    'wine': (pd.DataFrame(WINE.data), pd.Series(WINE.target)),
}


@patch('mlblocks.PRIMITIVES_PATHS', new=[PRIMITIVES_PATH])
def test_jsons():
    """Validate MLBlocks primitive jsons"""

    primitives = (f for f in os.listdir(PRIMITIVES_PATH) if f.endswith('.json'))
    for primitive_filename in primitives:
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

            # Run pipeline, when possible
            validation_dataset = primitive.get('validation_dataset')
            if validation_dataset:
                X, y = DATASETS[validation_dataset]
                mlpipeline.fit(X, y)
                mlpipeline.predict(X)

        except Exception:
            raise ValueError("Invalid JSON primitive: {}".format(primitive_filename))
