# -*- coding: utf-8 -*-

import pytest
from mlblocks import MLBlock
from mlblocks.discovery import _PRIMITIVES_PATHS, find_primitives, load_primitive

# Remove the $(cwd)/mlprimitives path from discovery
_PRIMITIVES_PATHS.pop(0)

HYPERPARAM_DEFAULTS = {
    'int': 1,
    'float': 1.,
    'bool': True,
    'list': [],
    'dict': dict(),
}


def get_init_params(metadata):
    fixed_hyperparams = metadata.get('hyperparameters', dict()).get('fixed', dict())
    init_params = dict()
    for name, hyperparameter in fixed_hyperparams.items():
        if 'default' not in hyperparameter:
            type_ = hyperparameter.get('type')
            init_params[name] = HYPERPARAM_DEFAULTS.get(type_)

    return init_params


@pytest.mark.parametrize("primitive_name", find_primitives())
def test_primitive(primitive_name):
    metadata = load_primitive(primitive_name)
    init_params = get_init_params(metadata)
    mlblock = MLBlock(primitive_name, **init_params)

    if mlblock._class:
        fit = metadata.get('fit')
        if fit:
            assert hasattr(mlblock.instance, fit['method'])

        produce = metadata['produce']
        assert hasattr(mlblock.instance, produce['method'])
