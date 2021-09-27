from unittest import TestCase

import numpy as np

from mlprimitives.adapters.keras import Sequential


class SequentialTest(TestCase):

    def test__setdefault_in_kwargs(self):
        # Setup
        sequential = Sequential(None, None, None, None)

        # Run
        kwargs = {'input_shape': [100, 1]}
        sequential._setdefault(kwargs, 'input_shape', 'whatever')

        # Assert
        assert kwargs['input_shape'] == [100, 1]

    def test__setdefault_not_in_hyperparameters(self):
        # Setup
        sequential = Sequential(None, None, None, None)

        # Run
        kwargs = dict()
        sequential._setdefault(kwargs, 'input_shape', 'whatever')

        # Assert
        assert kwargs == dict()

    def test__setdefault_not_none(self):
        # Setup
        sequential = Sequential(None, None, None, None, input_shape=[100, 1])

        # Run
        kwargs = dict()
        sequential._setdefault(kwargs, 'input_shape', 'whatever')

        # Assert
        assert kwargs == dict()

    def test__setdefault_none(self):
        # Setup
        sequential = Sequential(None, None, None, None, input_shape=None)

        # Run
        kwargs = dict()
        sequential._setdefault(kwargs, 'input_shape', [100, 1])

        # Assert
        assert kwargs == {'input_shape': [100, 1]}

    def test__augment_hyperparameters_3d_numpy(self):
        # Setup
        sequential = Sequential(None, None, None, None, input_shape=None)

        # Run
        kwargs = dict()
        X = np.array([
            [[1, 2, 3, 4],
             [1, 2, 3, 4],
             [1, 2, 3, 4]],
            [[1, 2, 3, 4],
             [1, 2, 3, 4],
             [1, 2, 3, 4]],
        ])
        Sequential._augment_hyperparameters(sequential, X, 'input', kwargs)

        # Assert
        assert kwargs == {'input_shape': (3, 4)}
