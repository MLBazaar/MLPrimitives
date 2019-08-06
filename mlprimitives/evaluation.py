# -*- coding: utf-8 -*-

"""
MLPrimitives Evaluation functions.

Collection of functions and tools to evaluate the performance
of a pipeline over a given dataset.
"""

import json
import logging
from copy import copy

import numpy as np
from mlblocks import MLPipeline
from sklearn import metrics

from mlprimitives.datasets import load_dataset

LOGGER = logging.getLogger(__name__)


def get_value(dataset, value):
    if isinstance(value, str) and value.startswith('$'):
        value = getattr(dataset, value[1:])
    elif isinstance(value, dict):
        value = get_context(dataset, value)
    elif isinstance(value, list):
        value = [get_value(dataset, v) for v in value]

    return copy(value)


def get_context(dataset, context_spec):
    context = dict()
    for key, value in context_spec.items():
        context[key] = get_value(dataset, value)

    return context


def get_scorer(name, kwargs):
    metric = getattr(metrics, name, None)
    if not metric:
        raise ValueError('Unknown metric: "{}"'.format(name))

    def scorer(obs, exp):
        return metric(obs, exp, **kwargs)

    return scorer


def score_pipeline(pipeline_metadata, n_splits=5, random_state=0, dataset=None):
    if isinstance(pipeline_metadata, str):
        LOGGER.info('Loading pipeline %s', pipeline_metadata)
        with open(pipeline_metadata, 'r') as pipeline_file:
            pipeline_metadata = json.load(pipeline_file)

    validation = pipeline_metadata['validation']
    if dataset is None:
        dataset = validation['dataset']

    LOGGER.info('Loading dataset %s', dataset)
    dataset = load_dataset(dataset)
    metric = validation.get('metric')
    metric_args = validation.get('metric_args', dict())
    if metric:
        scorer = get_scorer(metric, metric_args)
    else:
        scorer = dataset.score
        metric = dataset.metric

    scores = list()
    splits = dataset.get_splits(n_splits, random_state)
    if n_splits == 1:
        splits = [splits]

    for split, (X_train, X_test, y_train, y_test) in enumerate(splits):
        LOGGER.info('Scoring split %s', split + 1)
        context = get_context(dataset, validation.get('context', dict()))
        pipeline = MLPipeline.from_dict(pipeline_metadata)
        pipeline.fit(X_train, y_train, **context)
        predictions = pipeline.predict(X_test, **context)

        score = scorer(y_test, predictions)
        LOGGER.info('Split %s %s: %s', split + 1, metric, score)

        scores.append(score)

    return np.mean(scores), np.std(scores)
