# -*- coding: utf-8 -*-

import importlib
import math

import numpy as np


def import_object(object_name):
    """Import an object from its Fully Qualified Name."""
    package, name = object_name.rsplit('.', 1)
    return getattr(importlib.import_module(package), name)


def image_transform(X, function, reshape_before=False, reshape_after=False,
                    image_shape=None, **kwargs):
    """Apply a function image by image.

    Args:
        reshape_before: whether 1d array needs to be reshaped to a 2d image
        reshape_after: whether the returned values need to be reshaped back to a 1d array
        width: image width used to rebuild the 2d images. Required if the image is not square.
        heigth: image heigth used to rebuild the 2d images. Required if the image is not square.
    """

    if not callable(function):
        function = import_object(function)

    elif not callable(function):
        raise ValueError("function must be a str or a callable")

    image_length = X.shape[1]

    if reshape_before and not image_shape:
        side_length = math.sqrt(image_length)
        if side_length.is_integer():
            side_length = int(side_length)
            width = side_length
            heigth = side_length
            image_shape = (side_length, side_length)

        else:
            raise ValueError("Image sizes must be given for non-square images")

    def apply_function(image):
        if reshape_before:
            image = image.reshape(image_shape[:2])

        features = function(
            image,
            **kwargs
        )

        if reshape_after:
            features = np.reshape(features, image_length)

        return features

    return np.apply_along_axis(apply_function, axis=1, arr=X)


def graph_pairs_feature_extraction(X, functions, node_columns, graphs=None):
    functions = [import_objecT(function) for function in functions]

    pairs = X[node_columns].values

    for i, graph in enumerate(graphs):
        def apply(function):
            try:
                values = function(graph, pairs)
                return np.array(list(values))[:, 2]

            except ZeroDivisionError:
                LOGGER.warn("ZeroDivisionError captured running %s", function)
                return np.zeros(len(pairs))

        for function in functions:
            name = '{}_{}'.format(function.__name__, i)
            X[name] = apply(function)

    return X


def graph_feature_extraction(X, functions, graphs):
    functions = [import_objecT(function) for function in functions]

    for node_column, graph in graphs.items():
        index_type = type(X[node_column].values[0])

        features = pd.DataFrame(index=graph.nodes)
        features.index = features.index.astype(index_type)

        def apply(function):
            values = function(graph)
            return np.array(list(values.values()))

        for function in functions:
            name = '{}_{}'.format(function.__name__, node_column)
            features[name] = apply(function)

        X = X.merge(features, left_on=node_column, right_index=True, how='left')

        graph_data = pd.DataFrame(dict(graph.nodes.items())).T
        graph_data.index = graph_data.index.astype(index_type)

        X = X.merge(graph_data, left_on=node_column, right_index=True, how='left')

    return X
