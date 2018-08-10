# -*- coding: utf-8 -*-

import logging

import networkx as nx
import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)


LINK_PREDICTION_FUNCTIONS = [
    nx.jaccard_coefficient,
    nx.resource_allocation_index,
    nx.adamic_adar_index,
    nx.preferential_attachment
]


def link_prediction(X, node_columns, graphs=None):
    pairs = X[node_columns].values

    for i, graph in enumerate(graphs):
        def apply(function):
            try:
                values = function(graph, pairs)
                return np.array(list(values))[:, 2]

            except ZeroDivisionError:
                LOGGER.warn("ZeroDivisionError captured running %s", function)
                return np.zeros(len(pairs))

        for function in LINK_PREDICTION_FUNCTIONS:
            name = '{}_{}'.format(function.__name__, i)
            X[name] = apply(function)

    return X


GRAPH_FEATURIZATION_FUNCTIONS = [
    nx.degree_centrality,
    nx.closeness_centrality,
    nx.betweenness_centrality,
    nx.clustering
]


def graph_featurization(X, graphs):
    for node_column, graph in graphs.items():
        index_type = type(X[node_column].values[0])

        features = pd.DataFrame(index=graph.nodes)
        features.index = features.index.astype(index_type)

        def apply(function):
            values = function(graph)
            return np.array(list(values.values()))

        for function in GRAPH_FEATURIZATION_FUNCTIONS:
            name = '{}_{}'.format(function.__name__, node_column)
            features[name] = apply(function)

        X = X.merge(features, left_on=node_column, right_index=True, how='left')

        graph_data = pd.DataFrame(dict(graph.nodes.items())).T
        graph_data.index = graph_data.index.astype(index_type)

        X = X.merge(graph_data, left_on=node_column, right_index=True, how='left')

    return X
