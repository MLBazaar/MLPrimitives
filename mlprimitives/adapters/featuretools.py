# -*- coding: utf-8 -*-

import featuretools as ft
from featuretools.selection import remove_low_information_features


class DFS(object):

    features = None

    def __init__(self, encode=False, remove_low_information=False, index=None, time_index=None,
                 target_entity=None, agg_primitives=None, trans_primitives=None, max_depth=2,
                 max_features=-1, training_window=None, n_jobs=1, verbose=False, copy=True):
        self.encode = encode
        self.remove_low_information = remove_low_information
        self.index = index
        self.time_index = time_index
        self.target_entity = target_entity
        self.agg_primitives = agg_primitives
        self.trans_primitives = trans_primitives
        self.max_depth = max_depth
        self.max_features = max_features
        self.training_window = training_window
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.copy = copy

    def __repr__(self):
        return (
            "DFS(encode={encode},\n"
            "    remove_low_information={remove_low_information},\n"
            "    index={index},\n"
            "    time_index={time_index},\n"
            "    target_entity={target_entity},\n"
            "    agg_primitives={agg_primitives},\n"
            "    trans_primitives={trans_primitives},\n"
            "    max_depth={max_depth},\n"
            "    max_features={max_features},\n"
            "    training_window={training_window},\n"
            "    n_jobs={n_jobs},\n"
            "    verbose={verbose},\n"
            "    copy={copy})"
        ).format(**self.__dict__)

    def _get_index(self, X):
        if self.copy:
            X = X.copy()

        index = X.index.name or 'index'
        while index in X.columns:
            index = '_' + index

        X.index.name = index
        X.reset_index(inplace=True)

        return X, index

    def _get_entityset(self, X, target_entity, entities, relationships):
        if entities is None:
            X, index = self._get_index(X)
            entities = {
                'X': (X, index)
            }

        if relationships is None:
            relationships = []

        return ft.EntitySet('entityset', entities, relationships)

    def dfs(self, X=None, target_entity=None, entityset=None, entities=None, relationships=None):
        if not entities and not entityset:
            target_entity = 'X'
        else:
            target_entity = target_entity or self.target_entity

        if entityset is None:
            entityset = self._get_entityset(X, target_entity, entities, relationships)

        if self.training_window is not None:
            entityset.add_last_time_indexes()

        cutoff_time = None
        if self.time_index:
            cutoff_time = X[[self.index, self.time_index]]
            cutoff_time = cutoff_time.rename(columns={self.time_index: 'time'})

        self.features = ft.dfs(
            cutoff_time=cutoff_time,
            max_depth=self.max_depth,
            entityset=entityset,
            target_entity=target_entity,
            features_only=True,
            agg_primitives=self.agg_primitives,
            trans_primitives=self.trans_primitives,
            max_features=self.max_features,
            training_window=self.training_window,
            n_jobs=self.n_jobs,
            verbose=self.verbose,
        )

        if self.encode or self.remove_low_information:
            X = ft.calculate_feature_matrix(
                self.features,
                entityset=entityset,
                cutoff_time=cutoff_time,
                training_window=self.training_window,
                n_jobs=self.n_jobs,
                verbose=self.verbose,
            )

            if self.encode:
                X, self.features = ft.encode_features(X, self.features)

            if self.remove_low_information:
                X, self.features = remove_low_information_features(X, self.features)

    def calculate_feature_matrix(self, X, target_entity=None, entityset=None,
                                 entities=None, relationships=None):

        if entityset is None:
            entityset = self._get_entityset(X, target_entity, entities, relationships)

        if self.training_window is not None:
            entityset.add_last_time_indexes()

        cutoff_time = None
        if self.time_index:
            cutoff_time = X[[self.index, self.time_index]]
            cutoff_time = cutoff_time.rename(columns={self.time_index: 'time'})

        X = ft.calculate_feature_matrix(
            self.features,
            entityset=entityset,
            cutoff_time=cutoff_time,
            training_window=self.training_window,
            n_jobs=self.n_jobs,
            verbose=self.verbose,
        )

        return X


def entity_from_dataframe(entityset_id, entity_id, dataframe, entityset=None, index=None,
                          variable_types=None, make_index=False, time_index=None,
                          secondary_time_index=None, already_sorted=False):
    if entityset is None:
        entityset = ft.EntitySet(entityset_id)

    entityset.entity_from_dataframe(entity_id, dataframe.copy(), index, variable_types,
                                    make_index, time_index, secondary_time_index,
                                    already_sorted)

    return entityset


def add_relationship(entityset, parent, parent_column, child, child_column):
    parent_variable = entityset[parent][parent_column]
    child_variable = entityset[child][child_column]
    relationship = ft.Relationship(parent_variable, child_variable)
    entityset.add_relationship(relationship)

    return entityset
