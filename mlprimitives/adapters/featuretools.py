# -*- coding: utf-8 -*-

import featuretools as ft
from featuretools.selection import remove_low_information_features


class DFS(object):

    features = None

    def __init__(self, max_depth=None, encode=True, remove_low_information=True,
                 target_entity=None, index=None, time_index=None,
                 agg_primitives=None, trans_primitives=None, copy=True):
        self.copy = copy
        self.max_depth = max_depth
        self.encode = encode
        self.remove_low_information = remove_low_information
        self.target_entity = target_entity
        self.index = index
        self.time_index = time_index
        self.agg_primitives = agg_primitives
        self.trans_primitives = trans_primitives

    def __repr__(self):
        return (
            "DFS(max_depth={max_depth},\n"
            "    encode={encode},\n"
            "    remove_low_information={remove_low_information},\n"
            "    target_entity={target_entity},\n"
            "    index={index},\n"
            "    time_index={time_index},\n"
            "    agg_primitives={agg_primitives},\n"
            "    trans_primitives={trans_primitives})"
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

        instance_ids = None
        cutoff_time = None
        if self.time_index:
            cutoff_time = X[[self.index, self.time_index]]
        elif self.index:
            instance_ids = X[self.index]
        else:
            instance_ids = X.index.values

        self.features = ft.dfs(
            cutoff_time=cutoff_time,
            instance_ids=instance_ids,
            max_depth=self.max_depth,
            entityset=entityset,
            target_entity=target_entity,
            features_only=True,
            agg_primitives=self.agg_primitives,
            trans_primitives=self.trans_primitives
        )

        X = ft.calculate_feature_matrix(
            self.features,
            entityset=entityset,
            cutoff_time=cutoff_time,
            instance_ids=instance_ids,
        )

        if self.encode:
            X, self.features = ft.encode_features(X, self.features)

        if self.remove_low_information:
            X, self.features = remove_low_information_features(X, self.features)

    def calculate_feature_matrix(self, X, target_entity=None, entityset=None,
                                 entities=None, relationships=None):

        if entityset is None:
            entityset = self._get_entityset(X, target_entity, entities, relationships)

        instance_ids = None
        cutoff_time = None
        if self.time_index:
            cutoff_time = X[[self.index, self.time_index]]
        elif self.index:
            instance_ids = X[self.index]
        else:
            instance_ids = X.index.values

        X = ft.calculate_feature_matrix(
            self.features,
            entityset=entityset,
            cutoff_time=cutoff_time,
            instance_ids=instance_ids,
        )

        return X


def entity_from_dataframe(entityset, entityset_id, entity_id, dataframe, index=None,
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
