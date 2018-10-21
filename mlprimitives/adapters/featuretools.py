# -*- coding: utf-8 -*-

import featuretools as ft
from featuretools.selection import remove_low_information_features


class DFS(object):

    features = None

    def __init__(self, max_depth=None, encode=True, remove_low_information=True):
        self.max_depth = max_depth
        self.encode = encode
        self.remove_low_information = remove_low_information

    def __repr__(self):
        return (
            "DFS(max_depth={max_depth}, encode={encode},\n"
            "    remove_low_information={remove_low_information})"
        ).format(**self.__dict__)

    def _get_entityset(self, X, target_entity, entities, relationships):
        if entities is None:
            index = X.index.name
            X = X.reset_index()
            entities = {
                target_entity: (X, index)
            }

        if relationships is None:
            relationships = []

        return ft.EntitySet('entityset', entities, relationships)

    def dfs(self, X=None, target_entity=None, entityset=None, entities=None, relationships=None):
        if entityset is None:
            entityset = self._get_entityset(X, target_entity, entities, relationships)

        target = entityset[target_entity]
        time_index = target.time_index
        index = target.index

        cutoff_time = None
        if time_index:
            cutoff_time = target.df[[index, time_index]]

        instance_ids = X.index.values.copy()

        self.features = ft.dfs(
            cutoff_time=cutoff_time,
            max_depth=self.max_depth,
            entityset=entityset,
            target_entity=target_entity,
            features_only=True,
            instance_ids=instance_ids
        )

        X = ft.calculate_feature_matrix(
            self.features,
            entityset=entityset,
            instance_ids=instance_ids
        )

        if self.encode:
            X, self.features = ft.encode_features(X, self.features)

        if self.remove_low_information:
            X, self.features = remove_low_information_features(X, self.features)

    def calculate_feature_matrix(self, X, target_entity=None, entityset=None,
                                 entities=None, relationships=None):

        if entityset is None:
            entityset = self._get_entityset(X, target_entity, entities, relationships)

        X = ft.calculate_feature_matrix(
            self.features,
            entityset=entityset,
            instance_ids=X.index.values
        )

        return X
