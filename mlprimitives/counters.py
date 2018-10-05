# -*- coding: utf-8 -*-

import numpy as np


class Counter():

    def __init__(self, scalar=True, add=0):
        self.scalar = scalar
        self.add = add

    def _count(self, column):
        raise NotImplementedError

    def count(self, X):
        if len(X.shape) > 2:
            raise ValueError('Only 1d or 2d arrays are supported')

        elif self.scalar and len(X.shape) == 2 and X.shape[1] == 2:
            raise ValueError('If scalar is True, only single column arrays are supported')

        if len(X.shape) == 1:
            X = np.reshape(X, (-1, 1))

        self.counts = list()
        for column in X.T:
            count = self._count(column) + self.add
            self.counts.append(count)

    def get_counts(self):
        if self.scalar:
            return self.counts[0]
        else:
            return np.array(self.counts)


class UniqueCounter(Counter):

    def _count(self, column):
        return len(np.unique(column))


class VocabularyCounter(Counter):

    def _count(self, column):
        vocabulary = set()
        for text in column:
            words = text.split()
            vocabulary.update(words)

        return len(vocabulary)
