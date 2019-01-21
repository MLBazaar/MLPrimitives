# -*- coding: utf-8 -*-

from mlprimitives.utils import import_object


class Dummy(object):
    pass


def test_import_object():
    imported_dummy = import_object(__name__ + '.Dummy')

    assert Dummy is imported_dummy
