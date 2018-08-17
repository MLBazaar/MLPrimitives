# -*- coding: utf-8 -*-

import importlib
import logging
import math

import numpy as np

LOGGER = logging.getLogger(__name__)


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
