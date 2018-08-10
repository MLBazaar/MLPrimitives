# -*- coding: utf-8 -*-

import math

import numpy as np
from skimage.feature import hog


class HOG(object):

    def __init__(self, orientations, pixels_per_cell, cells_per_block,
                 image_size_x=None, image_size_y=None):

        self.orientations = orientations
        self.pixels_per_cell = pixels_per_cell
        self.cells_per_block = cells_per_block
        self.image_size_x = image_size_x
        self.image_size_y = image_size_y

    def make_hog_features(self, X):
        """Call the transform function of the HOG primitive.

        NOTE: Get a "ValueError: Negative sizes" with some settings
        of the hyperparameters.
        """

        if math.sqrt(X.shape[1]).is_integer():
            # We can set sizes if the image is square (default).
            image_size = int(math.sqrt(X.shape[1]))
            self.image_size_x = image_size
            self.image_size_y = image_size

        elif not self.image_size_x or not self.image_size_y:
            raise ValueError("Image sizes must be given for non-square image")

        def make_hog(image):
            image = image.reshape((self.image_size_x, self.image_size_y))
            features = hog(
                image,
                orientations=self.orientations,
                pixels_per_cell=(self.pixels_per_cell, self.pixels_per_cell),
                cells_per_block=(self.cells_per_block, self.cells_per_block),
                block_norm='L2-Hys',
                visualise=False
            )
            return features

        return np.apply_along_axis(lambda x: make_hog(x), axis=1, arr=X)
