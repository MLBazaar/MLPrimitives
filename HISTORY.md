# History

## 0.1.5

## New Primitives

* sklearn.impute.SimpleImputer
* sklearn.preprocessing.MinMaxScaler
* sklearn.preprocessing.MaxAbsScaler
* sklearn.preprocessing.RobustScaler
* sklearn.linear_model.LinearRegression

### General Improvements

* Separate curated from candidate primitives
* Setup `entry_points` in setup.py to improve compaitibility with MLBlocks
* Add a test-pipelines command to test all the existing pipelines
* Clean sklearn example pipelines
* Change the `author` entry to a `contributors` list
* Change the name of `mlblocks_primitives` folder
* Pip install `requirements_dev.txt` fail documentation

### Bug Fixes

* Fix LSTMTimeSeriesRegressor primitive. Issue #90
* Fix timeseries primitives. Issue #91
* Negative index anomalies in `timeseries_errors`. Issue #89
* Keep pandas version below 0.24.0. Issue #87

## 0.1.4

### New Primitives

* mlprimitives.timeseries primitives for timeseries data preprocessing
* mlprimitives.timeseres_error primitives for timeseries anomaly detection
* keras.Sequential.LSTMTimeSeriesRegressor
* sklearn.neighbors.KNeighbors Classifier and Regressor
* several sklearn.decomposition primitives
* several sklearn.ensemble primitives

### Bug Fixes

* Fix typo in mlprimitives.text.TextCleaner primitive
* Fix bug in index handling in featuretools.dfs primitive
* Fix bug in SingleLayerCNNImageClassifier annotation
* Remove old vlaidation tags from JSON annotations

## 0.1.3

### New Features

* Fix and re-enable featuretools.dfs primitive.

## 0.1.2

### New Features

* Add pipeline specification language and Evaluation utilities.
* Add pipelines for graph, text and tabular problems.
* New primitives ClassEncoder and ClassDecoder
* New primitives UniqueCounter and VocabularyCounter

### Bug Fixes

* Fix TrivialPredictor bug when working with numpy arrays
* Change XGB default learning rate and number of estimators


## 0.1.1

### New Features

* Add more keras.applications primitives.
* Add a Text Cleanup primitive.

### Bug Fixes

* Add keywords to `keras.preprocessing` primtives.
* Fix the `image_transform` method.
* Add `epoch` as a fixed hyperparameter for `keras.Sequential` primitives.

## 0.1.0

* First release on PyPI.
