# History

## 0.2.5 - 2020-07-29

### Primitive Improvements

* Accept timedelta `window_size` in `cutoff_window_sequences` - [Issue #239](https://github.com/MLBazaar/MLPrimitives/issues/239) by @joanvaquer

### Bug Fixes

* ImportError: Keras requires TensorFlow 2.2 or higher. Install TensorFlow via `pip install tensorflow` - [Issue #237](https://github.com/MLBazaar/MLPrimitives/issues/237) by @joanvaquer

### New Primitives

* Add `pandas.DataFrame.set_index` primitive - [Issue #222](https://github.com/MLBazaar/MLPrimitives/issues/222) by @JDTheRipperPC

## 0.2.4 - 2020-01-30

### New Primitives

* Add RangeScaler and RangeUnscaler primitives - [Issue #232](https://github.com/MLBazaar/MLPrimitives/issues/232) by @csala

### Primitive Improvements

* Extract input_shape from X in keras.Sequential - [Issue #223](https://github.com/MLBazaar/MLPrimitives/issues/223) by @csala

### Bug Fixes

* mlprimitives.custom.text.TextCleaner fails if text is empty - [Issue #228](https://github.com/MLBazaar/MLPrimitives/issues/228) by @csala
* Error when loading the reviews dataset - [Issue #230](https://github.com/MLBazaar/MLPrimitives/issues/230) by @csala
* Curate dependencies: specify an explicit prompt-toolkit version range - [Issue #224](https://github.com/MLBazaar/MLPrimitives/issues/224) by @csala

## 0.2.3 - 2019-11-14

### New Primitives

* Add primitive to make window_sequences based on cutoff times - [Issue #217](https://github.com/MLBazaar/MLPrimitives/issues/217) by @csala
* Create a keras LSTM based TimeSeriesClassifier primitive - [Issue #218](https://github.com/MLBazaar/MLPrimitives/issues/218) by @csala
* Add pandas DataFrame primitives - [Issue #214](https://github.com/MLBazaar/MLPrimitives/issues/214) by @csala
* Add featuretools.EntitySet.normalize_entity primitive - [Issue #209](https://github.com/MLBazaar/MLPrimitives/issues/209) by @csala

### Primitive Improvements

* Make featuretools.EntitySet.entity_from_dataframe entityset arg optional - [Issue #208](https://github.com/MLBazaar/MLPrimitives/issues/208) by @csala

* Add text regression dataset - [Issue #206](https://github.com/MLBazaar/MLPrimitives/issues/206) by @csala

### Bug Fixes

* pandas.DataFrame.resample crash when grouping by integer columns - [Issue #211](https://github.com/MLBazaar/MLPrimitives/issues/211) by @csala

## 0.2.2 - 2019-10-08

### New Primitives

* Add primitives for GAN based time-series anomaly detection - [Issue #200](https://github.com/MLBazaar/MLPrimitives/issues/200) by @AlexanderGeiger
* Add `numpy.reshape` and `numpy.ravel` primitives - [Issue #197](https://github.com/MLBazaar/MLPrimitives/issues/197) by @AlexanderGeiger
* Add feature selection primitive based on Lasso - [Issue #194](https://github.com/MLBazaar/MLPrimitives/issues/194) by @csala

### Primitive Improvements

* `feature_extraction.CategoricalEncoder` support dtype category - [Issue #196](https://github.com/MLBazaar/MLPrimitives/issues/196) by @csala

## 0.2.1 - 2019-09-09

### New Primitives

* Timeseries Intervals to Mask Primitive - [Issue #186](https://github.com/MLBazaar/MLPrimitives/issues/186) by @AlexanderGeiger
* Add new primitive: Arima model - [Issue #168](https://github.com/MLBazaar/MLPrimitives/issues/168) by @AlexanderGeiger

### Primitive Improvements

* Curate PCA primitive hyperparameters - [Issue #190](https://github.com/MLBazaar/MLPrimitives/issues/190) by @AlexanderGeiger
* Add option to drop rolling window sequences - [Issue #186](https://github.com/MLBazaar/MLPrimitives/issues/186) by @AlexanderGeiger

### Bug Fixes

* scikit-image==0.14.3 crashes when installed on Mac - [Issue #188](https://github.com/MLBazaar/MLPrimitives/issues/188) by @csala

## 0.2.0

### New Features

* Publish the pipelines as an `entry_point`
[Issue #175](https://github.com/MLBazaar/MLPrimitives/issues/175) by @csala

### Primitive Improvements

* Improve pandas.DataFrame.resample primitive [Issue #177](https://github.com/MLBazaar/MLPrimitives/issues/177) by @csala
* Improve `feature_extractor` primitives [Issue #183](https://github.com/MLBazaar/MLPrimitives/issues/183) by @csala
* Improve `find_anomalies` primitive [Issue #180](https://github.com/MLBazaar/MLPrimitives/issues/180) by @AlexanderGeiger

### Bug Fixes

* Typo in the primitive keras.Sequential.LSTMTimeSeriesRegressor [Issue #176](https://github.com/MLBazaar/MLPrimitives/issues/176) by @DanielCalvoCerezo


## 0.1.10

### New Features

* Add function to run primitives without a pipeline [Issue #43](https://github.com/MLBazaar/MLPrimitives/issues/43) by @csala

### New Pipelines

* Add pipelines for all the MLBlocks examples [Issue #162](https://github.com/MLBazaar/MLPrimitives/issues/162) by @csala

### Primitive Improvements

* Add Early Stopping to `keras.Sequential.LSTMTimeSeriesRegressor` primitive [Issue #156](https://github.com/MLBazaar/MLPrimitives/issues/156) by @csala
* Make FeatureExtractor primitives accept Numpy arrays [Issue #165](https://github.com/MLBazaar/MLPrimitives/issues/165) by @csala
* Add window size and pruning to the `timeseries_anomalies.find_anomalies` primitive [Issue #160](https://github.com/MLBazaar/MLPrimitives/issues/160) by @csala


## 0.1.9

### New Features

* Add a single table binary classification dataset [Issue #141](https://github.com/MLBazaar/MLPrimitives/issues/141) by @csala

### New Primitives

* Add Multilayer Perceptron (MLP) primitive for binary classification [Issue #140](https://github.com/MLBazaar/MLPrimitives/issues/140) by @Hector-hedb12
* Add primitive for Sequence classification with LSTM [Issue #150](https://github.com/MLBazaar/MLPrimitives/issues/150) by @Hector-hedb12
* Add VGG-like convnet primitive [Issue #149](https://github.com/MLBazaar/MLPrimitives/issues/149) by @Hector-hedb12
* Add Multilayer Perceptron (MLP) primitive for multi-class softmax classification [Issue #139](https://github.com/MLBazaar/MLPrimitives/issues/139) by @Hector-hedb12
* Add primitive to count feature matrix columns [Issue #146](https://github.com/MLBazaar/MLPrimitives/issues/146) by @csala

### Primitive Improvements

* Add additional fit and predict arguments to keras.Sequential [Issue #161](https://github.com/MLBazaar/MLPrimitives/issues/161) by @csala
* Add suport for keras.Sequential Callbacks [Issue #159](https://github.com/MLBazaar/MLPrimitives/issues/159) by @csala
* Add fixed hyperparam to control keras.Sequential verbosity [Issue #143](https://github.com/MLBazaar/MLPrimitives/issues/143) by @csala

## 0.1.8

### New Primitives

* mlprimitives.custom.timeseries_preprocessing.time_segments_average - [Issue #137](https://github.com/MLBazaar/MLPrimitives/issues/137)

### New Features

* Add target_index output in timseries_preprocessing.rolling_window_sequences - [Issue #136](https://github.com/MLBazaar/MLPrimitives/issues/136)

## 0.1.7

### General Improvements

* Validate JSON format in `make lint` -  [Issue #133](https://github.com/MLBazaar/MLPrimitives/issues/133)
* Add demo datasets - [Issue #131](https://github.com/MLBazaar/MLPrimitives/issues/131)
* Improve featuretools.dfs primitive - [Issue #127](https://github.com/MLBazaar/MLPrimitives/issues/127)

### New Primitives

* pandas.DataFrame.resample - [Issue #123](https://github.com/MLBazaar/MLPrimitives/issues/123)
* pandas.DataFrame.unstack - [Issue #124](https://github.com/MLBazaar/MLPrimitives/issues/124)
* featuretools.EntitySet.add_relationship - [Issue #126](https://github.com/MLBazaar/MLPrimitives/issues/126)
* featuretools.EntitySet.entity_from_dataframe - [Issue #126](https://github.com/MLBazaar/MLPrimitives/issues/126)

### Bug Fixes

* Bug in timeseries_anomalies.py - [Issue #119](https://github.com/MLBazaar/MLPrimitives/issues/119)

## 0.1.6

### General Improvements

* Add Contributing Documentation
* Remove upper bound in pandas version given new release of `featuretools` v0.6.1
* Improve LSTMTimeSeriesRegressor hyperparameters

### New Primitives

* mlprimitives.candidates.dsp.SpectralMask
* mlprimitives.custom.timeseries_anomalies.find_anomalies
* mlprimitives.custom.timeseries_anomalies.regression_errors
* mlprimitives.custom.timeseries_preprocessing.rolling_window_sequences
* mlprimitives.custom.timeseries_preprocessing.time_segments_average
* sklearn.linear_model.ElasticNet
* sklearn.linear_model.Lars
* sklearn.linear_model.Lasso
* sklearn.linear_model.MultiTaskLasso
* sklearn.linear_model.Ridge

## 0.1.5

### New Primitives

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
