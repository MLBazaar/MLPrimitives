<p align="left">
  <a href="https://dai.lids.mit.edu">
    <img width=15% src="https://dai.lids.mit.edu/wp-content/uploads/2018/06/Logo_DAI_highres.png" alt="DAI-Lab" />
  </a>
  <i>An Open Source Project from the <a href="https://dai.lids.mit.edu">Data to AI Lab, at MIT</a></i>
</p>

[![Development Status](https://img.shields.io/badge/Development%20Status-2%20--%20Pre--Alpha-yellow)](https://pypi.org/search/?c=Development+Status+%3A%3A+2+-+Pre-Alpha)
[![PyPi Shield](https://img.shields.io/pypi/v/mlprimitives.svg)](https://pypi.python.org/pypi/mlprimitives)
[![Tests](https://github.com/MLBazaar/MLPrimitives/workflows/Run%20Tests/badge.svg)](https://github.com/MLBazaar/MLPrimitives/actions?query=workflow%3A%22Run+Tests%22+branch%3Amaster)
[![Downloads](https://pepy.tech/badge/mlprimitives)](https://pepy.tech/project/mlprimitives)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/MLBazaar/MLBlocks/master?filepath=examples/tutorials)

# MLPrimitives

Pipelines and primitives for machine learning and data science.

* Documentation: https://MLBazaar.github.io/MLPrimitives
* Github: https://github.com/MLBazaar/MLPrimitives
* License: [MIT](https://github.com/MLBazaar/MLPrimitives/blob/master/LICENSE)
* Development Status: [Pre-Alpha](https://pypi.org/search/?c=Development+Status+%3A%3A+2+-+Pre-Alpha)

# Overview

This repository contains primitive annotations to be used by the MLBlocks library, as well as
the necessary Python code to make some of them fully compatible with the MLBlocks API requirements.

There is also a collection of custom primitives contributed directly to this library, which either
combine third party tools or implement new functionalities from scratch.

## Why did we create this library?

* Too many libraries in a fast growing field
* Huge societal need to build machine learning apps
* Domain expertise resides at several places (knowledge of math)
* No documented information about hyperparameters, behavior...

# Installation

## Requirements

**MLPrimitives** has been developed and tested on [Python 3.6, 3.7 and 3.8](https://www.python.org/downloads/)

Also, although it is not strictly required, the usage of a
[virtualenv](https://virtualenv.pypa.io/en/latest/) is highly recommended in order to avoid
interfering with other software installed in the system where **MLPrimitives** is run.

## Install with pip

The easiest and recommended way to install **MLPrimitives** is using [pip](https://pip.pypa.io/en/stable/):

```bash
pip install mlprimitives
```

This will pull and install the latest stable release from [PyPi](https://pypi.org/).

If you want to install from source or contribute to the project please read the
[Contributing Guide](https://MLBazaar.github.io/MLPrimitives/community/welcome.html).

# Quickstart

This section is a short series of tutorials to help you getting started with MLPrimitives.

In the following steps you will learn how to load and run a primitive on some data.

Later on you will learn how to evaluate and improve the performance of a primitive by tuning
its hyperparameters.

## Running a Primitive

In this first tutorial, we will be executing a single primitive for data transformation.

### 1. Load a Primitive

The first step in order to run a primitive is to load it.

This will be done using the `mlprimitives.load_primitive` function, which will
load the indicated primitive as an [MLBlock Object from MLBlocks](https://MLBazaar.github.io/MLBlocks/api/mlblocks.html#mlblocks.MLBlock)

In this case, we will load the `mlprimitives.custom.feature_extraction.CategoricalEncoder`
primitive.

```python3
from mlprimitives import load_primitive

primitive = load_primitive('mlprimitives.custom.feature_extraction.CategoricalEncoder')
```

### 2. Load some data

The CategoricalEncoder is a transformation primitive which applies one-hot encoding to all the
categorical columns of a `pandas.DataFrame`.

So, in order to be able to run our primitive, we will first load some data that contains
categorical columns.

This can be done with the `mlprimitives.datasets.load_census` function:

```python3
from mlprimitives.datasets import load_census

dataset = load_census()
```

This dataset object has an attribute `data` which contains a table with several categorical
columns.

We can have a look at this table by executing `dataset.data.head()`, which will return a
table like this:

```
                             0                    1                   2
age                         39                   50                  38
workclass            State-gov     Self-emp-not-inc             Private
fnlwgt                   77516                83311              215646
education            Bachelors            Bachelors             HS-grad
education-num               13                   13                   9
marital-status   Never-married   Married-civ-spouse            Divorced
occupation        Adm-clerical      Exec-managerial   Handlers-cleaners
relationship     Not-in-family              Husband       Not-in-family
race                     White                White               White
sex                       Male                 Male                Male
capital-gain              2174                    0                   0
capital-loss                 0                    0                   0
hours-per-week              40                   13                  40
native-country   United-States        United-States       United-States
```

### 3. Fit the primitive

In order to run our pipeline, we first need to fit it.

This is the process where it analyzes the data to detect which columns are categorical

This is done by calling its `fit` method and assing the `dataset.data` as `X`.

```python3
primitive.fit(X=dataset.data)
```

### 4. Produce results

Once the pipeline is fit, we can process the data by calling the `produce` method of the
primitive instance and passing agin the `data` as `X`.

```python3
transformed = primitive.produce(X=dataset.data)
```

After this is done, we can see how the transformed data contains the newly generated
one-hot vectors:

```
                                                0      1       2       3       4
age                                            39     50      38      53      28
fnlwgt                                      77516  83311  215646  234721  338409
education-num                                  13     13       9       7      13
capital-gain                                 2174      0       0       0       0
capital-loss                                    0      0       0       0       0
hours-per-week                                 40     13      40      40      40
workclass= Private                              0      0       1       1       1
workclass= Self-emp-not-inc                     0      1       0       0       0
workclass= Local-gov                            0      0       0       0       0
workclass= ?                                    0      0       0       0       0
workclass= State-gov                            1      0       0       0       0
workclass= Self-emp-inc                         0      0       0       0       0
...                                             ...    ...     ...     ...     ...
```

## Tuning a Primitive

In this short tutorial we will teach you how to evaluate the performance of a primitive
and improve its performance by modifying its hyperparameters.

To do so, we will load a primitive that can learn from the transformed data that we just
generated and later on make predictions based on new data.

### 1. Load another primitive

Firs of all, we will load the `xgboost.XGBClassifier` primitive that we will use afterwards.

```python3
primitive = load_primitive('xgboost.XGBClassifier')
```

### 2. Split the dataset

Before being able to evaluate the primitive perfomance, we need to split the data in two
parts: train, which will be used for the primitive to learn, and test, which will be used
to make the predictions that later on will be evaluated.

In order to do this, we will get the first 75% of rows from the transformed data that we
obtained above and call it `X_train`, and then set the next 25% of rows as `X_test`.

```python3
train_size = int(len(transformed) * 0.75)
X_train = transformed.iloc[:train_size]
X_test = transformed.iloc[train_size:]
```

Similarly, we need to obtain the `y_train` and `y_test` variables containing the corresponding
output values.

```python3
y_train = dataset.target[:train_size]
y_test = dataset.target[train_size:]
```

### 3. Fit the new primitive

Once we have have splitted the data, we can fit the primitive by passing `X_train` and `y_train`
to its `fit` method.

```python3
primitive.fit(X=X_train, y=y_train)
```

### 4. Make predictions

Once the primitive has been fitted, we can produce predictions using the `X_test` data as input.

```python3
predictions = primitive.produce(X=X_test)
```

### 5. Evalute the performance

We can now evaluate how good the predictions from our primitive are by using the `score`
method from the `dataset` object on both the expected output and the real output from the
primitive:

```python3
dataset.score(y_test, predictions)
```

This will output a float value between 0 and 1 indicating how good the predicitons are, being
0 the worst score possible and 1 the best one.

In this case we will obtain a score around 0.866

### 6. Set new hyperparameter values

In order to improve the performance of our primitive we will try to modify a couple of its
hyperparameters.

First we will see which hyperparameter values the primitive has by calling its
`get_hyperparameters` method.

```python3
primitive.get_hyperparameters()
```

which will return a dictionary like this:

```python
{
    "n_jobs": -1,
    "n_estimators": 100,
    "max_depth": 3,
    "learning_rate": 0.1,
    "gamma": 0,
    "min_child_weight": 1
}
```

Next, we will see which are the valid values for each one of those hyperparameters by calling its
`get_tunable_hyperparameters` method:

```python3
primitive.get_tunable_hyperparameters()
```

For example, we will see that the `max_depth` hyperparameter has the following specification:

```python
{
    "type": "int",
    "default": 3,
    "range": [
        3,
        10
    ]
}
```

Next, we will choose a valid value, for example 7, and set it into the pipeline using the
`set_hyperparameters` method:

```python3
primitive.set_hyperparameters({'max_depth': 7})
```

### 7. Re-evaluate the performance

Once the new hyperparameter value has been set, we repeat the fit/train/score cycle to
evaluate the performance of this new hyperparameter value:

```python3
primitive.fit(X=X_train, y=y_train)
predictions = primitive.produce(X=X_test)
dataset.score(y_test, predictions)
```

This time we should see that the performance has improved to a value around 0.724

## What's Next?

Do you want to [learn more about how the project](https://MLBazaar.github.io/MLPrimitives/getting_started/concepts.html),
about [how to contribute to it](https://MLBazaar.github.io/MLPrimitives/community/contributing.html)
or browse the [API Reference](https://MLBazaar.github.io/MLPrimitives/api/mlprimitives.html)?
Please check the corresponding sections of the [documentation](https://MLBazaar.github.io/MLPrimitives/)!
