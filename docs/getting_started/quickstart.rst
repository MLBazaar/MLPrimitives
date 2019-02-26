Quickstart
==========

Below is a short tutorial that will show you how to get started using MLPrimitives with MLBlocks.

In this tutorial we will learn how to:

* Create a pipeline using multiple primitives
* Obtain the list of tunable hyperparameters from the pipeline
* Specify hyperparameters for each primitive in the pipeline
* Fit the pipeline using training data
* Use the pipeline to make predictions from new data

Creating a pipeline
-------------------

With MLBlocks, creating a pipeline is as simple as specifying a list of MLPrimitives and passing
them to the ``MLPipeline``:

.. ipython:: python

    from mlblocks import MLPipeline
    primitives = [
        'mlprimitives.custom.feature_extraction.StringVectorizer',
        'sklearn.ensemble.RandomForestClassifier',
    ]
    pipeline = MLPipeline(primitives)

Optionally, specific hyperparameters can be also set by specifying them in a dictionary:

.. ipython:: python

    hyperparameters = {
        'sklearn.ensemble.RandomForestClassifier': {
            'n_estimators': 100
        }
    }
    pipeline = MLPipeline(primitives, hyperparameters)

Once the pipeline has been instantiated, we can easily see what hyperparameters have been set
for each block, by calling the ``get_hyperparameters``.

The output of this method is a dictionary which has the name of each block as keys and
a dictionary with the hyperparameters of the corresponding block as values.

.. ipython:: python

    pipeline.get_hyperparameters()

Tunable Hyperparameters
-----------------------

One of the main features of MLPrimitives is the possibility to indicate the type and possible
values that each primitive hyperparameter accepts.

The list of possible hyperparameters and their details can easily be obtained from the pipeline
instance by calling its ``get_tunable_hyperparameters``.

The output of this method is a dictionary that contains the list of tunable hyperparameters
for each block in the pipeline, ready to be passed to any hyperparameter tuning library such
as `BTB`_.

.. ipython:: python

    pipeline.get_tunable_hyperparameters()

Setting Hyperparameters
-----------------------

Modifying the hyperparameters of an already instantiated pipeline can be done using the
``set_hyperparameters method``, which expects a dictionary with the same format as the returned
by the ``get_hyperparameters method``.

Note that if a subset of the hyperparameters is passed, only these will be modified, and the
other ones will remain unmodified.

.. ipython:: python

    new_hyperparameters = {
        'sklearn.ensemble.RandomForestClassifier#1': {
            'max_depth': 15
        }
    }
    pipeline.set_hyperparameters(new_hyperparameters)
    hyperparameters = pipeline.get_hyperparameters()
    hyperparameters['sklearn.ensemble.RandomForestClassifier#1']['max_depth']

Making predictions
------------------

Once we have created the pipeline with the desired hyperparameters we can fit it
and then use it to make predictions on new data.

To do this, we first call the ``fit`` method passing the training data and the corresponding
labels.

.. ipython:: python

    from mlblocks.datasets import load_personae
    dataset = load_personae()
    X_train, X_test, y_train, y_test = dataset.get_splits(1)
    pipeline.fit(X_train, y_train)

Once we have fitted our model to our data, we can call the ``predict`` method passing new data
to obtain predictions from the pipeline.

.. ipython:: python

    predictions = pipeline.predict(X_test)
    predictions
    dataset.score(y_test, predictions)

.. _MLPrimitives JSON Annotations: ../getting_started/primitives.html#json-annotations
.. _BTB: https://github.com/HDI-Project/BTB
