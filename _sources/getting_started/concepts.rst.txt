.. _concepts:

Basic Concepts
==============

Before diving into advanced usage and contributions, let's review the basic concept of the
library to help you get started.

What is a primitive?
--------------------

A primitive is a data processing block. Along with a code that does the processing on the data,
a primitive also has an associated JSON file that has a number of annotations. These annotations
help automated algorithms to interpret the primitive, and data scientists to construct machine
learning pipelines with proper provenance and full transparency about each individual components.

Types of Primitives
-------------------

Not all primitives are the same, so in the following sections we review which types of
primitives there are.

Function Primitives
~~~~~~~~~~~~~~~~~~~

The most simple type of primitives are simple functions that can be called directly, without
the need to created any class instance before.

In most cases, if not all, these functions do not have any associated learning process, and their
behavior is always the same both during the fitting and the predicting phases of the pipeline.

A simple example of such a primitive would be the ``numpy.argmax`` function, which expects a 2
dimensional array as input, and returns a 1 dimensional array that indicates the index of
the maximum values along an axis.

Class Primitives
~~~~~~~~~~~~~~~~

A more complex type of primitives are classes which need to be instantiated before they can be
used.

In most cases, these classes will have an associated learning process, and they will have some
fit method or equivalent that will be called during the fitting phase but not during the
predicting one.

A simple example of such a primitive would be the ``sklearn.preprocessing.StandardScaler`` class,
which is used to standardize a set of values by calculating their z-score, which means centering
them around 0 and scaling them to unit variance.

This primitive has an associated learning process, where it calculates the mean and standard
deviation of the training data, to later on use them to transform the prediction data to the same
center and scale.

Types of Integrations
---------------------

Also, primitives can be classified depending on how they are integrated into the project.

Directly integrable primitives
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Some libraries already follow the `fit-produce` abstraction. That is, they already have several
data processing building blocks that have this abstraction. A good example of these type of
primitives are most of the estimators from the scikit-learn library. These building blocks usually
have these characteristics:

* Tunable hyperparameters are simple values of the supported basic types:
  * ``str``
  * ``bool``
  * ``int``
  * ``float``
* Creating the class instance or calling the fit or produce methods does not require building
  any complex structure before the call is made.
* The fitting and predicting phase consist of a single method or function call each.

In this case, no additional code is necessary to adapt them and those blocks can be brought into
MLPrimitives using nothing else than a single JSON annotation file, which can be found in the
`mlprimitives/primitives folder`_.

Examples
********

* `numpy.argmax`_
* `sklearn.preprocessing.StandardScaler`_
* `xgboost.XGBClassifier`_

.. note:: If the code is directly usable then why create a JSON annotation file? While the code is
          directly usable, most building blocks do not have an associated metadata we need for
          automation. Usually when using scikit-learn for example, a data scientist goes through
          the documentation to understand different hyperparameters, their ranges and has to do a
          lot of manual inference before they can use them.


Primitives that require a Python adapter
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The second type of primitives are the ones that need some kind of adaptation process to fit to our
API, but whose behaviour is not altered in any way by this process. The type of primitives that
are integrated in this way are the ones that have some of these characteristics:

* Need some additional steps after the instantiation in order to be prepared to run.
* The tunable hyperparameters need some kind of transformation or instantiation before they can be
  passed to the primitive.
* The primitive cannot be directly applied to the inputs or the outputs, we support, and need to
  be manipulated in some way before they can be passed to any other primitive.

Some examples of these primitives are the Keras models, which need to be built in several steps
and later on compiled before they can be used, or some image transformation primitives which need
to be applied to the images one by one. These primitives consist of some Python code which can be
found in the ``mlprimitives.adapters`` module, as well as JSON annotations that point at the
corresponding functions or classes, which can also be found in the `mlprimitives/primitives folder`_.

Examples
********

* LightFM
* Keras Sequential LSTMTextClassifier
* NetworkX Graph Feature Extraction


Custom primitives
~~~~~~~~~~~~~~~~~

The third type are custom primitives implemented specifically for this library. These custom
primitives may be implemented from scratch or they may be using third party tools in such a way
as to alter the third party tool’s native behavior to add new functionalities.

This type of primitives consist of Python code that can be found inside the `mlprimitives/custom module`_,
as well as the corresponding JSON annotations, which can also be found in the `mlprimitives/primitives folder`_.

Examples
********

* Preprocessing Class Encoder
* Vocabulary Counter
* Text Cleaner


Candidate primitives
********************

Since this is a project with a strong focus in community contributions, we want to make it easy
for everyone to contribute their own code without the need to have project maintainers that
carefully and thoroughly review all the new contributions, as this would make the contributing
process very slow. However, having all the new primitives accepted and merged without a proper
review, might compromise the project stability in some cases.

For this reason, we have created the special `mlprimitives/candidates module`_, which includes
all the primitives that have been recently contributed but haven't gone through a proper testing
and review yet.

So, does this it mean that these primitives do not work? Not at all!

All the candidate primitives have gone through an initial testing and review process before being
accepted, so they are always proved to work. The only difference between these primitives and
the ones that you can find in `mlprimitives/custom module`_ is that the later ones have gone
through a deeper code review in search of possible improvements in terms of performance and
functionality refinements


.. _mlprimitives/primitives folder: https://github.com/MLBazaar/MLPrimitives/blob/master/mlprimitives/primitives
.. _mlprimitives/custom module: https://github.com/MLBazaar/MLPrimitives/blob/master/mlprimitives/custom
.. _mlprimitives/candidates module: https://github.com/MLBazaar/MLPrimitives/blob/master/mlprimitives/candidates
.. _numpy.argmax: https://github.com/MLBazaar/MLPrimitives/blob/master/mlprimitives/primitives/numpy.argmax.json
.. _sklearn.preprocessing.StandardScaler: https://github.com/MLBazaar/MLPrimitives/blob/master/mlprimitives/primitives/sklearn.preprocessing.StandardScaler.json
.. _xgboost.XGBClassifier: https://github.com/MLBazaar/MLPrimitives/blob/master/mlprimitives/primitives/xgboost.XGBClassifier.json
