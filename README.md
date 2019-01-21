<p align="left">
<img width=15% src="https://dai.lids.mit.edu/wp-content/uploads/2018/06/Logo_DAI_highres.png" alt=“DAI-Lab” />
<i>An open source project from Data to AI Lab at MIT.</i>
</p>



[![PyPi Shield](https://img.shields.io/pypi/v/mlprimitives.svg)](https://pypi.python.org/pypi/mlprimitives)
[![Travis CI Shield](https://travis-ci.org/HDI-Project/MLPrimitives.svg?branch=master)](https://travis-ci.org/HDI-Project/MLPrimitives)


# MLPrimitives

MLBlocks Primitives

- Free software: MIT license
- Documentation: https://HDI-Project.github.io/MLPrimitives


## Overview

This repository contains JSON primitives to be used by the MLBlocks library, as well as the
necessary Python code to make some of them fully compatible with the MLBlocks API requirements.

There is also a collection of custom primitives contributed directly to this library, which
either combine third party tools or implement new functionalities from scratch.


## Project Structure

The project is divided in three parts:

### The `mlprimitives` package

The mlprimitives folder is where all the Python code can be found.

Several sub-modules exist inside it, for the different types of primitives implemented, including
the `mlprimitives.adapters` module, which has a special role in the integration of third
party tools that do not directly fit the MLBlocks requirements.

### The `mlprimitives/jsons` folder

The `mlprimitives/jsons` folder contains the JSON annotations for the primitives.

This folder has a flat structure, without subfolders, and all the primitive JSONs are named
after the Fully Qualified Name of the annotated primitive (function or class).

As a result of this, sorting the JSON files alphabetically shows them grouped by library, which
makes browsing them and seeing what tools are implemented easy.

### The `tests` folder

Here are the unit tests for the Python code, as well as some validation tests for the JSON
annotations.


## Primitive Types

Three types of primitives can be found in this repository:

### Primitives that can be directly integrated to MLBlocks

The simplest type of primitives are the ones that can be directly integrated to MLBlocks
using nothing else than a single JSON annotation file.

These JSON files can be found in the `mlblocks_primitives` folder, and integrate functions
or classes that comply with the following requirements:

* Tunable hyperparameters are simple values of the supported basic types: str, bool, int or float.
* Creating the class instance or calling the fit or produce methods does not require building
  any complex structure before the call is made.
* The fitting and predicting phase consist on a single method or function call each.

A good example of this type of primitives are most of the estimators from the scikit-learn
library.

### Primitives that need a Python adapter to be integrated to MLBlocks

The second simplest type of primitives are the ones that need some kind of adaptation process
to be integrated to MLBlocks, but whose behaviour is not altered in any way by this process.

These primitives consist of some Python code which can be found in the `mlprimitives.adapters`
module, as well as JSON annotations that point at the corresponding functions or classes,
which can be found in the `mlblocs_primitives` folder.

The type of primitives that are integrated in this way are the ones that have some of these
characteristics:

* Need some additional steps after the instantiation in order to be prepared to run.
* The tunable hyperparameters need some kind of transformation or instantiation before they can
  be passed to the primitive.
* The primitive cannot be directly applied to the inputs or the outputs need to be manipulated in
  some way before they can be passed to any other primitive.

Some examples of these primitives are the Keras models, which need to be built in several steps
and later on compiled before they can be used, or some image transformation primitives which
need to be applied to the images one by one.

### Custom primitives

The third type are custom primitives implemented specifically for this library.

These custom primitives may be using third party tools or implemented from scratch, but if they
use third party tools they alter in some way their native behavior to add new functionalities
to them.

This type of primitives consist of Python code from the `mlprimitives` module, as well as the
corresponding JSON annotations, which can also be found in the `mlblocks_primitives` folder.


## Contributing

This is a community driven project and all contributions are more than welcome, from simple
feedback to the most complex coding contributions.

If you have anything that you want to ask, request or contribute, please check the
[contributing section in the documentation][contributing-docs], and do not hesitate
to open [GitHub Issue](https://github.com/HDI-Project/MLPrimitives/issues), even if it is
to ask a simple question.

[contributing-docs]: https://hdi-project.github.io/MLPrimitives/contributing.html
