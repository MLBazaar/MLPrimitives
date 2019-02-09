What is MLPrimitives?
=====================

.. figure:: images/dai-logo.png
   :width: 300 px
   :alt: DAI-Lab Logo

   An open source project from Data to AI Lab at MIT.

Overview
--------

This repository contains primitive annotations to be used by the MLBlocks library, as well as
the necessary Python code to make some of them fully compatible with the MLBlocks API requirements.
There is also a collection of custom primitives contributed directly to this library, which either
combine third party tools or implement new functionalities from scratch.

Why did we create this library?
-------------------------------

* Too many libraries in a fast growing field
* Huge societal need to build machine learning apps
* Domain expertise resides at several places (knowledge of math)
* No documented information about hyperparameters, behavior...


.. toctree::
   :caption: Getting Started
   :maxdepth: 2

   self
   getting_started/install
   getting_started/quickstart
   getting_started/concepts
   getting_started/development

.. toctree::
   :caption: Community
   :maxdepth: 2

   Contributing <community/contributing>
   Annotations <community/annotations>
   Adapters <community/adapters>
   Custom Primitives <community/custom>

.. toctree::
   :caption: Resources
   :hidden:

   API Reference <api/mlprimitives>
   contributing
   authors
   history


Indices and tables
==================
* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
