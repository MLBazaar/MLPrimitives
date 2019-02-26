.. highlight:: shell

Installation
============

Stable release
--------------

To install MLPrimitives, run this command in your terminal:

.. code-block:: console

    $ pip install mlprimitives

This is the preferred method to install MLPrimitives, as it will always install the most recent
stable release.

If you don't have `pip`_ installed, this `Python installation guide`_ can guide
you through the process.

.. _pip: https://pip.pypa.io
.. _Python installation guide: http://docs.python-guide.org/en/latest/starting/installation/

From sources
------------

The sources for MLPrimitives can be downloaded from the `Github repo`_.

You can either clone the ``stable`` branch form the public repository:

.. code-block:: console

    $ git clone --branch stable git://github.com/HDI-Project/MLPrimitives

Or download the `tarball`_:

.. code-block:: console

    $ curl  -OL https://github.com/HDI-Project/MLPrimitives/tarball/stable

Once you have a copy of the source, you can install it with this command:

.. code-block:: console

    $ make install

.. _development:

Development Setup
-----------------

If you want to make changes in `MLPrimitives` and contribute them, you will need to prepare
your environment to do so.

These are the required steps:

1. Fork the MLPrimitives `Github repo`_.

2. Clone your fork locally::

    $ git clone git@github.com:your_name_here/MLPrimitives.git

3. Install your local copy into a virtualenv. Assuming you have virtualenvwrapper installed,
   this is how you set up your fork for local development::

    $ mkvirtualenv MLPrimitives
    $ cd MLPrimitives/
    $ make install-develop

.. _Github repo: https://github.com/HDI-Project/MLPrimitives
.. _tarball: https://github.com/HDI-Project/MLPrimitives/tarball/stable
