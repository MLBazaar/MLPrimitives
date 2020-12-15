.. highlight:: shell

Welcome to the Community
========================

MLPrimitive library is an open source compendium of all the possible data transforms
that are used by machine learning practitioners.

It is a community driven effort, so it relies on the community. For this reason, we designed it
thoughtfully so much of the contributions here can have shelf life greater than any of the
machine learning libraries it integrates, as it represents the combined knowledge of all the
contributors and allows many different systems to be built using the annotations themselves.

So, are you ready to join the community? If so, please feel welcome and keep reading!

Types of contributions
----------------------

There are several ways to contribute to a project like **MLPrimitives**, and they do not always
involve coding.

If you want to contribute but do not know where to start, consider one of the following options:

Reporting Issues
~~~~~~~~~~~~~~~~

If there is something that you would like to see changed in the project, or that you just want
to ask, please create an issue at https://github.com/MLBazaar/MLPrimitives/issues

If you do so, please:

* Explain in detail what you are requesting.
* Keep the scope as narrow as possible, to make it easier to implement or respond.
* Remember that this is a volunteer-driven project and that the maintainers will attend every
  request as soon as possible, but that in some cases this might take some time.

Below there are some examples of the types of issues that you might want to create.

Request new primitives
**********************

Sometimes you will feel that a necessary primitive is missing and should be added.

In this case, please create an issue indicating the name of the primitive and a link to
its documentation.

If the primitive documentation is unclear or not precise enough to know what needs to be
done only by reading it, please add as many details as necessary in the issue description.

Request new features
********************

If there is any other feature that you would like to see implemented, such as adding new
functionalities to the existing custom primitives, or changing their behavior to cover
a broader range of cases, you can also create an issue.

If you do so, please indicate all the details about what you request as well as some use
cases of the new feature.

Report Bugs
***********

If you find something that fails, please report it including:

* Your operating system name and version.
* Any details about your local setup that might be helpful in troubleshooting.
* Detailed steps to reproduce the bug.

Ask for Documentation
*********************

If there is something that is not documented well enough, do not hesitate to point at that
in a new issue and request the necessary changes.

Write Documentation
~~~~~~~~~~~~~~~~~~~

MLPrimitives could always use more documentation, whether as part of the official MLPrimitives
docs, in docstrings, or even on the web in blog posts, articles, and such, so feel free to
contribute any changes that you deem necessary, from fixing a simple typo, to writing whole
new pages of documentation.

Contribute code
~~~~~~~~~~~~~~~

Obviously, the main element in the MLPrimitives library is the code.

If you are willing to contribute to it, please check the documentation for more details about
how to proceed!


Release Workflow
================

The process of releasing a new version involves several steps combining both ``git`` and
``bumpversion`` which, briefly:

1. Merge what is in ``master`` branch into ``stable`` branch.
2. Update the version in the code and condiguration files.
3. Create a new git tag pointing at the corresponding commit in ``stable`` branch.
4. Merge the new commit from ``stable`` into ``master``.
2. Update the version in the code and condiguration files again to start the next development iteration.

.. note:: Before starting the process, make sure that ``HISTORY.md`` has been updated with a new
          entry that explains the changes that will be included in the new version.
          Normally this is just a list of the Pull Requests that have been merged to master
          since the last release.

Once this is done, run of the following commands:

1. If you are releasing a patch version::

    make release

2. If you are releasing a minor version::

    make release-minor

3. If you are releasing a major version::

    make release-major


Release Candidates
~~~~~~~~~~~~~~~~~~

Sometimes it is necessary or convenient to upload a release candidate to PyPi as a pre-release,
in order to make some of the new features available for testing on other projects before they
are included in an actual full-blown release.

In order to perform such an action, you can execute::

    make release-candidate

This will perform the following actions:

1. Build and upload the current version to PyPi as a pre-release, with the format ``X.Y.Z.devN``

2. Bump the current version to the next release candidate, ``X.Y.Z.dev(N+1)``

After this is done, the new pre-release can be installed by including the ``dev`` section in the
dependency specification, either in ``setup.py``::

    install_requires = [
        ...
        'mlprimitives>=X.Y.Z.dev',
        ...
    ]

or in the command line commands::

    pip install 'mlprimitives>=X.Y.Z.dev'
