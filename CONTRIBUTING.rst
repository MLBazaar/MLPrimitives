.. highlight:: shell

============
Contributing
============

Contributions are welcome, and they are greatly appreciated! Every little bit
helps, and credit will always be given.

You can contribute in many ways:

Types of Contributions
======================

Reporting Issues
----------------

If there is something that you would like to see changed in the project, or that you just want
to ask, please create an issue at https://github.com/HDI-Project/MLPrimitives/issues

If you do so, please:

* Explain in detail what you are requesting.
* Keep the scope as narrow as possible, to make it easier to implement or respond.
* Remember that this is a volunteer-driven project and that the maintainers will attend every
  request as soon as possible, but that in some cases this might take some time.

Below there are some examples of the types of issues that you might want to create.

Request new primitives
~~~~~~~~~~~~~~~~~~~~~~

Sometimes you will feel that a necessary primitive is missing and should be integrated.

In this case, please create an issue indicating the name of the primitive and a link to
its documentation.

If the primitive documentation is unclear or not precise enough to know what needs to be
integrated only by reading it, please add as many details as necessary in the issue description.

Request new features
~~~~~~~~~~~~~~~~~~~~

If there is any other feature that you would like to see implemented, such as adding new
functionalities to the existing custom primitives, or changing their behavior to cover
a broader range of cases, you can also create an issue.

If you do so, please indicate all the details about what you request as well as some use
cases of the new feature.

Report Bugs
~~~~~~~~~~~

If you find something that fails, please report it including:

* Your operating system name and version.
* Any details about your local setup that might be helpful in troubleshooting.
* Detailed steps to reproduce the bug.

Ask for Documentation
~~~~~~~~~~~~~~~~~~~~~

If there is something that is not documented well enough, do not hesitate to point at that
in a new issue and request the necessary changes.

Implementing Changes
--------------------

If you want to contribute to the project with your own changes, you are more than welcome
to do so! :)

In this case, please do the following steps:

1. Indicate your intentions in a GitHub issue, by saying so in the project description or in
   a comment. If no issue exists yet for the changes that you want to implement, please
   create one.
2. After you have done so, please wait for the feedback from the maintainers, who will approve
   the issue and assign it to you, before proceeding to implement any changes.
3. Implement the necessary changes in your own fork of the project. Please implement them in
   a branch named after the issue number and title.
4. Make sure that your changes include unit tests and that the existing tests and quality
   checks are all executed successfully.
5. Push all your changes to GitHub and open a Pull Request, indicating what was implemented
   in the description.

Below there are some more details about each type of contribution possible.

Integrate new primitives
~~~~~~~~~~~~~~~~~~~~~~~~

If you want to contribute integrating new third party primitives, you are welcome to contribute
the necessary JSON annotations and Python adapters.

Implement new primitives
~~~~~~~~~~~~~~~~~~~~~~~~

If what you want to implement is not available in any third party library, you can also contribute
the necessary Python code directly in any of the `mlprimitives` sub-modules.

In this case, please remember to also include the necessary JSON annotations, as well as the
corresponding documentation.

Write Documentation
~~~~~~~~~~~~~~~~~~~

MLPrimitives could always use more documentation, whether as part of the official MLPrimitives
docs, in docstrings, or even on the web in blog posts, articles, and such, so feel free to
contribute any changes that you deem necessary, from fixing a simple typo, to writing whole
new pages of documentation.

Get Started!
============

Ready to contribute? Here's how to set up `MLPrimitives` for local development.

1. Fork the `MLPrimitives` repo on GitHub.
2. Clone your fork locally::

    $ git clone git@github.com:your_name_here/MLPrimitives.git

3. Install your local copy into a virtualenv. Assuming you have virtualenvwrapper installed,
   this is how you set up your fork for local development::

    $ mkvirtualenv MLPrimitives
    $ cd MLPrimitives/
    $ make install-develop

4. Create a branch for local development::

    $ git checkout -b name-of-your-bugfix-or-feature

   Now you can make your changes locally.

5. While hacking your changes, make sure to cover all your developments with the required
   unit tests, and that none of the old tests fail as a consequence of your changes.
   For this, make sure to run the tests suite and check the code coverage::

    $ make test       # Run the tests
    $ make coverage   # Get the coverage report

6. When you're done making changes, check that your changes pass flake8 and the
   tests, including testing other Python versions with tox::

    $ make lint       # Check code styling
    $ make test-all   # Execute tests on all python versions

7. Make also sure to include the necessary documentation in the code as docstrings following
   the `google docstring`_ style.
   If you want to view how your documentation will look like when it is published, you can
   generate and view the docs with this command::

    $ make viewdocs

8. Commit your changes and push your branch to GitHub::

    $ git add .
    $ git commit -m "Your detailed description of your changes."
    $ git push origin name-of-your-bugfix-or-feature

9. Submit a pull request through the GitHub website.

.. _google docstring: https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html

Pull Request Guidelines
=======================

Before you submit a pull request, check that it meets these guidelines:

1. It resolves an open GitHub Issue and contains its reference in the title or
   the comment. If there is no associated issue, feel free to create one.
2. Whenever possible, it resolves only **one** issue. If your PR resolves more than
   one issue, try to split it in more than one pull request.
3. The pull request should include unit tests that cover all the changed code
4. If the pull request adds functionality, the docs should be updated. Put
   your new functionality into a function with a docstring, and add the
   feature to the list in README.rst.
5. The pull request should work for Python2.7, 3.4, 3.5 and 3.6. Check
   https://travis-ci.org/HDI-Project/MLPrimitives/pull_requests
   and make sure that all the checks pass.

Unit Testing Guidelines
=======================

All the Unit Tests should comply with the following requirements:

1. Unit Tests should be based only in unittest and pytest modules.

2. The tests that cover a module called ``mlprimitives/path/to/a_module.py`` should be
   implemented in a separated module called ``tests/mlprimitives/path/to/test_a_module.py``.
   Note that the module name has the ``test_`` prefix and is located in a path similar
   to the one of the tested module, just inside te ``tests`` folder.

3. Each method of the tested module should have at least one associated test method, and
   each test method should cover only **one** use case or scenario.

4. Test case methods should start with the ``test_`` prefix and have descriptive names
   that indicate which scenario they cover.
   Names such as ``test_some_methed_input_none``, ``test_some_method_value_error`` or
   ``test_some_method_timeout`` are right, but names like ``test_some_method_1``,
   ``some_method`` or ``test_error`` are not.

5. Each test should validate only what the code of the method being tested does, and not
   cover the behavior of any third party package or tool being used, which is assumed to
   work properly as far as it is being passed the right values.

6. Any third party tool that may have any kind of random behavior, such as some Machine
   Learning models, databases or Web APIs, will be mocked using the ``mock`` library, and
   the only thing that will be tested is that our code passes the right values to them.

7. Unit tests should not use anything from outside the test and the code being tested. This
   includes not reading or writting to any filesystem or database, which will be properly
   mocked.

Tips
====

To run a subset of tests::

    $ pytest tests.test_mlprimitives

Release Workflow
================

The process of releasing a new version involves several steps combining both ``git`` and
``bumpversion`` which, briefly:

1. Merge what is in ``master`` branch into ``stable`` branch.
2. Update the version in ``setup.cfg``, ``mlprimitives/__init__.py`` and ``HISTORY.md`` files.
3. Create a new TAG pointing at the correspoding commit in ``stable`` branch.
4. Merge the new commit from ``stable`` into ``master``.
5. Update the version in ``setup.cfg`` and ``mlprimitives/__init__.py`` to open the next
   development interation.

**Note:** Before starting the process, make sure that ``HISTORY.md`` has a section titled
after thew new version that is about to be released with the list of changes that will be
included in the new version, and that these changes are all committed and available in the
``master`` branch.
Normally this is just a list of the Issues that have been closed since the latest version.

Once this is done, just run the commands ``make release`` and insert the PyPi username and
password when required.
