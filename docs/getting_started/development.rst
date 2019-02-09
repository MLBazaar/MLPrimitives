Development Setup
=================

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
