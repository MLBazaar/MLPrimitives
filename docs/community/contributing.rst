Contributing to the Project
===========================

MLPrimitive library is an open source compendium of all possible data transforms that are used by
machine learning practitioners. It is a community driven effort, so it relies on the community.
We designed it thoughtfully so much of the contributions here can have shelf life greater than
any of the machine learning libraries it integrates. It represents the combined knowledge of all
the contributors and allows many systems to be built using the annotations themselves. A few
examples of such systems in healthcare, education will be soon released. You can contribute to
the library in several ways:

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

Write Documentation
-------------------

MLPrimitives could always use more documentation, whether as part of the official MLPrimitives
docs, in docstrings, or even on the web in blog posts, articles, and such, so feel free to
contribute any changes that you deem necessary, from fixing a simple typo, to writing whole
new pages of documentation.

Contribute code
---------------

If you want to contribute to the project with your own code, you are more than welcome
to do so! :)

The necessary steps depending on the type of contributions are thoroughly covered in the next
sections of this documentation. Keep reading!

Unit Testing Guidelines
~~~~~~~~~~~~~~~~~~~~~~~

If you are going to contribute Python code, we will ask you to write unit tests that cover
your development, following these requirements:

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
