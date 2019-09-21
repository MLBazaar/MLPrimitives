Contributing Custom Primitives
==============================

Sometimes, the functionality that you want to add is not implemented yet by any other third
party tool, which means that you will need to implement that from scratch.

In these cases, you can either a new custom primitive or modify one of the existing ones to
add the new functionality to it.

Creating a Custom Primitive
---------------------------

If you want to create a new custom primitive, please follow these steps:

1. If it does not exist yet, create a new GitHub issue requesting a primitive that does the
   desired functionality, providing as many details as possible about the new primitive, including
   a thorough description of what it does and what it is useful for.
2. Indicate in the issue description or in a comment that you are available to apply the changes
   yourself, and provide an initial implementation proposal as detailed as possible. Include in
   this description the modules, classes and functions that you will create, as well as
   a clear description of the inputs and outputs of the primitive.
3. Wait for the feedback from the maintainers, who will approve the issue and assign it to you,
   before proceeding to implement any changes. Be open to discuss with them about the need
   of adding this primitive, as maybe there are other primitives that offer the same functionality,
   or they want to suggest a different implementation.
4. Once the issue has been approved and assigned to you, implement the necessary changes in your
   own fork of the project. Please implement them in a branch named after the issue number and
   title, as this makes keeping track of the history of the project easier in the long run.

    1. If it does not exist yet, create a python module inside the ``mlprimitives/candidates/``
       folder named after the type of primitive that you want to implement. Some good names
       could be `text_preprocessing`, `feature_extraction` or `timeseries_anomalies`.
    2. Implement the new primitive inside the corresponding module following as closely as
       possible the implementation discussed in the GitHub issue. If you feel that you need to
       deviate from it, please make a comment in the GitHub issue explaining why.
    3. As usual, when writing python code, make sure to follow a coding style consistent with
       the rest of the library, and to follow all the guidelines form the :ref:`contributing`
       section.
    4. Do not forget to properly document your code and cover it with unit tests!
    5. Create at least on JSON annotation that uses your primitive. When doing so, make sure to
       follow the corresponding conventions:

        1. The name of the file should correspond to the fully qualified name of the class or
           function which the primitive consists of.
           For example, if you are adding the primitive ``YourPrimitive`` from the module
           ``mlprimitives.candidates.your_module``, the name of the file should be
           ``mlprimitives.candidates.your_module.YourPrimitive.json``.
        2. Add proper description of what the primitive does in the corresponding entry, as well
           as a link to its documentation. If there is no documentation available, put the link
           to its source code. If the implementation follows a proposal from a scientific paper,
           consider adding the link to the PDF as well. And don't forget to add you name and
           e-mail address to the ``contributors`` list!
        3. Add a pipeline annotation that uses your primitive inside the pipelines folder, named
           exactly like your primitive, and test it with the command
           ``mlprimitives test mlprimitives/pipelines/mlprimitives.candidates.your_module.YourPrimitive.json``.
           If adding a pipeline is not possible for any reason, please inform the maintainers, as
           this probably means that a new dataset needs to be added.

5. Review your changes and make sure that everything continues to work properly by executing the
   ``make test-all`` command.
6. Push all your changes to GitHub and open a Pull Request, indicating in the description which
   issue you are resolving and what the changes consist of.

Modifying a Custom Primitive
----------------------------

If there is a custom primitive that covers the functionality that you want but it does not
support some particularities of your use case, you might want to modify it to add some new
features or extend its functionality.

In this case, if you are sure that these modifications will not break previous functionality,
and that the existing primitive can be safely modified, do the following steps:

1. If it does not exist yet, create a new GitHub issue requesting the new feature, providing
   as many details as possible about why the change is needed.
2. Indicate in the issue description or in a comment that you are available to apply the changes
   yourself, and provide an implementation proposal as detailed as possible.
3. Wait for the feedback from the maintainers, who will approve the issue and assign it to you,
   before proceeding to implement any changes. Be open to discuss with them about the need
   of adding this new feature, as maybe there are other primitive that offer the same functionality,
   or they want to suggest a different implementation.
4. Once the issue has been approved and assigned to you, implement the necessary changes in your
   own fork of the project. Please implement them in a branch named after the issue number and
   title, as this makes keeping track of the history of the project easier in the long run.

    1. Do the necessary modifications in the existing primitive.
    2. As usual, when writing python code, make sure to follow a coding style consistent with
       the rest of the library, and to follow all the guidelines form the :ref:`contributing`
       section.
    3. Do not forget to properly document your code and cover it with proper unit testing!
    4. Make sure that at least one JSON annotation exists that uses the new feature.
       While doing so, make sure to follow the corresponding conventions:

        1. The name of the file should correspond to the fully qualified name of the class or
           function which the primitive consists of.
           For example, if you are adding the primitive ``YourPrimitive`` from the module
           ``mlprimitives.candidates.your_module``, the name of the file should be
           ``mlprimitives.candidates.your_module.YourPrimitive.json``.
        2. Add proper description of what the primitive does in the corresponding entry, as well
           as a link to its documentation. If there is no documentation available, put the link
           to its source code. If the implementation follows a proposal from a scientific paper,
           consider adding the link to the PDF as well. And don't forget to add you name and
           e-mail address to the ``contributors`` list!
        3. If you are creating a new annotation, also add a pipeline annotation that uses your
           primitive inside the pipelines folder, named exactly like your primitive, and test it
           with the command
           ``mlprimitives test mlprimitives/pipelines/mlprimitives.candidates.your_module.YourPrimitive.json``.
           If adding a pipeline is not possible for any reason, please inform the maintainers, as
           this probably means that a new dataset needs to be added.
        4. Make sure that all the annotations that existed before that use the same primitive still
           work by testing their corresponding pipelines with the command above.

5. Review your changes and make sure that everything continues to work properly by executing the
   ``make test-all`` command.
6. Push all your changes to GitHub and open a Pull Request, indicating in the description which
   issue you are resolving and what the changes consist of.
