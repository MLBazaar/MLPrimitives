Contributing Adapters
=====================

If the primitives that you want to add are not compliant with our `fit-produce` schema you will
probably need to either add an adapter or modify an existing one in order to add them.

Creating a new Adapter
----------------------

If you want to create a new adapter, please follow these steps:

1. If it does not exist yet, create a new GitHub issue requesting the primitive that requires it.
   As indicated previously, provide as many details as possible about the new primitive, like
   links to the documentation, what it does and what it is useful for, as well as details about
   why you think a new adapter is needed and how it would work.
2. Indicate in the issue description or in a comment that you are available to apply the changes
   yourself.
3. Wait for the feedback from the maintainers, who will approve the issue and assign it to you,
   before proceeding to implement any changes. Be open to discuss with them about the need
   of adding this primitive, as maybe there are other primitive that offer the same functionality,
   and about whether the adapter will be needed or not, or what is should look like.
4. Once the issue has been approved and assigned to you, implement the necessary changes in your
   own fork of the project. Please implement them in a branch named after the issue number and
   title, as this makes keeping track of the history of the project easier in the long run.

    1. Create an adapter python module and add it to the ``mlprimitives/adapters/`` directory.
       The name of the module should be the name of the library that you want to create an adapter
       for. For example, if you want to add an adapter to add primitives from the library called
       ``cool-ml``, the name of the module should be ``mlprimitives/adapters/cool_ml.py``.
       If the module already exists because there is another adapter for the same library, create
       the new adapter within the same module.
    2. Inside the adapter module, try to name the class or function that you create as similar
       as possible to the classes that you are writing the adapter for.
       For example, the adapter class for the ``keras.Sequential`` class is called
       ``mlprimitives.adapters.keras.Sequential``, while the adapter class for the
       ``featuretools.dfs`` method is called ``mlprimitives.adapters.featuretools.DFS``.
    3. As usual, when writing python code, make sure to follow a coding style consistent with
       the rest of the library, and to follow all the guidelines form the :ref:`contributing`
       section.
    4. Do not forget to properly document your code and cover it with proper unit testing!
    5. Create at least on JSON annotation that uses your adapter. When doing so, make sure to
       follow the corresponding conventions:

        1. The name of the file should correspond to the fully qualified name of the class or
           function that you are integrating ignoring the fact that you are using an adapter.
           For example, if you are adding the primitive ``CoolPrimitive`` from the module
           ``cool_ml.module`` by using the ``mlprimitives/adapters/cool_ml.CoolML``
           adapter, the name of the file should be ``cool_ml.module.CoolPrimitive.json``.
        2. Inside the JSON annotation, the ``primitive`` entry should have the fully qualified
           name of your adapter class, and the ``fixed`` hyperparameters should contain all
           the details that your adapter needs to know how to integrate the third party primitive.
        3. Add proper description of what the primitive does in the corresponding entry, as well
           as a link to its documentation. If there is no documentation available, put the link
           to its source code. And don't forget to add you name and e-mail address to the
           ``contributors`` list!
        4. Add a pipeline annotation that uses your primitive inside the pipelines folder, named
           exactly like your primitive, and test it with the command
           ``mlprimitives test mlprimitives/pipelines/your.pipeline.json``.
           If adding a pipeline is not possible for any reason, please inform the maintainers, as
           this probably means that a new dataset needs to be added.

5. Review your changes and make sure that everything continues to work properly by executing the
   ``make test-all`` command.
6. Push all your changes to GitHub and open a Pull Request, indicating in the description which
   issue you are resolving and what the changes consist of.

Modifying an existing Adapter
-----------------------------

If an adapter for the library already exists but it does not properly cover one of the primitives
that you want to integrate, you might find that modifying the existing adapter adds this coverage.

In this case, if you are sure that these modifications will not break previous functionality,
and the existing adapter can be safely modified, do the following steps:

1. If it does not exist yet, create a new GitHub issue requesting the primitive that requires it.
   As indicated previously, provide as many details as possible about the new primitive, like
   links to the documentation, what it does and what it is useful for, as well as details about
   why you think the current adapter needs to be modified and how.
2. Indicate in the issue description or in a comment that you are available to apply the changes
   yourself.
3. Wait for the feedback from the maintainers, who will approve the issue and assign it to you,
   before proceeding to implement any changes. Be open to discuss with them about the need
   of adding this primitive, as maybe there are other primitive that offer the same functionality,
   and about whether the adapter can be modified or a new one created.
4. Once the issue has been approved and assigned to you, implement the necessary changes in your
   own fork of the project. Please implement them in a branch named after the issue number and
   title, as this makes keeping track of the history of the project easier in the long run.

    1. Do the necessary modifications in the existing adapter.
    2. As usual, when writing python code, make sure to follow a coding style consistent with
       the rest of the library, and to follow all the guidelines form the :ref:`contributing`
       section.
    3. Do not forget to properly document your code and cover it with proper unit testing!
    4. Create at least one new JSON annotation that uses the adapter. When doing so, make sure to
       follow the corresponding conventions:

        1. The name of the file should correspond to the fully qualified name of the class or
           function that you are integrating ignoring the fact that you are using an adapter.
           For example, if you are adding the primitive ``CoolPrimitive`` from the module
           ``cool_ml.module`` by using the ``mlprimitives/adapters/cool_ml.CoolML``
           adapter, the name of the file should be ``cool_ml.module.CoolPrimitive.json``.
        2. Inside the JSON annotation, the ``primitive`` entry should have the fully qualified
           name of your adapter class, and the ``fixed`` hyperparameters should contain all
           the details that your adapter needs to know how to integrate the third party primitive.
        3. Add proper description of what the primitive does in the corresponding entry, as well
           as a link to its documentation. If there is no documentation available, put the link
           to its source code. And don't forget to add you name and e-mail address to the
           ``contributors`` list!
        4. Add a pipeline annotation that uses your primitive inside the pipelines folder, named
           exactly like your primitive, and test it with the command
           ``mlprimitives test mlprimitives/pipelines/your.pipeline.json``.
           If adding a pipeline is not possible for any reason, please inform the maintainers, as
           this probably means that a new dataset needs to be added.
        5. Make sure that all the primitives that existed before that use the same adapter still
           work by testing their corresponding pipelines with the command above.

5. Review your changes and make sure that everything continues to work properly by executing the
   ``make test-all`` command.
6. Push all your changes to GitHub and open a Pull Request, indicating in the description which
   issue you are resolving and what the changes consist of.
