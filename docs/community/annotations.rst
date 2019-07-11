Contributing Annotations
========================

The simplest type of contributions are the ones that only involve modifications on JSON
annotations.

These can modifications can come in different ways:

Creating an annotation for a new primitive
------------------------------------------

The most usual scenario will be adding a new primitive that does not exist yet in the repository
and that can be directly integrated by writing a simple JSON annotation.

In this case, please follow these steps:

1. If it does not exist yet, create a new GitHub issue requesting the new primitive. As indicated
   previously, provide as many details as possible about the new primitive, like links to the
   documentation, what it does and what it is useful for.
2. Indicate in the issue description or in a comment that you are available to apply the changes
   yourself.
3. Wait for the feedback from the maintainers, who will approve the issue and assign it to you,
   before proceeding to implement any changes. Be open to discuss with them about the need
   of adding this primitive, as maybe there are other primitive that offer the same functionality,
   and about the best approach to add it.
4. Once the issue has been approved and assigned to you, implement the necessary changes in your
   own fork of the project. Please implement them in a branch named after the issue number and
   title, as this makes keeping track of the history of the project easier in the long run.

    1. Create a new JSON annotation. This can be made from scratch, or you can copy another one
       and modify it.
    2. The name of the file should correspond to the fully qualified name of the class or function
       that you are referencing inside the primitive. For example, if you are adding a primitive
       that uses the class ``CoolPrimitive`` from the module ``super.cool.module``, the name of
       the file should be ``super.cool.module.CoolPrimitive.json``.
    3. Add proper description of what the primitive does in the corresponding entry, as well as a
       link to its documentation. If there is no documentation available, put the link to its
       source code. And don't forget to add you name and e-mail address to the ``contributors`` list!
    4. Add a pipeline annotation that uses your primitive inside the pipelines folder, named
       exactly like your primitive, and test it with the command
       ``mlprimitives test mlprimitives/pipelines/your.pipeline.json``.
       If adding a pipeline is not possible for any reason, please inform the maintainers, as
       this probably means that a new dataset needs to be added.

5. Review your changes and make sure that everything continues to work properly by executing the
   ``make test-all`` command.
6. Push all your changes to GitHub and open a Pull Request, indicating in the description which
   issue you are resolving and what the changes consist of.

Modifying an existing annotation
--------------------------------

Sometimes you might think that an existing annotation needs to be modified in some way.

Usually this is because one of the following reasons:

* There is an error in the JSON that prevents it from working properly
* Some hyperparameters are not properly exposed, or not exposed at all
* Documentation is not complete enough

In this case, please follow these steps:

1. Create a new GitHub issue explaining what needs to be changed and why.
2. Indicate in the issue description or in a comment that you are available to apply the changes
   yourself.
3. Wait for the feedback from the maintainers, who will approve the issue and assign it to you,
   before proceeding to implement any changes. Be open to discussion, as sometimes you might find
   out that some of the things that you considered an error are actually intentional. For example,
   a hyperparameter that you consider missing might have been intentionally left out for
   performance or compatibility issues.
4. Once the issue has been approved and assigned to you, implement the necessary changes in your
   own fork of the project. Please implement them in a branch named after the issue number and
   title, as this makes keeping track of the history of the project easier in the long run. Don't
   forget to add you name and e-mail address to the ``contributors`` list while you are at it!
5. Make sure that the annotation still works by testing the corresponding pipeline. Normally,
   this can be done by running the command ``mlprimitives test mlprimitives/pipelines/your.pipeline.json``.
6. Review your changes and make sure that everything continues to work properly by executing the
   ``make test-all`` command.
7. Push all your changes to GitHub and open a Pull Request, indicating in the description which
   issue you are resolving and what the changes consist of.

Creating a new version of an existing annotation
------------------------------------------------

Sometimes you might find that a primitive for a particular annotation already exists, but that
modifying it in some way allows adapting it more precisely to some particular scenarios while,
at the same time making this unusable for others

Some examples of this would include:

* Use the ``predict_proba`` method instead of the ``predict`` one in a scikit-learn classifier.
* Alter the hyperparameter ranges to make the primitive more efficient when working with certain
  type of data or problems.

In this cases, what you need to do is to create a new annotation which is basically a copy of
the other one with some modifications.

In this case, please follow these steps:

1. Create a new GitHub issue explaining what needs to be changed and why.
2. Indicate in the issue description or in a comment that you are available to apply the changes
   yourself.
3. Wait for the feedback from the maintainers, who will approve the issue and assign it to you,
   before proceeding to implement any changes. As always, be open to discussion, as sometimes you
   might find that the behavior which you want to cover is already achievable by using certain
   ``init_params``.
4. Once the issue has been approved and assigned to you, implement the necessary changes in your
   own fork of the project. Please implement them in a branch named after the issue number and
   title, as this makes keeping track of the history of the project easier in the long run.

    1. Make a copy of the original JSON annotation.
    2. The name of the file should be the same as the original one, with a suffix added after the
       last dot ``.`` indicating what the changes are. For example, if you are adapting the
       hyperparameters of the primitive named ``some.cool.primitive.json`` to improve its
       performance on huge datasets, you might name the new file
       ``some.cool.primitive.huge_datasets.json``.
    3. Apply the necessary changes to the new file and add it to the repository. Don't forget to
       add you name and e-mail address to the ``contributors`` list while you are at it!
    4. Add a pipeline annotation that uses your primitive inside the pipelines folder, named
       exactly like your primitive, and test it with the command
       ``mlprimitives test mlprimitives/pipelines/your.pipeline.json``.
       If adding a pipeline is not possible for any reason, please inform the maintainers, as
       this probably means that a new dataset needs to be added.

5. Review your changes and make sure that everything continues to work properly by executing the
   ``make test-all`` command.
6. Push all your changes to GitHub and open a Pull Request, indicating in the description which
   issue you are resolving and what the changes consist of.
