Contributing Adapters
=====================

Adapters are primitives that need some kind of adaptation process to fit to our API, but whose
behaviour is not altered in any way by this process.

In this case, please follow these steps:

1. If it does not exist yet, create a new GitHub issue requesting the new primitive. As indicated
   previuosly, provide as many details as possible about the new primitive, like links to the
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
       source code. And don't forget to add you name and e-mail address to the `contributors` list!
    4. Add a pipeline annotation that uses your primitive inside the pipelines folder, named exactly
       like your primitive. If adding a pipeline is not possible for any reason, please infirm the
       maintainers, as this probably means that a new dataset might need to be added.
    5. Create a adapter python module and add it to the ``mlprimitives/adapters/`` directory.  The
       name of the module should match the name of your primitive but be in snake_case format.
    6. Follow a coding style consistent with the rest of the library when writing your custom python
       module.  The code should follow all PEP-8 standards, include the necessary documentation in
       the code as docstrings, and use a valid JSON format.
    7. Write unit tests that follow the Unit Testing Guidelines as defined in :ref:`Contributing to the Project`

5. Review your changes and make sure that everything continues to work properly by executing the
   ``make test-all`` command.
6. Push all your changes to GitHub and open a Pull Request, indicating in the description which
   issue you are resolving and what the changes consit of.
