
*****************
How to contribute
*****************

The aim and aspiration is to get closer to the trunk mode of development with
the processing function library repository.
That is, work in small steps and merge correct code with the main branch often.
Frequent merges prevent large, complicated and hard-to-review merge
requests, and keep the number of branches to a minimum.
Long-lived branches tend to get old quickly. They are usually too large to go
through in one code review session and create a jungle of different code
versions containing different features.
This puts unnecessary strain on maintainers of the processing function library.
Furthermore, long-lived branches might be used to develop dependent code,
which requires more effort to merge in the future.

Workflow
========
We strongly suggest the following workflow:

1. Work on and merge small pieces of code which are correct.
   Correct code means that all tests are passing, and the CI/CD pipeline is
   also passing.

2. Any branch should not live longer than a sprint. Plan work accordingly.
   We suggest the following:

   - Add required files (documentation, header files, source files, etc.),
     update the CMake build system and documentation.
     Add a "Work in progress" label to the documentation of the processing
     function.

   - Add interfaces using processing function data models. 

   - Add tests for interfaces. It should be clear what precision and data
     the processing function will support at this point.
   
   - Feel free to create a draft merge request at this point to start the review process.

   - Add the body of the function.

   - Add tests of the processing function functionality.

   - Add documentation. 

3. Create a merge request and contact maintainers for a code review when
   the code is ready, relevant tests and the CI/CD pipeline are passing.
   Code reviews may be done in a pair programming session.

4. When your code breaks any test (not just yours) or the CI/CD pipeline after
   a merge to the main branch, you are responsible for fixing it. This situation should never arise if the branch is correct.


Code style guide
================

   - Use appropriate indentation, with 4 spaces per indentation level.
     (Tabs cause problems, as they render differently in different places or
     for different people).

   - Don't use excessively long lines, as this makes code hard to read.
     80 characters is good, but not a hard limit.

   - Use blank lines in appropriate places to separate small sections of code.
     Don't add them in random places.

   - Use snake_case to name variables and functions.
     Use descriptive but concise names and avoid cryptic or non-standard
     abbreviations.

   - Use a single space around binary operators for clarity.

   - Commit messages should start with the ticket ID
     (e.g. HIP-280: Add coordinate and visibility phase rotation functions).

   - Branch names should start with the lower case ticket ID, have no spaces or
     underscores, and use hyphens to separate words (e.g. hip-280-phase-rotate).
