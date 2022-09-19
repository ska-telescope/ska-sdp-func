
.. |br| raw:: html

   <br /><br />


************************
How to contribute to PFL
************************

The aim and aspiration is to get closer to the trunk mode of development with the processing function repository. That is, work in small steps and merge correct code with the main branch often. Frequent merges would prevent large, complicated, hard-to-review merge requests and limit the number of branches to a minimum.

Long-lived branches tend to get old quickly. They are usually too large to go through in one code review session and create a jungle of different code versions containing different features. This put unnecessary strain on maintainers of the processing function library. Furthermore, long-lived branches might be used to develop dependent code, which requires more support in the future. 

Workflow
========
We strongly suggest the following workflow:

1. Work and merge small pieces of code which is correct. Do not wait until the whole processing function is complete. Correct code means that all tests are passing, and the CI/CD pipeline is also passing.
   
2. Any branch should not live longer than two days. Plan work accordingly. We suggest the following:
   
   - Add required files (documentation, header files, source files, etc.) and modify CMakelist and documentation list. Add "Work in progress" to the documentation of the processing function.
   
   - Add interfaces using processing function data models.
   
   - Add tests for interfaces. It should be clear what precision and data processing function will support at this point.
   
   - Add the body of the function
   
   - Add tests of the processing function functionality
   
   - Add documentation
   
3. Contain an unfinished processing function, so it does not cause trouble for users. Mark the processing function as "Work in progress" in the documentation to indicate that the function as not ready to be used. You can contain an incomplete processing function by:
   
   - not creating a public interface, 
   
   - or create a public interface but issue an error when the function is called,
   
   - instead of an error, issue a warning, but the processing function runs through.
   
4) Create merge request and contact maintainers for code review when the code is ready, relevant tests are passing and the CI/CD pipeline. Code reviews may by in the form of the pair programming session.
   
5) When your code breaks any test (not just yours) or the CI/CD pipeline after the merge to the main, you are responsible for fixing it. 
   
6) It is fine if the master contains a partially completed processing function as long as this function is contained and will not cause failures.
   