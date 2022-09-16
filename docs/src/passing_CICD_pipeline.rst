
.. |br| raw:: html

   <br /><br />


**********************************
What to do to pass CI/CD pipeline?
**********************************

This guide describes what stages are performed by the CI\CD pipeline and 
what steps you should take to catch any errors that could cause the pipeline
to fail.

1. Tests of the processing functions must be passing.

   - To perform C/C++ tests:
   
   .. code-block:: bash

      make
      make test
   
   - To perform Python tests, from the repository root execute:
   
   .. code-block:: bash

      pip3 install .
      pytest
   
   - Make sure that all tests in C/C++ and Python are passing.
   
2. Check that documentation builds and is correct.
   
   - To build the documentation from the repository root execute:
   
   .. code-block:: bash

      pip install sphinx 
      pip install breathe 
      pip install sphinx-rtd-theme
      ls docs
      make html
   
   - Check that your processing function appears in the documentation at the correct place and that all parameters are mentioned and described.
   
3. Linting stage
   
   - C/C++ linting can be checked using clang-tidy:
   
   .. code-block:: bash

      apt-get -y install clang-tidy 
      clang-tidy --checks='*' file.cpp
   
   - For Python use black:
   
   .. code-block:: bash

      pip3 install black isort lint-python 
      black --line-length 79 file.py
   
Removing all errors from these stages should get you through CI/CD pipeline.