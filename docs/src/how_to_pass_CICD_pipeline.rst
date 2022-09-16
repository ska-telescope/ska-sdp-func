
.. |br| raw:: html

   <br /><br />


***************************
How to pass the CI pipeline
***************************

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

      apt-get -y install doxygen
      pip install sphinx 
      pip install breathe 
      pip install sphinx-rtd-theme
      cd docs
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
      isort --profile black -w 79  src/ tests/  
      black --line-length 79  src/ tests/  
      flake8 src/ tests/  
      pylint src/ tests/
   
Removing all errors from these stages should get you through CI/CD pipeline.