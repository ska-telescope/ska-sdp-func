***************************
How to pass the CI pipeline
***************************

This guide describes what stages are performed by the CI/CD pipeline and
what steps you should take to catch any errors that could cause the pipeline
to fail.

Removing all errors from these stages should get you through the CI pipeline.


Lint and Format
---------------

- C/C++ linting can be checked using ``clang-tidy``:

   .. code-block:: bash

      apt-get -y install build-essential clang-tidy cmake
      mkdir release
      cd release
      cmake .. -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
      run-clang-tidy -quiet

- For Python use ``isort``, ``black``, ``flake8`` and ``pylint``:

   .. code-block:: bash

      pip3 install black isort flake8 pylint
      isort --profile black -w 79  src/ tests/
      black --line-length 79  src/ tests/
      flake8 src/ tests/
      pylint src/ tests/

- C/C++ formatting is done using ``uncrustify``. We are currently using
  `uncrustify version 0.75.1 <https://github.com/uncrustify/uncrustify>`_,
  which is available in the repository for Alpine 3.16.
  It can be run inside a Docker image from the repository root using:

   .. code-block:: bash

      docker run --rm -it -v $PWD:/data alpine:3.16
      apk update
      apk add uncrustify
      cd /data
      find src tests -iname '*.h' -o -iname '*.cpp' -o -iname '*.c' -o -iname '*.cu' | xargs uncrustify -c uncrustify.cfg -l CPP --replace --if-changed


Build and Test
--------------

Tests of the processing functions must be passing.

- To perform C/C++ tests:

   .. code-block:: bash

      make
      make test

- To perform Python tests, from the repository root execute:

   .. code-block:: bash

      pip3 install .
      pytest


Documentation
-------------

Check that documentation builds and is correct.

- To build the documentation from the repository root execute:

   .. code-block:: bash

      apt-get -y install doxygen
      pip3 install sphinx breathe sphinx-rtd-theme
      cd docs
      make html

- Check that your processing function appears in the documentation
  at the correct place and that all parameters are mentioned and described.
