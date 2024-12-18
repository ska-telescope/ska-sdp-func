## The following should be standard includes
# include core makefile targets for release management
-include .make/base.mk
-include .make/conan.mk
-include .make/python.mk

# include your own private variables for custom deployment configuration
-include PrivateRules.mak

.DEFAULT_GOAL := help

clean:
	rm -rf build

# Use bash shell with pipefail option enabled so that the return status of a
# piped command is the value of the last (rightmost) commnand to exit with a
# non-zero status. This lets us pipe output into tee but still exit on test
# failures.
SHELL = /bin/bash
.SHELLFLAGS = -o pipefail -c

all: build test lint

.PHONY: all test lint

lint:
# outputs linting.xml
	cd build && ../scripts/run-clang-tidy/run-clang-tidy_nocolor -quiet '^(?:(?!extern/|test/).)*$\r?\n?' > clang-tidy.out && \
        cat clang-tidy.out | ../scripts/clang-tidy-to-junit/clang-tidy-to-junit.py ../ > linting.xml

test:
	cd build && ctest --output-junit unit-tests.xml

build:
	mkdir -p build && cd build && cmake -DCMAKE_EXPORT_COMPILE_COMMANDS=ON ../ && $(MAKE) -j 4

# Add further places where version needs to be propagated to
post-set-release:
	@. .make-support; setPyProjectRelease; setDocsRelease; setCMakeRelease; setConanRelease

cpp-format:
	docker run --rm -v ${PWD}:/code alpine:3.16 /bin/sh -c "apk update && apk add uncrustify && cd code && find src tests -iname '*.h' -o -iname '*.cpp' -o -iname '*.c' -o -iname '*.cu' | xargs uncrustify -c uncrustify.cfg -l CPP --replace --if-changed"
