## The following should be standard includes
# include core makefile targets for release management
-include .make/base.mk

# include your own private variables for custom deployment configuration
-include PrivateRules.mak

# Set default docker registry user.
ifeq ($(strip $(DOCKER_REGISTRY_USER)),)
	DOCKER_REGISTRY_USER=ci-cd
endif

ifeq ($(strip $(DOCKER_REGISTRY_HOST)),)
	DOCKER_REGISTRY_HOST=artefact.skao.int
endif

.DEFAULT_GOAL := help

clean:
	rm -rf build

# Use bash shell with pipefail option enabled so that the return status of a
# piped command is the value of the last (rightmost) commnand to exit with a
# non-zero status. This lets us pipe output into tee but still exit on test
# failures.
SHELL = /bin/bash
.SHELLFLAGS = -o pipefail -c

all: build test lint docs

.PHONY: all test lint doc_html

lint:
# outputs linting.xml
	cd build && ../scripts/run-clang-tidy/run-clang-tidy_nocolor -quiet '^(?:(?!extern/|test/).)*$\r?\n?' > clang-tidy.out && \
        cat clang-tidy.out | ../scripts/clang-tidy-to-junit/clang-tidy-to-junit.py ../ > linting.xml

doc_html:  ## build docs
# Outputs docs/build/html
	$(MAKE) -C docs html

test:
	cd build && ctest --output-junit unit-tests.xml

build:
	mkdir -p build && cd build && cmake -DCMAKE_EXPORT_COMPILE_COMMANDS=ON ../ && $(MAKE) -j 4
