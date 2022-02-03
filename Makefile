# simple makefile to simplify repetitive build env management tasks under posix
PYTHON ?= python3
PYLINT ?= pylint
MAKE_DBG ?= ""
CMAKE ?= cmake
CTEST ?= ctest
TESTS ?= tests/
FLAKE ?= flake8
NAME = rascil
IMG ?= $(NAME)
TAG ?= ubuntu18.04
DOCKER_IMAGE = $(IMG):$(TAG)
WORKER_DATA ?= /tmp/data
CURRENT_DIR = $(shell pwd)
JUPYTER_PASSWORD ?= changeme

CRED=\033[0;31m
CBLUE=\033[0;34m
CEND=\033[0m
LINE:=$(shell printf '=%.0s' {1..70})

# Set default docker registry user.
ifeq ($(strip $(DOCKER_REGISTRY_USER)),)
	DOCKER_REGISTRY_USER=ci-cd
endif

ifeq ($(strip $(DOCKER_REGISTRY_HOST)),)
	DOCKER_REGISTRY_HOST=artefact.skao.int
endif

# ska-sdp-func data directory usualy found in ./data
SKA_SDP_FUNC_DATA = $(CURRENT_DIR)/data

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
	cd build && $(CTEST) --output-junit unit-tests.xml

build:
	mkdir -p build && cd build && $(CMAKE) -DCMAKE_EXPORT_COMPILE_COMMANDS=ON ../ && $(MAKE) -j 4

