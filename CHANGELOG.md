# Changelog

## 1.2.1

- Improve load-balancing between CPU threads for w-towers gridder and degridder.
- Allow number of CPU threads to be set when calling the gridder and degridder.
- Use PocketFFT by default if MKL is not available.
- Fix crash encountered when using some sub-grid sizes.
- Apply optimisations to the degridder and grid correction functions.

## 1.2.0

- Add new version of RFI flagger.
- Add Hogbom clean and MS clean functions.
- Fix FFT shift to work with large images.
- Add helper functions for optimally choosing w-towers gridder parameters.
- Refactor w-towers gridder kernel interfaces for consistency with notebook
  versions.
- Optimise initialisation of large arrays.
- Fix clamp_channels functions used by w-towers.
- Add support for passing mixed precision arrays to w-towers functions.
- Add support for CUDA architectures 8.9 and 9.0.

## 1.1.7

- Add GPU support in w-towers wrapper functions.
- Use OpenMP dynamic scheduling for w-towers wrappers in loop over subgrids.
- Allow non-complex image types to be used in w-towers wrappers.

## 1.1.6

- Build a single wheel for all Python 3 versions on Linux.
- Use static linking of MKL libraries and libstdc++ for the wheel.

## 1.1.5

- Re-issue release of previous version.

## 1.1.4

- Re-issue release of previous version.

## 1.1.3

- Re-issue release of previous version.

## 1.1.2

- Attempt to fix build of Python wheels in CI/CD pipeline.

## 1.1.1

- Add error checking for MKL functions.
- Attempt to build Python wheels in CI/CD pipeline using MKL.

## 1.1.0

- Add initial version of w-towers sub-grid gridder and de-gridder functions.

## 1.0.1

- Fix SwiFTly accumulation functions.

## 1.0.0

- Add SwiFTly functions.
