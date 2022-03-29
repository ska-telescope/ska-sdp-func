# Build script using manylinux2014 base image (Centos 7).
#
# Launch Docker container with:
#
#     docker run --rm -it -v $PWD:/io quay.io/pypa/manylinux2014_x86_64
#
# Run from inside container:
#
#     sh /io/scripts/build-wheels.sh
#
yum install -y yum-utils
yum-config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel7/x86_64/cuda-rhel7.repo
yum clean all
yum install -y cuda-nvcc-11-4 cuda-cudart-devel-11-4 libcufft-devel-11-4

# Compile the Python wheels.
for PYBIN in /opt/python/cp*/bin; do
    "${PYBIN}/pip" wheel /io/ --no-deps -w wheelhouse/
done

# Bundle external shared libraries into the wheels.
for whl in wheelhouse/*.whl; do
    auditwheel repair "$whl" -w /io/wheelhouse/
done

# Install the wheels.
for PYBIN in /opt/python/cp*/bin/; do
    "${PYBIN}/pip" install numpy pytest
    "${PYBIN}/pip" install ska-sdp-func --no-index -f /io/wheelhouse
done
