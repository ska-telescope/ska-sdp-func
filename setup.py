import setuptools
from setuptools import setup

setup(
    name="ska-sdp-func",
    version="0.0.0",
    description="SKA SDP Processing Function Library (Python bindings)",
    packages=setuptools.find_namespace_packages(where="python", include=["ska.*"]),
    package_dir={"": "python"},
    include_package_data=True,
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Environment :: Console",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: BSD License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3"
    ],
    author="The SKA SDP Processing Function Library Developers",
    url="https://gitlab.com/ska-telescope/sdp/ska-sdp-func/",
    license="BSD"
)
