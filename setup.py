#!/usr/bin/env python3
import os
from setuptools import setup

# Author(s): David Ascienzo.
# Contributor(s): David Ascienzo.

with open(os.path.join(os.path.dirname(__file__), "README.md"), "r") as f:
    long_description = f.read()

setup(
    name="tensorfit",
    version="0.0.5",
    description="Python package for univariate curve fitting using TensorFlow.",
    author="David Ascienzo",
    author_email="dascienz@gmail.com",
    url="https://github.com/dascienz/tensorfit",
    keywords="curve fitting linear nonlinear regression parametric machine learning tensorflow",
    long_description=long_description,
    long_description_content_type="text/markdown",
    license="MIT",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
      "numpy",
      "tensorflow"
    ],
    python_requires=">=3.6"
)
