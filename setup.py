# -*- coding: utf-8 -*-
#
# Copyright 2018 Data61, CSIRO
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import setuptools

DESCRIPTION = "Python library for machine learning on graphs"
URL = "https://github.com/stellargraph/stellargraph"
VERSION = "0.5.0b"

# Required packages
REQUIRES = [
    "keras>=2.2.0",
    "tensorflow>=1.8",
    "numpy>=1.14, <1.15",
    "networkx>=2.1",
    "scikit_learn>=0.18",
    "matplotlib>=2.2",
    "gensim>=3.4.0",
    "pandas>=0.23"
]

EXTRAS_REQURES = {"demos": ["numba"], "test": ["pytest", "pandas"]}

# Long description
with open("README.md", "r") as fh:
    LONG_DESCRIPTION = fh.read()

setuptools.setup(
    name="stellargraph",
    version=VERSION,
    description=DESCRIPTION,
    author="Data61, CSIRO",
    author_email="stellar.admin@csiro.au",
    url=URL,
    license="Apache 2.0",
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    include_package_data=True,
    python_requires='>=3.6.0, <3.7.0',
    install_requires=REQUIRES,
    extras_require=EXTRAS_REQURES,
    packages=setuptools.find_packages(exclude=("tests",)),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
