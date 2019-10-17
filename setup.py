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

# Required packages
REQUIRES = [
    "tensorflow>=1.12,<1.15",
    "numpy>=1.14",
    "scipy>=1.1.0",
    "networkx>=2.2,<2.4", # FIXME (#503): NetworkX 2.4 removed some attributes
    "scikit_learn>=0.20",
    "matplotlib>=2.2",
    "gensim>=3.4.0",
    "pandas>=0.24",
]

# The demos requirements are as follows:
# * demos/community_detection: mplleaflet, python-igraph
#  *** For now these are not installed as compiled python-igraph is not available for all platforms
#
# * demos/ensembles/ensemble-node-classification-example.ipynb: seaborn
#
# * demos/link-prediction/hinsage/utils.py: numba
#
# Other demos do not have specific requirements
EXTRAS_REQURES = {
    "demos": ["numba", "jupyter", "seaborn"],
    "test": ["pytest", "pytest-benchmark"],
}

# Long description
with open("README.md", "r") as fh:
    LONG_DESCRIPTION = fh.read()

# Get global version
# see: https://packaging.python.org/guides/single-sourcing-package-version/
version = {}
with open("stellargraph/version.py", "r") as fh:
    exec(fh.read(), version)
VERSION = version["__version__"]

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
    python_requires=">=3.5.0, <3.8.0",
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
