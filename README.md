# Stellar-ML Graph Machine Learning Library

## Build Status
|Branch|Build|
|:-----|:----:|
|*master*|[![Build status](https://badge.buildkite.com/34d537a018c6bf27cf154aa5bcc287b2e170d6e3391cd40c64.svg)](https://buildkite.com/stellar/stellar-ml?branch=master)|
|*devel*|[![Build status](https://badge.buildkite.com/34d537a018c6bf27cf154aa5bcc287b2e170d6e3391cd40c64.svg)](https://buildkite.com/stellar/stellar-ml?branch=develop)|


## Getting Started

## Installation
Stellar ML is a Python 3 library and requires Python version 3.6 or greater to function. The required Python version can be downloaded and installed either from [python.org](http://python.org/) or use the Anaconda Python environment, available from [anaconda.com](https://www.anaconda.com/download/).

The machine learning components of Stellar-ML use the Keras machine learning library, and all models build with Stellar-ML can be extended and modified using standard Keras library code.

The Stellar-ML library requires Keras you'll need to install Keras and a selected backend (we recommend tensorflow, which is used to test Stellar-ML).  Other requirements are the NetworkX library (to create and modify graphs and networks), numpy (to manipulate numeric arrays), pandas (to manipulate tabular data), and gensim (to use the Node2Vec model), scikit-learn (to prepare datasets for machine learning), and matplotlib (for plotting).

To install the requirements for Stellar-ML, execute the following command in a your preferred Python 3 environment within the root directory of the Stellar-ML repository (which contains this README.md file): 

```buildoutcfg
pip install -r requirements.txt
```

Then to install the Stellar-ML libaray, execute the followng command within the root directory :
```buildoutcfg
pip install .
```


