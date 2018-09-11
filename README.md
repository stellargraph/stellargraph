# Stellar Graph Machine Learning Library

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/ambv/black)


## Build Status
|Branch|Build|
|:-----|:----:|
|*master*|[![Build status](https://badge.buildkite.com/34d537a018c6bf27cf154aa5bcc287b2e170d6e3391cd40c64.svg)](https://buildkite.com/stellar/stellar-ml?branch=master)|
|*devel*|[![Build status](https://badge.buildkite.com/34d537a018c6bf27cf154aa5bcc287b2e170d6e3391cd40c64.svg)](https://buildkite.com/stellar/stellar-ml?branch=develop)|

StellarGraph is a Python library for machine learning on graphs and networks. It uses Keras to build models for homogeneous graphs (graphs with a single type of node and edge) and heterogeneous graphs (graphs with multiple types of nodes and/or edges) to predict node or edge properties. 

## Getting Started

To get started with StellarGraph you'll need a graph and some features for the nodes. NetworkX is used to represent the graph and Pandas or Numpy are used to store the node features and targets.

StellarGraph supports different machine learning use-cases, including:

* Node classification and regression:
  - See the demos in demos/node-classification for simple examples of how to predict classes of nodes, given node features and training labels.

* Link prediction:
  - See the demo in demos/link-prediction for an example of how to predict the existance of links between nodes without node features, using the node2vec algorithm.
  - See the demo in demos/link-prediction_graphsage for an example of how to predict the existance of links between nodes with node features using the GraphSAGE algorithm.

* Recommender systems:
  - See the demo in demos/link-prediction_hinsage/movielens-recommender for an example of how to predict movie ratings between users and movies using a Heterogeneous GraphSAGE model.

The StellarGraph library currently includes the following algorithms for graph machine learning:

* Node2Vec:
  - Learns embeddings for nodes without node features from a homogeneous graph (see ...). Does not require node features. Classification and regression are supported with a secondary classifier or regressor that uses these embeddings.

* GraphSAGE:
  - Learns a graph transformation for a homogenous graph (see ...). Supports classification and regression on nodes and edges. Requires nodes to have numeric features.

* HinSAGE:
  - Learns a graph transformation for a heterogeneous graph. This is an extension of GraphSAGE for heterogeneous networks. Supports classification and regression on nodes and edges. Requires nodes to have numeric features.


## Installation
StellarGraph is a Python 3 library and requires Python version 3.6 to function (note that the library
uses Tensorflow backend, and thus does not currently work in python 3.7). The required Python version can be downloaded 
and installed either from [python.org](http://python.org/). Alternatively, use the Anaconda Python environment, available from [anaconda.com](https://www.anaconda.com/download/).

The machine learning components of StellarGraph use the Keras machine learning library, and all models build with StellarGraph can be extended and modified using standard Keras library code.

The StellarGraph library requires Keras, so you'll need to install Keras and a selected backend (we recommend tensorflow, which is used to test StellarGraph).  Other requirements are the NetworkX library (to create and modify graphs and networks), numpy (to manipulate numeric arrays), pandas (to manipulate tabular data), and gensim (to use the Node2Vec model), scikit-learn (to prepare datasets for machine learning), and matplotlib (for plotting).

To install the requirements for StellarGraph, execute the following command in a your preferred Python 3 environment within the root directory of the StellarGraph repository (which contains this README.md file):

```
pip install -r requirements.txt
```

Then to install the StellarGraph libaray, execute the followng command within the root directory of this repository:
```
pip install -e .
```

## Getting Help

Documentation for StellarGraph will be provided ...

## CI

### buildkite integration

Pipeline is defined in `.buildkite/pipeline.yml`

### Docker images

* Tests: Uses the official [python:3.6](https://hub.docker.com/_/python/) image.
* Style: Uses [black](https://hub.docker.com/r/stellargraph/black/) from the `stellargraph` docker hub organisation.
