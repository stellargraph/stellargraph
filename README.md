![StellarGraph Machine Learning library logo](https://raw.githubusercontent.com/stellargraph/stellargraph/develop/stellar-graph-banner.png)

# Stellar Graph Machine Learning Library

<p align="center">
  <a href="https://github.com/ambv/black" alt="Code style">
    <img src="https://img.shields.io/badge/code%20style-black-000000.svg" /></a>
  <a href="http://stellargraph.readthedocs.io/" alt="Docs">
    <img src="https://readthedocs.org/projects/stellargraph/badge/?version=latest" /></a>
  <a href="https://pypi.org/project/stellargraph/" alt="PyPI">
    <img src="https://img.shields.io/pypi/v/stellargraph.svg" /></a>
  <a href="https://buildkite.com/stellar/stellar-ml?branch=master/" alt="Build status: master">
    <img src="https://img.shields.io/buildkite/34d537a018c6bf27cf154aa5bcc287b2e170d6e3391cd40c64/master.svg?label=branch:+master"/></a>
  <a href="https://buildkite.com/stellar/stellar-ml?branch=develop/" alt="Build status: develop">
    <img src="https://img.shields.io/buildkite/34d537a018c6bf27cf154aa5bcc287b2e170d6e3391cd40c64/develop.svg?label=branch:+develop"/></a>
  <a href="https://github.com/stellargraph/stellargraph/blob/develop/CONTRIBUTING.md" alt="contributions welcome">
    <img src="https://img.shields.io/badge/contributions-welcome-brightgreen.svg"/></a>
  <a href="https://github.com/stellargraph/stellargraph/blob/develop/LICENSE" alt="license">
    <img src="https://img.shields.io/github/license/stellargraph/stellargraph.svg"/></a>
  <a href="https://coveralls.io/github/stellargraph/stellargraph" alt="code coverage">
    <img src="https://coveralls.io/repos/github/stellargraph/stellargraph/badge.svg"/></a>
</p>


## Introduction
**StellarGraph** is a Python library for machine learning on graph-structured (or equivalently, network-structured) data.

Graph-structured data represent entities, e.g., people, as nodes (or equivalently, vertices),
and relationships between entities, e.g., friendship, as links (or
equivalently, edges). Nodes and links may have associated attributes such as age, income, and time when
a friendship was established, etc. StellarGraph supports analysis of both homogeneous networks (with nodes and links of one type)
and heterogeneous networks (with more than one type of nodes and/or links).

The StellarGraph library implements several state-of-the-art algorithms for applying machine learning methods to
discover patterns and answer questions using graph-structured data.

The StellarGraph library can be used to solve tasks using graph-structured data, such as:
- Representation learning for nodes and edges, to be used for visualisation and various downstream machine learning tasks;
- Classification and attribute inference of nodes or edges;
- Link prediction.

We provide [examples](https://github.com/stellargraph/stellargraph/tree/master/demos/) of using `StellarGraph` to solve
such tasks using several real-world datasets.


## Guiding Principles

StellarGraph uses the [Keras](https://keras.io/) library and adheres to the same guiding principles
as Keras: user-friendliness, modularity, and easy extendability. Modules and layers
of StellarGraph library are designed so that they can be used together with
standard Keras layers and modules, if required. This enables flexibility in using existing
or creating new models and workflows for machine learning on graphs.

## Getting Started

To get started with StellarGraph you'll need data structured as a homogeneous or heterogeneous graph, including
attributes for the entities represented as graph nodes.
[NetworkX](https://networkx.github.io/) is used to represent the graph and [Pandas](https://pandas.pydata.org/)
or [Numpy](http://www.numpy.org/) are used to store node attributes.

Detailed and narrated [examples](https://github.com/stellargraph/stellargraph/tree/master/demos/) of various machine learning workflows on network data, supported by StellarGraph, from data ingestion into graph structure to inference, are given in the `demos` directory of this repository.

<!--
StellarGraph supports different machine learning use-cases, including:

* Representation learning for nodes
  - See the demos in folder `demos/embeddings` for examples of unsupervised node representation learning using the
  random walk-based methods Node2Vec [1], and Metapath2Vec [2].

* Node classification and regression
  - See the demo in folder `demos/node-classification-graphsage` for an example of how to predict attributes of nodes
  using the GraphSAGE [3] algorithm given node features and training labels.
  - See the demo in folder `demos/node-classification-node2vec` for an example of how to predict attributes of nodes
  using the Node2Vec [1] algorithm for nodes without features, unsupervised node representation learning, and
  supervised classifier training for the downstream task.
  - See the demo in folder `demos/node-classification-hinsage` for examples of how to predict attributes of nodes
  using the HinSAGE algorithm for given node features and training labels.

* Link prediction
  - See the demo in folder `demos/link-prediction-random-walks` for an example of how to predict the existence of links between nodes
  without node features, using the Node2Vec [1] and Metapath2Vec [2] algorithms.
  - See the demo in folder `demos/link-prediction-graphsage` for an example of how to predict the existence of links between
  nodes with node features using the GraphSAGE [3] algorithm.

* Recommender systems
  - See the demo in folder `demos/link-prediction-hinsage` for an example of how to predict
  movie ratings between users and movies using a Heterogeneous generalisation of GraphSAGE model, which we call HinSAGE.

-->


## Installation
StellarGraph is a Python 3 library and requires Python version 3.6 to function (note that the library
uses Keras with the Tensorflow backend, and thus does not currently work in python 3.7). The required Python version
can be downloaded and installed from [python.org](http://python.org/). Alternatively, use the Anaconda Python
environment, available from [anaconda.com](https://www.anaconda.com/download/).

<!--
The StellarGraph library requires [Keras](https://keras.io/), so you'll need to install Keras and a selected backend (we recommend tensorflow, which is used to test StellarGraph).  Other requirements are the NetworkX library (to create and modify graphs and networks), numpy (to manipulate numeric arrays), pandas (to manipulate tabular data), and gensim (to use the Word2Vec model), scikit-learn (to prepare datasets for machine learning), and matplotlib (for plotting).
-->

The StellarGraph library can be installed in one of two ways, described next.

#### Install StellarGraph using pip:
To install StellarGraph library from [PyPi](http://pypi.org) using `pip`, execute the following command:
```
pip install stellargraph
```

Some of the examples require installing additional dependencies as well as `stellargraph`.
To install these dependencies using `pip`, execute the following command:
```
pip install stellargraph[demos]
```


#### Install StellarGraph from Github source:
First, clone the StellarGraph repository using `git`:
```
git clone https://github.com/stellargraph/stellargraph.git
```

Then, `cd` to the StellarGraph folder, and install the library by executing the following commands:
```
cd stellargraph
pip install -r requirements.txt
pip install .
```

## Running the examples

See the [README](https://github.com/stellargraph/stellargraph/tree/master/demos/README.md) in the `demos` directory for more information about the examples and how to run them.

## Algorithms
The StellarGraph library currently includes the following algorithms for graph machine learning:

* GraphSAGE [1]
  - Supports representation learning, node classification/regression, and link prediction for homogeneous networks.
  The current implementation supports mean aggregation of neighbour nodes only.

* HinSAGE
  - Extension of GraphSAGE algorithm to heterogeneous networks.
  Supports representation learning, node classification/regression, and link prediction/regression for heterogeneous graphs.
  The current implementation supports mean aggregation of neighbour nodes,
  taking into account their types and the types of links between them.

* Node2Vec [2]
  - Unsupervised representation learning for homogeneous networks, taking into account network structure while ignoring
  node attributes. The node2vec algorithm is implemented by combining StellarGraph's random walk generator with the word2vec
  algorithm from [Gensim](https://radimrehurek.com/gensim/).
  Learned node representations can be used in downstream machine learning models
  implemented using [Scikit-learn](http://scikit-learn.org/stable/), [Keras](https://keras.io/),
  [Tensorflow](https://www.tensorflow.org/) or any other Python machine learning library.

* Metapath2Vec [3]
  - Unsupervised, metapath-guided representation learning for heterogeneous networks, taking into account network structure while ignoring
  node attributes. The implementation combines StellarGraph's metapath-guided random walk
  generator and [Gensim](https://radimrehurek.com/gensim/) word2vec algorithm.
  As with node2vec, the learned node representations (node embeddings) can be used in
  downstream machine learning models to solve tasks such as node classification, link prediction, etc,
  for heterogeneous networks.


## Getting Help

Documentation for StellarGraph can be found [here.](https://stellargraph.readthedocs.io)

## CI

### buildkite integration

Pipeline is defined in `.buildkite/pipeline.yml`

### Docker images

* Tests: Uses the official [python:3.6](https://hub.docker.com/_/python/) image.
* Style: Uses [black](https://hub.docker.com/r/stellargraph/black/) from the `stellargraph` docker hub organisation.

## References

1. Inductive Representation Learning on Large Graphs. W.L. Hamilton, R. Ying, and J. Leskovec arXiv:1706.02216
[cs.SI], 2017. ([link](http://snap.stanford.edu/graphsage/))

2. Node2Vec: Scalable Feature Learning for Networks. A. Grover, J. Leskovec. ACM SIGKDD International Conference on
Knowledge Discovery and Data Mining (KDD), 2016. ([link](https://snap.stanford.edu/node2vec/))

3. Metapath2Vec: Scalable Representation Learning for Heterogeneous Networks. Yuxiao Dong, Nitesh V. Chawla, and
Ananthram Swami. ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD), 135–144, 2017
([link](https://ericdongyx.github.io/metapath2vec/m2v.html))
