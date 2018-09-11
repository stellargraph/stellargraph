# Stellar Graph Machine Learning Library

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/ambv/black)


## Build Status
|Branch|Build|
|:-----|:----:|
|*master*|[![Build status](https://badge.buildkite.com/34d537a018c6bf27cf154aa5bcc287b2e170d6e3391cd40c64.svg)](https://buildkite.com/stellar/stellar-ml?branch=master)|
|*devel*|[![Build status](https://badge.buildkite.com/34d537a018c6bf27cf154aa5bcc287b2e170d6e3391cd40c64.svg)](https://buildkite.com/stellar/stellar-ml?branch=develop)|

**StellarGraph** is a Python library for machine learning on graph-structured (or equivalently, network-structured) data. 

Graph-structured data represent entities, e.g., people, as nodes and relationships between entities as links (or
equivalently edges), e.g., friendship. Nodes and edges may have associated attributes such as age, income, time when
a friendship was established, etc. StellarGraph supports analysis of both homogeneous (graphs with a single type of node
and edge) and heterogeneous (graphs with more than one types of nodes and/or edges) graphs.

The StellarGraph library implements several state-of-the-art algorithms for applying machine learning methods to
discover patterns and answer questions of graph-structured data. For example, StellarGraph provides algorithms for
representation learning for nodes and edges, node attribute inference, and link prediction. We provide several
examples of applying these algorithms to solve real-world, e.g., a movie recommender system, and research-oriented, e.g., 
predicting the topic of a research paper, problems.

Furthermore, StellarGraph is, where possible, using [Keras](https://keras.io/) for efficient computation and extendability. 

StellarGraph was originally developed by researchers and engineers working on the Investigative Analytics team at
[CSIRO's Data61](https://www.data61.csiro.au/) for the purpose of conducting machine learning research on 
graph-structured data.

## Getting Started

To get started with StellarGraph you'll need structured data as homogeneous or heterogeneous graph including 
features for the nodes. [NetworkX](https://networkx.github.io/) is used to represent the graph and [Pandas](https://pandas.pydata.org/) 
or [Numpy](http://www.numpy.org/) are used to store the node  features and target attributes.

StellarGraph supports different machine learning use-cases, including:

* Representation learning for nodes
  - See the demos in folder `demos/embeddings` for examples of unsupervised node representation learning using the
  random walk-based methods Node2Vec, [1], and Metapath2Vec, [2].

* Node classification and regression
  - See the demo in folder `demos/node-classification_graphsage` for an example of how to predict attributes of nodes 
  using the GraphSAGE, [3], algorithm given node features and training labels.
  - See the demo in folder `demos/node-classification_node2vec` for an example of how to predict attributes of nodes 
  using the Node2Vec, [1], algorithm for nodes without features, unsupervised node representation learning, and 
  supervised classifier training for the downstream task.
  - See the demo in folder `demos/node-classification-hinsage` for examples of how to predict attributes of nodes 
  using the HinSAGE algorithm for given node features and training labels.

* Link prediction
  - See the demo in folder `demos/link-prediction-random-walks` for an example of how to predict the existence of links between nodes 
  without node features, using the Node2Vec, [1], and Metapath2Vec, [2], algorithms.
  - See the demo in folder `demos/link-prediction_graphsage` for an example of how to predict the existence of links between 
  nodes with node features using the GraphSAGE, [3], algorithm.
  - See the demo in folder `demos/link-prediction_hinsage` for an example of how to predict the existence of links between 
  nodes with node features using the HinSAGE algorithm.

* Recommender systems
  - See the demo in folder `demos/link-prediction_hinsage/movielens-recommender` for an example of how to predict 
  movie ratings between users and movies using a Heterogeneous GraphSAGE, [3], model.

The StellarGraph library currently includes the following algorithms for graph machine learning:

* GraphSAGE [3]
  - Representation learning for homogeneous graphs in a supervised setting. The current implementation supports 
  classification and regression for node and edge attributes. It requires that nodes have numeric features.

* HinSAGE
  - Representation learning for heterogeneous graphs in a supervised setting. HinSAGE is an extension of GraphSAGE, [3], 
  for heterogeneous networks. The current implementation supports classification and regression for node and edge 
  attributes. It requires that nodes have numeric features.

* Node2Vec [1]
  - Representation learning for homogeneous graphs with nodes without features in an unsupervised setting. 
  StellarGraph is used together with [Gensim](https://radimrehurek.com/gensim/) in order to implement 
  the Node2Vec algorithm. Learned node representations can be used in downstream classification and regression tasks 
  implemented using [Scikit-learn](http://scikit-learn.org/stable/), [Keras](https://keras.io/), 
  [Tensorflow](https://www.tensorflow.org/) or any other Python machine learning library.

* Metapath2Vec [2]
  - Representation learning for heterogeneous graphs with nodes without features in an unsupervised setting. 
  StellarGraph is used together with [Gensim](https://radimrehurek.com/gensim/) in order to implement 
  the Metapath2Vec algorithm. Learned node representations can be used in downstream classification and regression tasks 
  implemented using [Scikit-learn](http://scikit-learn.org/stable/), [Keras](https://keras.io/), 
  [Tensorflow](https://www.tensorflow.org/) or any other Python machine learning library.


## Installation
StellarGraph is a Python 3 library and requires Python version 3.6 to function (note that the library
uses Keras with the Tensorflow backend, and thus does not currently work in python 3.7). The required Python version 
can be downloaded and installed either from [python.org](http://python.org/). Alternatively, use the Anaconda Python 
environment, available from [anaconda.com](https://www.anaconda.com/download/).

The machine learning components of StellarGraph use the Keras machine learning library, and all models build with StellarGraph can be extended and modified using standard Keras library code.

The StellarGraph library requires Keras, so you'll need to install Keras and a selected backend (we recommend tensorflow, which is used to test StellarGraph).  Other requirements are the NetworkX library (to create and modify graphs and networks), numpy (to manipulate numeric arrays), pandas (to manipulate tabular data), and gensim (to use the Word2Vec model), scikit-learn (to prepare datasets for machine learning), and matplotlib (for plotting).

To install the requirements for StellarGraph, execute the following command in a your preferred Python 3 environment within the root directory of the StellarGraph repository (which contains this README.md file):

```
pip install -r requirements.txt
```

Then to install the StellarGraph library, execute the following command within the root directory of this repository:
```
pip install -e .
```

## Getting Help

API Documentation for StellarGraph can be found [here.](https://stellargraph.readthedocs.io)

## CI

### buildkite integration

Pipeline is defined in `.buildkite/pipeline.yml`

### Docker images

* Tests: Uses the official [python:3.6](https://hub.docker.com/_/python/) image.
* Style: Uses [black](https://hub.docker.com/r/stellargraph/black/) from the `stellargraph` docker hub organisation.

## References

1. Node2Vec: Scalable Feature Learning for Networks. A. Grover, J. Leskovec. ACM SIGKDD International Conference on 
Knowledge Discovery and Data Mining (KDD), 2016. ([link](https://snap.stanford.edu/node2vec/))

2. Metapath2Vec: Scalable Representation Learning for Heterogeneous Networks. Yuxiao Dong, Nitesh V. Chawla, and 
Ananthram Swami. ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD), 135–144, 2017
([link](https://ericdongyx.github.io/metapath2vec/m2v.html))

3. Inductive Representation Learning on Large Graphs. W.L. Hamilton, R. Ying, and J. Leskovec arXiv:1706.02216 
[cs.SI], 2017. ([link](http://snap.stanford.edu/graphsage/))
