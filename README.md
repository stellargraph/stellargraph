![StellarGraph Machine Learning library logo](https://raw.githubusercontent.com/stellargraph/stellargraph/develop/stellar-graph-banner.png)

# StellarGraph Machine Learning Library

<p align="center">
  <a href="https://community.stellargraph.io" alt="Discourse Forum">
    <img src="https://img.shields.io/badge/help_forum-discourse-blue.svg"/>
  </a>
  <a href="https://github.com/ambv/black" alt="Code style">
    <img src="https://img.shields.io/badge/code%20style-black-000000.svg"/>
  </a>
  <a href="https://github.com/mvdan/sh/" alt="Shellcode style">
    <img src="https://img.shields.io/badge/shell%20style-shfmt-black.svg"/>
  </a>
  <a href="https://stellargraph.readthedocs.io/" alt="Docs">
    <img src="https://readthedocs.org/projects/stellargraph/badge/?version=latest"/>
  </a>
  <a href="https://pypi.org/project/stellargraph/" alt="PyPI">
    <img src="https://img.shields.io/pypi/v/stellargraph.svg"/>
  </a>
  <a href="https://buildkite.com/stellar/stellar-ml?branch=master/" alt="Build status: master">
    <img src="https://img.shields.io/buildkite/34d537a018c6bf27cf154aa5bcc287b2e170d6e3391cd40c64/master.svg?label=branch:+master"/>
  </a>
  <a href="https://buildkite.com/stellar/stellar-ml?branch=develop/" alt="Build status: develop">
    <img src="https://img.shields.io/buildkite/34d537a018c6bf27cf154aa5bcc287b2e170d6e3391cd40c64/develop.svg?label=branch:+develop"/>
  </a>
  <a href="https://github.com/stellargraph/stellargraph/blob/develop/CONTRIBUTING.md" alt="contributions welcome">
    <img src="https://img.shields.io/badge/contributions-welcome-brightgreen.svg"/>
  </a>
  <a href="https://github.com/stellargraph/stellargraph/blob/develop/LICENSE" alt="license">
    <img src="https://img.shields.io/github/license/stellargraph/stellargraph.svg"/>
  </a>
  <a href="https://coveralls.io/github/stellargraph/stellargraph" alt="code coverage">
    <img src="https://coveralls.io/repos/github/stellargraph/stellargraph/badge.svg"/>
  </a>
  <a href="https://cloud.docker.com/r/stellargraph/stellargraph" alt="docker hub">
    <img alt="Docker Pulls" src="https://img.shields.io/docker/pulls/stellargraph/stellargraph.svg">
  </a>
  <a href="https://pypi.org/project/stellargraph" alt="pypi downloads">
    <img alt="pypi downloads" src="https://pepy.tech/badge/stellargraph">
  </a>
</p>


# Table of Contents
   * [Introduction](#introduction)
   * [Guiding Principles](#guiding-principles)
   * [Getting Started](#getting-started)
   * [Installation](#installation)
       * [Install StellarGraph using pip](#install-stellargraph-using-pip)
       * [Install StellarGraph from Github source](#install-stellargraph-from-github-source)
   * [Docker Image](#docker-image)
   * [Running the examples](#running-the-examples)
       * [Running the examples with docker](#Running-the-examples-with-docker)
   * [Algorithms](#algorithms)
   * [Getting Help](#getting-help)
   * [Discourse Community](#discourse-community)
   * [CI](#ci)
   * [Citing](#citing)
   * [References](#references)

## Introduction
**StellarGraph** is a Python library for machine learning on graph-structured (or equivalently, network-structured) data.

Graph-structured data represent entities, e.g., people, as nodes (or equivalently, vertices),
and relationships between entities, e.g., friendship, as links (or
equivalently, edges). Nodes and links may have associated attributes such as age, income, and time when
a friendship was established, etc. StellarGraph supports analysis of both homogeneous networks (with nodes and links of one type) and heterogeneous networks (with more than one type of nodes and/or links).

The StellarGraph library implements several state-of-the-art algorithms for applying machine learning methods to discover patterns and answer questions using graph-structured data.

The StellarGraph library can be used to solve tasks using graph-structured data, such as:
- Representation learning for nodes and edges, to be used for visualisation and various downstream machine learning tasks;
- Classification and attribute inference of nodes or edges;
- Link prediction;
- Interpretation of node classification through calculated importances of edges and neighbour nodes for selected target nodes [8].

We provide [examples](https://github.com/stellargraph/stellargraph/tree/master/demos/) of using `StellarGraph` to solve such tasks using several real-world datasets.


## Guiding Principles

StellarGraph uses the [Keras](https://keras.io/) API as implemented in the [TensorFlow](https://tensorflow.org/) library and adheres to the same 
guiding principles as Keras: user-friendliness, modularity, and easy extendability. Modules and layers
of StellarGraph library are designed so that they can be used together with
standard Keras layers and modules, if required. This enables flexibility in using existing
or creating new models and workflows for machine learning on graphs.

## Getting Started

To get started with StellarGraph you'll need data structured as a homogeneous or heterogeneous graph, including
attributes for the entities represented as graph nodes.
[NetworkX](https://networkx.github.io/) is used to represent the graph and [Pandas](https://pandas.pydata.org/)
or [Numpy](https://www.numpy.org/) are used to store node attributes.

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
StellarGraph is a Python 3 library and we recommend using Python version `3.6.*`. The required Python version
can be downloaded and installed from [python.org](https://python.org/). Alternatively, use the Anaconda Python
environment, available from [anaconda.com](https://www.anaconda.com/download/).

*Note*: while the library works on Python 3.7 it is based on Keras which does not officially support Python 3.7.
Therefore, there may be unforseen bugs and you there are many warnings from the Python libraries that
StellarGraph depends upon.

<!--
The StellarGraph library requires [Keras](https://keras.io/), so you'll need to install Keras and a selected backend (we recommend tensorflow, which is used to test StellarGraph).  Other requirements are the NetworkX library (to create and modify graphs and networks), numpy (to manipulate numeric arrays), pandas (to manipulate tabular data), and gensim (to use the Word2Vec model), scikit-learn (to prepare datasets for machine learning), and matplotlib (for plotting).
-->

The StellarGraph library can be installed in one of two ways, described next.

#### Install StellarGraph using pip:
To install StellarGraph library from [PyPi](https://pypi.org) using `pip`, execute the following command:
```
pip install stellargraph
```

Some of the examples in the `demos` directory require installing additional dependencies as well as `stellargraph`. To install these dependencies as well as StellarGraph using `pip` execute the following command:
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
pip install .
```

Some of the examples in the `demos` directory require installing additional dependencies as well as `stellargraph`. To install these dependencies as well as StellarGraph using `pip` execute the following command:
```
pip install .[demos]
```


## Docker Image

* [stellargraph/stellargraph](https://hub.docker.com/r/stellargraph/stellargraph): Docker image with `stellargraph` installed.

Images can be pulled via `docker pull stellargraph/stellargraph`


## Running the examples

See the [README](https://github.com/stellargraph/stellargraph/tree/master/demos/README.md) in the `demos` directory for more information about the examples and how to run them.

## Algorithms
The StellarGraph library currently includes the following algorithms for graph machine learning:

| Algorithm | Description |
| --- | --- |
| GraphSAGE [1] | Supports supervised as well as unsupervised representation learning, node classification/regression, and link prediction for homogeneous networks. The current implementation supports multiple aggregation methods, including mean, maxpool, meanpool, and attentional aggregators. |
| HinSAGE | Extension of GraphSAGE algorithm to heterogeneous networks. Supports representation learning, node classification/regression, and link prediction/regression for heterogeneous graphs. The current implementation supports mean aggregation of neighbour nodes, taking into account their types and the types of links between them. |
| attri2vec [4] | Supports node representation learning, node classification, and out-of-sample node link prediction for homogeneous graphs with node attributes. |
| Graph ATtention Network (GAT) [5] | The GAT algorithm supports representation learning and node classification for homogeneous graphs. There are versions of the graph attention layer that support both sparse and dense adjacency matrices. |
| Graph Convolutional Network (GCN) [6] | The GCN algorithm supports representation learning and node classification for homogeneous graphs. There are versions of the graph convolutional layer that support both sparse and dense adjacency matrices. |
| Simplified Graph Convolutional network (SGC) [7] | The SGC network algorithm supports representation learning and node classification for homogeneous graphs. It is an extension of the GCN algorithm that smooths the graph to bring in more distant neighbours of nodes without using multiple layers. |
| (Approximate) Personalized Propagation of Neural Predictions (PPNP/APPNP) [9] | The (A)PPNP algorithm supports fast and scalable representation learning and node classification for attributed homogeneous graphs. In a semi-supervised setting, first a multilayer neural network is trained using the node attributes as input. The predictions from the latter network are then diffused across the graph using a method based on Personalized PageRank. |
| Node2Vec [2] | The Node2Vec and Deepwalk algorithms perform unsupervised representation learning for homogeneous networks, taking into account network structure while ignoring node attributes. The node2vec algorithm is implemented by combining StellarGraph's random walk generator with the word2vec algorithm from [Gensim](https://radimrehurek.com/gensim/). Learned node representations can be used in downstream machine learning models implemented using [Scikit-learn](https://scikit-learn.org/stable/), [Keras](https://keras.io/), [Tensorflow](https://www.tensorflow.org/) or any other Python machine learning library. |
| Metapath2Vec [3] | The metapath2vec algorithm performs unsupervised, metapath-guided representation learning for heterogeneous networks, taking into account network structure while ignoring node attributes. The implementation combines StellarGraph's metapath-guided random walk generator and [Gensim](https://radimrehurek.com/gensim/) word2vec algorithm. As with node2vec, the learned node representations (node embeddings) can be used in downstream machine learning models to solve tasks such as node classification, link prediction, etc, for heterogeneous networks. |


## Getting Help

Documentation for StellarGraph can be found [here](https://stellargraph.readthedocs.io).

## Discourse Community

Feel free to ask questions and discuss problems on the [StellarGraph Discourse forum](https://community.stellargraph.io).

## CI

### buildkite integration

Pipeline is defined in `.buildkite/pipeline.yml`

### Docker images

* Tests: Uses the official [python:3.6](https://hub.docker.com/_/python/) image.
* Style: Uses [black](https://hub.docker.com/r/stellargraph/black/) from the `stellargraph` docker hub organisation.

## Citing
StellarGraph is designed, developed and supported by [CSIRO's Data61](https://data61.csiro.au/).
If you use any part of this library in your research, please cite it using the following BibTex entry
```latex
@misc{StellarGraph,
  author = {CSIRO's Data61},
  title = {StellarGraph Machine Learning Library},
  year = {2018},
  publisher = {GitHub},
  journal = {GitHub Repository},
  howpublished = {\url{https://github.com/stellargraph/stellargraph}},
}
```

## References

1. Inductive Representation Learning on Large Graphs. W.L. Hamilton, R. Ying, and J. Leskovec.
Neural Information Processing Systems (NIPS), 2017. ([link](https://arxiv.org/abs/1706.02216) [webpage](https://snap.stanford.edu/graphsage/))

2. Node2Vec: Scalable Feature Learning for Networks. A. Grover, J. Leskovec. ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD), 2016. ([link](https://snap.stanford.edu/node2vec/))

3. Metapath2Vec: Scalable Representation Learning for Heterogeneous Networks. Yuxiao Dong, Nitesh V. Chawla, and Ananthram Swami.
ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD), 135–144, 2017
([link](https://ericdongyx.github.io/metapath2vec/m2v.html))

4. Attributed Network Embedding via Subspace Discovery. D. Zhang, Y. Jie, X. Zhu and C. Zhang, arXiv:1901.04095,
[cs.SI], 2019. ([link](https://arxiv.org/abs/1901.04095))

5. Graph Attention Networks. P. Velickovic et al.
International Conference on Learning Representations (ICLR) 2018 ([link](https://arxiv.org/abs/1710.10903))

6. Graph Convolutional Networks (GCN): Semi-Supervised Classification with Graph Convolutional Networks. Thomas N. Kipf, Max Welling.
International Conference on Learning Representations (ICLR), 2017
([link](https://github.com/tkipf/gcn))

7. Simplifying Graph Convolutional Networks. F. Wu, T. Zhang, A. H. de Souza, C. Fifty, T. Yu, and K. Q. Weinberger.
International Conference on Machine Learning (ICML), 2019. ([link](https://arxiv.org/abs/1902.07153))

8. Adversarial Examples on Graph Data: Deep Insights into Attack and Defense. H. Wu, C. Wang, Y. Tyshetskiy, A. Docherty, K. Lu, and L. Zhu. IJCAI 2019. ([link](https://arxiv.org/abs/1903.01610))

9. Predict then propagate: Graph neural networks meet personalized PageRank. J. Klicpera, A. Bojchevski, A., and S. Günnemann, ICLR, 2019, arXiv:1810.05997.([link](https://arxiv.org/abs/1810.05997))