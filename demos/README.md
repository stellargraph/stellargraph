## Table of Contents

This folder contains several examples of applying graph machine learning (ML) algorithms on network-structured
data to solve several common problems including node attribute inference (inferring 
entity properties) and link prediction (predicting relations and/or their properties). The
examples demonstrate using the `StellarGraph` library to build machine learning 
workflows on both homogeneous and heterogeneous networks.

Each folder contains one or more examples of using the StellarGraph implementations of the
state-of-the-art algorithms, GraphSAGE [3], HinSAGE, Node2Vec [1], and Metapath2Vec [2]. 
GraphSAGE and HinSAGE are variants of Graph Convolutional Neural networks [5]. Node2Vec and
Metapath2Vec are methods based on graph random walks and representation learning using the
Word2Vec [4] algorithm.

The examples folder structure is shown below. 

* [`/embeddings`](https://github.com/stellargraph/stellargraph/tree/master/demos/embeddings)

    Examples of unsupervised node representation learning for homogeneous and heterogeneous networks, 
    using the Node2Vec and Metapath2Vec algorithms.

* [`/link-prediction-random-walks`](https://github.com/stellargraph/stellargraph/tree/master/demos/link-prediction-random-walks)

    Examples of semi-supervised link prediction for homogeneous and heterogeneous networks, 
    using the Node2Vec and Metapath2vec algorithms.
    
* [`/link-prediction-graphsage`](https://github.com/stellargraph/stellargraph/tree/master/demos/link-prediction-graphsage)

    Example of semi-supervised link prediction for a homogeneous network with attributed nodes, 
    using the GraphSAGE algorithm.
    
* [`/link-prediction-hinsage`](https://github.com/stellargraph/stellargraph/tree/master/demos/link-prediction-hinsage)

    Example of supervised link attribute prediction for a heterogeneous network with attributed nodes of different types, 
    using the HinSAGE algorithm.
    
* [`/node-classification-graphsage`](https://github.com/stellargraph/stellargraph/tree/master/demos/node-classification-graphsage)

    Example of supervised node classification for a homogeneous network with attributed nodes, using the GraphSAGE algorithm.
    
* [`/node-classification-gat`](https://github.com/stellargraph/stellargraph/tree/master/demos/node-classification-gat)

    Example of supervised node classification for a homogeneous network with attributed nodes, using the GAT algorithm [6].

* [`/node-classification-node2vec`](https://github.com/stellargraph/stellargraph/tree/master/demos/node-classification-node2vec)

    Example of unsupervised node representation learning using Node2Vec and supervised classification using 
    the Scikit-learn library.

* [`/node-classification-hinsage`](https://github.com/stellargraph/stellargraph/tree/master/demos/node-classification-hinsage)

    Example of semi-supervised node classification for a heterogeneous network with multiple node and link types, 
    using the HinSAGE algorithm.


## References

1. Node2Vec: Scalable Feature Learning for Networks. A. Grover, J. Leskovec. ACM SIGKDD International Conference on 
Knowledge Discovery and Data Mining (KDD), 2016. ([link](https://snap.stanford.edu/node2vec/))

2. Metapath2Vec: Scalable Representation Learning for Heterogeneous Networks. Yuxiao Dong, Nitesh V. Chawla, and 
Ananthram Swami. ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD), 135–144, 2017
([link](https://ericdongyx.github.io/metapath2vec/m2v.html))

3. Inductive Representation Learning on Large Graphs. W.L. Hamilton, R. Ying, and J. Leskovec arXiv:1706.02216 
[cs.SI], 2017. ([link](http://snap.stanford.edu/graphsage/))

4. Distributed representations of words and phrases and their compositionality. T. Mikolov, 
I. Sutskever, K. Chen, G. S. Corrado, and J. Dean. In Advances in Neural Information Processing
 Systems (NIPS), pp. 3111-3119, 2013. ([link](https://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf))

5. Semi-Supervised Classification with Graph Convolutional Networks. T. Kipf, M. Welling. 
ICLR 2017. arXiv:1609.02907 [link](https://arxiv.org/abs/1609.02907)

6. Graph Attention Networks. P. Velickovic et al. ICLR 2018 ([link](https://arxiv.org/abs/1710.10903))