## Table of Contents

This folder contains several examples of applying graph machine learning (ML) algorithms on network-structured
data to solve link prediction (predicting relations and/or their properties) problems. The
examples demonstrate using the `StellarGraph` library to build machine learning
workflows on both homogeneous and heterogeneous networks.

Each folder contains one or more examples of using the StellarGraph implementations of the
state-of-the-art algorithms, attri2vec[4], GraphSAGE [3], HinSAGE, GCN [6], GAT [7], Node2Vec [1], and Metapath2Vec [2].
GraphSAGE, HinSAGE, and GAT are variants of Graph Convolutional Neural networks [6]. Node2Vec and
Metapath2Vec are methods based on graph random walks and representation learning using the
Word2Vec [5] algorithm. attri2vec[4] is also based on graph random walks and learns node
representations by performing a mapping on node attributes.

The examples folder structure is shown below.

* [`/random-walks`](https://github.com/stellargraph/stellargraph/tree/master/demos/link-prediction/random-walks)

    Examples of semi-supervised link prediction for homogeneous and heterogeneous networks,
    using the Node2Vec and Metapath2vec algorithms.

* [`/graphsage`](https://github.com/stellargraph/stellargraph/tree/master/demos/link-prediction/graphsage)

    Example of semi-supervised link prediction for a homogeneous network with attributed nodes,
    using the GraphSAGE algorithm.

* [`/hinsage`](https://github.com/stellargraph/stellargraph/tree/master/demos/link-prediction/hinsage)

    Example of supervised link attribute prediction for a heterogeneous network with attributed nodes of different types,
    using the HinSAGE algorithm.

* [`/attri2vec`] (https://github.com/stellargraph/stellargraph/tree/master/demos/link-prediction/attri2vec)

    Example of link prediction for out-of-sample nodes for a homogeneous network with attributed nodes,
    using the attri2vec algorithm.

## References

1. Node2Vec: Scalable Feature Learning for Networks. A. Grover, J. Leskovec. ACM SIGKDD International Conference on
Knowledge Discovery and Data Mining (KDD), 2016. ([link](https://snap.stanford.edu/node2vec/))

2. Metapath2Vec: Scalable Representation Learning for Heterogeneous Networks. Yuxiao Dong, Nitesh V. Chawla, and
Ananthram Swami. ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD), 135–144, 2017
([link](https://ericdongyx.github.io/metapath2vec/m2v.html))

3. Inductive Representation Learning on Large Graphs. W.L. Hamilton, R. Ying, and J. Leskovec arXiv:1706.02216
[cs.SI], 2017. ([link](http://snap.stanford.edu/graphsage/))

4. Attributed Network Embedding via Subspace Discovery. D. Zhang, J, Yin, X. Zhu and C. Zhang, arXiv:1901.04095,
[cs.SI], 2019. ([link](https://arxiv.org/abs/1901.04095))

5. Distributed representations of words and phrases and their compositionality. T. Mikolov,
I. Sutskever, K. Chen, G. S. Corrado, and J. Dean. In Advances in Neural Information Processing
 Systems (NIPS), pp. 3111-3119, 2013. ([link](https://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf))

6. Semi-Supervised Classification with Graph Convolutional Networks. T. Kipf, M. Welling.
ICLR 2017. arXiv:1609.02907 ([link](https://arxiv.org/abs/1609.02907))

7. Graph Attention Networks. P. Velickovic et al. ICLR 2018 ([link](https://arxiv.org/abs/1710.10903))

8. On Calibration of Modern Neural Networks. C. Guo, G. Pleiss, Y. Sun, and K. Q. Weinberger.
ICML 2017. ([link](https://geoffpleiss.com/nn_calibration))
