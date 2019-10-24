## Table of Contents

This folder contains several examples of applying graph machine learning (ML) algorithms on network-structured
data to solve node attribute inference (inferring entity properties) problems. The
examples demonstrate using the `StellarGraph` library to build machine learning
workflows on both homogeneous and heterogeneous networks.

Each folder contains one or more examples of using the StellarGraph implementations of the
state-of-the-art algorithms, attri2vec[4], GraphSAGE [3], HinSAGE, GCN [6], GAT [7], PPNP/APPNP [10], SGC [9], 
Node2Vec [1], and Metapath2Vec [2].
GraphSAGE, HinSAGE, and GAT are variants of Graph Convolutional Neural networks [6]. Node2Vec and
Metapath2Vec are methods based on graph random walks and representation learning using the
Word2Vec [5] algorithm. attri2vec[4] is also based on graph random walks, and learns node
representations by performing a mapping on node attributes.

The examples folder structure is shown below.

* [`/attri2vec`](https://github.com/stellargraph/stellargraph/tree/master/demos/node-classification/graphsage)

    Examples of supervised node classification for two homogeneous networks with attributes, using the attri2vec algorithm [4].

* [`/graphsage`](https://github.com/stellargraph/stellargraph/tree/master/demos/node-classification/graphsage)

    Example of supervised node classification for a homogeneous network with attributed nodes, using the GraphSAGE algorithm [3].

* [`/gcn`](https://github.com/stellargraph/stellargraph/tree/master/demos/node-classification/gcn)

    Example of semi-supervised node classification for a homogeneous network, using the GCN algorithm [6].

* [`/sgc`](https://github.com/stellargraph/stellargraph/tree/master/demos/node-classification/sgc)

    Example of semi-supervised node classification for a homogeneous network, using the SGC algorithm [9].

* [`/gat`](https://github.com/stellargraph/stellargraph/tree/master/demos/node-classification/gat)

    Example of supervised node classification for a homogeneous network with attributed nodes, using the GAT algorithm [7].
   
* [`/ppnp`](https://github.com/stellargraph/stellargraph/tree/master/demos/node-classification/ppnp)

    Example of supervised node classification for a homogeneous network with attributed nodes, using the PPNP and 
    APPNP algorithms [10].


* [`/node2vec`](https://github.com/stellargraph/stellargraph/tree/master/demos/node-classification/node2vec)

    Example of unsupervised node representation learning using Node2Vec and supervised classification using
    the Scikit-learn library.

* [`/hinsage`](https://github.com/stellargraph/stellargraph/tree/master/demos/node-classification/hinsage)

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

9. Simplifying Graph Convolutional Networks. F. Wu, T. Zhang, A. H. de Souza, C. Fifty, T. Yu, and K. Q. Weinberger.
arXiv:1902.07153. ([link](https://arxiv.org/abs/1902.07153))

10.	Predict then propagate: Graph neural networks meet personalized PageRank. J. Klicpera, A. Bojchevski, A., and S. Günnemann,
 2018, arXiv:1810.05997.([link](https://arxiv.org/abs/1810.05997))
