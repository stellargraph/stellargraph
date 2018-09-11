## Table of Contents

This folder contains several examples of applying machine learning algorithms on graph-structured
data to solve several common problems including node attribute inference and link prediction. The
examples demonstrate using the StellarGraph library on both homogeneous and heterogeneous tasks.

Each folder contains one or more examples of using the StellarGraph implementations of the
state-of-the-art algorithms, GraphSAGE [3], HinSAGE, Node2Vec [1], and Metapath2Vec [2]. 
GraphSAGE and HinSAGE are methods based on graph convolutional neural networks. Node2Vec and
Metapath2Vec are methods based on graph random walks and representation learning using the
Word2Vec [4] algorithm.

* `embeddings`

    Examples for unsupervised node representation learning using the Node2Vec and Metapath2Vec 
    algorithms.

* `link-prediction-random-walks`

    Examples for unsupervised link prediction using the Node2Vec and Metapath2vec
    algorithms.
    
* `link-prediction_graphsage`

    Example for supervised link prediction with node features using the GraphSAGE algorithm.
    
* `link-prediction_hinsage`

    Example for supervised link prediction with node features using the HinSAGE algorithm.
    
* `node-classification`

    Examples for supervised and unsupervised node classification using the GraphSAGE and
    Node2Vec algorithms respectively.

* `node-classification-hinsage`

    Example for supervised node classification with node features using the GraphSAGE algorithm.


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

