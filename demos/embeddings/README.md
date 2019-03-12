## Representation Learning Examples

This folder contains three [Jupyter](http://jupyter.org/) python notebooks demonstrating the use of unsupervised representation learning methods implemented in the `Stellargraph` library for homogeneous and hetrogenous graphs with or without node features. The original works are referenced below. 

**Node2Vec** and **Metapath2Vec** notebooks demonstrate the combined use of `stellargraph` and `Gensim` [4] libraries for representation learning on homogeneous and heterogeneous graphs. 
**Unsupervised GraphSAGE** notebook demonstrate the use of reimplementation in the `Stellargraph` library of the GraphSAGE method for unsupervised learning for homogeneous graphs with node features.

The notebooks demonstrate the following algorithms.
- `stellargraph-node2vec.ipynb` The **Node2Vec** algorithm [1] for representation learning on homogeneous graphs
- `stellargraph-metapath2vec.ipynb` The **Metapath2Vec** algorithm [2] for representation learning on heterogeneous graphs.
- `embeddings-unsupervised-graphsage-cora.ipynb` The **Unsupervised GraphSAGE** algorithm [5] for representation learning on homogeneous graphs with node features.

All examples demonstrate how to calculate embedding vectors for a graph's nodes in just a few lines of Python code. 
The learned node representations can be used in numerous downstream tasks such as node attribute inference, link
prediction, and community detection.


## References

**1.** Node2Vec: Scalable Feature Learning for Networks. A. Grover, J. Leskovec. ACM SIGKDD International Conference 
on Knowledge Discovery and Data Mining (KDD), 2016. ([link](https://snap.stanford.edu/node2vec/))

**2.**  Metapath2Vec: Scalable Representation Learning for Heterogeneous Networks. Yuxiao Dong, Nitesh V. Chawla, and 
Ananthram Swami. ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD), 135–144, 2017. 
([link](https://ericdongyx.github.io/papers/KDD17-dong-chawla-swami-metapath2vec.pdf))

**3.** Distributed representations of words and phrases and their compositionality. T. Mikolov, I. Sutskever, K. Chen, 
G. S. Corrado, and J. Dean.  In Advances in Neural Information Processing Systems (NIPS), pp. 3111-3119, 2013. 
([link](https://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf))

**4.** Gensim: Topic modelling for humans. ([link](https://radimrehurek.com/gensim/))

**5.** Inductive Representation Learning on Large Graphs. W.L. Hamilton, R. Ying, and J. Leskovec arXiv:1706.02216
[cs.SI], 2017. ([link](http://snap.stanford.edu/graphsage/))
