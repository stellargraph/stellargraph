## Representation Learning Examples

This folder contains two [Jupyter](http://jupyter.org/) python notebooks demonstrating the combined use of the 
`stellargraph` machine learning on graphs and `Gensim`, [4] libraries for representation learning on homogeneous and 
heterogeneous graphs. The two notebooks demonstrate the following algorithms,

- `stellargraph-node2vec.ipynb` The **Node2Vec** algorithm, [2], for representation learning on homogeneous graphs
- `stellargraph-metapath2vec.ipynb` The **Metapath2Vec** algorithm, [1], for representation learning on heterogeneous graphs.

Both examples demonstrate how to calculate embedding vectors for a graph's nodes in just a few lines of Python code. 
The learned node representations can be used to solve numerous downstream tasks such as node attribute inference, link
prediction, and community detection.


## References

**1.**  Metapath2Vec: Scalable Representation Learning for Heterogeneous Networks. Yuxiao Dong, Nitesh V. Chawla, and 
Ananthram Swami. ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD), 135–144, 2017. 
([link](https://ericdongyx.github.io/papers/KDD17-dong-chawla-swami-metapath2vec.pdf))

**2.** Node2Vec: Scalable Feature Learning for Networks. A. Grover, J. Leskovec. ACM SIGKDD International Conference 
on Knowledge Discovery and Data Mining (KDD), 2016. ([link](https://snap.stanford.edu/node2vec/))

**3.** Distributed representations of words and phrases and their compositionality. T. Mikolov, I. Sutskever, K. Chen, 
G. S. Corrado, and J. Dean.  In Advances in Neural Information Processing Systems (NIPS), pp. 3111-3119, 2013. 
([link](https://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf))

**4.** Gensim: Topic modelling for humans. ([link](https://radimrehurek.com/gensim/))

