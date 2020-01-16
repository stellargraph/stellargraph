## Node classification using Node2Vec [1]

This folder contains two [Jupyter](http://jupyter.org/) python notebooks demonstrating the combined use of
`stellargraph` (this library), `Gensim` [4], and `Scikit-learn` [3] libraries for node classification in a
homogeneous graph.

The examples demonstrate how to calculate node embedding vectors in just a few lines of Python code using the
`Node2Vec` [1] algorithm on both weighted and unweighted graphs.

The learned node representations are then used in a node classification task. Specifically, the example demonstrates
how to predict the subject of a research paper given a paper citation network. The latter is a homogeneous graph
with nodes representing research papers that have a single attribute, namely the subject of the paper. Links in the
graph represent a citation relationship between two papers, i.e., `paper A` cites `paper B`. The graph is
treated as undirected.

## References

**1.** Node2Vec: Scalable Feature Learning for Networks. A. Grover, J. Leskovec. ACM SIGKDD International Conference
on Knowledge Discovery and Data Mining (KDD), 2016. ([link](https://snap.stanford.edu/node2vec/))

**2.** Gensim: Topic modelling for humans. ([link](https://radimrehurek.com/gensim/))

**3.** Scikit-learn: Machine learning in Python ([link](http://scikit-learn.org/stable/))
