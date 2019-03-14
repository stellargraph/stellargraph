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

## Dataset

The examples in this directory uses the CORA dataset. 

The dataset can be downloaded from [here](https://linqs-data.soe.ucsc.edu/public/lbc/cora.tgz).

The following is the description of the dataset:
> The Cora dataset consists of 2708 scientific publications classified into one of seven classes.
> The citation network consists of 5429 links. Each publication in the dataset is described by a
> 0/1-valued word vector indicating the absence/presence of the corresponding word from the dictionary.
> The dictionary consists of 1433 unique words. The README file in the dataset provides more details.

Download and unzip the [cora.tgz](https://linqs-data.soe.ucsc.edu/public/lbc/cora.tgz) file to a location on your 
computer and pass this location as a command line argument to this script.

## References

**1.** Node2Vec: Scalable Feature Learning for Networks. A. Grover, J. Leskovec. ACM SIGKDD International Conference 
on Knowledge Discovery and Data Mining (KDD), 2016. ([link](https://snap.stanford.edu/node2vec/))

**2.** Gensim: Topic modelling for humans. ([link](https://radimrehurek.com/gensim/))

**3.** Scikit-learn: Machine learning in Python ([link](http://scikit-learn.org/stable/))
