## Node classification using attri2vec [1]

This folder contains a [Jupyter](http://jupyter.org/) python notebook demonstrating the combined use of
`stellargraph` (this library) and `Scikit-learn` [2] libraries for node classification in a homogeneous graph
attached with node attributes.

The example demonstrates node representation learning and node classification using the citeseer
paper citation network. This demo is included in the Jupyter notebook
`attri2vec-citeseer-node-classification-example.ipynb`.

The notebook includes all the information for downloading the corresponding dataset, training the attri2vec
model and using it to classify nodes with unknown (to the training algorithm) labels.

To run the notebook, install Jupyter to the same Python 3.6 environment as StellarGraph, following the instructions on
the Jupyter project website: http://jupyter.org/install.html

After starting the Jupyter server on your computer, load the notebook and follow the instructions inside.

## Requirements

The example uses Python 3.6 and the StellarGraph library. To install the StellarGraph library
follow the instructions at: https://github.com/stellargraph/stellargraph

Additional requirements are Pandas, Numpy and Scikit-Learn which are installed as dependencies
of the StellarGraph library. In addition, Juptyer is required to run the notebook version of
the example.

## References

**1.** Attributed Network Embedding via Subspace Discovery. D. Zhang, J, Yin, X. Zhu and C. Zhang, arXiv:1901.04095,
[cs.SI], 2019. ([link](https://arxiv.org/abs/1901.04095))

**2.** Scikit-learn: Machine learning in Python ([link](http://scikit-learn.org/stable/))
