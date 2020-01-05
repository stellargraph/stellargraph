# GraphSAGE Node Classification

This folder includes two examples of using the GraphSAGE algorithm [1] for semi-supervised node classification in
homogeneous networks.

The first example demonstrates transductive node classification using the Cora citation network. This demo is included
as the Python script `graphsage-cora-example.py` and as the Jupyter
notebook `graphsage-cora-node-classification-example.ipynb`.

The second example demonstrates inductive representation learning and node classification using the Pubmed-Diabetes
paper citation network. This demo is included in the Jupyter notebook
`graphsage-pubmed-inductive-node-classification-example.ipynb`.

The two Jupyter notebooks include all the information for downloading the corresponding datasets, training the GraphSAGE
models and using them to classify nodes with unknown (to the training algorithm) labels.

To run the notebooks install Jupyter to the same Python 3.6 environment as StellarGraph, following the instructions on
the Jupyter project website: http://jupyter.org/install.html

After starting the Jupyter server on your computer, load either of the two notebooks and follow the instructions inside.

Instructions for downloading the Cora dataset and running the script `graphsage-cora-example.py` follow.

## Requirements

All examples use Python 3.6 and the StellarGraph library. To install the StellarGraph library
follow the instructions at: https://github.com/stellargraph/stellargraph

Additional requirements are Pandas, Numpy and Scikit-Learn which are installed as depdendencies
of the StellarGraph library. In addition Juptyer is required to run the notebook version of
the example.

## CORA dataset

Currently the examples in this directory are tested on the CORA dataset. The GraphSAGE model assumes that node
features are available for all nodes in the graph.

The dataset can be downloaded from https://linqs-data.soe.ucsc.edu/public/lbc/cora.tgz

The following is the description of the dataset:

> The Cora dataset consists of 2708 scientific publications classified into one of seven classes.
> The citation network consists of 5429 links. Each publication in the dataset is described by a
> 0/1-valued word vector indicating the absence/presence of the corresponding word from the dictionary.
> The dictionary consists of 1433 unique words. The README file in the dataset provides more details.

Download and unzip the [cora.tgz](https://linqs-data.soe.ucsc.edu/public/lbc/cora.tgz) file to a location on your computer and pass this location
as a command line argument to this script.

## Running the script

The example script can be run on supplying the location of the downloaded CORA dataset
with the following command:

```
python graphsage-cora-example.py -l <path_to_cora_dataset>
```

Additional arguments can be specified that change the GraphSAGE model and training parameters, a
description of these arguments is displayed using the help option to the script:

```
python graphsage-cora-example.py --help
```

## References

[1] W. L. Hamilton, R. Ying, and J. Leskovec, “Inductive representation learning on large graphs,” presented at NIPS 2017
([arXiv:1706.02216](https://arxiv.org/abs/1706.02216)).
