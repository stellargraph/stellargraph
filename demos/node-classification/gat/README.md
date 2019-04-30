# GAT for Node Classification

This is an example of using the Graph Attention network (GAT) algorithm [1] for semi-supervised node classification 
in a homogeneous network.

## Requirements
All examples use Python 3.6 and the StellarGraph library. To install the StellarGraph library
follow the instructions at: https://github.com/stellargraph/stellargraph

Additional requirements are Pandas, Numpy and Scikit-Learn which are installed as depdendencies
of the StellarGraph library. In addition Juptyer is required to run the notebook version of
the example.

## CORA dataset

Currently the examples in this directory are tested on the CORA dataset. The GAT model assumes that node
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
python gat-cora-example.py -l <path_to_cora_dataset>
```

Additional arguments can be specified that change the GAT model architecture and training parameters, a
description of these arguments is displayed using the help option to the script:
```
python gat-cora-example.py --help
```

## Running the notebook

The same example is also available as a Juptyer notebook. To use this install Jupyter to the
same Python 3.6 environment as StellarGraph, following the instructions on the Jupyter project
website: http://jupyter.org/install.html

After starting the Jupyter server on your computer, load the notebook
`gat-cora-node-classification-example.ipynb` and follow the instructions inside.


## References

[1]	Graph Attention Networks. P. Velickovic et al. ICLR 2018 ([link](https://arxiv.org/abs/1710.10903))
