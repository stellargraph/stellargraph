# PPNP and APPNP for Node Classification

This is an example of using the Personalized Propogation of Neural Predictions (PPNP) and Approximate PPNP (APPNP)
algorithms [1] for semi-supervised node classification in a homogeneous network.

## Requirements
All examples use Python 3.6 and the StellarGraph library. To install the StellarGraph library
follow the instructions at: https://github.com/stellargraph/stellargraph

Additional requirements are Pandas, Numpy, Keras, and Scikit-Learn which are installed as depdendencies
of the StellarGraph library.

## CORA dataset

Currently the examples in this directory are tested on the CORA dataset. The PPNP model assumes that node
features are available for all nodes in the graph.

The dataset can be downloaded from https://linqs-data.soe.ucsc.edu/public/lbc/cora.tgz

The following is the description of the dataset:
> The Cora dataset consists of 2708 scientific publications classified into one of seven classes.
> The citation network consists of 5429 links. Each publication in the dataset is described by a
> 0/1-valued word vector indicating the absence/presence of the corresponding word from the dictionary.
> The dictionary consists of 1433 unique words. The README file in the dataset provides more details.

Download and unzip the [cora.tgz](https://linqs-data.soe.ucsc.edu/public/lbc/cora.tgz) file to a location on your computer and pass this location
as a command line argument to this script.

## References

[1]	Predict then propagate: Graph neural networks meet personalized PageRank. J. Klicpera, A. Bojchevski, and S. Günnemann, S., ICLR, 2019. ([link](https://arxiv.org/abs/1810.05997))
