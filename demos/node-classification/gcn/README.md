# GCN for Node Classification

This is an example of using the Graph Convolutional network (GCN) algorithm [1] for semi-supervised node classification
in a homogeneous network.

## Requirements
All examples use Python 3.6 and the StellarGraph library. To install the StellarGraph library
follow the instructions at: https://github.com/stellargraph/stellargraph

Additional requirements are Pandas, Numpy and Scikit-Learn which are installed as depdendencies
of the StellarGraph library.

## Running the script

The example script can be run with the following command:
```
python gcn-cora-example.py
```

Additional arguments can be specified that change the GCN model architecture and training parameters, a
description of these arguments is displayed using the help option to the script:
```
python gcn-cora-example.py --help
```

## References

[1]	Semi-Supervised Classification with Graph Convolutional Networks. T. Kipf, M. Welling.
ICLR 2017. arXiv:1609.02907 ([link](https://arxiv.org/abs/1609.02907))
