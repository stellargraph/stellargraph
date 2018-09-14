# HinSAGE Node Classification

This is an example of using a Heterogenous extension to the GraphSAGE algorithm [1], called HinSAGE,
to classify the nodes in a heterogeneous network (a network with multiple node and link types).

This example uses the Yelp dataset and aims to predict the 'elite' status of users. It does this
as a binary classification task predicting if the user has had 'elite' status in any year or has
never had 'elite' status.

## Requirements
Install the StellarGraph library following the instructions at:
https://github.com/stellargraph/stellargraph

Additional requirements are Pandas, Numpy and Scikit-Learn. These are installed as depdendencies
of the StellarGraph library.

## Yelp dataset

Currently the examples in this directory use the Yelp dataset.
The Yelp dataset can be obtained by navigating to https://www.yelp.com/dataset,
selecting "Download Dataset" and signing the licence agreement.
Then, download the JSON dataset and uncompress it to an appropriate location 
in your filesystem.

The example code uses a preprocessed version of the dataset that is generated
by the `yelp-preprocessing.py` script.
To run this script, supply the path to the Yelp dataset that you downloaded
(this location should contain `yelp_academic_dataset_user.json`)
and the output directory (-o):

Example usage:
```
python yelp_preprocessing.py -l <path_to_yelp_dataset> -o .
```

By default the script will filter the graph to contain only businesses in the state
of Wisconsin. To change this to another state, or to "false" to use the entire dataset
(warning: this will make the machine learning example run very slowly and will require a lot of
memory as the entire graph will be loaded).

Example usage to run without filtering:
```
python yelp_preprocessing.py -l <path_to_yelp_dataset> -o . --filter_state=false
```

## Running the example

The example script can be run on supplying the location of the preprocessed Yelp dataset.

Example usage:
```
python yelp-example.py -l <location_of_preprocessed_data>
```

Additional command line arguments are available to tune the training of the model, to see a
description of these arguments use the `--help` argument:
```
python yelp-example.py --help
```

## References

[1]	W. L. Hamilton, R. Ying, and J. Leskovec, “Inductive representation learning on large graphs,” presented at NIPS 2017
([arXiv:1706.02216](https://arxiv.org/abs/1706.02216)).
