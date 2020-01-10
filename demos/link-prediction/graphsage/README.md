# Graphsage Link Prediction

This is an example of using the GraphSAGE [1] model, with a link classifier on top,
to predict links in a homogeneous citation network.

The link prediction problem is treated as a supervised binary classification problem for
`(src, dst)` node pairs that make up links in the graph, with positive examples
representing links that do exist in the graph, and negative examples representing
links that don't.

In this example, we learn to predict citation links between papers in a Cora dataset (see below).

## Requirements
This example assumes the `stellargraph` library and its requirements have been
installed by following the installation instructions in the README
of the library's [root directory](https://github.com/stellargraph/stellargraph).

## CORA dataset

Currently this example is tested on the CORA dataset. The GraphSAGE model assumes that node
features are available.

The following is the description of the dataset:
> The Cora dataset consists of 2708 scientific publications classified into one of seven classes.
> The citation network consists of 5429 links. Each publication in the dataset is described by a
> 0/1-valued word vector indicating the absence/presence of the corresponding word from the dictionary.
> The dictionary consists of 1433 unique words. The README file in the dataset provides more details.

Download and unzip the [cora.tgz](https://linqs-data.soe.ucsc.edu/public/lbc/cora.tgz) file to a location on your computer and pass this location
as a command line argument to this script.

## Running the notebook
The narrated version of this example is available in the `cora-links-example.ipynb` notebook.
To run the notebook:
 - Activate the python 3.6 environment in which the
`stellargraph` library is installed
 - Start `jupyter-notebook`
   - note: you may need to first install `jupyter` by running `pip install jupyter` in your python environment
 - Navigate to the notebook (`/demos/link-prediction-hinsage/movielens-recommender.ipynb`), and click on
 it to launch the notebook.

## Running the script

The example should be run by specifying the location of the downloaded CORA dataset using the `-g` command line
argument. You can run the script using the following command:
```
python cora-links-example.py -l <path_to_cora_dataset>
```
The above command runs the link prediction on Cora dataset with default parameters. There is a number of other command
line options that affect the architecture and training of the odel. For help on how to set parameters of the run, and
on parameter meaning, run
```
python cora-links-example.py --help
```

## References

[1]	W. L. Hamilton, R. Ying, and J. Leskovec, “Inductive representation learning on large graphs,” presented at NIPS 2017
([arXiv:1706.02216](https://arxiv.org/abs/1706.02216)).
