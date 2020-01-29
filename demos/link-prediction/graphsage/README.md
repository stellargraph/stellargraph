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

## Running the notebook
The narrated version of this example is available in the `cora-links-example.ipynb` notebook.
To run the notebook:
 - Activate the python 3.6 environment in which the
`stellargraph` library is installed
 - Start `jupyter-notebook`
   - note: you may need to first install `jupyter` by running `pip install jupyter` in your python environment
 - Navigate to the notebook (`/demos/link-prediction/graphsage/cora-links-example.ipynb`), and click on
 it to launch the notebook.

## Running the script

You can run the script using the following command:
```
python cora-links-example.py
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
