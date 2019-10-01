# attri2vec Link Prediction for Out-of-sample Nodes

This is an example of using the attri2vec [1] model, with a link classifier on top,
to predict links for out-of-sample nodes in a homogeneous citation network.

In this demo, we first train the attri2vec model on the in-sample subgraph and infer
representations for out-of-sample nodes with the trained attri2vec model. Then we use the
obtained node representations to perform link prediction for out-of-sample nodes.

The link prediction problem is treated as a supervised binary classification problem for
`(src, dst)` node pairs that make up links in the graph, with positive examples
representing links that do exist in the graph, and negative examples representing
links that don't.

In this example, we learn to predict citation links between papers in a DBLP dataset (see below).

## Requirements
This example assumes the `stellargraph` library and its requirements have been
installed by following the installation instructions in the README
of the library's [root directory](https://github.com/stellargraph/stellargraph).

## DBLP dataset

This example is tested on the DBLP dataset. The attri2vec model assumes that node
features are available.

The following is the description of the dataset:
> The DBLP citation network is a subgraph extracted from DBLP-Citation-network V3 (https://aminer.org/citation).
> To form this subgraph, papers from four subjects are extracted according to their venue information:
> Database, Data Mining, Artificial Intelligence and Computer Vision, and papers with no citations are removed.
> The DBLP network contains 18,448 papers and 45,661 citation relations. From paper titles, we construct
> 2,476-dimensional binary node feature vectors, with each element indicating the presence/absence of the corresponding word.
> By ignoring the citation direction, we take the DBLP subgraph as an undirected network.

Download and unzip the [DBLP.zip](https://www.kaggle.com/daozhang/dblp-subgraph) file to a location on your computer
and pass this location as a command line argument to this script.

## Running the notebook
The narrated version of this example is available in the `stellargraph-attri2vec-DBLP.ipynb` notebook.
To run the notebook:
 - Activate the python 3.6 environment in which the
`stellargraph` library is installed
 - Start `jupyter-notebook`
   - note: you may need to first install `jupyter` by running `pip install jupyter` in your python environment
 - Navigate to the notebook (`/demos/link-prediction/attri2vec/stellargraph-attri2vec-DBLP.ipynb`), and click on
 it to launch the notebook.

## References

[1] Attributed Network Embedding via Subspace Discovery. D. Zhang, J, Yin, X. Zhu and C. Zhang, arXiv:1901.04095,
[cs.SI], 2019. ([link](https://arxiv.org/abs/1901.04095))
