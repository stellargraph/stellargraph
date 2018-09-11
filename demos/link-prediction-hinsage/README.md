# Hinsage Recommender -- Movielens Example

This is an example of using Heterogeneous GraphSAGE [1] (HinSAGE) as a hybrid recommender
system, predicting user-movie ratings for the Movielens dataset.

## Requirements
This example assumes the `stellargraph` library and its requirements have been 
installed by following the installation instructions in the README 
of the library's [root directory](https://github.com/stellargraph/stellargraph).

## MovieLens Data

The MovieLens data contains a bipartite graph of Movies and Users, and edges
between them denoting reviews with scores  ranging from 1 to 5.

### Getting the data

The data for this example is the MovieLens dataset of movie ratings
collected from the MovieLens web site (http://movielens.org). 
The dataset of 100,000 ratings from 943 users on 1682 movies 
can be downloaded from [this link](https://grouplens.org/datasets/movielens/100k/).

To run the examples, extract the data into a directory, 
and adjust the command line below to have `--data_path` pointing
to the data directory.

## Running the notebook
The narrated version of this example is available in the `movielens-recommender.ipynb` notebook.
To run through the notebook, you need to launch jupyter notebook:
 - Activate the python 3.6 environment in which the 
`stellargraph` library is installed 
 - Run the following command `jupyter-notebook`, and note the ip address and port
 number at which is it listening (normally it should be http://127.0.0.1:8888/)
   - note: you may need to first install `jupyter` by running `pip install jupyter` in your python environment
 - Copy-paste the ip address obtained in the previous step into your browser. You should see 
 a directory structure of the `stellargraph` library.
 - Navigate to `/demos/link-prediction-hinsage/movielens-recommender.ipynb`, and click on
 it to launch the notebook.


## Running the script

Run the script example for ML-100k dataset with movie & user features using the following command:
```
python movielens-recommender.py --data_path=<path_to_ml-100k_dataset>
```

This examples trains HinSAGE to predict the "score" attribute on links. This
example runs for 10 epochs, training a heterogeneous GraphSAGE (HinSAGE) model
with a link regression predictor on top, with default parameters 
that specify the model architecture.

There are a number of other command line options that affect the training of the
model.  Use the `--help` option to see the list of these commands.

## References

[1]	W. L. Hamilton, R. Ying, and J. Leskovec, “Inductive representation learning on large graphs,” presented at NIPS 2017
([arXiv:1706.02216](https://arxiv.org/abs/1706.02216)).
