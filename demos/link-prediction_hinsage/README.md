# Hinsage Recommender -- Movielens Example

This is an example of using Heterogeneous GraphSAGE (HinSAGE) as a hybrid recommender
system for the Movielens dataset.

## Requirements
Install the Stellar ML library.

In top-level stellar-ml directory install using pip:

```
pip install -e .
```

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

## Running the script

Run the example for ML-100k with movie & user features using the following command:
```
python movielens-recommender.py --data_path=../data/ml-100k
```

This examples trains HinSAGE to predict the "score" attribute on links. This
example runs for 10 epochs, training a heterogeneous GraphSAGE (HinSAGE) model
with a link regression predictor on top, with default parameters 
that specify the model architecture.

There are a number of other command line options that affect the training of the
model.  Use the `--help` option to see the list of these commands.

## References

```
 @inproceedings{hamilton2017inductive,
     author = {Hamilton, William L. and Ying, Rex and Leskovec, Jure},
     title = {Inductive Representation Learning on Large Graphs},
     booktitle = {NIPS},
     year = {2017}
   }
```
