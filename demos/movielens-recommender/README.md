# Graphsage Recommender -- Movielens Example

This is an example of using Heterogeneous GraphSAGE as a hybrid recommender
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

The data for the movielens example has been pre-processed and stored as pickle
files to be directly loaded into Python. The input files required are the
graph file as a networkx graph and the input node features stored as a numpy
array.

In the examples below the data is expected to be in the current directory.
To run the examples, copy the relevant files to the directory containing
the examples folder, or adjust the command line below.

There are two movielens datasets that have been preprepared:

1. Movelens 100k

    This dataset contains a subset of 100,000 ratings. The files for this dataset
    are as follows:

     * ml-100k_split_graphnx.pkl:
        The networkx graph of the movielens data, this file includes the ratings
        and the test/train split for all nodes and the index of each node in the
        features array.  

     * ml-100k_embeddings.pkl:
        This is a numpy matrix of size (N_nodes x N_features) containing the
        node2vec embeddings calculated for each node in the graph.

2. Movelens 1m

    This dataset contains a subset of 1 million ratings. The files for this dataset
    are as follows:

     * ml-1m_split_graphnx.pkl:
        The networkx graph of the movielens data, this file includes the ratings
        and the test/train split for all nodes and the index of each node in the
        features array.  

     * ml-1m_embeddings.pkl:
        This is a numpy matrix of size (N_nodes x N_features) containing the
        node2vec embeddings calculated for each node in the graph.

     * ml-1m_features.pkl:
        This is a numpy matrix of size (N_nodes x N_features) containing the
        node2vec embeddings calculated for each node in the graph.

## Running the script

Run the example for ML-1m with movie & user features using the following command:
```
python movielens-example.py --graph=ml-1m_split_graphnx.pkl
    --features=ml-1m_features.pkl --epochs 10
```

Run the example for ML-1m with node2vec embeddings using the following command:
```
python movielens-example.py --graph=ml-1m_split_graphnx.pkl
    --features=ml-1m_embeddings.pkl --epochs 10
```

These examples trains HinSAGE to predict the "score" attribute on links. This
example runs for 10 epochs and may run for  around 1 hour on a laptop. The
baseline option is required to perform well in a collaborative filtering
comparison, this learns a baseline offset for each movie & user.

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
