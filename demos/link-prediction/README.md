# Link prediction using StellarGraph

[StellarGraph](https://github.com/stellargraph/stellargraph) provides numerous algorithms for doing link prediction on graphs. This folder contains demos of all of them to explain how they work and how to use them as part of a TensorFlow Keras data science workflow.

A link prediction task predicts an attribute of links/edges in a graph. For instance, predicting whether a link/edge that isn't already in the graph should exist (binary classification, or recommendation, or knowledge base completion, in a knowledge graph), or even labelling existing links with a categorical class (binary classification or multiclass classification), or predicting a continuous number (regression). It is supervised or semi-supervised, where the model is trained using a subset of links/edges that have ground-truth labels. For predicting edge existence, the ground-truth may just be whether the edge exists in the original data, rather than a separate label.

Link prediction can also be done as a downstream task from node representation learning/embeddings, by combining node embedding vectors for the source and target nodes of the edge and training a supervised or semi-supervised classifier against the result. Unsupervised algorithms that can be used in this manner include random walk-based methods like Metapath2Vec. StellarGraph provides [demos of unsupervised algorithms](../embeddings).

## Find algorithms and demos for a graph

This table lists all node classification demos, including the algorithms trained and the types of graph used.

| Demo | Algorithm | Node features | Heterogeneous | Temporal |
|---|---|---|---|---|
| [GCN][gcn] | GCN | yes | | |
| [Attri2Vec][attri2vec] | Attri2Vec | yes | | |
| [GraphSAGE][graphsage] | GraphSAGE | yes | | |
| [HinSAGE][hinsage] | HinSAGE | yes | yes | |
| [Node2Vec][node2vec] | Node2Vec | | | |
| [CTDNE][ctdne] | CTDNE | | | yes |
| [ComplEx][complex] | ComplEx | | yes, multiple edge types | |
| [DistMult][distmult] | DistMult | | yes, multiple edge types | |

[gcn]: gcn/cora-gcn-links-example.ipynb
[attri2vec]: attri2vec/stellargraph-attri2vec-DBLP.ipynb
[graphsage]: graphsage/cora-links-example.ipynb
[hinsage]: hinsage/movielens-recommender.ipynb
[node2vec]: random-walks/cora-lp-demo.ipynb
[ctdne]: random-walks/ctdne-link-prediction.ipynb
[complex]: knowledge-graphs/complex.ipynb
[distmult]: knowledge-graphs/distmult.ipynb

See [the root README](../../README.md) or each algorithm's documentation for the relevant citation(s). See [the demo README](../README.md) for more tasks, and a summary of each algorithm.
