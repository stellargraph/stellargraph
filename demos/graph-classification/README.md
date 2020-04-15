# Graph classification using StellarGraph

[StellarGraph](https://github.com/stellargraph/stellargraph) provides an algorithm for graph classification. This folder contains a demo to explain how it works and how to use it as part of a TensorFlow Keras data science workflow.

A graph classification task predicts an attribute of each graph in a collection of graphs. For instance, labelling each graph with a categorical class (binary classification or multiclass classification), or predicting a continuous number (regression). It is supervised or semi-supervised, where the model is trained using a subset of graphs that have ground-truth labels.

## Find algorithms and demos for a collection of graphs

This table lists all graph classification demos, including the algorithms trained and the types of graphs used.

| demo | algorithm(s) | node features | inductive |
|---|---|---|---|
| [GCN Supervised Graph Classification][supervised-gcn] | GCN, mean pooling | yes | yes |

[supervised-gcn]: supervised-graph-classification.ipynb

See [the demo README](../README.md) for more tasks, and a summary of each algorithm.
