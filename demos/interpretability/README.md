# Interpretability of node classification results using StellarGraph

[StellarGraph](https://github.com/stellargraph/stellargraph) has support for inspecting several different algorithms for [node classification][nc] to understand or interpret how they came to a decision. This folder contains demos of all of them to explain how they work and how to use them as part of a TensorFlow Keras data science workflow.

Interpreting a model involves training and making predictions for a model, and then analysing the predictions and the model to find which neighbours and which features had the largest influence on the prediction.

## Find algorithms and demos

- [GCN (dense)][gcn-dense]
- [GCN (sparse)][gcn-sparse]
- [GAT][GAT]

[gcn-dense]: gcn/node-link-importance-demo-gcn.ipynb
[gcn-sparse]: gcn/node-link-importance-demo-gcn-sparse.ipynb
[gat]: node-link-importance-demo-gat.ipynb

See [the root README](../../README.md) or each algorithm's documentation for the relevant citation(s), and [the node classification demos][nc] for more details on the base task.

[nc]: ../node-classification/README.md
