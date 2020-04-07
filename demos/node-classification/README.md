# Node classification using StellarGraph

[StellarGraph](https://github.com/stellargraph/stellargraph) provides numerous algorithms for doing node classification on graphs. This folder contains demos of all of them to explain how they work and how to use them as part of a TensorFlow Keras data science workflow.

A node classification task predicts an attribute of each node in a graph. For instance, labelling each node with a categorical class (binary classification or multiclass classification), or predicting a continuous number (regression). It is supervised or semi-supervised, where the model is trained using a subset of nodes that have ground-truth labels.

Node classification can also be done as a downstream task from node representation learning/embeddings, by training a supervised or semi-supervised classifier against the embedding vectors. Unsupervised algorithms that can be used in this manner include random walk-based methods like Metapath2Vec. StellarGraph provides [demos of unsupervised algorithms](../embeddings), some of which include a node classification downstream task.

## Find algorithms and demos for a graph

This table lists all node classification demos, including the algorithms trained, the types of graph used, and the tasks demonstrated.

| Demo | Algorithm(s) | Node features | Hetereogeneous | Directed | Edge weights | Inductive | Node embeddings |
|---|---|---|---|---|---|---|---|
| [GCN][gcn] | GCN | yes | | | | | yes |
| [Cluster-GCN][cluster-gcn] | Cluster-GCN | yes | | | | | yes |
| [RGCN][rgcn] | RGCN | yes | yes, multiple edge types | | | | yes |
| [GAT][gat] | GAT | yes | | | | | yes |
| [SGC][sgc] | SGC | yes | | | | | yes |
| [PPNP & APPNP][ppnp] | PPNP, APPNP | yes | | | | | |
| [Attri2Vec][attri2vec] | Attri2Vec | yes | | | | | yes |
| [GraphSAGE on Cora][graphsage] | GraphSAGE | yes | | | | | yes |
| [Inductive GraphSAGE][graphsage-inductive] | GraphSAGE | yes | | | | yes | yes |
| [Directed GraphSAGE][graphsage-directed] | GraphSAGE | yes | | yes | | | yes |
| [HinSAGE][hinsage] | HinSAGE | yes | yes | | | | |
| [Node2Vec][node2vec] | Node2Vec | | | | | | yes |
| [Weighted Node2Vec][node2vec-weighted] | Node2Vec | | | | yes | | yes |

[gcn]: gcn/gcn-cora-node-classification-example.ipynb
[cluster-gcn]: cluster-gcn/cluster-gcn-node-classification.ipynb
[rgcn]: rgcn/rgcn-aifb-node-classification-example.ipynb
[gat]: gat/gat-cora-node-classification-example.ipynb
[sgc]: sgc/sgc-node-classification-example.ipynb
[ppnp]: ppnp/ppnp-cora-node-classification-example.ipynb
[attri2vec]: attri2vec/attri2vec-citeseer-node-classification-example.ipynb
[graphsage]: graphsage/graphsage-cora-node-classification-example.ipynb
[graphsage-inductive]: graphsage/graphsage-pubmed-inductive-node-classification-example.ipynb
[graphsage-directed]: graphsage/directed-graphsage-on-cora-example.ipynb
[hinsage]: hinsage/README.md
[node2vec]: node2vec/stellargraph-node2vec-node-classification.ipynb
[node2vec-weighted]: node2vec/stellargraph-node2vec-weighted-random-walks.ipynb

See [the root README](../../README.md) or each algorithm's documentation for the relevant citation(s). See [the demo README](../README.md) for more tasks, and a summary of each algorithm.
