# Node classification using StellarGraph

[StellarGraph](https://github.com/stellargraph/stellargraph) provides numerous algorithms for doing node classification on graphs. This folder contains demos of all of them to explain how they work and how to use them.

A node classification task predicts an attribute of about each node in a graph. For instance, labelling each node with a categorical class (binary classification or multiclass classification), or predicting a continuous number (regression). It is supervised or semi-supervised, where the model is trained using a subset of nodes that have ground-truth labels.

Node classification can also be done as a downstream task from node representation learning/embeddings, by training a supervised or semi-supervised classifier against the embedding vectors. Unsupervised algorithms that can be used in this manner include random walk-based methods like Metapath2Vec. StellarGraph provides [demos of unsupervised algorithms](../embeddings), some of which include a node classification downstream task.

## Find algorithms and demos for a graph

| algorithm & demo | hetereogeneous | directed | edge weights | node features | inductive |
|---|---|---|---|---|---|
| [GCN][gcn] | see RGCN | | | yes | see Cluster-GCN |
| [Cluster-GCN][cluster-gcn] | | | | yes | yes |
| [RGCN][rgcn] | yes, multiple edge types | | | yes | |
| [GAT][gat] | | | | yes | |
| [SGC][sgc] | | | | yes | |
| [PPNP][ppnp] | | | | yes | |
| [APPNP][ppnp] | | | | yes | |
| [Attri2Vec][attri2vec] | | | | yes | |
| [GraphSAGE][graphsage] | see HinSAGE | [demo][graphsage-directed] | | yes | [demo][graphsage-inductive] |
| [HinSAGE][hinsage] | yes | | | yes | yes |
| [Node2Vec][node2vec] | | | [demo][node2vec-weighted] | | |

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

Example: suppose graph `G` has node features and multiple edge types; the algorithms that can use all this information directly are RGCN and HinSAGE. One can use additional algorithms on `G` by ignoring the edge types, allowing the use of GCN, Cluster-GCN, etc.

See [the root README](../../README.md) or each algorithm's documentation for the relevant citation(s).
