## Representation learning using StellarGraph

[StellarGraph](https://github.com/stellargraph/stellargraph) provides numerous algorithms for doing node and edge representation learning on graphs. This folder contains demos of all of them to explain how they work and how to use them as part of a TensorFlow Keras data science workflow.

A node representation learning task computes a representation or embedding vector for each node in a graph. These vectors capture latent/hidden information about the nodes and edges, and can be used for (semi-)supervised downstream tasks like [node classification][nc] and [link prediction][lp], or unsupervised ones like [community detection][cd] or similarity searches. Representation learning is typically an unsupervised task, where the model is trained on data that does not have any ground-truth labels.

Node representations can also be computed from (semi-)supervised models, using the output of a hidden layer as the embedding vector for nodes or edges. StellarGraph provides some [demonstrations of node classification][nc] and [link prediction][lp], some of which include computing and visualising node or edge embeddings.

## Find algorithms and demos for a graph

| demo | algorithm(s) | training method | downstream tasks shown | node features |
|---|---|---|---|---|
| [Deep Graph Infomax][dgi] | GCN, GAT, PPNP, APPNP, GraphSAGE | `DeepGraphInfomax` (mutual information) | visualisation, node classification | yes |
| [Unsupervised GraphSAGE][graphsage] | GraphSAGE | `UnsupervisedSampler` (link prediction) | visualisation, node classification | yes |
| [Attri2Vec][attri2vec] | Attri2Vec | `UnsupervisedSampler` (link prediction) | visualisation | yes |
| [Metapath2Vec][metapath2vec] | Metapath2Vec | natively unsupervised | visualisation | |
| [Node2Vec][node2vec] | Node2Vec | natively unsupervised | visualisation | |
| [Watch Your Step][wys] | Watch Your Step | natively unsupervised | visualisation, node classification | |
| [GraphWave][graphwave] | GraphWave | natively unsupervised | visualisation, node classification | |

[dgi]: deep-graph-infomax-cora.ipynb
[graphsage]: embeddings-unsupervised-graphsage-cora.ipynb
[graphwave]: graphwave-barbell.ipynb
[attri2vec]: stellargraph-attri2vec-citeseer.ipynb
[metapath2vec]: stellargraph-metapath2vec.ipynb
[node2vec]: stellargraph-node2vec.ipynb
[wys]: watch-your-step-cora-demo.ipynb
