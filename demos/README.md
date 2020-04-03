# StellarGraph demos


[StellarGraph](https://github.com/stellargraph/stellargraph) provides numerous algorithms for graph machine learning. This folder contains demos of all of them to explain how they work and how to use them.

The demo notebooks can be run without any installation of Python by using Binder or Google Colab - these both provide a cloud-based notebook environment.  The whole set of demos can be opened in Binder here: [![Open in Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/stellargraph/stellargraph/master?urlpath=lab/tree/demos/README.md) or you can click the Binder and Colab badges within each notebook.

## Find algorithms for a task

- Predicting attributes, such as classifying as a class or label, or regressing to calculate a continuous number:
  - For nodes/vertices/entities: [**node classification**](./node-classification)
  - For edges/links/connections: [**link prediction**](./link-prediction) (includes knowledge graph completion)
  - For graphs/networks: [**graph classification**](./graph-classification)
  - Adjusting predictions scores to be probabilities (for any model): [**calibration**](./calibration)
- Representation learning or computing embedding vectors (including unsupervised tasks):
  - For nodes/vertices/entities and edges/links/connections: [**embeddings**](./embeddings) (and [**community detection**](./community_detection) for using these to cluster nodes)
  - For graphs/networks: [**graph classification**](./graph-classification) (only supervised)
- Ensembling models to reduce prediction variance: [**ensembles**](./ensembles)
- Loading data into a `StellarGraph` object, with Pandas or NetworkX: [**basics**](./basics)
- Experimental: running GraphSAGE on data stored in Neo4j: [**neo4j connector**](./connector/neo4j)

## Find a demo for an algorithm

| algorithm                                                                 | *heter.*      | *EW* | *NF* | NC | I(NC)  | LP | RL        | ind.            | GC |
|---------------------------------------------------------------------------|-------------|----|----|----|---|----|-----------|-----------------|----|
| GCN (Graph Convolutional Network)                                         | see RGCN    |    | ✔️ | ✅ |   | ✅ | ✅ US DGI | see Cluster-GCN | ✅ |
| Cluster-GCN                                                               |             |    | ✔️ | ✅ |   | ✅ |           | ✅              |    |
| RGCN (Relational GCN)                                                     | ✔️         |    | ✔️ | ✅ |   | ✅ |           |                 |    |
| GAT (Graph ATtention Network)                                             |             |    | ✔️ | ✅ |   | ✅ | ✅ US DGI |                 |    |
| SGC (Simplified Graph Convolution)                                        |             |    | ✔️| ✅ |   | ✅ |           |                 |    |
| APPNP/PPNP ((Approximate) Personalized Propagation of Neural Predictions) |             |    | ✔️ | ✅ |   | ✅ | ✅ US DGI |                 |    |
| GraphWave                                                                 |             |    |    | ☑️  |   | ☑️  | ✅        |                 |    |
| Attri2Vec                                                                 |             |    | ✔️ | ☑️  |   | ☑️  | ✅        |                 |    |
| **Sampling methods**                                                      |             |    |    |    |   |    |           |                 |    |
| GraphSAGE                                                                 | see HinSAGE |    | ✔️ | ✅ |   | ✅ | ✅ US DGI | ✅              |    |
| HinSAGE                                                                   | ✔️         |    | ✔️ | ✅ |   | ✅ |           | ✅              |    |
| **Random walks**                                                          |             |    |    |    |   |    |           |                 |    |
| Node2Vec                                                                  |             | ✔️ |    | ☑️  |   | ☑️  | ✅        |                 |    |
| MetaPath2Vec                                                              | ✔️          |    |    | ☑️  |   | ☑️  | ✅        |                 |    |
| CTDNE (Continuous-Time Dynamic Network Embeddings)                        |             |    |    | ☑️  |   | ☑️  | ✅        |                 |    |
| Watch Your Step (simulated random walks)                                  |             |    |    | ☑️  |   | ☑️  | ✅        |                 |    |
| **Knowledge graphs**                                                      |             |    |    |    |   |    |           |                 |    |
| ComplEx                                                                   | ✔️          |    |    | ☑️  |   | ✅ | ☑️         |                 |    |
| DistMult                                                                  | ✔️          |    |    | ☑️  |   | ✅ | ☑️         |                 |    |

| abbreviation | explanation                                                                                                                                                                                                             |
|--------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| heter.       | Heterogeneous graphs. Algorithms without this support can still be used with heterogeneous graphs by ignoring the types.                                                                                                |
| EW           | Edge weights. Algorithms without this support still work on weighted graphs, by ignoring the weights.                                                                                                                   |
| NF           | Node feature vectors. Algorithms without this support still work on graphs with features, by ignoring the features.                                                                                                     |
| ✔️            | Algorithm can use this information about a graph                                                                                                                                                                        |
| NC           | Node classification, predicting attributes on nodes/vertices/entities.                                                                                                                                                  |
| INC          | Interpretability for node classification, providing insight into which features and links influence the prediction of an attribute.                                                                                     |
| LP           | Link prediction, predicting attributes on links/edges/connections.                                                                                                                                                      |
| RL           | Representation learning, computing embedding vectors for nodes and links (by combining node embeddings), usually unsupervised. These embeddings can be used for downstream tasks like NC and LP, and GC (with pooling). |
| GC           | Graph classification, predicting attributes                                                                                                                                                                             |
| ind.         | Inductive, the algorithm generalises to new entities not seen during training.                                                                                                                                          |
| DGI          | `DeepGraphInfomax`, a method for doing unsupervised training using mutual information.                                                                                                                                  |
| US           | `UnsupervisedSampler`, a method for doing unsupervised training by creating a link prediction problem with random walks.                                                                                                |
| ✅           | Demo available (and link).                                                                                                                                                                                              |
| ☑️            | Supported without an explicit demo, such as training a logistic regression model on node embedding vectors (RL) to do node classification (NC).                                                                         |

See [the root README](../README.md) or each algorithm's documentation for the relevant citation(s).
