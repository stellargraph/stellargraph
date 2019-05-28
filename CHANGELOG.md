# Change Log

## [Unreleased](https://github.com/stellargraph/stellargraph/tree/HEAD)

[Full Changelog](https://github.com/stellargraph/stellargraph/compare/v0.6.1...HEAD)

**Implemented enhancements:**

- SGC Implementation [\#361](https://github.com/stellargraph/stellargraph/pull/361) ([PantelisElinas](https://github.com/PantelisElinas))

**Fixed bugs:**

- Removed label from features for pubmed dataset. [\#362](https://github.com/stellargraph/stellargraph/pull/362)
- Python igraph requirement fixed [\#392](https://github.com/stellargraph/stellargraph/pull/392)

## [0.6.1](https://github.com/stellargraph/stellargraph/tree/v0.6.1) (1 Apr 2019)

**Fixed bugs:**
- a bug in passing graph adjacency matrix to the optional `func_opt` function in `FullBatchNodeGenerator` class
- a bug in `demos/node-classification/gcn/gcn-cora-example.py:144`: incorrect argument was used to pass
the optional function to the generator for GCN

**Enhancements:**
- separate treatment of `gcn` and `gat` models in `demos/ensembles/ensemble-node-classification-example.ipynb` 

## [0.6.0](https://github.com/stellargraph/stellargraph/tree/v0.6.0) (14 Mar 2019)

**Implemented new features and enhancements:**
- Graph Attention (GAT) layer and model (stack of GAT layers), with demos [\#216](https://github.com/stellargraph/stellargraph/issues/216), 
[\#315](https://github.com/stellargraph/stellargraph/pull/315)
- Unsupervised GraphSAGE [\#331](https://github.com/stellargraph/stellargraph/pull/331) with a demo [\#335](https://github.com/stellargraph/stellargraph/pull/335)
- Model Ensembles [\#343](https://github.com/stellargraph/stellargraph/pull/343)
- Community detection based on unsupervised graph representation learning [\#354](https://github.com/stellargraph/stellargraph/pull/354)
- Saliency maps and integrated gradients for model interpretability [\#345](https://github.com/stellargraph/stellargraph/pull/345)
- Shuffling of head nodes/edges in node and link generators at each epoch [\#298](https://github.com/stellargraph/stellargraph/issues/298)

**Fixed bugs:**
- a bug where seed was not passed to sampler in `GraphSAGELinkGenerator` constructor [\#337](https://github.com/stellargraph/stellargraph/pull/337)
- UniformRandomMetaPathWalk doesn't update the current node neighbors [\#340](https://github.com/stellargraph/stellargraph/issues/340)
- seed value for link mapper [\#336](https://github.com/stellargraph/stellargraph/issues/336)

## [0.5.0](https://github.com/stellargraph/stellargraph/tree/v0.5.0) (11 Feb 2019)

**Implemented new features and enhancements:**

- Added model calibration [\#326](https://github.com/stellargraph/stellargraph/pull/326)
- Added `GraphConvolution` layer, `GCN` class for a stack of `GraphConvolution` layers,
  and `FullBatchNodeGenerator` class for feeding data into `GCN` models [\#318](https://github.com/stellargraph/stellargraph/pull/318)
- Added GraphSAGE attention aggregator [\#317](https://github.com/stellargraph/stellargraph/pull/317)
- Added GraphSAGE MaxPoolAggregator and MeanPoolAggregator [\#278](https://github.com/stellargraph/stellargraph/pull/278)
- Added shuffle option to all `flow` methods for GraphSAGE and HinSAGE generators [\#328](https://github.com/stellargraph/stellargraph/pull/328)
- GraphSAGE and HinSAGE: ensure that a MLP can be created by using zero samples [\#301](https://github.com/stellargraph/stellargraph/issues/301)
- Handle isolated nodes in GraphSAGE [\#294](https://github.com/stellargraph/stellargraph/issues/294)
- Ensure isolated nodes are handled correctly by GraphSAGENodeMapper and GraphSAGELinkMapper [\#182](https://github.com/stellargraph/stellargraph/issues/182)
- EdgeSplitter: introduce a switch for keeping the reduced graph connected [\#285](https://github.com/stellargraph/stellargraph/issues/285)
- Node2vec for weighted graphs [\#241](https://github.com/stellargraph/stellargraph/issues/241)
- Fix edge types in demos [\#237](https://github.com/stellargraph/stellargraph/issues/237)
- Add docstrings to StellarGraphBase class [\#175](https://github.com/stellargraph/stellargraph/issues/175)
- Make L2-normalisation of the final embeddings in GraphSAGE and HinSAGE optional [\#115](https://github.com/stellargraph/stellargraph/issues/115)
- Check/change the GraphSAGE mapper's behaviour for isolated nodes [\#100](https://github.com/stellargraph/stellargraph/issues/100)
- Added GraphSAGE node embedding extraction and visualisation [\#290](https://github.com/stellargraph/stellargraph/pull/290)

**Fixed bugs:**

- Fixed the bug in running demos when no options given [\#271](https://github.com/stellargraph/stellargraph/issues/271)
- Fixed the bug in LinkSequence that threw an error when no link targets were given [\#273](https://github.com/stellargraph/stellargraph/pull/273)

**Refactoring:**
- Refactored link inference classes to use `edge_embedding_method` instead of `edge_feature_method` [\#327](https://github.com/stellargraph/stellargraph/pull/327)
