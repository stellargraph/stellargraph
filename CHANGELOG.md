# Change Log

## [HEAD](https://github.com/stellargraph/stellargraph/tree/HEAD)

[Full Changelog](https://github.com/stellargraph/stellargraph/compare/HEAD...v0.8.2)

**New algorithms:**

- Cluster-GCN algorithm (an extension of GCN that can be trained using SGD) + demo [\#487](https://github.com/stellargraph/stellargraph/issues/487)

**Implemented enhancements:** 

**Refactoring:**
- Changed `GraphSAGE` and `HinSAGE` class API to accept generator objects the same as GCN/GAT models. Passing a `NodeSequence` or `LinkSequence` object is now deprecated.  [\#498](https://github.com/stellargraph/stellargraph/pull/498)

**Breaking changes:**
- The `StellarGraph` class no longer exposes `NetworkX` internals, only required functionality.
In particular, calls like `list(G)` will no longer return a list of nodes; use `G.nodes()` instead.
[\#297](https://github.com/stellargraph/stellargraph/issues/297)

- Passing a `NodeSequence` or `LinkSequence` object to `GraphSAGE` and `HinSAGE` classes is now deprecated and no longer supported [\#498](https://github.com/stellargraph/stellargraph/pull/498).
Users might need to update their calls of `GraphSAGE` and `HinSAGE` classes by passing `generator` objects instead of `generator.flow()` objects.

**Fixed bugs:**
- Removed an incorrect assertion in `test_normalized_laplacian` and replaced it with a valid check of the symmetrically normalized Laplacian's egienvalues 

## [0.8.3](https://github.com/stellargraph/stellargraph/tree/v0.8.3)

**Fixed bugs:**
- Fixed the issue in the APPNP class that causes appnp to propagate excessive dropout layers. [\#525](https://github.com/stellargraph/stellargraph/pull/525)
- Added a fix into the PPNP node classification demo so that the softmax layer is no longer propagated. [\#525](https://github.com/stellargraph/stellargraph/pull/525)

## [0.8.2](https://github.com/stellargraph/stellargraph/tree/v0.8.2)

**Fixed bugs:**
- Updated requirements to Tensorflow>=1.14, as tensorflow with lower versions causes errors with sparse full batch node methods: GCN, APPNP, and GAT. [\#519](https://github.com/stellargraph/stellargraph/issues/519)

## [0.8.1](https://github.com/stellargraph/stellargraph/tree/v0.8.1)

**Fixed bugs:**
- Reverted erroneous demo notebooks.


## [0.8.0](https://github.com/stellargraph/stellargraph/tree/v0.8.0)

[Full Changelog](https://github.com/stellargraph/stellargraph/compare/v0.8.0...v0.7.3)

**New algorithms:**
- Directed GraphSAGE algorithm (a generalisation of GraphSAGE to directed graphs) + demo [\#479](https://github.com/stellargraph/stellargraph/pull/479)
- Attri2vec algorithm + demo [\#470](https://github.com/stellargraph/stellargraph/pull/470) [\#455](https://github.com/stellargraph/stellargraph/issues/455)
- PPNP and APPNP algorithms + demos [\#485](https://github.com/stellargraph/stellargraph/pull/485)
- GAT saliency maps for interpreting node classification with Graph Attention Networks + demo [\#435](https://github.com/stellargraph/stellargraph/pull/435)

**Implemented enhancements:** 
- New demo of node classification on Twitter hateful users [\430](https://github.com/stellargraph/stellargraph/pull/430)
- New demo of graph saliency on Twitter hateful users [\#448](https://github.com/stellargraph/stellargraph/pull/448)
- Added Directed SampledBFS walks on directed graphs [\#464](https://github.com/stellargraph/stellargraph/issues/464)
- Unified API of GCN, GAT, GraphSAGE, and HinSAGE classses by adding `build()` method to GCN and GAT classes [\#439](https://github.com/stellargraph/stellargraph/issues/439)
- Added `activations` argument to GraphSAGE and HinSAGE classes [\#381](https://github.com/stellargraph/stellargraph/issues/381)
- Unified activations for GraphSAGE, HinSAGE, GCN and GAT [\#493](https://github.com/stellargraph/stellargraph/pull/493) [\#381](https://github.com/stellargraph/stellargraph/issues/381)
- Added optional regularisation on the weights for GCN, GraphSage, and HinSage [\#172](https://github.com/stellargraph/stellargraph/issues/172) [\#469](https://github.com/stellargraph/stellargraph/issues/469)
- Unified regularisation of GraphSAGE, HinSAGE, GCN and GAT [\#494](https://github.com/stellargraph/stellargraph/pull/494) ([geoffj-d61](https://github.com/geoffj-d61))
- Unsupervised GraphSage speed up via multithreading [\#474](https://github.com/stellargraph/stellargraph/issues/474) [\#477](https://github.com/stellargraph/stellargraph/pull/477)
- Support of sparse generators in the GCN saliency map implementation. [\#432](https://github.com/stellargraph/stellargraph/issues/432)

**Refactoring:**
- Refactored Ensemble class into Ensemble and BaggingEnsemble. The former implements naive ensembles and the latter bagging ensembles. [\#459](https://github.com/stellargraph/stellargraph/pull/459)
- Changed from using `keras` to use `tensorflow.keras` [\#471](https://github.com/stellargraph/stellargraph/pull/471)
- Removed `flatten_output` arguments for all models [\#447](https://github.com/stellargraph/stellargraph/pull/447)

**Fixed bugs:**
- Updated Yelp example to support new dataset version [\#442](https://github.com/stellargraph/stellargraph/pull/442)
- Fixed bug where some nodes and edges did not get a default type [\#451](https://github.com/stellargraph/stellargraph/pull/451)
- Inconsistency in `Ensemble.fit_generator()` argument [\#461](https://github.com/stellargraph/stellargraph/issues/461)
- Fixed source--target node designations for code using Cora dataset [\#444](https://github.com/stellargraph/stellargraph/issues/444)
- IndexError: index 1 is out of bounds for axis 1 with size 1 in: demos/node-classification/hinsage [\#434](https://github.com/stellargraph/stellargraph/issues/434)
- GraphSAGE and GAT/GCN predictions have different shapes [\#425](https://github.com/stellargraph/stellargraph/issues/425)


## [0.7.3](https://github.com/stellargraph/stellargraph/tree/v0.7.3)
Limited NetworkX version to <2.4 and Tensorflow version to <1.15 in requirements, to avoid errors due to API changes
in the recent versions of NetworkX and Tensorflow.

## [0.7.2](https://github.com/stellargraph/stellargraph/tree/v0.7.2)
Limited Keras version to <2.2.5 and Tensorflow version to <2.0 in requirements, 
to avoid errors due to API changes in the recent versions of Keras and Tensorflow.


## [0.7.1](https://github.com/stellargraph/stellargraph/tree/v0.7.1)

[Full Changelog](https://github.com/stellargraph/stellargraph/compare/v0.7.0...v0.7.1)

**Fixed bugs:**
- Removed igraph and mplleaflet from `demos` requirements in `setup.py`. Python-igraph doesn't install on many systems and is only required for the clustering notebook. See the `README.md` in that directory for requirements and installation directions.
- Updated GCN interpretability notebook to work with new FullBatchGenerator API [\#429](https://github.com/stellargraph/stellargraph/pull/429)

## [0.7.0](https://github.com/stellargraph/stellargraph/tree/v0.7.0)

[Full Changelog](https://github.com/stellargraph/stellargraph/compare/v0.6.1...v0.7.0)

**Implemented enhancements:**
- SGC Implementation [\#361](https://github.com/stellargraph/stellargraph/pull/361) ([PantelisElinas](https://github.com/PantelisElinas))
- Updated to support Python 3.7 [\#348](https://github.com/stellargraph/stellargraph/pull/348)
- FullBatchNodeGenerator now supports a simpler interface to apply different adjacency matrix pre-processing options [\#405](https://github.com/stellargraph/stellargraph/pull/405)
- Full-batch models (GCN, GAT, and SGC) now return predictions for only those nodes provided to the generator in the same order [\#417](https://github.com/stellargraph/stellargraph/pull/417)
- GAT now supports using a sparse adjacency matrix making execution faster [\#420](https://github.com/stellargraph/stellargraph/pull/420)
- Added interpretability of GCN models and a demo of finding important edges for a node prediction [\#383](https://github.com/stellargraph/stellargraph/pull/383)
- Added a demo showing inductive classification with the PubMed dataset [\#372](https://github.com/stellargraph/stellargraph/pull/372)


**Refactoring:**
- Added build\(\) method for GraphSAGE and HinSAGE model classes [\#385](https://github.com/stellargraph/stellargraph/pull/385)
This replaces the node_model\(\) and link_model\(\) methods, which will be deprecated in future versions (deprecation warnings added).
- Changed the `FullBatchNodeGenerator` to accept simpler `method` and `transform` arguments [\#405](https://github.com/stellargraph/stellargraph/pull/405)


**Fixed bugs:**
- Removed label from features for pubmed dataset. [\#362](https://github.com/stellargraph/stellargraph/pull/362)
- Python igraph requirement fixed [\#392](https://github.com/stellargraph/stellargraph/pull/392)
- Simplified random walks to not require passing a graph [\#408](https://github.com/stellargraph/stellargraph/pull/408)


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
