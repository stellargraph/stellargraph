# Change Log

## [HEAD](https://github.com/stellargraph/stellargraph/tree/HEAD)

[Full Changelog](https://github.com/stellargraph/stellargraph/compare/v0.10.0...HEAD)

### Major features and improvements

- New algorithms:

### Breaking changes

### Experimental features

Some new algorithms and features are still under active development, and are available as an experimental preview. However, they may not be easy to use: their documentation or testing may be incomplete, and they may change dramatically from release to release. The experimental status is noted in the documentation and at runtime via prominent warnings.

### Bug fixes and other changes

- `StellarGraph.to_adjacency_matrix` is at least 15× faster on undirected graphs [\#932](http://github.com/stellargraph/stellargraph/pull/932)
- DevOps changes:

## [0.10.0](https://github.com/stellargraph/stellargraph/tree/v0.10.0)

[Full Changelog](https://github.com/stellargraph/stellargraph/compare/v0.9.0...v0.10.0)

### Major features and improvements

- The `StellarGraph` and `StellarDiGraph` classes are now backed by NumPy and Pandas [\#752](https://github.com/stellargraph/stellargraph/issues/752). The `StellarGraph(...)` and `StellarDiGraph(...)` constructors now consume Pandas DataFrames representing node features and the edge list. This significantly reduces the memory use and construction time for these `StellarGraph` objects.

  The following table shows some measurements of the memory use of `g = StellarGraph(...)`, and the time required for that constructor call, for several real-world datasets of different sizes, for both the old form backed by NetworkX code and the new form backed by NumPy and Pandas (both old and new store node features similarly, using 2D NumPy arrays, so the measurements in this table include only graph structure: the edges and nodes themselves):

  | dataset |  nodes |    edges | size old (MiB) | size new (MiB) | size change | time old (s) | time new (s) | time change |
  |---------|-------:|---------:|---------------:|---------------:|------------:|-------------:|-------------:|------------:|
  | Cora    |   2708 |     5429 |            4.1 |        **1.3** |    **-69%** |        0.069 |    **0.034** |    **-50%** |
  | FB15k   |  14951 |   592213 |            148 |         **28** |    **-81%** |          5.5 |      **1.2** |    **-77%** |
  | Reddit  | 231443 | 11606919 |           6611 |        **493** |    **-93%** |          154 |       **33** |    **-82%** |

  The old backend has been removed, and conversion from a NetworkX graph should be performed via the `StellarGraph.from_networkx` function (the existing form `StellarGraph(networkx_graph)` is supported in this release but is deprecated, and may be removed in a future release).
- More detailed information about Heterogeneous GraphSAGE (HinSAGE) has been added to StellarGraph's readthedocs documentation [\#839](https://github.com/stellargraph/stellargraph/pull/839).
- New algorithms:
  - Link prediction with directed GraphSAGE, via `DirectedGraphSAGELinkGenerator` [\#871](https://github.com/stellargraph/stellargraph/issues/871)
  - GraphWave: computes structural node embeddings by using wavelet transforms on the graph Laplacian [\#822](https://github.com/stellargraph/stellargraph/issues/822)

### Breaking changes

- Some layers and models had many parameters move from `**kwargs` to real arguments: `GraphConvolution`, `GCN`. [\#801](https://github.com/stellargraph/stellargraph/issues/801) Invalid (e.g. incorrectly spelled) arguments would have been ignored previously, but now may fail with a `TypeError`; to fix, remove or correct the arguments.
- The `stellargraph.data.load_dataset_BlogCatalog3` function has been replaced by the `load` method on `stellargraph.datasets.BlogCatalog3` [\#888](https://github.com/stellargraph/stellargraph/pull/888). Migration: replace `load_dataset_BlogCatalog3(location)` with `BlogCatalog3().load()`; code required to find the location or download the dataset can be removed, as `load` now does this automatically.
- `stellargraph.data.train_test_val_split` and `stellargraph.data.NodeSplitter` have been removed. [\#887](https://github.com/stellargraph/stellargraph/pull/887) Migration: this functionality should be replaced with `pandas` and `sklearn` (for instance, `sklearn.model_selection.train_test_split`).
- Most of the submodules in `stellargraph.utils` have been moved to top-level modules: `stellargraph.calibration`, `stellargraph.ensemble`, `stellargraph.losses` and `stellargraph.interpretability` [\#938](http://github.com/stellargraph/stellargraph/pull/938). Imports from the old location are now deprecated, and may stop working in future releases. See the linked issue for the full list of changes.

### Experimental features

Some new algorithms and features are still under active development, and are available as an experimental preview. However, they may not be easy to use: their documentation or testing may be incomplete, and they may change dramatically from release to release. The experimental status is noted in the documentation and at runtime via prominent warnings.

- Temporal Random Walks: random walks that respect the time that each edge occurred (stored as edge weights) [\#787](https://github.com/stellargraph/stellargraph/pull/787). The implementation does not have an example or thorough testing and documentation.
- Watch Your Step: computes node embeddings by simulating the effect of random walks, rather than doing them. [\#750](https://github.com/stellargraph/stellargraph/pull/750). The implementation is not fully tested.
- ComplEx: computes embeddings for nodes and edge types in knowledge graphs, and use these to perform link prediction [\#756](https://github.com/stellargraph/stellargraph/issues/756). The implementation hasn't been validated to match the paper.
- Neo4j connector: the GraphSAGE algorithm can execute doing neighbourhood sampling in a Neo4j database, so that the edges of a graph do not have to fit entirely into memory [\#799](https://github.com/stellargraph/stellargraph/pull/799). The implementation is not automatically tested, and doesn't support functionality like loading node feature vectors from Neo4j.

### Bug fixes and other changes

- StellarGraph now supports [TensorFlow 2.1](https://github.com/tensorflow/tensorflow/releases/tag/v2.1.0), which includes GPU support by default: [\#875](https://github.com/stellargraph/stellargraph/pull/875)
- Demos now focus on Jupyter notebooks, and demo scripts that duplicate notebooks have been removed: [\#889](https://github.com/stellargraph/stellargraph/pull/889)
- The following algorithms are now reproducible:
  - Supervised GraphSAGE Node Attribute Inference [\#844](https://github.com/stellargraph/stellargraph/pull/844)
  - GraphSAGE Link Prediction [\#925](https://github.com/stellargraph/stellargraph/pull/925)
- Randomness can be more easily controlled using `stellargraph.random.set_seed` [\#806](https://github.com/stellargraph/stellargraph/pull/806)
- `StellarGraph.edges()` can return edge weights as a separate NumPy array with `include_edge_weights=True` [\#754](https://github.com/stellargraph/stellargraph/pull/754)
- `StellarGraph.to_networkx` supports ignoring node features (and thus being a little more efficient) with `feature_name=None` [\#841](https://github.com/stellargraph/stellargraph/pull/841)
- `StellarGraph.to_adjacency_matrix` now ignores edge weights (that is, defaults every weight to `1`) by default, unless `weighted=True` is specified [\#857](https://github.com/stellargraph/stellargraph/pull/857)
- `stellargraph.utils.plot_history` visualises the model training history as a plot for each metric (such as loss) [\#902](https://github.com/stellargraph/stellargraph/pull/902)
- the saliency maps/interpretability code has been refactored to have more sharing as well as to make it cleaner and easier to extend [\#855](https://github.com/stellargraph/stellargraph/pull/855)
- DevOps changes:
  - Most demo notebooks are now tested on CI using Papermill, and so won't become out of date [\#575](https://github.com/stellargraph/stellargraph/issues/575)
  - CI: [\#698](https://github.com/stellargraph/stellargraph/pull/698), [\#760](https://github.com/stellargraph/stellargraph/pull/760), [\#788](https://github.com/stellargraph/stellargraph/pull/788), [\#817](https://github.com/stellargraph/stellargraph/pull/817), [\#860](https://github.com/stellargraph/stellargraph/pull/860), [\#874](https://github.com/stellargraph/stellargraph/pull/874), [\#877](https://github.com/stellargraph/stellargraph/pull/877), [\#878](https://github.com/stellargraph/stellargraph/pull/878), [\#906](https://github.com/stellargraph/stellargraph/pull/906), [\#908](https://github.com/stellargraph/stellargraph/pull/908), [\#915](https://github.com/stellargraph/stellargraph/pull/915), [\#916](https://github.com/stellargraph/stellargraph/pull/916), [\#918](https://github.com/stellargraph/stellargraph/pull/918)
  - Other: [\#708](https://github.com/stellargraph/stellargraph/pull/708), [\#746](https://github.com/stellargraph/stellargraph/pull/746), [\#791](https://github.com/stellargraph/stellargraph/pull/791)


## [0.9.0](https://github.com/stellargraph/stellargraph/tree/v0.9.0)

### Major features and improvements

- StellarGraph is now available as [a conda package on Anaconda Cloud](https://anaconda.org/stellargraph/stellargraph) [\#516](https://github.com/stellargraph/stellargraph/pull/516)
- New algorithms:
  - Cluster-GCN: an extension of GCN that can be trained using SGD, with demo [\#487](https://github.com/stellargraph/stellargraph/issues/487)
  - Relational-GCN (RGCN): a generalisation of GCN to relational/multi edge type graphs, with demo [\#490](https://github.com/stellargraph/stellargraph/issues/490)
  - Link prediction for full-batch models: `FullBatchLinkGenerator` allows doing link prediction with algorithms like GCN, GAT, APPNP and PPNP [\#543](https://github.com/stellargraph/stellargraph/pull/543)
- Unsupervised GraphSAGE has now been updated and tested for reproducibility. Ensuring all seeds are set, running the same pipeline should give reproducible embeddings. [\#620](https://github.com/stellargraph/stellargraph/pull/620)
- A `datasets` subpackage provides easier access to sample datasets with inbuilt downloading. [\#690](https://github.com/stellargraph/stellargraph/pull/690)


### Breaking changes

- The stellargraph library now only supports `tensorflow` version 2.0 [\#518](https://github.com/stellargraph/stellargraph/pull/518), [\#732](https://github.com/stellargraph/stellargraph/pull/732). Backward compatibility with earlier versions of `tensorflow` is not guaranteed.
- The stellargraph library now only supports Python versions 3.6 and above [\#641](https://github.com/stellargraph/stellargraph/pull/641). Backward compatibility with earlier versions of Python is not guaranteed.
- The `StellarGraph` class no longer exposes `NetworkX` internals, only required functionality. In particular, calls like `list(G)` will no longer return a list of nodes; use `G.nodes()` instead. [\#297](https://github.com/stellargraph/stellargraph/issues/297) If NetworkX functionality is required, use the new `.to_networkx()` method to convert to a normal `networkx.MultiGraph` or `networkx.MultiDiGraph`.
- Passing a `NodeSequence` or `LinkSequence` object to `GraphSAGE` and `HinSAGE` classes is now deprecated and no longer supported [\#498](https://github.com/stellargraph/stellargraph/pull/498). Users might need to update their calls of `GraphSAGE` and `HinSAGE` classes by passing `generator` objects instead of `generator.flow()` objects.
- Various methods on `StellarGraph` have been renamed to be more succinct and uniform:
   - `get_feature_for_nodes` is now `node_features`
   - `type_for_node` is now `node_type`
- Neighbourhood methods in `StellarGraph` class (`neighbors`, `in_nodes`, `out_nodes`) now return a list of neighbours instead of a set. This addresses [\#653](https://github.com/stellargraph/stellargraph/issues/653). This means multi-edges are no longer collapsed into one in the return value. There will be an implicit change in behaviour for explorer classes used for algorithms like GraphSAGE, Node2Vec, since a neighbour connected via multiple edges will now be more likely to be sampled. If this doesn't sound like the desired behaviour, consider pruning the graph of multi-edges before running the algorithm.
- `GraphSchema` has been simplified to remove type look-ups for individual nodes and edges [\#702](https://github.com/stellargraph/stellargraph/pull/702) [\#703](https://github.com/stellargraph/stellargraph/pull/703). Migration: for nodes, use `StellarGraph.node_type`; for edges, use the `triple` argument to the `edges` method, or filter when doing neighbour queries using the `edge_types` argument.
- `NodeAttributeSpecification` and the supporting `Converter` classes have been removed [\#707](https://github.com/stellargraph/stellargraph/pull/707). Migration: use the more powerful and flexible preprocessing tools from pandas and sklearn (see the linked PR for specifics)

### Experimental features

Some new algorithms and features are still under active development, and are available as an experimental preview. However, they may not be easy to use: their documentation or testing may be incomplete, and they may change dramatically from release to release. The experimental status is noted in the documentation and at runtime via prominent warnings.

- The `StellarGraph` and `StellarDiGraph` classes supports using a backend based on NumPy and Pandas that uses dramatically less memory for large graphs than the existing NetworkX-based backend [\#668](https://github.com/stellargraph/stellargraph/pull/668). The new backend can be enabled by constructing with `StellarGraph(nodes=..., edges=...)` using Pandas DataFrames, instead of a NetworkX graph.

### Bug fixes and other changes

- Documentation for every relased version is published under a permanent URL, in addition to the `stable` alias for the latest release, e.g. <https://stellargraph.readthedocs.io/en/v0.8.4/> for `v0.8.4` [#612](https://github.com/stellargraph/stellargraph/issues/612)
- Neighbourhood methods in `StellarGraph` class (`neighbors`, `in_nodes`, `out_nodes`) now support additional parameters to include edge weights in the results or filter by a set of edge types. [\#646](https://github.com/stellargraph/stellargraph/pull/646)
- Changed `GraphSAGE` and `HinSAGE` class API to accept generator objects the same as GCN/GAT models. Passing a `NodeSequence` or `LinkSequence` object is now deprecated.  [\#498](https://github.com/stellargraph/stellargraph/pull/498)
- `SampledBreadthFirstWalk`, `SampledHeterogeneousBreadthFirstWalk` and `DirectedBreadthFirstNeighbours` have been made 1.2-1.5× faster [\#628](https://github.com/stellargraph/stellargraph/pull/628)
- `UniformRandomWalk` has been made 2× faster [\#625](https://github.com/stellargraph/stellargraph/pull/625)
- `FullBatchNodeGenerator.flow` has been reduced from `O(n^2)` quadratic complexity to `O(n)`, where `n` is the number of nodes in the graph, making it orders of magnitude faster for large graphs [\#513](https://github.com/stellargraph/stellargraph/pull/513)
- The dependencies required for demos and testing have been included as "extras" in the main package: `demos` and `igraph` for demos, and `test` for testing. For example, `pip install stellargraph[demos,igraph]` will install the dependencies required to run every demo. [\#661](https://github.com/stellargraph/stellargraph/pull/661)
- The `StellarGraph` and `StellarDiGraph` constructors now list their arguments explicitly for clearer documentation (rather than using `*arg` and `**kwargs` splats) [\#659](https://github.com/stellargraph/stellargraph/pull/659)
- `sys.exit(0)` is no longer called on failure in `load_dataset_BlogCatalog3` [\#648](https://github.com/stellargraph/stellargraph/pull/648)
- Warnings are printed using the Python `warnings` module [\#583](https://github.com/stellargraph/stellargraph/pull/583)
- Numerous DevOps changes:
  - CI results are now publicly viewable: <https://buildkite.com/stellar/stellargraph-public>
  - CI: [\#524](https://github.com/stellargraph/stellargraph/pull/524), [\#534](https://github.com/stellargraph/stellargraph/pull/534), [\#544](https://github.com/stellargraph/stellargraph/pull/544), [\#550](https://github.com/stellargraph/stellargraph/pull/550), [\#551](https://github.com/stellargraph/stellargraph/pull/551), [\#557](https://github.com/stellargraph/stellargraph/pull/557), [\#562](https://github.com/stellargraph/stellargraph/pull/562), [\#574](https://github.com/stellargraph/stellargraph/issues/574) [\#578](https://github.com/stellargraph/stellargraph/pull/578), [\#579](https://github.com/stellargraph/stellargraph/pull/579), [\#587](https://github.com/stellargraph/stellargraph/pull/587), [\#592](https://github.com/stellargraph/stellargraph/pull/592), [\#595](https://github.com/stellargraph/stellargraph/pull/595), [\#596](https://github.com/stellargraph/stellargraph/pull/596), [\#602](https://github.com/stellargraph/stellargraph/issues/602), [\#609](https://github.com/stellargraph/stellargraph/pull/609), [\#613](https://github.com/stellargraph/stellargraph/pull/613), [\#615](https://github.com/stellargraph/stellargraph/pull/615), [\#631](https://github.com/stellargraph/stellargraph/pull/631), [\#637](https://github.com/stellargraph/stellargraph/pull/637), [\#639](https://github.com/stellargraph/stellargraph/pull/639), [\#640](https://github.com/stellargraph/stellargraph/pull/640), [\#652](https://github.com/stellargraph/stellargraph/pull/652), [\#656](https://github.com/stellargraph/stellargraph/pull/656), [\#663](https://github.com/stellargraph/stellargraph/pull/663), [\#675](https://github.com/stellargraph/stellargraph/pull/675)
  - Git and Github configuration: [\#516](https://github.com/stellargraph/stellargraph/pull/516), [\#588](https://github.com/stellargraph/stellargraph/pull/588), [\#624](https://github.com/stellargraph/stellargraph/pull/624), [\#672](https://github.com/stellargraph/stellargraph/pull/672), [\#682](https://github.com/stellargraph/stellargraph/pull/682), [\#683](https://github.com/stellargraph/stellargraph/pull/683),
  - Other: [\#523](https://github.com/stellargraph/stellargraph/pull/523), [\#582](https://github.com/stellargraph/stellargraph/pull/582), [\#590](https://github.com/stellargraph/stellargraph/pull/590), [\#654](https://github.com/stellargraph/stellargraph/pull/654)


## [0.8.4](https://github.com/stellargraph/stellargraph/tree/v0.8.4)

**Fixed bugs:**
- Fix `DirectedGraphSAGENodeGenerator` always hitting `TypeError` exception. [#695](https://github.com/stellargraph/stellargraph/issues/695)

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
