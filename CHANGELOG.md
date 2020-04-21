# Change Log

## [1.0.0rc1](https://github.com/stellargraph/stellargraph/tree/v1.0.0rc1)

[Full Changelog](https://github.com/stellargraph/stellargraph/compare/v0.11.0...v1.0.0rc1)

This is the first release candidate for StellarGraph 1.0. The 1.0 release will be the culmination of 2 years of activate development, and this release candidate is the first milestone for that release.

Jump in to this release, with the new demos and examples:

- [More helpful indexing and guidance in demo READMEs](demos/)
- [Loading from Neo4j][neo4j]
- [More explanatory Node2Vec link prediction][n2v-lp]
- [Unsupervised `GraphSAGE` and `HinSAGE` via `DeepGraphInfomax`][dgi]
- [Graph classification with `GCNSupervisedGraphClassification`][gc]
- [Time series prediction using spatial information, using `GraphConvolutionLSTM`][gcn-lstm] (experimental)

[neo4j]: demos/basics/loading-saving-neo4j.ipynb
[n2v-lp]: demos/link-prediction/random-walks/cora-lp-demo.ipynb
[dgi]: demos/embeddings/deep-graph-infomax-cora.ipynb
[gc]: demos/graph-classification/supervised-graph-classification.ipynb
[gcn-lstm]: demos/spatio-temporal/gcn-lstm-LA.ipynb

### Major features and improvements

- Better demonstration notebooks and documentation to make the library more accessible to new and existing users:
  - The [demos READMEs](demos/) now contain more guidance and explanation to make it easier to find a relevant example [\#1200](https://github.com/stellargraph/stellargraph/pull/1200)
  - A [demo for loading data from Neo4j][neo4j] has been added [\#1184](https://github.com/stellargraph/stellargraph/pull/1184)
  - The [demo for link prediction using Node2Vec][n2v-lp] has been rewritten to be clearer [\#1190](https://github.com/stellargraph/stellargraph/pull/1190)
  - Notebooks are [now included in the API documentation](https://stellargraph.readthedocs.io/en/latest/demos/index.html), for more convenient access [\#1279](https://github.com/stellargraph/stellargraph/pull/1279)
  - Notebooks now detect if they're being used with an incorrect version of the StellarGraph library, elimanting confusion about version mismatches [\#1242](https://github.com/stellargraph/stellargraph/pull/1242)
- New algorithms:
  - `GCNSupervisedGraphClassification`: supervised graph classification model based on Graph Convolutional layers (GCN) [\#929](https://github.com/stellargraph/stellargraph/issues/929), [demo][gc].
- `DeepGraphInfomax` can be used to train almost any model in an unsupervised way, via the `corrupt_index_groups` parameter to `CorruptedGenerator` [\#1243](https://github.com/stellargraph/stellargraph/pull/1243), [demo][dgi]. Additionally, many algorithms provide defaults and so can be used with `DeepGraphInfomax` without specifying this parameter:
  - any model using `FullBatchNodeGenerator`, including models supported in StellarGraph 0.11: `GCN`, `GAT`, `PPNP` and `APPNP`
  - `GraphSAGE` [\#1162](https://github.com/stellargraph/stellargraph/pull/1162)
  - `HinSAGE` for heterogeneous graphs with node features [\#1254](https://github.com/stellargraph/stellargraph/pull/1254)
- `UnsupervisedSampler` supports a `walker` parameter to use other random walking algorithms such as `BiasedRandomWalk`, in addition to the default `UniformRandomWalk`. [\#1187](https://github.com/stellargraph/stellargraph/pull/1187)
- The `StellarGraph` class is now smaller, faster and easier to construct:
  - The `StellarGraph(..., edge_type_column=...)` parameter can be used to construct a heterogeneous graph from a single flat `DataFrame`, containing a column of the edge types [\#1284](https://github.com/stellargraph/stellargraph/pull/1284). This avoids the need to build separate `DataFrame`s for each type, and is significantly faster when there are many types. Using `edge_type_column` gives a 2.6× speedup for loading the `stellargraph.datasets.FB15k` dataset (with almost 600 thousand edges across 1345 types).
  - `StellarGraph`'s internal cache of node adjacencies now uses the smallest integer type it can [\#1289](https://github.com/stellargraph/stellargraph/pull/1289). This reduces memory use by 31% on the `FB15k` dataset, and 36% on a reddit dataset (with 11.6 million edges).

### Breaking changes

- Edge weights are now validated to be numeric when creating a `StellarGraph`, previously edge weights could be any type, but all algorithms that use them would fail. [\#1191](https://github.com/stellargraph/stellargraph/pull/1191)
- Full batch layers no longer support an "output indices" tensor to filter the output rows to a selected set of nodes [\#1204](https://github.com/stellargraph/stellargraph/pull/1204) (this does **not** affect models like `GCN`, only the layers within them: `APPNPPropagationLayer`, `ClusterGraphConvolution`, `GraphConvolution`, `GraphAttention`, `GraphAttentionSparse`, `PPNPPropagationLayer`, `RelationalGraphConvolution`). Migration: post-process the output using `tf.gather` manually or the new `sg.layer.misc.GatherIndices` layer.
- `GraphConvolution` has been generalised to work with batch size > 1, subsuming the functionality of the now-deprecated `ClusterGraphConvolution` (and `GraphClassificationConvolution`) [\#1205](https://github.com/stellargraph/stellargraph/pull/1205). Migration: replace `stellargraph.layer.ClusterGraphConvolution` with `stellargraph.layer.GraphConvolution`.

### Experimental features

Some new algorithms and features are still under active development, and are available as an experimental preview. However, they may not be easy to use: their documentation or testing may be incomplete, and they may change dramatically from release to release. The experimental status is noted in the documentation and at runtime via prominent warnings.

- `DeepGraphConvolutionalNeuralNetwork`: supervised graph classification based on GCN, the new `SortPooling` pooling layer and asymmetric adjacency normalisation [\#1210](https://github.com/stellargraph/stellargraph/pull/1210) [\#1212](https://github.com/stellargraph/stellargraph/pull/1212) [\#1265](https://github.com/stellargraph/stellargraph/pull/1265)
- `GraphConvolutionLSTM`: time series prediction on spatio-temporal data, combining GCN with a [LSTM](https://en.wikipedia.org/wiki/Long_short-term_memory) model to augment the conventional time-series model with information from nearby data points [\#1085](https://github.com/stellargraph/stellargraph/pull/1085), [demo][gcn-lstm]

### Bug fixes and other changes

- Random walk classes like `UniformRandomWalk` and `BiasedRandomWalk` can have their hyperparameters set on construction, in addition to in each call to `run` [\#1179](https://github.com/stellargraph/stellargraph/pull/1179)
- Node feature sampling was made ~4× faster by ensuring a better data layout, this makes some configurations of `GraphSAGE` (and `HinSAGE`) noticably faster [\#1225](https://github.com/stellargraph/stellargraph/pull/1225)
- The `PROTEINS` dataset has been added to `stellargraph.datasets`, for graph classification [\#1282](https://github.com/stellargraph/stellargraph/pull/1282)
- The `BlogCatalog3` dataset can now be successfully downloaded again [\#1283](https://github.com/stellargraph/stellargraph/pull/1283)
- Knowledge graph model evaluation via `rank_edges_against_all_nodes` now defaults to the `random` strategy for breaking ties, and supports `top` (previous default) and `bottom` as alternatives [\#1223](https://github.com/stellargraph/stellargraph/pull/1223)
- Creating a `RelationalFullBatchNodeGenerator` is now significantly faster and requires much less memory (18× speedup and 560× smaller for the `stellargraph.datasets.AIFB` dataset) [\#1274](https://github.com/stellargraph/stellargraph/pull/1274)
- Various documentation, demo and error message fixes and improvements: [\#1141](https://github.com/stellargraph/stellargraph/pull/1141), [\#1219](https://github.com/stellargraph/stellargraph/pull/1219), [\#1246](https://github.com/stellargraph/stellargraph/pull/1246), [\#1260](https://github.com/stellargraph/stellargraph/pull/1260), [\#1266](https://github.com/stellargraph/stellargraph/pull/1266)
- DevOps changes:
  - CI: [\#1161](https://github.com/stellargraph/stellargraph/pull/1161), [\#1189](https://github.com/stellargraph/stellargraph/pull/1189), [\#1230](https://github.com/stellargraph/stellargraph/pull/1230), [\#1122](https://github.com/stellargraph/stellargraph/pull/1122)
  - Other: [\#1197](https://github.com/stellargraph/stellargraph/pull/1197)

## [0.11.1](https://github.com/stellargraph/stellargraph/tree/v0.11.1)

[Full Changelog](https://github.com/stellargraph/stellargraph/compare/v0.11.0...v0.11.1)

This bugfix release contains the same code as 0.11.0, and just fixes the metadata in the Anaconda package so that it can be installed successfully.

### Bug fixes and other changes

- The [Conda package for StellarGraph](https://anaconda.org/stellargraph/stellargraph) has been updated to require TensorFlow 2.1, as TensorFlow 2.0 is no longer supported.  As a result, StellarGraph will currently install via Conda on Linux and Windows - Mac support is waiting on the [Tensorflow 2.1 osx-64 release to Conda](https://github.com/ContinuumIO/anaconda-issues/issues/11697). [\#1165](https://github.com/stellargraph/stellargraph/pull/1165)

## [0.11.0](https://github.com/stellargraph/stellargraph/tree/v0.11.0)

[Full Changelog](https://github.com/stellargraph/stellargraph/compare/v0.10.0...v0.11.0)

### Major features and improvements

- The onboarding/getting-started process has been optimised and improved:
  - The README has been rewritten to highlight our numerous demos, and how to get help [\#1081](https://github.com/stellargraph/stellargraph/pull/1081)
  - [Example Jupyter notebooks](https://github.com/stellargraph/stellargraph/tree/master/demos/) can now be run directly in [Google Colab](https://colab.research.google.com) and [Binder](https://mybinder.org), providing an easy way to get started with StellarGraph - simply click the ![](https://colab.research.google.com/assets/colab-badge.svg) and ![](https://mybinder.org/badge_logo.svg) badges within each notebook. [\#1119](https://github.com/stellargraph/stellargraph/pull/1119).
  - The [new `demos/basics` directory](demos/basics) contains two notebooks demonstrating how to construct a `StellarGraph` object from Pandas, and from NetworkX [\#1074](https://github.com/stellargraph/stellargraph/pull/1074)
  - The [GCN node classification demo](demos/node-classification/gcn/gcn-cora-node-classification-example.ipynb) now has more explanation, to serve as an introduction to graph machine learning using StellarGraph [\#1125](https://github.com/stellargraph/stellargraph/pull/1125)
- New algorithms:
  - Watch Your Step: computes node embeddings by simulating the effect of random walks, rather than doing them. [\#750](https://github.com/stellargraph/stellargraph/pull/750).
  - Deep Graph Infomax: performs unsupervised node representation learning [\#978](https://github.com/stellargraph/stellargraph/issues/978).
  - Temporal Random Walks (Continuous-Time Dynamic Network Embeddings): random walks that respect the time that each edge occurred (stored as edge weights) [\#1120](https://github.com/stellargraph/stellargraph/issues/1120).
  - ComplEx: computes multiplicative complex-number embeddings for entities and relationships (edge types) in knowledge graphs, which can be used for link prediction. [\#901](https://github.com/stellargraph/stellargraph/pull/901) [\#1080](https://github.com/stellargraph/stellargraph/pull/1080)
  - DistMult: computes multiplicative real-number embeddings for entities and relationships (edge types) in knowledge graphs, which can be used for link prediction. [\#755](https://github.com/stellargraph/stellargraph/issues/755) [\#865](https://github.com/stellargraph/stellargraph/pull/865) [\#1136](https://github.com/stellargraph/stellargraph/pull/1136)

### Breaking changes

- StellarGraph now requires TensorFlow 2.1 or greater, TensorFlow 2.0 is no longer supported [\#1008](https://github.com/stellargraph/stellargraph/pull/1008)
- The legacy constructor using NetworkX graphs has been deprecated [\#1027](https://github.com/stellargraph/stellargraph/pull/1027). Migration: replace `StellarGraph(some_networkx_graph)` with `StellarGraph.from_networkx(some_networkx_graph)`, and similarly for `StellarDiGraph`.
- The `build` method on model classes (such as `GCN`) has been renamed to `in_out_tensors` [\#1140](https://github.com/stellargraph/stellargraph/pull/1140). Migration: replace `model.build()` with `model.in_out_tensors()`.
- The `node_model` and `link_model` methods on model classes has been replaced by `in_out_tensors` [\#1140](https://github.com/stellargraph/stellargraph/pull/1140) (see that PR for the exact list of types). Migration: replace `model.node_model()` with `model.in_out_tensors()` or `model.in_out_tensors(multiplicity=1)`, and `model.node_model()` with `model.in_out_tensors()` or `model.in_out_tensors(multiplicity=2)`.
- Re-exports of calibration and ensembling functionality from the top-level of the `stellargraph` module were deprecated, in favour of importing from the `stellargraph.calibration` or `stellargraph.ensemble` submodules directly [\#1107](https://github.com/stellargraph/stellargraph/pull/1107). Migration: replace uses of `stellargraph.Ensemble` with `stellargraph.ensemble.Ensemble`, and similarly for the other names (see [\#1107](https://github.com/stellargraph/stellargraph/pull/1107) for all replacements).
- `StellarGraph.to_networkx` parameters now use `attr` to refer to NetworkX attributes, not `name` or `label` [\#973](https://github.com/stellargraph/stellargraph/pull/973). Migration: for any named parameters in `graph.to_networkx(...)`, change `node_type_name=...` to `node_type_attr=...` and similarly `edge_type_name` to `edge_type_attr`, `edge_weight_label` to `edge_weight_attr`, `feature_name` to `feature_attr`.
- `StellarGraph.nodes_of_type` is deprecated in favour of the `nodes` method [\#1111](https://github.com/stellargraph/stellargraph/pull/1111). Migration: replace `some_graph.nodes_of_type(some_type)` with `some_graph.nodes(node_type=some_type)`.
- `StellarGraph.info` parameters `show_attributes` and `sample` were deprecated [\#1110](https://github.com/stellargraph/stellargraph/pull/1110)
- Some more layers and models had many parameters move from `**kwargs` to real arguments: `Attri2Vec` ([\#1128](https://github.com/stellargraph/stellargraph/pull/1128)), `ClusterGCN` ([\#1129](https://github.com/stellargraph/stellargraph/pull/1129)), `GraphAttention` & `GAT` ([\#1130](https://github.com/stellargraph/stellargraph/pull/1130)), `GraphSAGE` & its aggregators ([\#1142](https://github.com/stellargraph/stellargraph/pull/1142)), `HinSAGE` & its aggregators ([\#1143](https://github.com/stellargraph/stellargraph/pull/1143)), `RelationalGraphConvolution` & `RGCN` ([\#1148](https://github.com/stellargraph/stellargraph/pull/1148)). Invalid (e.g. incorrectly spelled) arguments would have been ignored previously, but now may fail with a `TypeError`; to fix, remove or correct the arguments.
- The `method="chebyshev"` option to `FullBatchNodeGenerator`, `FullBatchLinkGenerator` and `GCN_Aadj_feats_op` has been removed for now, because it needed significant revision to be correctly implemented [\#1028](https://github.com/stellargraph/stellargraph/pull/1028)
- The `fit_generator`, `evaluate_generator` and `predict_generator` methods on `Ensemble` and `BaggingEnsemble` have been renamed to `fit`, `evaluate` and `predict`, to match the deprecation in TensorFlow 2.1 of the `tensorflow.keras.Model` methods of the same name [\#1065](https://github.com/stellargraph/stellargraph/pull/1065). Migration: remove the `_generator`  suffix on these methods.
- The `default_model` method on `Attri2Vec`, `GraphSAGE` and `HinSAGE` has been deprecated, in favour of `in_out_tensors` [\#1145](https://github.com/stellargraph/stellargraph/pull/1145). Migration: replace `model.default_model()` with `model.in_out_tensors()`.

### Experimental features

Some new algorithms and features are still under active development, and are available as an experimental preview. However, they may not be easy to use: their documentation or testing may be incomplete, and they may change dramatically from release to release. The experimental status is noted in the documentation and at runtime via prominent warnings.

- GCNSupervisedGraphClassification: supervised graph classification model based on Graph Convolutional layers (GCN) [\#929](https://github.com/stellargraph/stellargraph/issues/929).

### Bug fixes and other changes

- `StellarGraph.to_adjacency_matrix` is at least 15× faster on undirected graphs [\#932](https://github.com/stellargraph/stellargraph/pull/932)
- `ClusterNodeGenerator` is now noticably faster, which makes training and predicting with a `ClusterGCN` model faster [\#1095](https://github.com/stellargraph/stellargraph/pull/1095). On a random graph with 1000 nodes and 5000 edges and 10 clusters, iterating over an epoch with `q=1` (each clusters individually) is 2× faster, and is even faster for larger `q`. The model in the Cluster-GCN demo notebook using Cora trains 2× faster overall.
- The `node_features=...` parameter to `StellarGraph.from_networkx` now only needs to mention the node types that have features, when passing a dictionary of Pandas DataFrames. Node types that aren't mentioned will automatically have no features (zero-length feature vectors). [\#1082](https://github.com/stellargraph/stellargraph/pull/1082)
- A `subgraph` method was added to `StellarGraph` for computing a node-induced subgraph [\#958](https://github.com/stellargraph/stellargraph/pull/958)
- A `connected_components` method was added to `StellarGraph` for computing the nodes involved in each connected component in a `StellarGraph` [\#958](https://github.com/stellargraph/stellargraph/pull/958)
- The `info` method on `StellarGraph` now shows only 20 node and edge types by default to be more useful for graphs with many types [\#993](https://github.com/stellargraph/stellargraph/pull/993). This behaviour can be customized with the `truncate=...` parameter.
- The `info` method on `StellarGraph` now shows information about the size and type of each node type's feature vectors [\#979](https://github.com/stellargraph/stellargraph/pull/979)
- The `EdgeSplitter` class supports `StellarGraph` input (and will output `StellarGraph`s in this case), in addition to NetworkX graphs [\#1032](https://github.com/stellargraph/stellargraph/pull/1032)
- The `Attri2Vec` model class stores its weights statefully, so they are shared between all tensors computed by `build` [\#1101](https://github.com/stellargraph/stellargraph/pull/1101)
- The `GCN` model defaults for some parameters now match the `GraphConvolution` layer's defaults: specifically `kernel_initializer` (`glorot_uniform`) and `bias_initializer` (`zeros`) [\#1147](https://github.com/stellargraph/stellargraph/pull/1147)
- The `datasets` submodule is now accessible as `stellargraph.datasets`, after just `import stellargraph` [\#1113](https://github.com/stellargraph/stellargraph/pull/1113)
- All datasets in `stellargraph.datasets` now support a `load` method to create a `StellarGraph` object (and other information): `AIFB` ([\#982](https://github.com/stellargraph/stellargraph/pull/982)), `CiteSeer` ([\#989](https://github.com/stellargraph/stellargraph/pull/989)), `Cora` ([\#913](https://github.com/stellargraph/stellargraph/pull/913)), `MovieLens` ([\#947](https://github.com/stellargraph/stellargraph/pull/947)), `PubMedDiabetes` ([\#986](https://github.com/stellargraph/stellargraph/pull/986)). The demo notebooks using these datasets are now cleaner.
- Some new datasets were added to `stellargraph.datasets`:
  - `MUTAG`: a collection of graphs representing chemical compounds [\#960](https://github.com/stellargraph/stellargraph/pull/960)
  - `WN18`, `WN18RR`: knowledge graphs based on the WordNet linguistics data [\#977](https://github.com/stellargraph/stellargraph/pull/977)
  - `FB15k`, `FB15k_237`: knowledge graphs based on the FreeBase knowledge base [\#977](https://github.com/stellargraph/stellargraph/pull/977)
  - `IAEnronEmployees`: a small set of employees of Enron, and the many emails between them [\#1058](https://github.com/stellargraph/stellargraph/pull/1058)
- Warnings now point to the call site of the function causing the warning, not the `warnings.warn` call inside StellarGraph; this means `DeprecationWarning`s will be visible in Jupyter notebooks and scripts run with Python 3.7 [\#1144](https://github.com/stellargraph/stellargraph/pull/1144)
- Some code that triggered warnings from other libraries was fixed or removed [\#995](https://github.com/stellargraph/stellargraph/pull/995) [\#1008](https://github.com/stellargraph/stellargraph/pull/1008), [\#1051](https://github.com/stellargraph/stellargraph/pull/1051), [\#1064](https://github.com/stellargraph/stellargraph/pull/1064), [\#1066](https://github.com/stellargraph/stellargraph/pull/1066)
- Some demo notebooks have been updated or fixed: `demos/use-cases/hateful-twitters.ipynb` ([\#1019](https://github.com/stellargraph/stellargraph/pull/1019)), `rgcn-aifb-node-classification-example.ipynb` ([\#983](https://github.com/stellargraph/stellargraph/pull/983))
- The documentation "quick start" guide duplicated a lot of the information in the README, and so has been replaced with the latter [\#1096](https://github.com/stellargraph/stellargraph/pull/1096)
- API documentation now lists items under their recommended import path, not their definition. For instance, `stellargraph.StellarGraph` instead of `stellargraph.core.StellarGraph` ([\#1127](https://github.com/stellargraph/stellargraph/pull/1127)), `stellargraph.layer.GCN` instead of `stellargraph.layer.gcn.GCN` ([\#1150](https://github.com/stellargraph/stellargraph/pull/1150)) and `stellargraph.datasets.Cora` instead of `stellargraph.datasets.datasets.Cora` ([\#1157](https://github.com/stellargraph/stellargraph/pull/1157))
- Some API documentation is now formatted better [\#1061](https://github.com/stellargraph/stellargraph/pull/1061), [\#1068](https://github.com/stellargraph/stellargraph/pull/1068), [\#1070](https://github.com/stellargraph/stellargraph/pull/1070), [\#1071](https://github.com/stellargraph/stellargraph/pull/1071)
- DevOps changes:
  - Neo4j functionality is now tested on CI, and so will continue working [\#1046](https://github.com/stellargraph/stellargraph/pull/1046) [\#1050](https://github.com/stellargraph/stellargraph/pull/1050)
  - CI: [\#967](https://github.com/stellargraph/stellargraph/pull/967), [\#968](https://github.com/stellargraph/stellargraph/pull/968), [\#1036](https://github.com/stellargraph/stellargraph/pull/1036), [\#1067](https://github.com/stellargraph/stellargraph/pull/1067), [\#1097](https://github.com/stellargraph/stellargraph/pull/1097)
  - Other: [\#956](https://github.com/stellargraph/stellargraph/pull/956), [\#962](https://github.com/stellargraph/stellargraph/pull/962), [\#974](https://github.com/stellargraph/stellargraph/pull/974)

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
