StellarGraph API
========================

Core
----------------

.. automodule:: stellargraph
  :members: StellarGraph, StellarDiGraph, GraphSchema, IndexedArray

.. autodata:: custom_keras_layers
   :annotation: = {...}


Data
----------------

.. automodule:: stellargraph.data
  :members: UniformRandomWalk, BiasedRandomWalk, UniformRandomMetaPathWalk, SampledBreadthFirstWalk, SampledHeterogeneousBreadthFirstWalk, TemporalRandomWalk, UnsupervisedSampler, EdgeSplitter, from_epgm


Generators
-----------

.. automodule:: stellargraph.mapper
  :members: Generator, FullBatchNodeGenerator, FullBatchLinkGenerator, GraphSAGENodeGenerator, DirectedGraphSAGENodeGenerator, DirectedGraphSAGELinkGenerator, ClusterNodeGenerator, GraphSAGELinkGenerator, HinSAGENodeGenerator, HinSAGELinkGenerator, Attri2VecNodeGenerator, Attri2VecLinkGenerator, Node2VecNodeGenerator, Node2VecLinkGenerator, RelationalFullBatchNodeGenerator, AdjacencyPowerGenerator, GraphWaveGenerator, CorruptedGenerator, PaddedGraphGenerator, KGTripleGenerator


Layers and models
-----------------

.. automodule:: stellargraph.layer

GraphSAGE
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: GraphSAGE
  :members:

.. autoclass:: DirectedGraphSAGE
  :members:

.. autoclass:: MeanAggregator
  :members:

.. autoclass:: MeanPoolingAggregator
  :members:

.. autoclass:: MaxPoolingAggregator
  :members:

.. autoclass:: AttentionalAggregator
  :members:


HinSAGE
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: HinSAGE
  :members:

.. autoclass:: MeanHinAggregator
  :members:


Node2Vec
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: Node2Vec
  :members:


Attri2Vec
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: Attri2Vec
  :members:


GCN
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: GCN
  :members:

.. autoclass:: GraphConvolution
  :members:

Cluster-GCN
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: ClusterGCN
  :members:

.. autoclass:: ClusterGraphConvolution
  :members:

RGCN
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: RGCN
  :members:

.. autoclass:: RelationalGraphConvolution
  :members:

PPNP
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: PPNP
  :members:

.. autoclass:: PPNPPropagationLayer
  :members:

APPNP
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: APPNP
  :members:

.. autoclass:: APPNPPropagationLayer
  :members:

GAT
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: GAT
  :members:

.. autoclass:: GraphAttention
  :members:

.. autoclass:: GraphAttentionSparse
  :members:

Watch Your Step
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass: WatchYourStep
  :members:

Knowledge Graph models
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: ComplEx
  :members:

.. autoclass:: ComplExScore
  :members:

.. autoclass:: DistMult
  :members:

.. autoclass:: DistMultScore
  :members:

.. autoclass:: RotatE
  :members:

.. autoclass:: RotatEScore
  :members:

GCN Supervised Graph Classification
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: GCNSupervisedGraphClassification
  :members:

Deep Graph Convolutional Neural Network
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: SortPooling
  :members:

.. autoclass:: DeepGraphCNN
  :members:

Graph Convolution LSTM
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: GraphConvolutionLSTM
  :members:

.. autoclass:: FixedAdjacencyGraphConvolution
  :members:

Deep Graph Infomax
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: DeepGraphInfomax
  :members:

.. autoclass:: DGIDiscriminator
  :members:

Link prediction
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: LinkEmbedding
  :members:

.. autofunction:: link_classification

.. autofunction:: link_regression

.. autofunction:: link_inference

Ensembles
------------------------

.. automodule:: stellargraph.ensemble
  :members: Ensemble, BaggingEnsemble


Calibration
------------------------

.. automodule:: stellargraph.calibration
  :members: expected_calibration_error, plot_reliability_diagram, TemperatureCalibration, IsotonicCalibration


Neo4j Connector
----------------

.. automodule:: stellargraph.connector.neo4j
  :members: Neo4jStellarGraph, Neo4jStellarDiGraph, Neo4JGraphSAGENodeGenerator, Neo4JDirectedGraphSAGENodeGenerator, Neo4jSampledBreadthFirstWalk, Neo4jDirectedBreadthFirstNeighbors


Utilities
------------------------

.. automodule:: stellargraph.utils
  :members: plot_history


Datasets
----------------

.. automodule:: stellargraph.datasets
  :members:
  :inherited-members:


Random
------------------------

.. automodule:: stellargraph.random
  :members: set_seed
