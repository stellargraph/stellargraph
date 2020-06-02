Link prediction
==================================

`StellarGraph <https://github.com/stellargraph/stellargraph>`_ provides numerous algorithms for doing link prediction on graphs. This folder contains demos of all of them to explain how they work and how to use them as part of a TensorFlow Keras data science workflow.

A link prediction task predicts an attribute of links/edges in a graph. For instance, predicting whether a link/edge that isn't already in the graph should exist (binary classification, or recommendation, or knowledge base completion, in a knowledge graph), or even labelling existing links with a categorical class (binary classification or multiclass classification), or predicting a continuous number (regression). It is supervised or semi-supervised, where the model is trained using a subset of links/edges that have ground-truth labels. For predicting edge existence, the ground-truth may just be whether the edge exists in the original data, rather than a separate label.

Link prediction can also be done as a downstream task from node representation learning/embeddings, by combining node embedding vectors for the source and target nodes of the edge and training a supervised or semi-supervised classifier against the result. Unsupervised algorithms that can be used in this manner include random walk-based methods like Metapath2Vec. StellarGraph provides :doc:`demos of unsupervised algorithms <../embeddings/index>`.

Find algorithms and demos for a graph
-------------------------------------

This table lists all node classification demos, including the algorithms trained and the types of graph used.

.. list-table::
   :header-rows: 1

   * - Demo
     - Algorithm
     - Node features
     - Heterogeneous
     - Temporal
   * - :doc:`GCN <gcn-link-prediction>`
     - GCN
     - yes
     -
     -
   * - :doc:`Attri2Vec <attri2vec-link-prediction>`
     - Attri2Vec
     - yes
     -
     -
   * - :doc:`GraphSAGE <graphsage-link-prediction>`
     - GraphSAGE
     - yes
     -
     -
   * - :doc:`HinSAGE <hinsage-link-prediction>`
     - HinSAGE
     - yes
     - yes
     -
   * - :doc:`Node2Vec <node2vec-link-prediction>`
     - Node2Vec
     -
     -
     -
   * - :doc:`Metapath2Vec <metapath2vec-link-prediction>`
     - Metapath2Vec
     -
     - yes
     -
   * - :doc:`CTDNE <ctdne-link-prediction>`
     - CTDNE
     -
     -
     - yes
   * - :doc:`ComplEx <complex-link-prediction>`
     - ComplEx
     -
     - yes, multiple edge types
     -
   * - :doc:`DistMult <distmult-link-prediction>`
     - DistMult
     -
     - yes, multiple edge types
     -


See :doc:`the root README <../../README>` or each algorithm's documentation for the relevant citation(s). See :doc:`the demo index <../index>` for more tasks, and a summary of each algorithm.

Table of contents
-----------------

.. toctree::
    :titlesonly:
    :glob:

    ./*
