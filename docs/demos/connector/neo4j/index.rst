Neo4j
====================================

`StellarGraph <https://github.com/stellargraph/stellargraph>`_ provides various algorithms that can be run on graphs in Neo4J. This folder contains demos of all of them to explain how to connect to Neo4J and how to run these algorithms as part of a TensorFlow Keras data science workflow.

Required Installations:

- **Neo4J**: `Instruction to download <https://neo4j.com/docs/operations-manual/current/installation/>`_.
- **APOC library** plug-in: APOC provides utilities for common procedures and functions in Neo4J. `Instruction to download and install <https://neo4j.com/developer/neo4j-apoc/>`_.
- **py2neo**: A client library and toolkit to connect to Neo4J database from within python applications. Install ``py2neo`` by using pip: ``pip install py2neo``. `documentation <https://py2neo.org/v4/>`_.

.. warning::

   All functionalities demonstrated in the notebooks below are still experimental. They have not been tested thoroughly and the implementation might be dramatically changed.

Find algorithms and demos for a graph
-------------------------------------

This table lists all Neo4J demos, including the algorithms trained, the types of graph used, and the tasks demonstrated.

.. list-table::
   :header-rows: 1

   * - Demo
     - Algorithm(s)
     - Task
     - Node features
     - Directed
   * - :doc:`Load Cora <load-cora-into-neo4j>`
     -
     - Dataset Loading
     -
     -
   * - :doc:`GraphSAGE <undirected-graphsage-on-cora-neo4j-example>`
     - GraphSAGE
     - Node classification
     - yes
     -
     -
   * - :doc:`Directed GraphSAGE <directed-graphsage-on-cora-neo4j-example>`
     - GraphSAGE
     - Node classification
     - yes
     - yes
   * - :doc:`Cluster-GCN <cluster-gcn-on-cora-neo4j-example>`
     - Cluster-GCN
     - Node classification
     - yes
     -


See :doc:`the root README <../../../README>` or each algorithm's documentation for the relevant citation(s). See :doc:`the demo index <../../index>` for more tasks, and a summary of each algorithm.

There is also :doc:`a demonstration of loading data into memory from Neo4J <../../basics/loading-saving-neo4j>`.  This allows using any StellarGraph algorithm on data from Neo4J.

Table of contents
-----------------

.. toctree::
    :titlesonly:
    :glob:

    ./*
