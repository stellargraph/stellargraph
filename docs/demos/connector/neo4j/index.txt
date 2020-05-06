Neo4j
====================================

In this folder:

- :doc:`directed-graphsage-on-cora-neo4j-example` and :doc:`undirected-graphsage-on-cora-neo4j-example` provide examples of running GraphSAGE with connection to graph database.

- :doc:`load-cora-into-neo4j` provides demo of loading Cora network citation dataset into neo4j database.

Required Installations:

- **Neo4J**: `Instruction to download <https://neo4j.com/docs/operations-manual/current/installation/>`_.
- **APOC library** plug-in: APOC provides utilities for common procedures and functions in Neo4J. `Instruction to download and install <https://neo4j.com/developer/neo4j-apoc/>`_.
- **py2neo**: A client library and toolkit to connect to Neo4J database from within python applications. Install ``py2neo`` by using pip: ``pip install py2neo``. `documentation <https://py2neo.org/v4/>`_.


.. warning::

   All functionalities demonstrated in the above-mentioned notebooks are experimental. They have not been tested thoroughly and the implementation might be dramatically changed.

There is also :doc:`a demonstration of loading data into memory from Neo4j <../../basics/loading-saving-neo4j>`.  This allows using any StellarGraph algorithm on data from Neo4j.

Table of contents
-----------------

.. toctree::
    :titlesonly:
    :glob:

    */index
    ./*
