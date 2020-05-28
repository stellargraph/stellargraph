StellarGraph basics
===================

`StellarGraph <https://github.com/stellargraph/stellargraph>`_ has support for loading data via Pandas, NetworkX and Neo4j. This folder contains examples of the loading data into a ``StellarGraph`` object, which is the format used by the machine learning algorithms in this library.

Find demos for a format
-----------------------

.. list-table::
   :header-rows: 1

   * - Demo
     - Data formats
     - Performance
     - Data preprocessing
   * - :doc:`loading-pandas`
     - Anything `supported by Pandas <https://pandas.pydata.org/pandas-docs/stable/user_guide/io.html>`__: CSV, TSV, Excel, JSON, SQL, HDF5, many more
     - Good
     - Good, using Pandas and `scikit-learn <http://scikit-learn.github.io/stable>`__ and more
   * - :doc:`loading-networkx`
     - Anything `supported by NetworkX <https://networkx.github.io/documentation/stable/reference/readwrite/index.html>`__: GEXF, GML, GraphML, Shapefiles, many more
     - Poor
     - Good, for graph-related preprocessing
   * - :doc:`loading-saving-neo4j`
     - Any Cypher query supported by `Neo4j <https://neo4j.com>`__
     - Good for subgraphs and other queries
     - Good, using Cypher functionality

See :doc:`all demos for machine learning algorithms <../index>`.

Table of contents
-----------------

.. toctree::
    :titlesonly:
    :glob:

    ./*
