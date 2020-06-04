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
     - Via Pandas, `scikit-learn <http://scikit-learn.github.io/stable>`__ and more
   * - :doc:`loading-numpy`
     - Anything supported by `NumPy <https://numpy.org/doc/1.18/reference/routines.io.html>`__, `SciPy <https://docs.scipy.org/doc/scipy/reference/io.html>`__ or other libraries: CSV, TSV, MATLAB ``.mat``, NetCDF, many more
     - Best
     - Via NumPy, `scikit-learn <http://scikit-learn.github.io/stable>`__ and more
   * - :doc:`loading-networkx`
     - Anything `supported by NetworkX <https://networkx.github.io/documentation/stable/reference/readwrite/index.html>`__: Adjacency lists, GEXF, GML, GraphML, Shapefiles, many more
     - Poor
     - Via graph-focused transforms and functions in NetworkX
   * - :doc:`loading-saving-neo4j`
     - Any Cypher query supported by `Neo4j <https://neo4j.com>`__
     - Good for subgraphs and other queries
     - Via Cypher functionality

See :doc:`all demos for machine learning algorithms <../index>`.

badspelling

badlink_

Table of contents
-----------------

.. toctree::
    :titlesonly:
    :glob:

    ./*
