StellarGraph basics
===================

`StellarGraph <https://github.com/stellargraph/stellargraph>`_ has support for loading data via Pandas and NetworkX. This folder contains examples of the loading data into a ``StellarGraph`` object, which is the format used by the machine learning algorithms in this library.

Find demos for a format
-----------------------


* :doc:`loading-pandas <loading-pandas>` shows the recommended way to load data, using Pandas (supporting any input format that Pandas supports, including CSV files and SQL databases)
* :doc:`loading-numpy <loading-numpy>` shows the lowest-overhead way to load data, using NumPy
* :doc:`loading-networkx <loading-networkx>` shows how to load data from a `NetworkX <https://networkx.github.io>`_ graph
* :doc:`loading-saving-neo4j <loading-saving-neo4j>` shows how to load data from a `Neo4j <https://neo4j.com>`_ database, and save results back to it

See :doc:`all demos for machine learning algorithms <../index>`.

Table of contents
-----------------

.. toctree::
    :titlesonly:
    :glob:

    ./*
