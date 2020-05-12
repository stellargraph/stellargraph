Interpretability of node classification results
==================================================================

`StellarGraph <https://github.com/stellargraph/stellargraph>`_ has support for inspecting several different algorithms for :doc:`node classification <../node-classification/index>` to understand or interpret how they came to a decision. This folder contains demos of all of them to explain how they work and how to use them as part of a TensorFlow Keras data science workflow.

Interpreting a model involves training and making predictions for a model, and then analysing the predictions and the model to find which neighbours and which features had the largest influence on the prediction.

Find algorithms and demos
-------------------------

This table lists interpretability demos, including the algorithms used.

.. list-table::
   :header-rows: 1

   * - demo
     - algorithm(s)
   * - :doc:`GCN (dense) <gcn-node-link-importance>`
     - GCN, Integrated Gradients
   * - :doc:`GCN (sparse) <gcn-sparse-node-link-importance>`
     - GCN, Integrated Gradients
   * - :doc:`GAT <gat-node-link-importance>`
     - GAT, Integrated Gradients


See :doc:`the root README <../../README>` or each algorithm's documentation for the relevant citation(s). See :doc:`the demo index <../index>` for more tasks, and a summary of each algorithm. See :doc:`the node classification demos <../node-classification/index>` for more details on the base task.

Table of contents
-----------------

.. toctree::
    :titlesonly:
    :glob:

    ./*
