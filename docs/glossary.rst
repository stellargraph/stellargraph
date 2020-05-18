Glossary
========

.. glossary::
   :sorted:

   edge
   link
   relationship
     An edge connects two :term:`nodes <node>` in a graph.

   entity
   node
   vertex
     The entities or objects in a graph, that are connected by :term:`edges <edge>`.


   node attribute inference
   node classification
     A task that predicts properties of individual :term:`nodes <node>` in a graph. For example, predicting the subject of an academic paper, or the flowering time of a crop. Despite the "classification" in the name, this includes regression tasks.

     .. seealso:: :doc:`Demos of algorithms <demos/node-classification/index>` for node classification.

   link prediction
     A task that predicts properties of :term:`edges <edge>` in a graph. For example, predicting the probability of a "likes" edge between a user and a product (recommendation), or the strength of an atomic bond. This is typically framed as predicting new edges in a graph (where the predicted property is some measure of probability/likelihood of the edge existing), but includes predicting properties on existing edges.

     .. seealso:: :doc:`Demos of algorithms <demos/link-prediction/index>` for link prediction.

   inductive
     A task is inductive if it generalises to unseen input. For :term:`node classification` and :term:`link prediction` tasks, this typically means a model trained on one graph (or subgraph) can be used for prediction on new nodes (whether a whole new graph or the larger graph that contains the training subgraph).

   iloc
   integer location
      Similar to Pandas, ilocs are sequential integers that allow for efficient storage and indexing. The :class:`stellargraph.StellarGraph` class typically stores external IDs as ilocs internally.
