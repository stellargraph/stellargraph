__all__ = ["Neo4jStellarGraph"]

import numpy as np
import scipy.sparse as sps
from ...core.experimental import experimental


@experimental(reason="the class is not fully tested")
class Neo4jStellarGraph:
    def __init__(self, graph_db, is_directed=False):

        self.graph_db = graph_db
        self._is_directed = is_directed

    def nodes(self):
        # FIXME: don't assume stellargraphs "ID"s are in the Neo4j graph
        # use id() instead
        node_ids_query = f"""    
            CALL apoc.cypher.run(
                'MATCH (n)
                RETURN n.ID as node_ids',
                {{}}
            ) YIELD value

            RETURN collect(value.node_ids) as node_ids
            """
        result = self.graph_db.run(node_ids_query)

        return np.array(result.data()[0]["node_ids"])

    def node_features(self, node_ids):
        feature_query = f"""
            UNWIND $node_id_list AS node_id

            // for each node id in every row, collect the random list of its neighbors.
            CALL apoc.cypher.run(

                'MATCH(cur_node) WHERE cur_node.ID = $node_id
                RETURN cur_node.features as features',
                {{node_id: node_id}}
            ) YIELD value

            RETURN collect(value.features) as features
            """

        result = self.graph_db.run(
            feature_query, parameters={"node_id_list": node_ids},
        )

        return np.array(result.data()[0]["features"])

    def to_adjacency_matrix(self, node_ids):
        subgraph_query = f"""
            UNWIND $node_id_list AS node_id

            // for each node id in every row, collect the random list of its neighbors.
            CALL apoc.cypher.run(

                'MATCH(cur_node) WHERE cur_node.ID = $node_id

                // find the neighbors
                MATCH (cur_node)--(neighbors)
                WITH collect(neighbors.ID) AS neigh_ids
                RETURN neigh_ids',
                {{node_id: node_id}}) YIELD value

            RETURN collect(value.neigh_ids) as neighbors
            """
        result = self.graph_db.run(
            subgraph_query, parameters={"node_id_list": node_ids}
        )
        adj_list = result.data()[0]["neighbors"]
        index = dict(zip(node_ids, range(len(node_ids))))

        def _remove_invalid(arr):
            return arr[arr != -1]

        def _numpy_indexer(arr):
            return np.array([index.get(x, -1) for x in arr])

        adj_list = [_remove_invalid(_numpy_indexer(neighs)) for neighs in adj_list]

        indptr = np.cumsum([0] + [len(neighs) for neighs in adj_list])
        indices = np.concatenate(adj_list)
        data = np.ones(len(indices), dtype=np.float32)
        shape = (len(node_ids), len(node_ids))
        adj = sps.csr_matrix((data, indices, indptr), shape=shape)

        if not self.is_directed() and len(data) > 0:
            # in an undirected graph, the adjacency matrix should be symmetric: which means counting
            # weights from either "incoming" or "outgoing" edges, but not double-counting self loops

            # FIXME https://github.com/scipy/scipy/issues/11949: these operations, particularly the
            # diagonal, don't work for an empty matrix (n == 0)
            backward = adj.transpose(copy=True)
            # this is setdiag(0), but faster, since it doesn't change the sparsity structure of the
            # matrix (https://github.com/scipy/scipy/issues/11600)
            (nonzero,) = backward.diagonal().nonzero()
            backward[nonzero, nonzero] = 0

            adj += backward

        # this is a multigraph, let's eliminate any duplicate entries
        adj.sum_duplicates()
        return adj

    def is_directed(self):
        return self._is_directed


class Neo4jStellarDiGraph(Neo4jStellarGraph):
    def __init__(self, graph_db):
        super().__init__(graph_db, is_directed=True)
