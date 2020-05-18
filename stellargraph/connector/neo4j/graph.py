__all__ = ["Neo4jStellarGraph"]

import numpy as np
import scipy.sparse as sps
import pandas as pd
from ...core.experimental import experimental


@experimental(reason="the class is not fully tested and lacks documentation")
class Neo4jStellarGraph:
    def __init__(self, graph_db, is_directed=False):

        self.graph_db = graph_db
        self._is_directed = is_directed

    def nodes(self):
        node_ids_query = f"""    
            MATCH (n)
            RETURN n.ID as node_id
            """

        result = self.graph_db.run(node_ids_query)
        data = result.data()
        return np.array([row["node_id"] for row in data])

    def node_features(self, node_ids):
        feature_query = f"""
            UNWIND $node_id_list AS node_id
            MATCH(node) WHERE node.ID = node_id
            RETURN node.features as features
            """
        result = self.graph_db.run(
            feature_query, parameters={"node_id_list": node_ids},
        )
        features = np.array([row["features"] for row in result.data()])
        return features

    def to_adjacency_matrix(self, node_ids):
        # neo4j optimizes this query to be O(edges incident to nodes)
        # not O(E) as it appears
        subgraph_query = f"""
            MATCH (source)-->(target)
            WHERE source.ID IN $node_id_list AND target.ID IN $node_id_list
            RETURN collect(source.ID) AS sources, collect(target.ID) as targets
            """

        result = self.graph_db.run(
            subgraph_query, parameters={"node_id_list": node_ids}
        )

        data = result.data()[0]
        sources = np.array(data["sources"])
        targets = np.array(data["targets"])

        index = pd.Index(node_ids)

        src_idx = index.get_indexer(sources)
        tgt_idx = index.get_indexer(targets)

        weights = np.ones(len(sources), dtype=np.float32)
        shape = (len(node_ids), len(node_ids))
        adj = sps.csr_matrix((weights, (src_idx, tgt_idx)), shape=shape)

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
