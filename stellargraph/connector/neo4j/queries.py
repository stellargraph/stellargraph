import numpy as np
import scipy.sparse as sps


def neo4j_adjacency_matrix(graph, nodes):
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
    result = graph.run(subgraph_query, parameters={"node_id_list": nodes})
    adj_list = result.data()[0]['neighbors']
    index = dict(zip(nodes, range(len(nodes))))

    def _remove_invalid(arr):
        return arr[arr != -1]

    def _numpy_indexer(arr):
        return np.array([index.get(x, -1) for x in arr])

    adj_list = [_remove_invalid(_numpy_indexer(neighs)) for neighs in adj_list]

    indptr = np.cumsum([0] + [len(neighs) for neighs in adj_list])
    indices = np.concatenate(adj_list)
    data = np.ones(len(indices))
    shape = (len(nodes), len(nodes))
    adj_mat = sps.csr_matrix((data, indices, indptr), shape=shape)
    return adj_mat


def neo4j_features(graph, nodes):
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

    result = graph.run(
        feature_query,
        parameters={"node_id_list": nodes},
    )

    return np.array(result.data()[0]['features'])


def neo4j_node_ids(graph):
    node_ids_query = f"""
        CALL apoc.cypher.run(
            'MATCH (n)
            RETURN n.ID as node_ids',
            {{}}
        ) YIELD value

        RETURN collect(value.node_ids) as node_ids
        """
    result = graph.run(node_ids_query)

    return np.array(result.data()[0]['node_ids'])


def neo4j_node_ids_to_ilocs(graph, node_ids):
    node_ids_query = f"""
        CALL apoc.cypher.run(
            'MATCH (n)
            WHERE n.ID in $node_id_list
            RETURN id(n) as ilocs',
            {{node_id_list: $node_id_list}}
        ) YIELD value

        RETURN collect(value.ilocs) as ilocs
        """
    result = graph.run(
        node_ids_query,
        parameters={"node_id_list": node_ids}
    )

    return np.array(result.data()[0]['ilocs'])