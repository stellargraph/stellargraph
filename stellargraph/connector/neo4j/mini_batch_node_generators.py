from ...core.utils import is_real_iterable
import copy
import numpy as np
from .queries import neo4j_adjacency_matrix, neo4j_features
import random


class Neo4jClusterNodeGenerator:

    def __init__(self, neo4j_graphdb, clusters=1, q=1, lam=0.1, name=None):
        try:
            import py2neo
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                f"{e.msg}. StellarGraph can only connect to Neo4j using the 'py2neo' module; please install it",
                name=e.name,
                path=e.path,
            ) from None

        if not isinstance(neo4j_graphdb, py2neo.Graph):
            raise TypeError(
                f"neo4j_graphdb: expected py2neo.Graph, found {type(neo4j_graphdb)}"
            )

        if not isinstance(clusters, list):
            raise TypeError(f"{clusters}: expect list found {str(type(clusters))}.")

        self.name = name
        self.q = q  # The number of clusters to sample per mini-batch
        self.lam = lam
        self.clusters = clusters
        self.neo4j_graphdb = neo4j_graphdb

    def flow(self, node_ids, targets=None, name=None):
        if targets is not None:
            # Check targets is an iterable
            if not is_real_iterable(targets):
                raise TypeError(
                    "{}: Targets must be an iterable or None".format(
                        type(self).__name__
                    )
                )

            # Check targets correct shape
            if len(targets) != len(node_ids):
                raise ValueError(
                    "{}: Targets must be the same length as node_ids".format(
                        type(self).__name__
                    )
                )

        return Neo4jClusterNodeSequence(
            self.neo4j_graphdb,
            self.clusters,
            targets=targets,
            node_ids=node_ids,
            q=self.q,
            lam=self.lam,
            name=name,
        )


class Neo4jClusterNodeSequence:

    def __init__(
            self,
            neo4j_graphdb,
            clusters,
            targets=None,
            node_ids=None,
            normalize_adj=True,
            q=1,
            lam=0.1,
            name=None
    ):
        self.name = name
        self.clusters = list()
        self.clusters_original = copy.deepcopy(clusters)
        self.neo4j_graphdb = neo4j_graphdb
        self.normalize_adj = normalize_adj
        self.q = q
        self.lam = lam
        self.node_order = list()
        self._node_order_in_progress = list()
        self.__node_buffer = dict()
        self.target_ids = list()

        if node_ids is not None:
            self.target_ids = list(node_ids)

        if targets is not None:
            if node_ids is None:
                raise ValueError(
                    "Since targets is not None, node_ids must be given and cannot be None."
                )

            if len(node_ids) != len(targets):
                raise ValueError(
                    "When passed together targets and indices should be the same length."
                )

            self.targets = np.asanyarray(targets)
            self.target_node_lookup = dict(
                zip(self.target_ids, range(len(self.target_ids)))
            )
        else:
            self.targets = None

        self.on_epoch_end()

    def __len__(self):
        num_batches = len(self.clusters_original) // self.q
        return num_batches

    def _diagonal_enhanced_normalization(self, adj_cluster):
        # Cluster-GCN normalization is:
        #     A~ + λdiag(A~) where A~ = N(A + I) with normalization factor N = (D + I)^(-1)
        #
        # Expands to:
        #     NA + NI + λN(diag(A) + I) =
        #     NA + N(I + λ(diag(A) + I)) =
        #     NA + λN(diag(A) + (1 + 1/λ)I))
        #
        # (This could potentially become a layer, to benefit from a GPU.)
        degrees = np.asarray(adj_cluster.sum(axis=1)).ravel()
        normalization = 1 / (degrees + 1)

        # NA: multiply rows manually
        norm_adj = adj_cluster.multiply(normalization[:, None]).toarray()

        # λN(diag(A) + (1 + 1/λ)I): work with the diagonals directly
        diag = np.diag(norm_adj)
        diag_addition = (
            normalization * self.lam * (adj_cluster.diagonal() + (1 + 1 / self.lam))
        )
        np.fill_diagonal(norm_adj, diag + diag_addition)
        return norm_adj

    def __getitem__(self, index):
        # The next batch should be the adjacency matrix for the cluster and the corresponding feature vectors
        # and targets if available.
        cluster = self.clusters[index]

        adj_cluster = neo4j_adjacency_matrix(self.neo4j_graphdb, cluster)

        if self.normalize_adj:
            adj_cluster = self._diagonal_enhanced_normalization(adj_cluster)
        else:
            adj_cluster = adj_cluster.toarray()

        g_node_list = list(cluster)

        # Determine the target nodes that exist in this cluster
        target_nodes_in_cluster = np.asanyarray(
            list(set(g_node_list).intersection(self.target_ids))
        )

        self.__node_buffer[index] = target_nodes_in_cluster

        # Dictionary to store node indices for quicker node index lookups
        node_lookup = dict(zip(g_node_list, range(len(g_node_list))))

        # The list of indices of the target nodes in self.node_list
        target_node_indices = np.array(
            [node_lookup[n] for n in target_nodes_in_cluster]
        )

        if index == (len(self.clusters_original) // self.q) - 1:
            # last batch
            self.__node_buffer_dict_to_list()

        cluster_targets = None
        #
        if self.targets is not None:
            # Dictionary to store node indices for quicker node index lookups
            # The list of indices of the target nodes in self.node_list
            cluster_target_indices = np.array(
                [self.target_node_lookup[n] for n in target_nodes_in_cluster]
            )
            cluster_targets = self.targets[cluster_target_indices]
            cluster_targets = cluster_targets.reshape((1,) + cluster_targets.shape)

        features = neo4j_features(self.neo4j_graphdb, g_node_list)

        features = np.reshape(features, (1,) + features.shape)
        adj_cluster = adj_cluster.reshape((1,) + adj_cluster.shape)
        target_node_indices = target_node_indices[np.newaxis, :]

        return [features, target_node_indices, adj_cluster], cluster_targets

    def __node_buffer_dict_to_list(self):
        self.node_order = []
        for k, v in self.__node_buffer.items():
            self.node_order.extend(v)

    def on_epoch_end(self):
        """
         Shuffle all nodes at the end of each epoch
        """
        if self.q > 1:
            # combine clusters
            cluster_indices = list(range(len(self.clusters_original)))
            random.shuffle(cluster_indices)
            self.clusters = []

            for i in range(0, len(cluster_indices) - 1, self.q):
                cc = cluster_indices[i : i + self.q]
                tmp = []
                for l in cc:
                    tmp.extend(list(self.clusters_original[l]))
                self.clusters.append(tmp)
        else:
            self.clusters = copy.deepcopy(self.clusters_original)

        self.__node_buffer = dict()

        random.shuffle(self.clusters)

