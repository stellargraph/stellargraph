import keras
import numpy as np
import pandas as pd
from typing import List


class GeneGraph:
    def __init__(self, edge_data_file, gene_attr_file):
        # Read edge data
        edge_data = pd.read_csv(edge_data_file, delim_whitespace=True).drop(
            ["score", "data_source"], axis=1
        )
        edge_data["ensg1"] = edge_data["ensg1"].str.replace("ENSG", "").astype(int)
        edge_data["ensg2"] = edge_data["ensg2"].str.replace("ENSG", "").astype(int)

        # Read attribute data
        gene_attr = pd.read_csv(gene_attr_file, sep="\t").drop_duplicates("ensg")
        gene_attr = gene_attr[gene_attr.ensg.str.contains("LRG") == False]
        gene_attr["ensg"] = gene_attr["ensg"].str.replace("ENSG", "").astype(int)
        gene_attr.drop(
            ["gene_name", "biotype", "SNP_id", "GWAS_pvalue", "ps_pval"],
            axis=1,
            inplace=True,
        )
        gene_attr.set_index("ensg", inplace=True)

        # List of IDs
        rnd = np.random.uniform(0, 1, len(gene_attr))
        self.ids_train = gene_attr.loc[rnd <= 0.5].index.values
        self.ids_val = gene_attr.loc[(rnd > 0.5) & (rnd < 0.7)].index.values
        self.ids_test = gene_attr.loc[rnd >= 0.7].index.values

        # Features
        self.feats = gene_attr.drop(["node_type"], axis=1)
        self.feats.loc[-1] = [0] * self.feats.shape[
            1
        ]  # create an all-zeros feature vector for the special node with ind -1, i.e., a non-existant node

        # Labels
        self.labels = gene_attr["node_type"].map(lambda x: x == "alz")

        # YT: create separate adjacency lists, one per edge type:
        edge_types = np.unique(edge_data["int_type"])
        self.adj = dict()
        for edge_type in edge_types:
            self.adj[edge_type] = (
                edge_data.loc[edge_data["int_type"] == edge_type]
                .groupby(["ensg1"])["ensg2"]
                .apply(list)
            )
            # YT: add special (non-existent) node's entry into adj lists for all edge types: neighbours of node [-1] (non-existent)
            # is an empty list
            self.adj[edge_type][-1] = []

        # YT: add empty neighbour lists to adj lists, for nodes that don't have neighbours via the corresponding edge type
        for edge_type in edge_types:
            missing_genes = set(gene_attr.index).difference(
                set(self.adj[edge_type].index)
            )
            adj_ext = pd.Series(
                [list()] * len(missing_genes), index=missing_genes
            )  # form a series to append to self.adj[edge_type]
            self.adj[edge_type] = self.adj[edge_type].append(
                adj_ext, verify_integrity=True
            )  # append adj_ext

    def get_feats(self, indices: List[int]):
        """Get features of nodes whose list is given in indices"""
        return self.feats.loc[indices].fillna(0).as_matrix()

    def get_labels(self, indices: List[int]):
        return np.array(self.labels[indices], dtype=np.float64)

    def sample_neighs(self, indices: List[int], ns: int):
        """Neighbour sampling method"""

        def with_adj(adj_curr):
            if ns > 0:
                return [
                    [-1] * ns
                    if len(adj) == 0 or ((not isinstance(adj, list)) and pd.isnull(adj))
                    else [
                        adj[i] for i in np.random.randint(len(adj), size=ns)
                    ]  # YT: sample ns neighbours of each node in indices
                    for adj in adj_curr.loc[indices].values
                ]
            else:  # YT: if ns=0, do not sample neighbours and return a special node -1 (non-existent) with all-zero feature vector for neighbour aggregation
                return [[-1] for adj in adj_curr.loc[indices].values]

        return tuple([with_adj(adj) for adj in self.adj.values()])

    def get_batch(
        self, indices: List[int], ns: List[int]
    ):  # This will soon be replaced by the Mapper class
        nb = len(indices)
        flatten = lambda l: [item for sublist in l for item in sublist]

        # ppi, coex, epis = self.sample_neighs(indices, ns[0])
        # ppi_1 = self.sample_neighs(flatten(ppi), ns[1])
        # coex_1 = self.sample_neighs(flatten(coex), ns[1])
        # epis_1 = self.sample_neighs(flatten(epis), ns[1])
        # return (
        #     self.get_labels(indices),
        #     [
        #         self.get_feats(flatten(inds)).reshape([nb, -1, self.feats.shape[1]])
        #         for inds in [[indices], ppi, coex, epis, *ppi_1, *coex_1, *epis_1]
        #     ],
        # )

        # YT: somewhat generalized version of the commented-out snippet above:
        neigh_1hop = self.sample_neighs(indices, ns[0])
        neigh_2hop = dict()
        for i, et in enumerate(self.adj.keys()):
            neigh_2hop[et] = self.sample_neighs(flatten(neigh_1hop[i]), ns[1])

        return (
            self.get_labels(indices),
            [
                self.get_feats(flatten(inds)).reshape([nb, -1, self.feats.shape[1]])
                for inds in [
                    [indices],
                    *neigh_1hop,
                    *[n for n in neigh_2hop[et] for et in self.adj.keys()],
                ]
            ],
        )


class DataGenerator(keras.utils.Sequence):
    """Generates data for Keras"""

    def __init__(
        self,
        g: GeneGraph,
        nf: int,
        ns: List[int],
        batch_size: int = 1000,
        name: str = "train",
    ):
        """Initialization"""
        if isinstance(batch_size, int):
            self.batch_size = batch_size
        else:
            raise Exception(
                "DataGenerator: batch_size should be of type int, got {}".format(
                    type(batch_size)
                )
            )

        if isinstance(nf, int):
            self.nf = nf
        else:
            raise Exception(
                "DataGenerator: nf should be of type int, got {}".format(type(nf))
            )

        self.g = g
        self.ns = ns
        if name == "train":
            self.ids = g.ids_train
        elif name == "validate":
            self.ids = g.ids_val
        else:
            raise Exception(
                'DataGenerator: name is {}; should be either "train" or "validate"'.format(
                    name
                )
            )

        self.data_size = len(self.ids)
        self.idx = 0
        self.name = name
        self.on_epoch_end()  # shuffle the data entries

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return int(np.ceil(self.data_size / self.batch_size))

    def __getitem__(self, index):
        """Generate one batch of data"""
        if self.idx > self.data_size:
            raise Exception(
                "DataGenerator: index {} exceeds data size {}. This shouldn't happen!".format(
                    self.idx, self.data_size
                )
            )
        elif self.idx == self.data_size:
            # print(
            #     "DataGenerator: index {} is equal to data size {}. Calling self.on_epoch_end()...".format(
            #         self.idx, self.data_size
            #     )
            # )
            self.on_epoch_end()
        else:
            pass

        end = min(self.idx + self.batch_size, self.data_size)
        indices = list(self.ids[range(self.idx, end)])
        tgt, inp = self.g.get_batch(indices, self.ns)
        self.idx = end

        # print("{}DataGenerator: len(inp) = {}, len(tgt) = {}".format(self.name, len(inp), len(tgt)))
        # print('{}DataGenerator: index={}, batch size={}, self.idx={}, self.data_size={}'.format(self.name,index,len(indices),self.idx,self.data_size))

        return inp, tgt

    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        self.idx = 0
        if self.name == "train":
            np.random.shuffle(self.ids)


class TestDataGenerator(keras.utils.Sequence):
    """Generates data for Keras"""

    def __init__(self, g: GeneGraph, nf: int, ns: List[int], batch_size: int = 1000):
        """Initialization"""
        self.batch_size = batch_size
        self.g = g
        self.ids = g.ids_test
        self.data_size = len(self.ids)
        self.nf = nf
        self.ns = ns
        self.idx = 0
        self.y_true = []

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return int(np.ceil(self.data_size / self.batch_size))

    def __getitem__(self, index):
        """Generate one batch of data"""
        end = min(self.idx + self.batch_size, self.data_size)
        indices = list(self.ids[range(self.idx, end)])
        tgt, inp = self.g.get_batch(indices, self.ns)
        self.y_true += [tgt]
        self.idx = end

        # print('TestDataGenerator: index={}, batch size={}, self.idx={}, self.data_size={}'.format(index, len(indices),
        #                                                                                            self.idx,
        #                                                                                            self.data_size))

        return inp

    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        self.idx = 0
