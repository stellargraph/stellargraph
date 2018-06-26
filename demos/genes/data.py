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
        self.feats.loc[-1] = [0] * self.feats.shape[1]  # create an all-zeros feature vector for the special node with ind -1, i.e., a non-existant node

        # Labels
        self.labels = gene_attr["node_type"].map(lambda x: x == "alz")

        # YT: create separate adjacency lists, one per edge type:
        self.adj_coex = (
            edge_data.loc[edge_data["int_type"] == "coexpression"]
            .groupby(["ensg1"])["ensg2"]
            .apply(list)
        )
        self.adj_ppi = (
            edge_data.loc[edge_data["int_type"] == "PPI"]
            .groupby(["ensg1"])["ensg2"]
            .apply(list)
        )
        self.adj_epis = (
            edge_data.loc[edge_data["int_type"] == "epistasis"]
            .groupby(["ensg1"])["ensg2"]
            .apply(list)
        )
        self.adj_coex[-1] = [-1]  # YT: special (non-existent) node's adj list for coex edge type: neighbour of node [-1] (non-existent)
                                  # is node [-1] (non-existent) with all-zeros feature vector
        self.adj_ppi[-1] = [-1]   # YT: same for ppi edge type
        self.adj_epis[-1] = [-1]  # YT: same for epis edge type
        for gene in gene_attr.index:
            for adj in [self.adj_coex, self.adj_ppi, self.adj_epis]:
                if gene not in adj.index:
                    adj[gene] = []  # YT: add empty neighbour lists to adj lists, for nodes who don't have neighbours via the corresponding edge type

    def get_feats(self, indices: List[int]):
        return self.feats.loc[indices].fillna(0).as_matrix()

    def get_labels(self, indices: List[int]):
        return np.array(self.labels[indices], dtype=np.float64)

    def sample_neighs(self, indices: List[int], ns: int):
        def with_adj(adj_curr):
            return [
                [-1] * ns
                if len(adj) == 0 or ((not isinstance(adj, list)) and pd.isnull(adj))
                else [adj[i] for i in np.random.randint(len(adj), size=ns)]
                for adj in adj_curr.loc[indices].values
            ]

        return with_adj(self.adj_coex), with_adj(self.adj_ppi), with_adj(self.adj_epis)

    def get_batch(self, indices: List[int], ns: List[int]):
        nb = len(indices)
        flatten = lambda l: [item for sublist in l for item in sublist]
        coex, ppi, epis = self.sample_neighs(indices, ns[0])
        coex_1 = self.sample_neighs(flatten(coex), ns[1])
        ppi_1 = self.sample_neighs(flatten(ppi), ns[1])
        epis_1 = self.sample_neighs(flatten(epis), ns[1])
        return (
            self.get_labels(indices),
            [
                self.get_feats(flatten(inds)).reshape([nb, -1, 414])
                for inds in [[indices], coex, ppi, epis, *coex_1, *ppi_1, *epis_1]
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
        if isinstance(batch_size,int):
            self.batch_size = batch_size
        else:
            raise Exception("DataGenerator: batch_size should be of type int, got {}".format(type(batch_size)))

        if isinstance(nf,int):
            self.nf = nf
        else:
            raise Exception("DataGenerator: nf should be of type int, got {}".format(type(nf)))

        self.g = g
        self.ns = ns
        if name == "train":
            self.ids = g.ids_train
        elif name == "validate":
            self.ids = g.ids_val
        else:
            raise Exception("DataGenerator: name is {}; should be either \"train\" or \"validate\"".format(name))

        self.data_size = len(self.ids)
        self.idx = 0
        self.name = name
        self.on_epoch_end()   # shuffle the data entries

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
            print(
                "DataGenerator: index {} is equal to data size {}. Calling self.on_epoch_end()...".format(
                    self.idx, self.data_size
                )
            )
            self.on_epoch_end()
        else:
            pass

        end = min(self.idx + self.batch_size, self.data_size)
        indices = list(self.ids[range(self.idx, end)])
        tgt, inp = self.g.get_batch(indices, self.ns)
        self.idx = end

        # print('DataGenerator: index={}, batch size={}, self.idx={}, self.data_size={}'.format(index,len(indices),self.idx,self.data_size))

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
