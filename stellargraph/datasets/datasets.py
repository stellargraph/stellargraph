# -*- coding: utf-8 -*-
#
# Copyright 2019-2020 Data61, CSIRO
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
`stellargraph.datasets` contains classes to download sample network datasets.

The default download path of ``stellargraph-datasets`` within the user's home directory can be changed by setting the
``STELLARGRAPH_DATASETS_PATH`` environment variable, and each dataset will be downloaded to a subdirectory within this path.
"""

from .dataset_loader import DatasetLoader
from ..core.graph import StellarGraph, StellarDiGraph
import itertools
import logging
import os
import pandas as pd
import numpy as np
from sklearn import preprocessing
from ..core.experimental import experimental


log = logging.getLogger(__name__)


def _load_cora_or_citeseer(
    dataset,
    directed,
    largest_connected_component_only,
    subject_as_feature,
    edge_weights,
    nodes_dtype,
):
    assert isinstance(dataset, (Cora, CiteSeer))

    if nodes_dtype is None:
        nodes_dtype = dataset._NODES_DTYPE

    dataset.download()

    # expected_files should be in this order
    cites, content = [dataset._resolve_path(name) for name in dataset.expected_files]

    feature_names = ["w_{}".format(ii) for ii in range(dataset._NUM_FEATURES)]
    subject = "subject"
    if subject_as_feature:
        feature_names.append(subject)
        column_names = feature_names
    else:
        column_names = feature_names + [subject]

    node_data = pd.read_csv(
        content, sep="\t", header=None, names=column_names, dtype={0: nodes_dtype}
    )

    edgelist = pd.read_csv(
        cites, sep="\t", header=None, names=["target", "source"], dtype=nodes_dtype
    )

    valid_source = node_data.index.get_indexer(edgelist.source) >= 0
    valid_target = node_data.index.get_indexer(edgelist.target) >= 0
    edgelist = edgelist[valid_source & valid_target]

    subjects = node_data[subject]

    cls = StellarDiGraph if directed else StellarGraph

    features = node_data[feature_names]
    if subject_as_feature:
        # one-hot encode the subjects
        features = pd.get_dummies(features, columns=[subject])

    graph = cls({"paper": features}, {"cites": edgelist})

    if edge_weights is not None:
        # A weighted graph means computing a second StellarGraph after using the unweighted one to
        # compute the weights.
        edgelist["weight"] = edge_weights(graph, subjects, edgelist)
        graph = cls({"paper": node_data[feature_names]}, {"cites": edgelist})

    if largest_connected_component_only:
        cc_ids = next(graph.connected_components())
        return graph.subgraph(cc_ids), subjects[cc_ids]

    return graph, subjects


class Cora(
    DatasetLoader,
    name="Cora",
    directory_name="cora",
    url="https://linqs-data.soe.ucsc.edu/public/lbc/cora.tgz",
    url_archive_format="gztar",
    expected_files=["cora.cites", "cora.content"],
    description="The Cora dataset consists of 2708 scientific publications classified into one of seven classes. "
    "The citation network consists of 5429 links. Each publication in the dataset is described by a 0/1-valued word vector "
    "indicating the absence/presence of the corresponding word from the dictionary. The dictionary consists of 1433 unique words.",
    source="https://linqs.soe.ucsc.edu/data",
):

    _NUM_FEATURES = 1433

    def load(
        self,
        directed=False,
        largest_connected_component_only=False,
        subject_as_feature=False,
        edge_weights=None,
        str_node_ids=False,
    ):
        """
        Load this dataset into a homogeneous graph that is directed or undirected, downloading it if
        required.

        The node feature vectors are included, and the edges are treated as directed or undirected
        depending on the ``directed`` parameter.

        Args:
            directed (bool): if True, return a directed graph, otherwise return an undirected one.
            largest_connected_component_only (bool): if True, returns only the largest connected
                component, not the whole graph.
            edge_weights (callable, optional): a function that accepts three parameters: an
                unweighted StellarGraph containing node features, a Pandas Series of the node
                labels, a Pandas DataFrame of the edges (with `source` and `target` columns). It
                should return a sequence of numbers (e.g. a 1D NumPy array) of edge weights for each
                edge in the DataFrame.
            str_node_ids (bool): if True, load the node IDs as strings, rather than integers.
            subject_as_feature (bool): if True, the subject for each paper (node) is included in the
                node features, one-hot encoded (the subjects are still also returned as a Series).

        Returns:
            A tuple where the first element is the :class:`StellarGraph` object (or
            :class:`StellarDiGraph`, if ``directed == True``) with the nodes, node feature vectors
            and edges, and the second element is a pandas Series of the node subject class labels.
        """
        nodes_dtype = str if str_node_ids else int

        return _load_cora_or_citeseer(
            self,
            directed,
            largest_connected_component_only,
            subject_as_feature,
            edge_weights,
            nodes_dtype,
        )


class CiteSeer(
    DatasetLoader,
    name="CiteSeer",
    directory_name="citeseer",
    url="https://linqs-data.soe.ucsc.edu/public/lbc/citeseer.tgz",
    url_archive_format="gztar",
    expected_files=["citeseer.cites", "citeseer.content"],
    description="The CiteSeer dataset consists of 3312 scientific publications classified into one of six classes. "
    "The citation network consists of 4732 links, although 17 of these have a source or target publication that isn't in the dataset and only 4715 are included in the graph. "
    "Each publication in the dataset is described by a 0/1-valued word vector "
    "indicating the absence/presence of the corresponding word from the dictionary. The dictionary consists of 3703 unique words.",
    source="https://linqs.soe.ucsc.edu/data",
):
    _NUM_FEATURES = 3703

    def load(self, largest_connected_component_only=False):
        """
        Load this dataset into an undirected homogeneous graph, downloading it if required.

        The node feature vectors are included.

        Args:
            largest_connected_component_only (bool): if True, returns only the largest connected
                component, not the whole graph.

        Returns:
            A tuple where the first element is the :class:`StellarGraph` object with the nodes, node
            feature vectors and edges, and the second element is a pandas Series of the node subject
            class labels.
        """
        # some node IDs are integers like 100157 and some are strings like
        # bradshaw97introduction. Pandas can get confused, so it's best to explicitly force them all
        # to be treated as strings.
        nodes_dtype = str

        return _load_cora_or_citeseer(
            self, False, largest_connected_component_only, False, None, nodes_dtype
        )


class PubMedDiabetes(
    DatasetLoader,
    name="PubMed Diabetes",
    directory_name="Pubmed-Diabetes",
    url="https://linqs-data.soe.ucsc.edu/public/Pubmed-Diabetes.tgz",
    url_archive_format="gztar",
    expected_files=[
        "data/Pubmed-Diabetes.DIRECTED.cites.tab",
        "data/Pubmed-Diabetes.GRAPH.pubmed.tab",
        "data/Pubmed-Diabetes.NODE.paper.tab",
    ],
    description="The PubMed Diabetes dataset consists of 19717 scientific publications from PubMed database "
    "pertaining to diabetes classified into one of three classes. The citation network consists of 44338 links. "
    "Each publication in the dataset is described by a TF/IDF weighted word vector from a dictionary which consists of 500 unique words.",
    source="https://linqs.soe.ucsc.edu/data",
    data_subdirectory_name="data",
):
    def load(self):
        """
        Load this graph into an undirected homogeneous graph, downloading it if required.

        Returns:
            A tuple where the first element is a :class:`StellarGraph` instance containing the graph
            data and features, and the second element is a pandas Series of node class labels.
        """
        self.download()

        directed, _graph, node = [self._resolve_path(f) for f in self.expected_files]
        edgelist = pd.read_csv(
            directed,
            sep="\t",
            skiprows=2,
            header=None,
            names=["id", "source", "pipe", "target"],
            usecols=["source", "target"],
        )
        edgelist.source = edgelist.source.str.lstrip("paper:").astype(int)
        edgelist.target = edgelist.target.str.lstrip("paper:").astype(int)

        def parse_feature(feat):
            name, value = feat.split("=")
            return name, float(value)

        def parse_line(line):
            pid, raw_label, *raw_features, _summary = line.split("\t")
            features = dict(parse_feature(feat) for feat in raw_features)
            features["pid"] = int(pid)
            features["label"] = int(parse_feature(raw_label)[1])
            return features

        with open(node) as fp:
            node_data = pd.DataFrame(
                parse_line(line) for line in itertools.islice(fp, 2, None)
            )

        node_data.fillna(0, inplace=True)
        node_data.set_index("pid", inplace=True)

        labels = node_data["label"]

        nodes = node_data.drop(columns="label")

        return StellarGraph({"paper": nodes}, {"cites": edgelist}), labels


class BlogCatalog3(
    DatasetLoader,
    name="BlogCatalog3",
    directory_name="BlogCatalog-dataset",
    url="https://ndownloader.figshare.com/files/22349970",
    url_archive_format="zip",
    expected_files=[
        "data/edges.csv",
        "data/group-edges.csv",
        "data/groups.csv",
        "data/nodes.csv",
    ],
    description="This dataset is crawled from a social blog directory website BlogCatalog "
    "http://www.blogcatalog.com and contains the friendship network crawled and group memberships.",
    source="https://figshare.com/articles/BlogCatalog_dataset/11923611",
    data_subdirectory_name="data",
):
    def load(self):
        """
        Load this dataset into an undirected heterogeneous graph, downloading it if required.

        The graph has two types of nodes, 'user' and 'group', and two types of edges, 'friend' and 'belongs'.
        The 'friend' edges connect two 'user' nodes and the 'belongs' edges connects 'user' and 'group' nodes.

        The node and edge types are not included in the dataset that is a collection of node and group ids along with
        the list of edges in the graph.

        Important note about the node IDs: The dataset uses integers for node ids. However, the integers from 1 to 39 are
        used as IDs for both users and groups. This would cause a confusion when constructing the graph object.
        As a result, we convert all IDs to string and append the character 'u' to the integer ID for user nodes and the
        character 'g' to the integer ID for group nodes.

        Returns:
            A :class:`StellarGraph` object.
        """
        self.download()

        # load the raw data
        edges, group_edges, groups, nodes = [
            self._resolve_path(name) for name in self.expected_files
        ]

        user_node_ids = pd.read_csv(nodes, header=None)
        group_ids = pd.read_csv(groups, header=None)
        edges = pd.read_csv(edges, header=None, names=["source", "target"])
        group_edges = pd.read_csv(group_edges, header=None, names=["source", "target"])

        # The dataset uses integers for node ids. However, the integers from 1 to 39 are used as IDs
        # for both users and groups. This is disambiguated by converting everything to strings and
        # prepending u to user IDs, and g to group IDs.
        def u(users):
            return "u" + users.astype(str)

        def g(groups):
            return "g" + groups.astype(str)

        # nodes:
        user_node_ids = u(user_node_ids)
        group_ids = g(group_ids)

        # node IDs in each edge:
        edges = u(edges)
        group_edges["source"] = u(group_edges["source"])
        group_edges["target"] = g(group_edges["target"])

        # arrange the DataFrame indices appropriately: nodes use their node IDs, which have
        # been made distinct above, and the group edges have IDs after the other edges
        user_node_ids.set_index(0, inplace=True)
        group_ids.set_index(0, inplace=True)

        start = len(edges)
        group_edges.index = range(start, start + len(group_edges))

        return StellarGraph(
            nodes={"user": user_node_ids, "group": group_ids},
            edges={"friend": edges, "belongs": group_edges},
        )


class MovieLens(
    DatasetLoader,
    name="MovieLens",
    directory_name="ml-100k",
    url="http://files.grouplens.org/datasets/movielens/ml-100k.zip",
    url_archive_format="zip",
    expected_files=["u.data", "u.user", "u.item", "u.genre", "u.occupation"],
    description="The MovieLens 100K dataset contains 100,000 ratings from 943 users on 1682 movies.",
    source="https://grouplens.org/datasets/movielens/100k/",
):
    def load(self):
        """
        Load this dataset into an undirected heterogeneous graph, downloading it if required.

        The graph has two types of nodes (``user`` and ``movie``) and one type of edge (``rating``).

        The dataset includes some node features on both users and movies: on users, they consist of
        categorical features (``gender`` and ``job``) which are one-hot encoded into binary
        features, and an ``age`` feature that is scaled to have mean = 0 and standard deviation = 1.

        Returns:
            A tuple where the first element is a :class:`StellarGraph` instance containing the graph
            data and features, and the second element is a pandas DataFrame of edges, with columns
            ``user_id``, ``movie_id`` and ``rating`` (a label from 1 to 5).
        """
        self.download()

        ratings, users, movies, *_ = [
            self._resolve_path(path) for path in self.expected_files
        ]

        edges = pd.read_csv(
            ratings,
            sep="\t",
            header=None,
            names=["user_id", "movie_id", "rating", "timestamp"],
            usecols=["user_id", "movie_id", "rating"],
        )

        users = pd.read_csv(
            users,
            sep="|",
            header=None,
            names=["user_id", "age", "gender", "job", "zipcode"],
            usecols=["user_id", "age", "gender", "job"],
        )

        movie_columns = [
            "movie_id",
            "title",
            "release_date",
            "video_release_date",
            "imdb_url",
            # features from here:
            "unknown",
            "action",
            "adventure",
            "animation",
            "childrens",
            "comedy",
            "crime",
            "documentary",
            "drama",
            "fantasy",
            "film_noir",
            "horror",
            "musical",
            "mystery",
            "romance",
            "sci_fi",
            "thriller",
            "war",
            "western",
        ]
        movies = pd.read_csv(
            movies,
            sep="|",
            header=None,
            names=movie_columns,
            usecols=["movie_id"] + movie_columns[5:],
        )

        # manage the IDs
        def u(users):
            return "u_" + users.astype(str)

        def m(movies):
            return "m_" + movies.astype(str)

        users_ids = u(users["user_id"])

        movies["movie_id"] = m(movies["movie_id"])
        movies.set_index("movie_id", inplace=True)

        edges["user_id"] = u(edges["user_id"])
        edges["movie_id"] = m(edges["movie_id"])

        # convert categorical user features to numeric, and normalize age
        feature_encoding = preprocessing.OneHotEncoder(sparse=False)
        onehot = feature_encoding.fit_transform(users[["gender", "job"]])
        scaled_age = preprocessing.scale(users["age"])
        encoded_users = pd.DataFrame(onehot, index=users_ids).assign(
            scaled_age=scaled_age
        )

        g = StellarGraph(
            {"user": encoded_users, "movie": movies},
            {"rating": edges[["user_id", "movie_id"]]},
            source_column="user_id",
            target_column="movie_id",
        )
        return g, edges


class AIFB(
    DatasetLoader,
    name="AIFB",
    directory_name="aifb",
    url="https://ndownloader.figshare.com/files/1118822",
    url_archive_format=None,
    expected_files=["aifbfixed_complete.n3"],
    description="The AIFB dataset describes the AIFB research institute in terms of its staff, research group, and publications. "
    'First used for machine learning with RDF in Bloehdorn, Stephan and Sure, York, "Kernel Methods for Mining Instance Data in Ontologies", '
    "The Semantic Web (2008), http://dx.doi.org/10.1007/978-3-540-76298-0_5. "
    "It contains ~8k entities, ~29k edges, and 45 different relationships or edge types. In (Bloehdorn et al 2007) the dataset "
    "was first used to predict the affiliation (i.e., research group) for people in the dataset. The dataset contains 178 "
    "members of a research group with 5 different research groups. The goal is to predict which research group a researcher belongs to.",
    source="https://figshare.com/articles/AIFB_DataSet/745364",
):
    _AFFILIATION_TYPE = "http://swrc.ontoware.org/ontology#affiliation"
    _EMPLOYS_TYPE = "http://swrc.ontoware.org/ontology#employs"

    def load(self):
        """
        Loads the dataset into a directed heterogeneous graph.

        The nodes features are the node's position after being one-hot encoded; for example, the
        first node has features ``[1, 0, 0, ...]``, the second has ``[0, 1, 0, ...]``.

        This requires the ``rdflib`` library to be installed.

        Returns:
            A tuple where the first element is a graph containing all edges except for those with
            type ``affiliation`` and ``employs`` (the inverse of ``affiliation``), and the second
            element is a DataFrame containing the one-hot encoded affiliation of the 178 nodes that
            have an affiliation.
        """
        try:
            import rdflib
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                f"{e.msg}. Loading the AIFB dataset requires the 'rdflib' module; please install it",
                name=e.name,
                path=e.path,
            ) from None

        self.download()

        graph = rdflib.Graph()
        graph.parse(self._resolve_path(self.expected_files[0]), format="n3")

        triples = pd.DataFrame(
            ((s.n3(), str(p), o.n3()) for s, p, o in graph),
            columns=["source", "label", "target"],
        )

        all_nodes = pd.concat([triples.source, triples.target])
        nodes = pd.DataFrame(index=pd.unique(all_nodes))
        nodes_onehot_features = pd.get_dummies(nodes.index).set_index(nodes.index)

        edges = {
            edge_type: df.drop(columns="label")
            for edge_type, df in triples.groupby("label")
        }

        affiliation = edges.pop(self._AFFILIATION_TYPE)
        # 'employs' is the inverse relation, so it should be removed
        del edges[self._EMPLOYS_TYPE]

        onehot_affiliation = pd.get_dummies(affiliation.set_index("source")["target"])

        graph = StellarDiGraph(nodes_onehot_features, edges)

        return graph, onehot_affiliation


def _load_graph_kernel_dataset(dataset):

    dataset.download()

    def _load_from_txt_file(filename, names=None, dtype=None, index_increment=None):
        df = pd.read_csv(
            dataset._resolve_path(filename=f"{dataset.name}_{filename}.txt"),
            header=None,
            index_col=False,
            dtype=dtype,
            names=names,
        )
        # We optional increment the index by 1 because indexing, e.g. node IDs, for this dataset starts
        # at 1 whereas the Pandas DataFrame implicit index starts at 0 potentially causing confusion selecting
        # rows later on.
        if index_increment:
            df.index = df.index + index_increment
        return df

    # edge information:
    df_graph = _load_from_txt_file(filename="A", names=["source", "target"])

    if dataset._edge_labels_as_weights:
        # there's some edge labels, that can be used as edge weights
        df_edge_labels = _load_from_txt_file(
            filename="edge_labels", names=["weight"], dtype=int
        )
        df_graph = pd.concat([df_graph, df_edge_labels], axis=1)

    # node information:
    df_graph_ids = _load_from_txt_file(
        filename="graph_indicator", names=["graph_id"], index_increment=1
    )

    df_node_labels = _load_from_txt_file(
        filename="node_labels", dtype="category", index_increment=1
    )
    # One-hot encode the node labels because these are used as node features in graph classification
    # tasks.
    df_node_features = pd.get_dummies(df_node_labels)

    if dataset._node_attributes:
        # there's some actual node attributes
        df_node_attributes = _load_from_txt_file(
            filename="node_attributes", dtype=np.float32, index_increment=1
        )

        df_node_features = pd.concat([df_node_features, df_node_attributes], axis=1)

    # graph information:
    df_graph_labels = _load_from_txt_file(
        filename="graph_labels", dtype="category", names=["label"], index_increment=1
    )

    # split the data into each of the graphs, based on the nodes in each one
    def graph_for_nodes(nodes):
        # each graph is disconnected, so the source is enough to identify the graph for an edge
        edges = df_graph[df_graph["source"].isin(nodes.index)]
        return StellarGraph(nodes, edges)

    groups = df_node_features.groupby(df_graph_ids["graph_id"])
    graphs = [graph_for_nodes(nodes) for _, nodes in groups]

    return graphs, df_graph_labels["label"]


class MUTAG(
    DatasetLoader,
    name="MUTAG",
    directory_name="MUTAG",
    url="https://ls11-www.cs.tu-dortmund.de/people/morris/graphkerneldatasets/MUTAG.zip",
    url_archive_format="zip",
    expected_files=[
        "MUTAG_A.txt",
        "MUTAG_graph_indicator.txt",
        "MUTAG_node_labels.txt",
        "MUTAG_edge_labels.txt",
        "MUTAG_graph_labels.txt",
        "README.txt",
    ],
    description="Each graph represents a chemical compound and graph labels represent 'their mutagenic effect on a specific gram negative bacterium.'"
    "The dataset includes 188 graphs with 18 nodes and 20 edges on average for each graph. Graph nodes have 7 labels and each graph is labelled as belonging to 1 of 2 classes.",
    source="https://ls11-www.cs.tu-dortmund.de/staff/morris/graphkerneldatasets",
):
    _edge_labels_as_weights = False
    _node_attributes = False

    def load(self):
        """
        Load this dataset into a list of StellarGraph objects with corresponding labels, downloading it if required.

        Note: Edges in MUTAG are labelled as one of 4 values: aromatic, single, double, and triple indicated by integers
        0, 1, 2, 3 respectively. The edge labels are included in the  :class:`StellarGraph` objects as edge weights in
        integer representation.

        Returns:
            A tuple that is a list of :class:`StellarGraph` objects and a Pandas Series of labels one for each graph.
        """
        return _load_graph_kernel_dataset(self)


class PROTEINS(
    DatasetLoader,
    name="PROTEINS",
    directory_name="PROTEINS",
    url="https://ls11-www.cs.tu-dortmund.de/people/morris/graphkerneldatasets/PROTEINS.zip",
    url_archive_format="zip",
    expected_files=[
        "PROTEINS_A.txt",
        "PROTEINS_graph_indicator.txt",
        "PROTEINS_node_labels.txt",
        "PROTEINS_node_attributes.txt",
        "PROTEINS_graph_labels.txt",
        "README.txt",
    ],
    description="Each graph represents a protein and graph labels represent whether they are are enzymes or non-enzymes. "
    "The dataset includes 1113 graphs with 39 nodes and 73 edges on average for each graph. "
    "Graph nodes have 4 attributes (including a one-hot encoding of their label), and each graph is labelled as belonging to 1 of 2 classes.",
    source="https://ls11-www.cs.tu-dortmund.de/staff/morris/graphkerneldatasets",
):

    _edge_labels_as_weights = False
    _node_attributes = True

    def load(self):
        """
        Load this dataset into a list of StellarGraph objects with corresponding labels, downloading it if required.

        Returns:
            A tuple that is a list of :class:`StellarGraph` objects and a Pandas Series of labels one for each graph.
        """
        return _load_graph_kernel_dataset(self)


def _load_tsv_knowledge_graph(dataset):
    dataset.download()

    train, test, valid = [
        pd.read_csv(
            dataset._resolve_path(name), sep="\t", names=["source", "label", "target"]
        )
        for name in dataset.expected_files
    ]

    all_data = pd.concat([train, test, valid], ignore_index=True)

    return StellarDiGraph(edges=all_data, edge_type_column="label"), train, test, valid


class WN18(
    DatasetLoader,
    name="WN18",
    directory_name="wordnet-mlj12",
    url="https://ndownloader.figshare.com/files/21768732",
    url_archive_format="zip",
    expected_files=[
        "wordnet-mlj12-train.txt",
        "wordnet-mlj12-test.txt",
        "wordnet-mlj12-valid.txt",
    ],
    description="The WN18 dataset consists of triplets from WordNet 3.0 (http://wordnet.princeton.edu). There are "
    "40,943 synsets and 18 relation types among them. The training set contains 141442 triplets, the "
    "validation set 5000 and the test set 5000. "
    "Antoine Bordes, Xavier Glorot, Jason Weston and Yoshua Bengio “A Semantic Matching Energy Function for Learning with Multi-relational Data” (2014).\n\n"
    "Note: this dataset contains many inverse relations, and so should only be used to compare against published results. Prefer WN18RR. See: "
    "Kristina Toutanova and Danqi Chen “Observed versus latent features for knowledge base and text inference” (2015), and "
    "Dettmers, Tim, Pasquale Minervini, Pontus Stenetorp and Sebastian Riedel “Convolutional 2D Knowledge Graph Embeddings” (2017).",
    source="https://everest.hds.utc.fr/doku.php?id=en:transe",
):
    def load(self):
        """
        Load this data into a directed heterogeneous graph.

        Returns:
            A tuple ``(graph, train, test, validation)`` where ``graph`` is a
            :class:`StellarDiGraph` containing all the data, and the remaining three elements are
            DataFrames of triplets, with columns ``source`` & ``target`` (synsets) and ``label``
            (the relation type). The three DataFrames together make up the edges included in
            ``graph``.
        """
        return _load_tsv_knowledge_graph(self)


class WN18RR(
    DatasetLoader,
    name="WN18RR",
    directory_name="WN18RR",
    url="https://ndownloader.figshare.com/files/21844185",
    url_archive_format="zip",
    expected_files=["train.txt", "test.txt", "valid.txt"],
    description="The WN18RR dataset consists of triplets from WordNet 3.0 (http://wordnet.princeton.edu). There are "
    "40,943 synsets and 11 relation types among them. The training set contains 86835 triplets, the validation set 3034 and the test set 3134. "
    "It is a reduced version of WN18 where inverse relations have been removed."
    "Tim Dettmers, Pasquale Minervini, Pontus Stenetorp and Sebastian Riedel “Convolutional 2D Knowledge Graph Embeddings” (2017).",
    source="https://github.com/TimDettmers/ConvE",
):
    def load(self):
        """
        Load this data into a directed heterogeneous graph.

        Returns:
            A tuple ``(graph, train, test, validation)`` where ``graph`` is a
            :class:`StellarDiGraph` containing all the data, and the remaining three elements are
            DataFrames of triplets, with columns ``source`` & ``target`` (synsets) and ``label``
            (the relation type). The three DataFrames together make up the edges included in
            ``graph``.
        """
        return _load_tsv_knowledge_graph(self)


class FB15k(
    DatasetLoader,
    name="FB15k",
    directory_name="FB15k",
    url="https://ndownloader.figshare.com/files/21768729",
    url_archive_format="zip",
    expected_files=[
        "freebase_mtr100_mte100-train.txt",
        "freebase_mtr100_mte100-test.txt",
        "freebase_mtr100_mte100-valid.txt",
    ],
    description="This FREEBASE FB15k DATA consists of a collection of triplets (synset, relation_type, triplet)"
    "extracted from Freebase (http://www.freebase.com). There are 14,951 mids and 1,345 relation types among them. "
    "The training set contains 483142 triplets, the validation set 50000 and the test set 59071. "
    "Antoine Bordes, Nicolas Usunier, Alberto Garcia-Durán, Jason Weston and Oksana Yakhnenko “Translating Embeddings for Modeling Multi-relational Data” (2013).\n\n"
    "Note: this dataset contains many inverse relations, and so should only be used to compare against published results. Prefer FB15k_237. See: "
    "Kristina Toutanova and Danqi Chen “Observed versus latent features for knowledge base and text inference” (2015), and "
    "Dettmers, Tim, Pasquale Minervini, Pontus Stenetorp and Sebastian Riedel “Convolutional 2D Knowledge Graph Embeddings” (2017).",
    source="https://everest.hds.utc.fr/doku.php?id=en:transe",
):
    def load(self):
        """
        Load this data into a directed heterogeneous graph.

        Returns:
            A tuple ``(graph, train, test, validation)`` where ``graph`` is a
            :class:`StellarDiGraph` containing all the data, and the remaining three elements are
            DataFrames of triplets, with columns ``source`` & ``target`` (synsets) and ``label``
            (the relation type). The three DataFrames together make up the edges included in
            ``graph``.
        """
        return _load_tsv_knowledge_graph(self)


class FB15k_237(
    DatasetLoader,
    name="FB15k-237",
    directory_name="FB15k-237",
    url="https://ndownloader.figshare.com/files/21844209",
    url_archive_format="zip",
    expected_files=["train.txt", "test.txt", "valid.txt"],
    description="This FREEBASE FB15k DATA consists of a collection of triplets (synset, relation_type, triplet)"
    "extracted from Freebase (http://www.freebase.com). There are 14541 mids and 237 relation types among them. "
    "The training set contains 272115 triplets, the validation set 17535 and the test set 20466."
    "It is a reduced version of FB15k where inverse relations have been removed."
    "Kristina Toutanova and Danqi Chen “Observed versus latent features for knowledge base and text inference” (2015).",
    source="https://github.com/TimDettmers/ConvE",
):
    def load(self):
        """
        Load this data into a directed heterogeneous graph.

        Returns:
            A tuple ``(graph, train, test, validation)`` where ``graph`` is a
            :class:`StellarDiGraph` containing all the data, and the remaining three elements are
            DataFrames of triplets, with columns ``source`` & ``target`` (synsets) and ``label``
            (the relation type). The three DataFrames together make up the edges included in
            ``graph``.
        """
        return _load_tsv_knowledge_graph(self)


class IAEnronEmployees(
    DatasetLoader,
    name="ia-enron-employees",
    directory_name="ia-enron-employees",
    url="http://nrvis.com/download/data/dynamic/ia-enron-employees.zip",
    url_archive_format="zip",
    url_archive_contains_directory=False,
    expected_files=["ia-enron-employees.edges", "readme.html"],
    description="A dataset of edges that represent emails sent from one employee to another."
    "There are 50572 edges, and each of them contains timestamp information. "
    "Edges refer to 151 unique node IDs in total."
    "Ryan A. Rossi and Nesreen K. Ahmed “The Network Data Repository with Interactive Graph Analytics and Visualization” (2015)",
    source="http://networkrepository.com/ia-enron-employees.php",
):
    def load(self):
        """
        Load this data into a set of nodes and edges

        Returns:
            A tuple ``(graph, edges)``

            ``graph`` is a :class:`StellarGraph` containing all the data. Timestamp information on
            edges are encoded as edge weights.

            ``edges`` are the original edges from the dataset which are sorted in ascending
            order of time - these can be used to create train/test splits based on time values.

            Node IDs in the returned data structures are all converted to strings to allow for
            compatibility with with ``gensim``'s ``Word2Vec`` model.
        """
        self.download()

        edges_path = self._resolve_path("ia-enron-employees.edges")
        edges = pd.read_csv(
            edges_path,
            sep=" ",
            header=None,
            names=["source", "target", "x", "time"],
            usecols=["source", "target", "time"],
        )
        edges[["source", "target"]] = edges[["source", "target"]].astype(str)

        nodes = pd.DataFrame(
            index=np.unique(
                pd.concat([edges["source"], edges["target"]], ignore_index=True)
            )
        )

        return StellarGraph(nodes=nodes, edges=edges, edge_weight_column="time"), edges


@experimental(reason="tests and documentation missing", issues=[1303])
class METR_LA(
    DatasetLoader,
    name="METR-LA",
    directory_name="T-GCN-8427128f04157e6fd0b239a8734a468d923cd0c9",
    url="https://github.com/lehaifeng/T-GCN/archive/8427128f04157e6fd0b239a8734a468d923cd0c9.zip",
    url_archive_format="zip",
    expected_files=["data/los_speed.csv", "data/los_adj.csv"],
    description="This traffic dataset contains traffic information collected from loop detectors in the highway of Los Angeles County (Jagadish et al., 2014).",
    source="https://github.com/lehaifeng/T-GCN/tree/master/data",
):
    def load(self):
        self.download()
        los_adj = pd.read_csv(self._resolve_path(self.expected_files[1]), header=None)
        adj = np.mat(los_adj)
        los_tf = pd.read_csv(self._resolve_path(self.expected_files[0])).transpose()
        return los_tf, adj

    def train_test_split(self, data, train_portion):
        time_len = data.shape[0]
        train_size = int(time_len * train_portion)
        train_data = np.array(data[:train_size])
        test_data = np.array(data[train_size:])
        return train_data, test_data

    def scale_data(self, train_data, test_data):
        max_speed = train_data.max()
        min_speed = train_data.min()
        train_scaled = (train_data - min_speed) / (max_speed - min_speed)
        test_scaled = (test_data - min_speed) / (max_speed - min_speed)
        return train_scaled, test_scaled

    def sequence_data_preparation(self, seq_len, pre_len, train_data, test_data):
        trainX, trainY, testX, testY = [], [], [], []

        for i in range(len(train_data) - int(seq_len + pre_len - 1)):
            a = train_data[
                i : i + seq_len + pre_len,
            ]
            trainX.append(a[:seq_len])
            trainY.append(a[-1])

        for i in range(len(test_data) - int(seq_len + pre_len - 1)):
            b = test_data[
                i : i + seq_len + pre_len,
            ]
            testX.append(
                b[:seq_len,]
            )
            testY.append(b[-1])

        trainX = np.array(trainX)
        trainY = np.array(trainY)
        testX = np.array(testX)
        testY = np.array(testY)

        return trainX, trainY, testX, testY
