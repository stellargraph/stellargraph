import numpy as np
import networkx as nx
from stellargraph.layer.rgcn import RelationalGraphConvolution, RGCN
from stellargraph.mapper.node_mappers import RelationalFullBatchNodeGenerator
import pytest
from scipy import sparse as sps
from stellargraph.core.utils import normalize_adj
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input, Lambda
from stellargraph import StellarDiGraph
from stellargraph.layer.misc import SqueezedSparseConversion
import pandas as pd


def create_graph_features():
    G = nx.MultiDiGraph()
    G.add_nodes_from(["a", "b", "c"])
    G.add_edges_from([("a", "b", "r1"), ("b", "c", "r1"), ("a", "c", "r2")])
    return G, np.array([[1, 1], [1, 0], [0, 1]])


def test_RelationalGraphConvolution_config():
    rgcn_layer = RelationalGraphConvolution(units=16, num_relationships=5)
    conf = rgcn_layer.get_config()

    assert conf["units"] == 16
    assert conf["activation"] == "linear"
    assert conf["num_bases"] == 0
    assert conf["num_relationships"] == 5
    assert conf["use_bias"] == True
    assert conf["kernel_initializer"]["class_name"] == "GlorotUniform"
    assert conf["bias_initializer"]["class_name"] == "Zeros"
    assert conf["kernel_regularizer"] == None
    assert conf["bias_regularizer"] == None
    assert conf["kernel_constraint"] == None
    assert conf["bias_constraint"] == None


def test_RelationalGraphConvolution_init():
    rgcn_layer = RelationalGraphConvolution(
        units=16, num_relationships=5, num_bases=0, activation="relu"
    )

    assert rgcn_layer.units == 16
    assert rgcn_layer.use_bias == True
    assert rgcn_layer.num_bases == 0
    assert rgcn_layer.get_config()["activation"] == "relu"

    rgcn_layer = RelationalGraphConvolution(
        units=16, num_relationships=5, num_bases=10, activation="relu"
    )

    assert rgcn_layer.units == 16
    assert rgcn_layer.use_bias == True
    assert rgcn_layer.num_bases == 10
    assert rgcn_layer.get_config()["activation"] == "relu"


def test_RelationalGraphConvolution_sparse():
    G, features = create_graph_features()
    edge_types = sorted(set(e[-1] for e in G.edges))
    n_edge_types = len(edge_types)

    # We need to specify the batch shape as one for the GraphConvolutional logic to work
    n_nodes = features.shape[0]
    n_feat = features.shape[1]

    # Inputs for features & target indices
    x_t = Input(batch_shape=(1, n_nodes, n_feat))
    out_indices_t = Input(batch_shape=(1, None), dtype="int32")

    # Create inputs for sparse or dense matrices

    # Placeholders for the sparse adjacency matrix
    As_indices = [
        Input(batch_shape=(1, None, 2), dtype="int64") for i in range(n_edge_types)
    ]
    As_values = [Input(batch_shape=(1, None)) for i in range(n_edge_types)]
    A_placeholders = As_indices + As_values

    # Test with final_layer=False
    Ainput = [
        SqueezedSparseConversion(shape=(n_nodes, n_nodes), dtype=As_values[i].dtype)(
            [As_indices[i], As_values[i]]
        )
        for i in range(n_edge_types)
    ]

    x_inp_model = [x_t, out_indices_t] + A_placeholders
    x_inp_conv = [x_t, out_indices_t] + Ainput

    out = RelationalGraphConvolution(
        2, num_relationships=n_edge_types, final_layer=False
    )(x_inp_conv)

    # Note we add a batch dimension of 1 to model inputs
    As = []
    node_list = list(G.nodes)
    node_index = dict(zip(node_list, range(len(node_list))))
    for edge_type in edge_types:
        col_index = [node_index[n1] for n1, n2, etype in G.edges if etype == edge_type]
        row_index = [node_index[n2] for n1, n2, etype in G.edges if etype == edge_type]
        data = np.ones(len(col_index), np.float64)

        A = sps.coo_matrix(
            (data, (row_index, col_index)), shape=(len(node_list), len(node_list))
        )

        A = normalize_adj(A, symmetric=False)
        A = A.tocoo()
        As.append(A)

    A_indices = [
        np.expand_dims(np.hstack((A.row[:, None], A.col[:, None])), 0) for A in As
    ]
    A_values = [np.expand_dims(A.data, 0) for A in As]

    out_indices = np.array([[0, 1]], dtype="int32")
    x = features[None, :, :]

    model = keras.Model(inputs=x_inp_model, outputs=out)
    preds = model.predict([x, out_indices] + A_indices + A_values, batch_size=1)
    assert preds.shape == (1, 3, 2)

    # Now try with final_layer=True
    out = RelationalGraphConvolution(
        2, num_relationships=n_edge_types, final_layer=True
    )(x_inp_conv)
    model = keras.Model(inputs=x_inp_model, outputs=out)
    preds = model.predict([x, out_indices] + A_indices + A_values, batch_size=1)
    assert preds.shape == (1, 2, 2)


def test_RelationalGraphConvolution_dense():
    G, features = create_graph_features()
    edge_types = sorted(set(e[-1] for e in G.edges))
    n_edge_types = len(edge_types)

    # We need to specify the batch shape as one for the GraphConvolutional logic to work
    n_nodes = features.shape[0]
    n_feat = features.shape[1]

    # Inputs for features & target indices
    x_t = Input(batch_shape=(1, n_nodes, n_feat))
    out_indices_t = Input(batch_shape=(1, None), dtype="int32")

    # Create inputs for sparse or dense matrices

    # Placeholders for the sparse adjacency matrix
    A_placeholders = [
        Input(batch_shape=(1, n_nodes, n_nodes)) for _ in range(n_edge_types)
    ]

    A_in = [Lambda(lambda A: K.squeeze(A, 0))(A_p) for A_p in A_placeholders]

    x_inp_model = [x_t, out_indices_t] + A_placeholders
    x_inp_conv = [x_t, out_indices_t] + A_in

    out = RelationalGraphConvolution(
        2, num_relationships=n_edge_types, final_layer=False
    )(x_inp_conv)

    # Note we add a batch dimension of 1 to model inputs
    As = []
    node_list = list(G.nodes)
    node_index = dict(zip(node_list, range(len(node_list))))
    for edge_type in edge_types:
        col_index = [node_index[n1] for n1, n2, etype in G.edges if etype == edge_type]
        row_index = [node_index[n2] for n1, n2, etype in G.edges if etype == edge_type]
        data = np.ones(len(col_index), np.float64)

        A = sps.coo_matrix(
            (data, (row_index, col_index)), shape=(len(node_list), len(node_list))
        )

        A = normalize_adj(A, symmetric=False)
        A = A.todense()[None, :, :]
        As.append(A)

    out_indices = np.array([[0, 1]], dtype="int32")
    x = features[None, :, :]

    model = keras.Model(inputs=x_inp_model, outputs=out)
    preds = model.predict([x, out_indices] + As, batch_size=1)
    assert preds.shape == (1, 3, 2)

    # Now try with final_layer=True
    out = RelationalGraphConvolution(
        2, num_relationships=n_edge_types, final_layer=True
    )(x_inp_conv)
    model = keras.Model(inputs=x_inp_model, outputs=out)
    preds = model.predict([x, out_indices] + As, batch_size=1)
    assert preds.shape == (1, 2, 2)


def test_RGCN_init():
    G, features = create_graph_features()
    nodes = G.nodes()
    node_features = pd.DataFrame.from_dict(
        {n: f for n, f in zip(nodes, features)}, orient="index"
    )
    G = StellarDiGraph(G, node_type_name="node", node_features=node_features)

    generator = RelationalFullBatchNodeGenerator(G)
    rgcnModel = RGCN([2], generator, num_bases=10, activations=["relu"], dropout=0.5)

    assert rgcnModel.layer_sizes == [2]
    assert rgcnModel.activations == ["relu"]
    assert rgcnModel.dropout == 0.5
    assert rgcnModel.num_bases == 10


def test_RGCN_apply_sparse():
    G, features = create_graph_features()

    As = []
    edge_types = sorted(set(e[-1] for e in G.edges))
    node_list = list(G.nodes)
    node_index = dict(zip(node_list, range(len(node_list))))
    for edge_type in edge_types:
        col_index = [node_index[n1] for n1, n2, etype in G.edges if etype == edge_type]
        row_index = [node_index[n2] for n1, n2, etype in G.edges if etype == edge_type]
        data = np.ones(len(col_index), np.float64)

        A = sps.coo_matrix(
            (data, (row_index, col_index)), shape=(len(node_list), len(node_list))
        )

        A = normalize_adj(A, symmetric=False)
        A = A.tocoo()
        As.append(A)

    A_indices = [
        np.expand_dims(np.hstack((A.row[:, None], A.col[:, None])), 0) for A in As
    ]
    A_values = [np.expand_dims(A.data, 0) for A in As]

    nodes = G.nodes()
    node_features = pd.DataFrame.from_dict(
        {n: f for n, f in zip(nodes, features)}, orient="index"
    )
    G = StellarDiGraph(G, node_features=node_features)

    generator = RelationalFullBatchNodeGenerator(G, sparse=True)
    rgcnModel = RGCN([2], generator, num_bases=10, activations=["relu"], dropout=0.5)

    x_in, x_out = rgcnModel.node_model()
    model = keras.Model(inputs=x_in, outputs=x_out)

    # Check fit method
    out_indices = np.array([[0, 1]], dtype="int32")
    preds_1 = model.predict([features[None, :, :], out_indices] + A_indices + A_values)
    assert preds_1.shape == (1, 2, 2)

    # Check fit_generator method
    preds_2 = model.predict_generator(generator.flow(["a", "b"]))
    assert preds_2.shape == (1, 2, 2)

    assert preds_1 == pytest.approx(preds_2)


def test_RGCN_apply_dense():
    G, features = create_graph_features()

    As = []
    edge_types = sorted(set(e[-1] for e in G.edges))
    node_list = list(G.nodes)
    node_index = dict(zip(node_list, range(len(node_list))))
    for edge_type in edge_types:
        col_index = [node_index[n1] for n1, n2, etype in G.edges if etype == edge_type]
        row_index = [node_index[n2] for n1, n2, etype in G.edges if etype == edge_type]
        data = np.ones(len(col_index), np.float64)

        A = sps.coo_matrix(
            (data, (row_index, col_index)), shape=(len(node_list), len(node_list))
        )

        A = normalize_adj(A, symmetric=False)
        A = A.todense()[None, :, :]
        As.append(A)

    nodes = G.nodes()
    node_features = pd.DataFrame.from_dict(
        {n: f for n, f in zip(nodes, features)}, orient="index"
    )
    G = StellarDiGraph(G, node_features=node_features)

    generator = RelationalFullBatchNodeGenerator(G, sparse=False)
    rgcnModel = RGCN([2], generator, num_bases=10, activations=["relu"], dropout=0.5)

    x_in, x_out = rgcnModel.node_model()
    model = keras.Model(inputs=x_in, outputs=x_out)

    # Check fit method
    out_indices = np.array([[0, 1]], dtype="int32")
    preds_1 = model.predict([features[None, :, :], out_indices] + As)
    assert preds_1.shape == (1, 2, 2)

    # Check fit_generator method
    preds_2 = model.predict_generator(generator.flow(["a", "b"]))
    assert preds_2.shape == (1, 2, 2)

    assert preds_1 == pytest.approx(preds_2)


def test_RelationalGraphConvolution_edge_cases():


    try:
        rgcn_layer = RelationalGraphConvolution(
            units=16, num_relationships=5, num_bases=0.5, activation="relu"
        )
    except TypeError as e:
        error = e
    assert str(error) == "num_bases should be an int"

    rgcn_layer = RelationalGraphConvolution(
        units=16, num_relationships=5, num_bases=-1, activation="relu"
    )
    rgcn_layer.build(input_shapes=[(1,)])
    assert rgcn_layer.bases is None
