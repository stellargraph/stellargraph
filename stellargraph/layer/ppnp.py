from keras.layers import Dense, Lambda, Softmax, Dropout, Input
import keras.backend as K
import numpy as np

from ..mapper import FullBatchNodeGenerator
from .preprocessing_layer import GraphPreProcessingLayer


class PPNP:
    """
    Implementation of Personalized Propagation of Neural Predictions (PPNP)
    as in https://openreview.net/pdf?id=H1gL-2A9Ym.

    The model minimally requires specification of the fully connected layer sizes as a list of ints
    corresponding to the feature dimensions for each hidden layer,
    activation functions for each hidden layers, and a generator object.

    To use this class as a Keras model, the features and pre-processed adjacency matrix
    should be supplied using the :class:`FullBatchNodeGenerator` class. To have the appropriate
    pre-processing the generator object should be instantiated as follows::

        generator = FullBatchNodeGenerator(G, method="gcn")

    Notes:
      - The inputs are tensors with a batch dimension of 1. These are provided by the \
        :class:`FullBatchNodeGenerator` object.

      - This assumes that the normalized Lapalacian matrix is provided as input to
        Keras methods. When using the :class:`FullBatchNodeGenerator` specify the
        ``method='gcn'`` argument to do this pre-processing.

      - The nodes provided to the :class:`FullBatchNodeGenerator.flow` method are
        used by the final layer to select the predictions for those nodes in order.
        However, the intermediate layers before the final layer order the nodes
        in the same way as the adjacency matrix.

      - The size of the final fully connected layer must be equal to the number of classes

    Args:
        layer_sizes (list of int): list of output sizes of fully connected layers in the stack
        activations (list of str): list of activations applied to each fully connected layer's output
        generator (FullBatchNodeGenerator): an instance of FullBatchNodeGenerator class constructed on the graph of interest
        bias (bool): toggles an optional bias in fully connected layers
        dropout (float): dropout rate applied to input features of each layer
        kernel_regularizer (str): normalization applied to the kernels of fully connetcted layers
        transport_probability: "probability" of returning to the starting node in the propogation step as desribed  in
        the paper (alpha in the paper)
    """

    def __init__(
            self,
            layer_sizes,
            activations,
            generator,
            bias=True,
            dropout=0.0,
            transport_probability=0.1,
            kernel_regularizer=None
    ):

        if not isinstance(generator, FullBatchNodeGenerator):
            raise TypeError("Generator should be a instance of FullBatchNodeGenerator")

        assert len(layer_sizes) == len(activations)

        self.layer_sizes = layer_sizes
        self.transport_probability = transport_probability
        self.activations = activations
        self.bias = bias
        self.dropout = dropout
        self.kernel_regularizer = kernel_regularizer
        self.generator = generator
        self.support = 1
        self.method = generator.method

        # Check if the generator is producing a sparse matrix
        self.use_sparse = generator.use_sparse
        if self.method == "none":
            self.graph_norm_layer = GraphPreProcessingLayer(
                num_of_nodes=self.generator.Aadj.shape[0]
            )

        # Initialize a stack of fully connected layers
        n_layers = len(self.layer_sizes)
        self._layers = []
        for ii in range(n_layers):
            l = self.layer_sizes[ii]
            a = self.activations[ii]
            self._layers.append(Dropout(self.dropout))
            self._layers.append(
                Dense(
                    l,
                    activation=a,
                    use_bias=self.bias,
                    kernel_regularizer=self.kernel_regularizer,
                )
            )

        # Create propogation matrix and propogation layer
        Aadj = generator.Aadj.todense()

        # propagation matrix is non-sparse : create and store at initalization to save time
        propogation_matrix = self.transport_probability * np.linalg.inv(np.eye(Aadj.shape[0]) -
                                                                  (1 - self.transport_probability) * Aadj)
        propogation_matrix = K.tf.convert_to_tensor(propogation_matrix, K.tf.float32)

        self._layers.append(Dropout(self.dropout))
        self._layers.append(Lambda(lambda x: K.dot(propogation_matrix, x[0, :])))
        self._layers.append(Softmax())
        self._layers.append(Lambda(lambda x: K.gather(x[0], K.tf.cast(x[1], K.tf.int32))))


    def __call__(self, x):
        """
        Apply PPNP to the inputs.
        The input tensors are expected to be a list of the following:
        [
            Node features shape (1, N, F),
            Adjacency indices (1, E, 2),
            Adjacency values (1, E),
            Output indices (1, O)
        ]
        where N is the number of nodes, F the number of input features,
              E is the number of edges, O the number of output nodes.
        Args:
            x (Tensor): input tensors
        Returns:
            Output tensor
        """
        x_in, out_indices, *As = x

        # Currently we require the batch dimension to be one for full-batch methods
        batch_dim, n_nodes, _ = K.int_shape(x_in)
        if batch_dim != 1:
            raise ValueError(
                "Currently full-batch methods only support a batch dimension of one"
            )

        h_layer = x_in
        for layer in self._layers[:-1]:
            h_layer = layer(h_layer)
        h_layer = self._layers[-1]([h_layer, out_indices])

        return h_layer

    def node_model(self):
        """
        Builds a PPNP model for node prediction
        Returns:
            tuple: `(x_inp, x_out)`, where `x_inp` is a list of two Keras input tensors for the PPNP model (containing node features and graph adjacency),
            and `x_out` is a Keras tensor for the PPNP model output.
        """
        # Placeholder for node features
        N_nodes = self.generator.features.shape[0]
        N_feat = self.generator.features.shape[1]

        # Inputs for features & target indices
        x_t = Input(batch_shape=(1, N_nodes, N_feat))
        out_indices_t = Input(batch_shape=(1, None), dtype="int32")

        # Create inputs for sparse or dense matrices
        if self.use_sparse:
            # Placeholders for the sparse adjacency matrix
            A_indices_t = Input(batch_shape=(1, None, 2), dtype="int64")
            A_values_t = Input(batch_shape=(1, None))
            A_placeholders = [A_indices_t, A_values_t]

        else:
            # Placeholders for the dense adjacency matrix
            A_m = Input(batch_shape=(1, N_nodes, N_nodes))
            A_placeholders = [A_m]

        # TODO: Support multiple matrices

        x_inp = [x_t, out_indices_t] + A_placeholders
        x_out = self(x_inp)

        # Flatten output by removing singleton batch dimension
        if x_out.shape[0] == 1:
            self.x_out_flat = Lambda(lambda x: K.squeeze(x, 0))(x_out)
        else:
            self.x_out_flat = x_out

        return x_inp, x_out



