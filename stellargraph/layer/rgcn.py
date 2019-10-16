import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer, Lambda, Dropout, Input
from tensorflow.keras import activations, initializers, constraints, regularizers
from .misc import SqueezedSparseConversion
from ..mapper.node_mappers import RelationalFullBatchNodeGenerator


class RelationalGraphConvolution(Layer):

    def __init__(
            self,
            units,
            num_relationships,
            num_bases=0,
            activation=None,
            use_bias=True,
            final_layer=False,
            kernel_initializer="glorot_uniform",
            bias_initializer="zeros",
            kernel_regularizer=None,
            bias_regularizer=None,
            activity_regularizer=None,
            kernel_constraint=None,
            bias_constraint=None,
            **kwargs
    ):
        if "input_shape" not in kwargs and "input_dim" in kwargs:
            kwargs["input_shape"] = (kwargs.get("input_dim"),)

        super().__init__(**kwargs)

        self.units = units
        self.num_relationships = num_relationships
        self.num_bases = num_bases
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

        self.final_layer = final_layer

    def get_config(self):
        """
        Gets class configuration for Keras serialization.
        Used by keras model serialization.

        Returns:
            A dictionary that contains the config of the layer
        """

        config = {
            "units": self.units,
            "use_bias": self.use_bias,
            "final_layer": self.final_layer,
            "activation": activations.serialize(self.activation),
            "kernel_initializer": initializers.serialize(self.kernel_initializer),
            "bias_initializer": initializers.serialize(self.bias_initializer),
            "kernel_regularizer": regularizers.serialize(self.kernel_regularizer),
            "bias_regularizer": regularizers.serialize(self.bias_regularizer),
            "activity_regularizer": regularizers.serialize(self.activity_regularizer),
            "kernel_constraint": constraints.serialize(self.kernel_constraint),
            "bias_constraint": constraints.serialize(self.bias_constraint),
            "num_relationships": self.num_relationships,
            "num_bases": self.num_bases
        }

        base_config = super().get_config()
        return {**base_config, **config}

    def compute_output_shape(self, input_shapes):
        """
        Computes the output shape of the layer.
        Assumes the following inputs:

        Args:
            input_shapes (tuple of ints)
                Shape tuples can include None for free dimensions, instead of an integer.

        Returns:
            An input shape tuple.
        """
        feature_shape, out_shape, A_shape = input_shapes

        batch_dim = feature_shape[0]
        if self.final_layer:
            out_dim = out_shape[1]
        else:
            out_dim = feature_shape[1]

        return batch_dim, out_dim, self.units

    def build(self, input_shapes):
        """
        Builds the layer

        Args:
            input_shapes (list of int): shapes of the layer's inputs (node features and adjacency matrix)

        """
        feat_shape = input_shapes[0]
        input_dim = int(feat_shape[-1])

        if self.num_bases != 0:

            self.bases = self.add_weight(
                shape=(input_dim, self.units, self.num_bases),  # hyperparametr B
                initializer=self.kernel_initializer,
                name="bases",
                regularizer=self.kernel_regularizer,
                constraint=self.kernel_constraint,
            )

            self.coefficients = [
                self.add_weight(
                    shape=(self.num_bases, ),  # hyperparametr B
                    initializer=self.kernel_initializer,
                    name="coeff",
                    regularizer=self.kernel_regularizer,
                    constraint=self.kernel_constraint,
                ) for _ in range(self.num_relationships)]

            self.relational_kernels = [tf.einsum("ijk,k->ij", self.bases, coeff) for
                                       coeff in self.coefficients]


        else:
            self.bases = None
            self.coefficients = None
            self.relational_kernels = [
                self.add_weight(
                    shape=(input_dim, self.units),
                    initializer=self.kernel_initializer,
                    regularizer=self.kernel_regularizer,
                    constraint=self.kernel_constraint,
                ) for _ in range(self.num_relationships)
            ]

        self.self_kernel = self.add_weight(
            shape=(input_dim, self.units),  # hyperparametr B
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
        )
        # TODO: add bias
        self.bias = None
        self.built = True

    def call(self, inputs):
        """
        Applies the layer.

        Args:
            inputs (list): a list of 3 input tensors that includes
                node features (size 1 x N x F),
                output indices (size 1 x M)
                graph adjacency matrix (size N x N),
                where N is the number of nodes in the graph, and
                F is the dimensionality of node features.

        Returns:
            Keras Tensor that represents the output of the layer.
        """
        features, out_indices, *As = inputs
        batch_dim, n_nodes, _ = K.int_shape(features)
        if batch_dim != 1:
            raise ValueError(
                "Currently full-batch methods only support a batch dimension of one"
            )

        # Remove singleton batch dimension
        features = K.squeeze(features, 0)
        out_indices = K.squeeze(out_indices, 0)

        # Calculate the layer operation of GCN
        output = K.dot(features, self.self_kernel)

        for i in range(self.num_relationships):
            h_graph = K.dot(As[i], features)
            output += K.dot(h_graph, self.relational_kernels[i])

        # Add optional bias & apply activation
        if self.bias is not None:
            output += self.bias
        output = self.activation(output)

        # On the final layer we gather the nodes referenced by the indices
        if self.final_layer:
            output = K.gather(output, out_indices)

        # Add batch dimension back if we removed it
        # print("BATCH DIM:", batch_dim)
        if batch_dim == 1:
            output = K.expand_dims(output, 0)

        return output


class RGCN:

    """
    """

    def __init__(
        self, layer_sizes, generator, bias=True,
        num_bases=0, dropout=0.0, activations=None, **kwargs
    ):
        if not isinstance(generator, RelationalFullBatchNodeGenerator):
            raise TypeError("Generator should be a instance of FullBatchNodeGenerator")

        n_layers = len(layer_sizes)
        self.layer_sizes = layer_sizes
        self.activations = activations
        self.bias = bias
        self.num_bases = num_bases
        self.dropout = dropout
        self.generator = generator
        self.support = 1
        self.method = generator.method

        # Check if the generator is producing a sparse matrix
        self.use_sparse = generator.use_sparse

        # Activation function for each layer
        if activations is None:
            activations = ["relu"] * n_layers
        elif len(activations) != n_layers:
            raise ValueError(
                "Invalid number of activations; require one function per layer"
            )

        self.activations = activations
        self.num_bases = num_bases
        # Optional regulariser, etc. for weights and biases
        self._get_regularisers_from_keywords(kwargs)

        # Initialize a stack of GCN layers
        self._layers = []
        for ii in range(n_layers):
            self._layers.append(Dropout(self.dropout))
            self._layers.append(
                RelationalGraphConvolution(
                    self.layer_sizes[ii],
                    num_relationships=len(generator.As),
                    num_bases=self.num_bases,
                    activation=self.activations[ii],
                    use_bias=self.bias,
                    final_layer=ii == (n_layers - 1),
                    **self._regularisers
                )
            )

    def _get_regularisers_from_keywords(self, kwargs):
        regularisers = {}
        for param_name in [
            "kernel_initializer",
            "kernel_regularizer",
            "kernel_constraint",
            "bias_initializer",
            "bias_regularizer",
            "bias_constraint",
        ]:
            param_value = kwargs.pop(param_name, None)
            if param_value is not None:
                regularisers[param_name] = param_value
        self._regularisers = regularisers

    def __call__(self, x):
        """
        Apply a stack of RGCN layers to the inputs.
        The input tensors are expected to be a list of the following:
        [Node features shape (1, N, F), Output indices (1, O)] +
        [Adjacency indices for each relationship (1, E, 2) for _ in range(R)]
        [Adjacency values for each relationshiop (1, E) for _ in range(R)]


        where N is the number of nodes, F the number of input features,
              E is the number of edges, O the number of output nodes,
              R is the number of relationships in the graph (edge types).

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

        N_edge_types = len(self.generator.As)

        # Convert input indices & values to a sparse matrix
        if self.use_sparse:

            As_indices = As[:N_edge_types]
            As_values = As[N_edge_types:]

            Ainput = [
                SqueezedSparseConversion(
                    shape=(n_nodes, n_nodes), dtype=As_values[i].dtype
                )([As_indices[i], As_values[i]])
                for i in range(N_edge_types)
            ]

        # Otherwise, create dense matrix from input tensor
        else:
            raise NotImplementedError("Currently dense is not supported.")

        h_layer = x_in

        for layer in self._layers:
            if isinstance(layer, RelationalGraphConvolution):
                # For a GCN layer add the matrix and output indices
                # Note that the output indices are only used if `final_layer=True`
                h_layer = layer([h_layer, out_indices] + Ainput)
            else:
                # For other (non-graph) layers only supply the input tensor
                h_layer = layer(h_layer)

        return h_layer

    def node_model(self):
        """
        Builds a GCN model for node prediction

        Returns:
            tuple: `(x_inp, x_out)`, where `x_inp` is a list of two Keras input tensors for the GCN model (containing node features and graph laplacian),
            and `x_out` is a Keras tensor for the GCN model output.
        """
        # Placeholder for node features
        N_nodes = self.generator.features.shape[0]
        N_feat = self.generator.features.shape[1]
        N_edge_types = len(self.generator.As)

        # Inputs for features & target indices
        x_t = Input(batch_shape=(1, N_nodes, N_feat))
        out_indices_t = Input(batch_shape=(1, None), dtype="int32")

        # Create inputs for sparse or dense matrices
        if self.use_sparse:
            # Placeholders for the sparse adjacency matrix
            A_indices_t = [Input(batch_shape=(1, None, 2), dtype="int64") for i in range(N_edge_types)]
            A_values_t = [Input(batch_shape=(1, None)) for i in range(N_edge_types)]
            A_placeholders = A_indices_t + A_values_t


        else:
            raise NotImplementedError("Dense is not currently supported.")
            # Placeholders for the dense adjacency matrix
            A_m = Input(batch_shape=(1, N_nodes, N_nodes))
            A_placeholders = [A_m]

        x_inp = [x_t, out_indices_t] + A_placeholders
        x_out = self(x_inp)

        # Flatten output by removing singleton batch dimension
        if x_out.shape[0] == 1:
            self.x_out_flat = Lambda(lambda x: K.squeeze(x, 0))(x_out)
        else:
            self.x_out_flat = x_out

        return x_inp, x_out

