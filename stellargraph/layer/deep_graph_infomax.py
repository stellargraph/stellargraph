from . import GraphSAGE, GCN
from tensorflow.keras.layers import Input, Dense, Lambda, Layer
import tensorflow as tf
from ..mapper import CorruptedGraphSAGENodeGenerator
from tensorflow.keras import backend as K

__all__ = [
    "GraphSAGEInfoMax",
    "GCNInfoMax",
]


class Discriminator(Layer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def build(self, input_shapes):
        self.kernel = self.add_weight(
            shape=(input_shapes[0][1], input_shapes[0][1]),
            initializer="glorot_uniform",
            name="kernel",
            regularizer=None,
            constraint=None,
        )
        self.built = True

    def call(self, inputs):
        """
        """
        features, summary = inputs

        score = tf.linalg.matvec(
            tf.linalg.matmul(features, self.kernel),
            summary,
        )

        return score


class GraphSAGEInfoMax(GraphSAGE):

    def __init__(
            self,
            layer_sizes,
            generator=None,
            aggregator=None,
            bias=True,
            dropout=0.0,
            normalize="l2",
            activations=None,
            **kwargs,
    ):
        super().__init__(
            layer_sizes,
            generator=generator,
            aggregator=aggregator,
            bias=bias,
            dropout=dropout,
            normalize=normalize,
            activations=activations,
            **kwargs,
        )

    def _get_sizes_from_generator(self, generator):
        """
        Sets n_samples and input_feature_size from the generator.
        Args:
             generator: The supplied generator.
        """
        if not isinstance(generator, CorruptedGraphSAGENodeGenerator):
            errmsg = "Generator should be an instance of CorruptedGraphSAGENodeGenerator"

            raise TypeError(errmsg)

        self.n_samples = generator.num_samples
        # Check the number of samples and the layer sizes are consistent
        if len(self.n_samples) != self.max_hops:
            raise ValueError(
                "Mismatched lengths: neighbourhood sample sizes {} versus layer sizes {}".format(
                    self.n_samples, self.layer_sizes
                )
            )

        self.multiplicity = generator.multiplicity
        feature_sizes = generator.graph.node_feature_sizes()
        if len(feature_sizes) > 1:
            raise RuntimeError(
                "GraphSAGE called on graph with more than one node type."
            )
        self.input_feature_size = feature_sizes.popitem()[1]

    def unsupervised_node_model(self):
        """
        """
        # Create tensor inputs for neighbourhood sampling
        x_inp = [
            Input(shape=(s, self.input_feature_size)) for s in self.neighbourhood_sizes
        ]

        x_inp_corrupted = [
            Input(shape=(s, self.input_feature_size)) for s in self.neighbourhood_sizes
        ]

        # Output from GraphSAGE model
        node_feats = self(x_inp)
        node_feats_corrupted = self(x_inp_corrupted)

        summary = Lambda(
            lambda x: tf.math.sigmoid(tf.math.reduce_mean(x, axis=0))
        )(node_feats)

        discriminator = Discriminator()
        scores = discriminator([node_feats, summary])
        scores_corrupted = discriminator([node_feats_corrupted, summary])

        x_out = tf.stack([scores, scores_corrupted], axis=1)
        # Returns inputs and outputs
        return x_inp + x_inp_corrupted, x_out

    def embedding_model(self, model):

        pass


class GCNInfoMax(GCN):

    def __init__(
            self,
            layer_sizes,
            generator,
            bias=True,
            dropout=0.0,
            activations=None,
            kernel_initializer=None,
            kernel_regularizer=None,
            kernel_constraint=None,
            bias_initializer=None,
            bias_regularizer=None,
            bias_constraint=None,
    ):

        super().__init__(
            layer_sizes,
            generator,
            bias=bias,
            dropout=dropout,
            activations=activations,
            kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer,
            kernel_constraint=kernel_constraint,
            bias_initializer=bias_initializer,
            bias_regularizer=bias_regularizer,
            bias_constraint=bias_constraint,
        )

    def unsupervised_node_model(self):
        """
        """
        # Inputs for features
        x_t = Input(batch_shape=(1, self.n_nodes, self.n_features))
        # Inputs for shuffled features
        x_corr = Input(batch_shape=(1, self.n_nodes, self.n_features))

        out_indices_t = Input(batch_shape=(1, None), dtype="int32")

        # Create inputs for sparse or dense matrices
        if self.use_sparse:
            # Placeholders for the sparse adjacency matrix
            A_indices_t = Input(batch_shape=(1, None, 2), dtype="int64")
            A_values_t = Input(batch_shape=(1, None))
            A_placeholders = [A_indices_t, A_values_t]

        else:
            # Placeholders for the dense adjacency matrix
            A_m = Input(batch_shape=(1, self.n_nodes, self.n_nodes))
            A_placeholders = [A_m]

        x_inp = [x_t, x_corr, out_indices_t] + A_placeholders

        node_feats = self([x_t, out_indices_t] + A_placeholders)
        node_feats = Lambda(
            lambda x: K.squeeze(x, axis=0,),
            name="NODE_FEATURES",
        )(node_feats)

        node_feats_corrupted = self([x_corr, out_indices_t] + A_placeholders)
        node_feats_corrupted = Lambda(
            lambda x: K.squeeze(x, axis=0,),
        )(node_feats_corrupted)

        summary = Lambda(
            lambda x: tf.math.sigmoid(tf.math.reduce_mean(x, axis=0))
        )(node_feats)

        discriminator = Discriminator()
        scores = discriminator([node_feats, summary])
        scores_corrupted = discriminator([node_feats_corrupted, summary])

        x_out = tf.stack([scores, scores_corrupted], axis=1)

        x_out = K.expand_dims(x_out, axis=0)
        return x_inp, x_out

    def embedding_model(self, model):

        x_emb_in = model.inputs
        x_emb_out = model.get_layer("NODE_FEATURES").output

        return x_emb_in, x_emb_out


