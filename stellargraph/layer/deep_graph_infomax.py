from . import GraphSAGE
from tensorflow.keras.layers import Input, Dense
import tensorflow as tf
from ..mapper import CorruptedGraphSAGENodeGenerator

__all__ = [
    "GraphSAGEInfoMax",
]

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

        summary = tf.math.sigmoid(tf.math.reduce_mean(node_feats, axis=0))

        D = Dense(summary.shape[0], use_bias=False)

        scores = tf.math.sigmoid(tf.linalg.matvec(D(node_feats), summary))
        scores_corrupted = 1 - tf.math.sigmoid(tf.linalg.matvec(D(node_feats_corrupted), summary))

        x_out = tf.stack([scores, scores_corrupted], axis=1)
        # Returns inputs and outputs
        return x_inp + x_inp_corrupted, x_out
