from keras import activations, initializers, constraints
from keras import regularizers
from keras.regularizers import l2
from keras.engine import Layer
from keras import backend as K
from keras import Input
from keras.layers import Lambda, Dropout, Reshape
from ..mapper import gcn_mappers as gm

from typing import List, Tuple, Callable, AnyStr


class GraphConvolution(Layer):
    """Implementation of the graph convolution layer as in https://arxiv.org/abs/1609.02907"""

    def __init__(self, units,
                 support=1,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):

        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)

        super(GraphConvolution, self).__init__(**kwargs)

        self.units = units
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.support = support

    def compute_output_shape(self, input_shapes):
        features_shape = input_shapes[0]
        output_shape = (features_shape[0], self.units)
        return output_shape  # (batch_size, output_dim)

    def build(self, input_shapes):
        features_shape = input_shapes[0]
        assert len(features_shape) == 2
        input_dim = features_shape[1]

        self.kernel = self.add_weight(shape=(input_dim * self.support,
                                             self.units),
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        if self.use_bias:
            self.bias = self.add_weight(shape=(self.units,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None
        self.built = True

    def call(self, inputs, mask=None):
        features = inputs[0]
        basis = inputs[1:]

        supports = list()
        for i in range(self.support):
            supports.append(K.dot(basis[i], features))
        supports = K.concatenate(supports, axis=1)
        output = K.dot(supports, self.kernel)

        if self.bias:
            output += self.bias
        return self.activation(output)

class GCN:
    """
    To create GCN layers with Keras layers.

    """
    def __init__(
        self,
        layer_sizes,
        generator=None,
        bias=True,
        dropout=0.0,
        normalize="l2"):

        if isinstance(generator, gm.FullBatchNodeGenerator):
            raise TypeError("Generator should be a instance of FullBatchNodeGenerator")

        self.layer_sizes = layer_sizes
        self.bias = bias
        self.dropout = dropout
        self.normalize = normalize

        self.generator = generator
        self.X_in = Input(shape=(generator.features.shape[1],))
        self.support = generator.support
        self.suppG = generator.suppG

        #FixedMe: Assuming layer_sizes [16,7] for now
        H = Dropout(self.dropout)(self.X_in)
        H = GraphConvolution(16, self.support, activation='relu', kernel_regularizer=l2(5e-4))([H]+self.suppG)
        H = Dropout(self.dropout)(H)
        self.gcn_layers = GraphConvolution(7, self.support, activation='softmax')([H]+self.suppG)

    def __call__(self, x: List):
        """
        Apply aggregator layers

        Args:
            x (list of Tensor): input features

        Returns:
            Output tensor
        """

        return self.gcn_layers

    def node_model(self):
        return [self.X_in]+self.suppG, self.gcn_layers

    def link_model(self, flatten_output=False):
        return None
