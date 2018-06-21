from stellar.layer.hinsage import Hinsage
import keras
from keras import backend as K
import numpy as np
from data import GeneGraph, DataGenerator, TestDataGenerator


def create_model(nf, n_samples):
    def n_at(i):
        return np.product(n_samples[:i])

    def create_weighted_binary_crossentropy(zero_weight, one_weight):
        def weighted_binary_crossentropy(y_true, y_pred):
            b_ce = K.binary_crossentropy(y_true, y_pred)
            weight_vector = y_true * one_weight + (1. - y_true) * zero_weight
            weighted_b_ce = weight_vector * b_ce
            return K.mean(weighted_b_ce)

        return weighted_binary_crossentropy

    hs = Hinsage(
        output_dims=[256, 256],
        n_samples=n_samples,
        input_neigh_tree=[('gene', [1, 2, 3]),
                          ('gene', [4, 5, 6]), ('gene', [7, 8, 9]), ('gene', [10, 11, 12]),
                          ('gene', []), ('gene', []), ('gene', []),
                          ('gene', []), ('gene', []), ('gene', []),
                          ('gene', []), ('gene', []), ('gene', [])],
        input_dim={'gene': nf}
    )

    x_inp = [
        keras.Input(shape=(1, nf)),
        keras.Input(shape=(n_at(1), nf)),
        keras.Input(shape=(n_at(1), nf)),
        keras.Input(shape=(n_at(1), nf)),
        keras.Input(shape=(n_at(2), nf)),
        keras.Input(shape=(n_at(2), nf)),
        keras.Input(shape=(n_at(2), nf)),
        keras.Input(shape=(n_at(2), nf)),
        keras.Input(shape=(n_at(2), nf)),
        keras.Input(shape=(n_at(2), nf)),
        keras.Input(shape=(n_at(2), nf)),
        keras.Input(shape=(n_at(2), nf)),
        keras.Input(shape=(n_at(2), nf))
    ]

    x_out = keras.layers.Reshape((256,))(hs(x_inp))
    pred = keras.layers.Activation('sigmoid')(keras.layers.Dense(1)(x_out))

    model = keras.Model(inputs=x_inp, outputs=pred)
    model.compile(
        optimizer=keras.optimizers.Adam(lr=0.01),
        loss=create_weighted_binary_crossentropy(0.6, 9),
        metrics=['accuracy']
    )

    return model


def train(g: GeneGraph, nf, n_samples):
    batch_iter = DataGenerator(g, nf, n_samples)
    model = create_model(nf, n_samples)
    model.fit_generator(batch_iter, epochs=10, verbose=2)
    return model


def test(g: GeneGraph, nf, n_samples, model):
    test_iter = TestDataGenerator(g, nf, n_samples)
    y_preds = model.predict_generator(test_iter)
    y_trues_bin = np.concatenate(test_iter.y_true).ravel()[:len(g.ids_test)]
    y_preds_bin = np.array(np.reshape(y_preds, (-1,)) >= 0.5, dtype=np.float64)
    met = lambda f, k: sum([f(y_trues_bin[i], y_preds_bin[i]) and y_preds_bin[i] == k for i in range(len(g.ids_test))])
    return {'tn': met(lambda t, p: t == p, 0),
            'fp': met(lambda t, p: t != p, 1),
            'fn': met(lambda t, p: t != p, 0),
            'tp': met(lambda t, p: t == p, 1)}


def main():
    print("Reading graph...")
    g = GeneGraph(
        "/Users/jun021/work/datasets/genes/data_small_whole_graph/interactions_alz_nonalz_gwas.txt",
        "/Users/jun021/work/datasets/genes/data_small_whole_graph/nodes_alz_nonalz_gwas_filt.txt"
    )
    nf = 414
    n_samples = [5, 5]
    model = train(g, nf, n_samples)
    confu_mat = test(g, nf, n_samples, model)
    print(confu_mat)


if __name__ == "__main__":
    main()

