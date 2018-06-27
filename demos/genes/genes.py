from stellar.layer.hinsage import Hinsage
import keras
from keras import backend as K
import numpy as np
from data import GeneGraph, DataGenerator, TestDataGenerator
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    roc_auc_score,
)
import os


class GeneHinSAGEClassifier(object):
    """Gene classification model with HinSAGE+logistic regression architecture"""

    def __init__(self, nf, n_samples, emb_dim=256):
        """

        :param nf: number of node features
        :param n_samples: list of numbers of node samples, per edge type, for each layer (hop)
        :param emb_dim: dimensionality of the hidden node representations (embeddings)
        """
        self.nf = nf
        self.n_samples = n_samples
        self.emb_dim = emb_dim
        # YT: create a proxy of n_samples to be passed to create_model(), for model building.
        # If none of ns in n_samples is 0, then n_samples_model is just a copy of n_samples
        # If any of ns in n_samples is 0, replace it with 1 (1 neighbour sampled per node), but ensure in sample_neighs()
        # that the sampled neighbour in this case is a special "non-node" with index -1 and all-zero feature vector
        # This allows flexibility of setting any element of n_samples to 0, to avoid sampling neighbours in that hop
        # In particular, this allows to set ALL elements of n_samples to 0, thus ignoring the graph structure completely
        n_samples_model = [ns if ns > 0 else 1 for ns in n_samples]
        self.build_model(n_samples_model)  # build the model

    def build_model(self, n_samples):
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
            output_dims=[self.emb_dim] * len(n_samples),
            n_samples=n_samples,
            input_neigh_tree=[
                ("gene", [1, 2, 3]),
                ("gene", [4, 5, 6]),
                ("gene", [7, 8, 9]),
                ("gene", [10, 11, 12]),
                ("gene", []),
                ("gene", []),
                ("gene", []),
                ("gene", []),
                ("gene", []),
                ("gene", []),
                ("gene", []),
                ("gene", []),
                ("gene", []),
            ],
            input_dim={"gene": self.nf},
        )

        x_inp = [
            keras.Input(shape=(1, self.nf)),
            keras.Input(shape=(n_at(1), self.nf)),
            keras.Input(shape=(n_at(1), self.nf)),
            keras.Input(shape=(n_at(1), self.nf)),
            keras.Input(shape=(n_at(2), self.nf)),
            keras.Input(shape=(n_at(2), self.nf)),
            keras.Input(shape=(n_at(2), self.nf)),
            keras.Input(shape=(n_at(2), self.nf)),
            keras.Input(shape=(n_at(2), self.nf)),
            keras.Input(shape=(n_at(2), self.nf)),
            keras.Input(shape=(n_at(2), self.nf)),
            keras.Input(shape=(n_at(2), self.nf)),
            keras.Input(shape=(n_at(2), self.nf)),
        ]

        x_out = keras.layers.Reshape((self.emb_dim,))(hs(x_inp))
        pred = keras.layers.Activation("sigmoid")(keras.layers.Dense(1)(x_out))

        self.model = keras.Model(
            inputs=x_inp, outputs=pred
        )  # stack the layers together into a model
        self.model.compile(
            optimizer=keras.optimizers.Adam(lr=0.01),
            loss=create_weighted_binary_crossentropy(0.6, 9),
            metrics=["accuracy"],
        )

    def train(self, g: GeneGraph, epochs=1):
        train_iter = DataGenerator(g, self.nf, self.n_samples, name="train")
        valid_iter = DataGenerator(g, self.nf, self.n_samples, name="validate")

        self.model.fit_generator(
            train_iter,
            epochs=epochs,
            validation_data=valid_iter,
            max_queue_size=10,
            shuffle=True,
            verbose=2,
        )
        # print("Final train_iter.idx: {}, train_iter.data_size: {}".format(train_iter.idx, train_iter.data_size))   #this is not necessarily 0 at the end of training, since the iterator can be called more than needed, to fill the queue
        # print("Final valid_iter.idx: {}, valid_iter.data_size: {}".format(valid_iter.idx, valid_iter.data_size))   #this is not necessarily 0, since the iterator can be called more than needed, to fill the queue

    def test(self, g: GeneGraph, threshold=0.5):
        test_iter = TestDataGenerator(g, self.nf, self.n_samples)
        y_preds_proba = self.model.predict_generator(test_iter)
        y_preds_proba = np.reshape(y_preds_proba, (-1,))
        y_preds = np.array(y_preds_proba >= threshold, dtype=np.float64)
        y_trues = np.concatenate(test_iter.y_true).ravel()[
            : len(g.ids_test)
        ]  # test_iter can be called more times than needed, to fill the queue, hence test_iter.y_true might be longer than needed and thus needs truncating

        # Evaluate metrics (binary classification task):
        precision = precision_score(y_trues, y_preds)
        recall = recall_score(y_trues, y_preds)
        average_precision = average_precision_score(y_trues, y_preds_proba)
        f1score = f1_score(y_trues, y_preds)
        roc_auc = roc_auc_score(y_trues, y_preds_proba)

        met = lambda f, k: sum(
            [
                f(y_trues[i], y_preds[i]) and y_preds[i] == k
                for i in range(len(g.ids_test))
            ]
        )
        conf_matrix = {
            "tn": met(lambda t, p: t == p, 0),
            "fp": met(lambda t, p: t != p, 1),
            "fn": met(lambda t, p: t != p, 0),
            "tp": met(lambda t, p: t == p, 1),
        }

        return precision, recall, f1score, average_precision, roc_auc, conf_matrix


def main():
    print("Reading graph...")
    data_dir = "/Users/tys017/Projects/Graph_Analytics/data/Alzheimer_genes/data_small_whole_graph/"
    edge_data_fname = os.path.join(data_dir, "interactions_alz_nonalz_gwas.txt")
    gene_attr_fname = os.path.join(data_dir, "nodes_alz_nonalz_gwas_filt.txt")
    # Create the gene "graph" with genes as nodes and 3 adjacency lists, 1 per edge type
    # Split the nodes into train/validation/test sets, and store them in g.ids_train, g.ids_val, and g.ids_test
    g = GeneGraph(edge_data_fname, gene_attr_fname)
    nf = g.feats.shape[1]  # number of node features
    # YT: number of sampled neighbours (per edge type) for 1st and 2nd hop neighbourhoods of each node
    # length of n_samples list defines the number of layers in HinSAGE model
    n_samples = [2, 5]

    # Create a model:
    gene_model = GeneHinSAGEClassifier(nf, n_samples, emb_dim=256)

    print("Training the model...")
    gene_model.train(g, epochs=3)

    print("Evaluating the model on test set...")
    threshold = 0.5
    precision, recall, f1score, average_precision, roc_auc, conf_matrix = gene_model.test(
        g, threshold
    )
    print("Precision score: {0:0.2f}".format(precision))
    print("Recall score: {0:0.2f}".format(recall))
    print("F1 score: {0:0.2f}".format(f1score))
    print("Average precision-recall score: {0:0.2f}".format(average_precision))
    print("ROC AUC score: {0:0.2f}".format(roc_auc))
    print("Confusion matrix for threshold={}:".format(threshold))
    print(conf_matrix)


if __name__ == "__main__":
    main()
