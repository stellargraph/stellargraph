"""
Link Attribute Inference on Heterogeneous Graph using MovieLens data

"""
import time
import os
import argparse
import tensorflow as tf
from keras.layers import Dense, Concatenate, Multiply
import numpy as np
import pandas as pd
from typing import AnyStr

from model.hinsage import hinsage_supervised, MeanAggregatorHin
from util.redis_movielens import MovielensRedis


def create_log_dir():
    """
    Create log directory

    :return: log directory as string path
    """

    log_dir = './log'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return log_dir


def regression_pred_mul(x: tf.Tensor, b0: tf.Tensor, b1: tf.Tensor) -> tf.Tensor:
    """
    Function to transform HinSAGE output to score predictions for MovieLens graph.

    The edge feature is formed from a Hadamard product of transformed
    movie and user embeddings.

    :param x:   HinSAGE output tensor
    :return:    Score predictions
    """
    x0 = Dense(32, activation='relu')(x[0])
    x1 = Dense(32, activation='relu')(x[1])
    le = Multiply()([x0, x1])

    return b0 + b1 + Dense(1, activation='linear')(le)


def regression_pred_concat(x: tf.Tensor, b0: tf.Tensor, b1: tf.Tensor) -> tf.Tensor:
    """
    Function to transform HinSAGE output to score predictions for MovieLens graph

    The edge feature is formed from a concatenation of transformed movie
    and user embeddings followed by a dense NN layer.

    :param x:   HinSAGE output tensor
    :return:    Score predictions
    """
    x0 = Dense(16, activation='relu')(x[0])
    x1 = Dense(16, activation='relu')(x[1])
    le = Concatenate()([x0, x1])
    le = Dense(32, activation='relu')(le)

    return b0 + b1 + Dense(1, activation='linear')(le)


def regression_pred_ip(x: tf.Tensor, b0: tf.Tensor, b1: tf.Tensor) -> tf.Tensor:
    """
    Function to transform HinSAGE output to score predictions for MovieLens graph

    This is a direct inner product between the user and movie embedddings.

    :param x:   HinSAGE output tensor
    :return:    Score predictions
    """
    x0 = Dense(16, activation='relu')(x[0])
    x1 = Dense(16, activation='relu')(x[1])
    return b0 + b1 + tf.reduce_sum(x0 * x1, axis=1, keepdims=True)


def mse_loss(pred: tf.Tensor, true: tf.Tensor) -> tf.Tensor:
    """
    Function to compute loss from score prediction for MovieLens graph

    :param pred:    Predicted scores
    :param true:    True scores
    :return:        Loss
    """
    return tf.losses.mean_squared_error(true, pred)


def train(ml_data: MovielensRedis, batch_size: int = 1000, num_epochs: int = 10):
    """
    Main training loop
    """
    schema = ml_data.create_schema()

    tf_batch_iter, tf_train_iter_init, tf_test_iter_init \
        = ml_data.create_iterators(batch_size=batch_size)

    # The edge regressor to use
    if ml_data.edge_regressor == "ip":
        regression_pred = regression_pred_ip

    elif ml_data.edge_regressor == "concat":
        regression_pred = regression_pred_concat

    elif ml_data.edge_regressor == "mul":
        regression_pred = regression_pred_mul

    # create tf model
    tf_batch_in = tf_batch_iter.get_next()
    tf_loss, tf_opt, tf_pred, tf_true = hinsage_supervised(
        schema=schema,
        batch_in=tf_batch_in,
        agg=MeanAggregatorHin,
        f_pred=regression_pred,
        f_loss=mse_loss,
        bias=ml_data.use_bias,
        node_bias=ml_data.node_baseline,
        n_nodes=ml_data.n_nodes,
        learning_rate=0.005
    )

    with tf.Session() as sess:
        tf_summ = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(create_log_dir(), sess.graph)
        saver = tf.train.Saver()

        # initialize variables
        sess.run(tf.global_variables_initializer())

        # Runs for ne epochs
        for epoch in range(num_epochs):
            # initialize iterator with train data
            sess.run(tf_train_iter_init)
            it = 1
            t = time.time()
            while True:
                try:
                    loss, _, pred, true, summ = sess.run(
                        [tf_loss, tf_opt, tf_pred, tf_true, tf_summ]
                    )
                    summary_writer.add_summary(summ, it)
                    time_per_batch = (time.time() - t)/it
                    it += 1
                    print("epoch {}, train_loss={:.3f}, batch_time={:.3f}"
                        .format(epoch, loss, time_per_batch))

                except tf.errors.OutOfRangeError:
                    print("End of epoch. Processed in {}s."
                          .format(time.time() - t))
                    break

            # Save model each epoch
            saver.save(sess, "./ml_{}_e{}".format(ml_data.info_str(), epoch))

            # Use current model to predict
            sess.run(tf_test_iter_init)
            predictions = []
            while True:
                try:
                    pred_mb = sess.run(tf_pred)
                    predictions.extend(np.ravel(pred_mb))
                except tf.errors.OutOfRangeError:
                    print("End of test data. Processed in {}s.".format(time.time() - t))
                    break

            ml_data.calc_test_metrics(predictions)

        # Save predictions & GT
        ml_data.save_predictions(predictions, "predictions_{}.csv"
                                 .format(ml_data.info_str()))


def test(ml_data: MovielensRedis, model_file: AnyStr):
    """
    Predict and measure the test performance
    """
    schema = ml_data.create_schema()

    tf_batch_iter, tf_train_iter_init, tf_test_iter_init \
        = ml_data.create_iterators()

    # The edge regressor to use
    if ml_data.edge_regressor == "ip":
        regression_pred = regression_pred_ip

    elif ml_data.edge_regressor == "concat":
        regression_pred = regression_pred_concat

    elif ml_data.edge_regressor == "mul":
        regression_pred = regression_pred_mul

    # create tf model
    tf_batch_in = tf_batch_iter.get_next()
    tf_loss, tf_opt, tf_pred, tf_true = hinsage_supervised(
        schema=schema,
        batch_in=tf_batch_in,
        agg=MeanAggregatorHin,
        f_pred=regression_pred,
        f_loss=mse_loss,
        bias=ml_data.use_bias,
        node_bias=ml_data.node_baseline,
        n_nodes=ml_data.n_nodes,
    )

    saver = tf.train.Saver()
    with tf.Session() as sess:
        # initialize variables
        #sess.run(tf.global_variables_initializer())

        # Restore weights from file
        saver.restore(sess, model_file)

        # initialize iterator with test data
        sess.run(tf_test_iter_init)

        it = 1
        t = time.time()
        predictions = []
        while True:
            try:
                pred_mb = sess.run(tf_pred)
                predictions.extend(np.ravel(pred_mb))
                it += 1
            except tf.errors.OutOfRangeError:
                print("End of training data. Processed in {}s.".format(time.time() - t))
                break

        ml_data.calc_test_metrics(predictions)

    # Save predictions & GT
    ml_data.save_predictions(predictions, "predictions_{}.csv"
                             .format(ml_data.info_str()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run GraphSAGE on movielens")
    parser.add_argument('-c', '--checkpoint', nargs='?', type=str, default=None,
                        help="Load a save checkpoint file")
    parser.add_argument('-n', '--batch_size', type=int, default=500,
                        help="Load a save checkpoint file")
    parser.add_argument('-e', '--epochs', type=int, default=10,
                        help="Number of epochs to train for")
    parser.add_argument('-s', '--neighbour_samples', type=int, nargs='*', default=[30, 10],
                        help="The number of nodes sampled at each layer")
    parser.add_argument('-l', '--layer_size', type=int, nargs='*', default=[50, 50],
                        help="The number of hidden features at each layer")
    parser.add_argument('-m', '--method', type=str, default='ip',
                        help="The edge regression method: 'concat', 'mul', or 'ip")
    parser.add_argument('-b', '--baseline', action='store_true',
                        help="Use a learned offset for each node.")
    parser.add_argument('-g', '--graph', type=str, default='data/ml-1m_split_graphnx.pkl',
                        help="The graph stored in networkx pickle format.")
    parser.add_argument('-f', '--features', type=str, default='data/ml-1m_embeddings.pkl',
                        help="The node features to use, stored as a pickled numpy array.")
    parser.add_argument('-t', '--target', type=str, default='score',
                        help="The target edge attribute, default is 'score'")

    args, cmdline_args = parser.parse_known_args()

    print("Running GraphSAGE recommender:")

    # Node2vec embeddings
    ml_data = MovielensRedis(args.graph, args.features, args.target)

    # Training: batch size & epochs
    batch_size = args.batch_size
    num_epochs = args.epochs

    # number of samples per additional layer,
    ml_data.node_samples = args.neighbour_samples

    # number of features per additional layer
    ml_data.layer_size = args.layer_size

    # The edge regressor to use
    ml_data.edge_regressor = args.method

    # Per-node baselines - learns a baseline for movies and users
    # requires fixed set of train/test movies + users
    ml_data.node_baseline = args.baseline

    # Layer bias
    ml_data.use_bias = True

    if args.checkpoint is None:
        train(ml_data, batch_size, num_epochs)
    else:
        test(ml_data, args.checkpoint)

