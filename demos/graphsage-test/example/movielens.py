"""
Link Attribute Inference on Heterogeneous Graph using MovieLens data

"""

import tensorflow as tf
from keras.layers import Dense, Concatenate
from model.hinsage import hinsage_supervised, MeanAggregatorHin, Schema
from graph.redisgraph import RedisHin
from util.redis_movielens import write_to_redis
from redis import StrictRedis
import time
import os


def print_stats(t, loss):
    """
    Print result statistics

    :param t:       Time taken
    :param loss:    Loss
    :param mic:     Micro-average F1-score
    :param mac:     Macro-average F1-score
    """

    print("time={:.5f}, loss={:.5f}".format(
        t, loss
    ))


def create_log_dir():
    """
    Create log directory

    :return: log directory as string path
    """

    log_dir = './log'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return log_dir


def create_iterator(graph, nb: int, schema: Schema):
    """
    Creates a Tensorflow iterator and its initializers with given graph object. Tensorflow initializer objects are used
    to initialize the iterator with training or test set.

    :param graph:   Graph object containing generator methods for traversing through train and test sets
    :param schema:  Graph and sampling schema
    :return: Tuple of batch iterator, training set initializer, test set initializer
    """

    # input types
    inp_types = (tf.int32, tf.float32, *[tf.float32]*schema.xlen[0])

    # input shapes
    inp_shapes = (
        tf.TensorShape(()),
        tf.TensorShape((None, 1)),
        *[tf.TensorShape((None, schema.dims[0][t][0])) for t in schema.types]
    )

    # train and test data
    ds_train = tf.data.Dataset.from_generator(graph.train_gen(nb, schema), inp_types, inp_shapes).prefetch(1)
    ds_test = tf.data.Dataset.from_generator(graph.test_gen(nb, schema), inp_types, inp_shapes)

    tf_batch_iter = tf.data.Iterator.from_structure(inp_types, inp_shapes)
    return (
        tf_batch_iter,
        tf_batch_iter.make_initializer(ds_train),
        tf_batch_iter.make_initializer(ds_test)
    )


def _pred(x):
    """
    Function to transform HinSAGE output to score predictions for MovieLens graph

    :param x:   HinSAGE output tensor
    :return:    Score predictions
    """

    x0 = Dense(32, activation='sigmoid')(x[0])
    x1 = Dense(32, activation='sigmoid')(x[1])
    le = Concatenate()([x0, x1])
    return Dense(1, activation='linear')(le)


def _loss(pred: tf.Tensor, true: tf.Tensor) -> tf.Tensor:
    """
    Function to compute loss from score prediction for MovieLens graph

    :param pred:    Predicted scores
    :param true:    True scores
    :return:        Loss
    """

    return tf.losses.mean_squared_error(true, pred)


def main():
    """
    Main training loop setup

    """

    # batch size, number of samples per additional layer, number of feats per layer, number of epochs
    nb, ns, nf, ne = 1000, [25, 10], [256, 256, 256], 10

    # create schema for HinSAGE
    schema = Schema(
        types=['user', 'movie', 'movie', 'user', 'user', 'movie'],
        n_samples=[n for n in ns+[1] for _ in range(2)][::-1],
        neighs=[[(2, 'USM')], [(3, 'MSU')], [(4, 'MSU')], [(5, 'USM')], [], []],
        dims=[{'user': (d, 1), 'movie': (d, 1)} for d in nf],
        n_layers=2,
        xlen=[6, 4, 2]
    )

    # data graph
    graph = RedisHin(StrictRedis())

    # create iterator and its initializers
    tf_batch_iter, tf_train_iter_init, tf_test_iter_init = create_iterator(graph, nb, schema)

    # create tf model
    tf_batch_in = tf_batch_iter.get_next()
    tf_loss, tf_opt, tf_pred, tf_true = hinsage_supervised(
        schema=schema,
        batch_in=tf_batch_in,
        agg=MeanAggregatorHin,
        f_pred=_pred,
        f_loss=_loss
    )

    with tf.Session() as sess:
        # logs
        tf_summ = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(create_log_dir(), sess.graph)

        # initialize variables
        sess.run(tf.global_variables_initializer())
        it = 0

        # Runs for ne epochs
        for epoch in range(ne):
            print("Epoch", epoch)
            # initialize iterator with train data
            sess.run(tf_train_iter_init)
            while True:
                try:
                    t = time.time()
                    loss, _, pred, true, summ = sess.run([tf_loss, tf_opt, tf_pred, tf_true, tf_summ])
                    print_stats(time.time() - t, loss)
                    summary_writer.add_summary(summ, it)
                    it += 1
                except tf.errors.OutOfRangeError:
                    print("End of iterator...")
                    break

        # Final run with test set
        sess.run(tf_test_iter_init)
        print("Showing final test run results...")
        loss, pred, true = sess.run([tf_loss, tf_pred, tf_true])
        print_stats(-1, loss)

    print("Done")


if __name__ == '__main__':
    write_to_redis('./data')
    main()

