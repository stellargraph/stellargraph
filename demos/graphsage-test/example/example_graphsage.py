import tensorflow as tf
from model.graphsage import graphsage_nai, MeanAggregator
from graph.redisgraph import RedisGraph
from util.evaluation import calc_f1
from redis import StrictRedis
from util.redisutil import write_to_redis
import time
import os


def print_stats(t, loss, mic, mac):
    """
    Print result statistics

    :param t:       Time taken
    :param loss:    Loss
    :param mic:     Micro-average F1-score
    :param mac:     Macro-average F1-score
    """

    print("time={:.5f}, loss={:.5f}, f1_micro={:.5f}, f1_macro={:.5f}".format(
        t, loss, mic, mac
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


def create_iterator(graph, nl, nf):
    """
    Creates a Tensorflow iterator and its initializers with given graph object. Tensorflow initializer objects are used
    to initialize the iterator with training or test set.

    :param graph:   Graph object containing generator methods for traversing through train and test sets
    :param nl:      Total number of labels
    :param nf:      Length of feature vector
    :return: Tuple of batch iterator, training set initializer, test set initializer
    """

    # input types
    inp_types = (tf.int32, tf.float32, tf.float32, tf.float32, tf.float32)

    # input shapes
    inp_shapes = (
        tf.TensorShape(()),
        tf.TensorShape((None, nl)),
        tf.TensorShape((None, nf)),
        tf.TensorShape((None, nf)),
        tf.TensorShape((None, nf))
    )

    # train and test data
    ds_train = tf.data.Dataset.from_generator(graph.train_gen, inp_types, inp_shapes).prefetch(1)
    ds_test = tf.data.Dataset.from_generator(graph.test_gen, inp_types, inp_shapes)

    tf_batch_iter = tf.data.Iterator.from_structure(inp_types, inp_shapes)
    return (
        tf_batch_iter,
        tf_batch_iter.make_initializer(ds_train),
        tf_batch_iter.make_initializer(ds_test)
    )


def main():
    """
    Main training loop setup

    """

    # batch size, number of samples per additional layer, number of feats per layer, number of epochs
    nb, ns, nf, ne = 1000, [25, 10], [50, 128, 128], 10

    # data graph
    graph = RedisGraph(StrictRedis(), nb, ns)

    # number of labels
    nl = int(graph.num_labels)

    # create iterator and its initializers
    tf_batch_iter, tf_train_iter_init, tf_test_iter_init = create_iterator(graph, nl, nf[0])

    # create tf model
    tf_batch_in = tf_batch_iter.get_next()
    tf_loss, tf_opt, tf_pred, tf_true = graphsage_nai(
        num_labels=nl,
        dims=nf,
        num_samples=ns,
        batch_in=tf_batch_in,
        agg=MeanAggregator,
        sigmoid=True
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
                    print_stats(time.time() - t, loss, *calc_f1(true, pred))
                    summary_writer.add_summary(summ, it)
                    it += 1
                except tf.errors.OutOfRangeError:
                    print("End of iterator...")
                    break

        # Final run with test set
        sess.run(tf_test_iter_init)
        print("Showing final test run results...")
        loss, pred, true = sess.run([tf_loss, tf_pred, tf_true])
        print_stats(-1, loss, *calc_f1(true, pred))

    print("Done")


if __name__ == '__main__':
    r = write_to_redis("./data/ppi")
    main()

