import tensorflow as tf
from model.graphsage import supervised_graphsage, MeanAggregator
from graph.redisgraph import RedisGraph
from util.evaluation import calc_f1
from redis import StrictRedis
from util.redisutil import write_to_redis
import time
import os


def print_stats(t, loss, mic, mac):
    print("time={:.5f}, loss={:.5f}, f1_micro={:.5f}, f1_macro={:.5f}".format(
        t, loss, mic, mac
    ))


def create_log_dir():
    log_dir = './log'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return log_dir


def create_iterator(graph, nl, nf):
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
    # batch size, number of samples per layer, number of feats
    nb, ns, nf = 1000, [25, 10], 50

    # data graph
    graph = RedisGraph(StrictRedis(), nb, ns)

    # number of labels
    nl = int(graph.num_labels)

    # create iterator and its initializers
    tf_batch_iter, tf_train_iter_init, tf_test_iter_init = create_iterator(graph, nl, nf)

    # create tf model
    tf_batch_in = tf_batch_iter.get_next()
    tf_loss, tf_opt, tf_pred, tf_true = supervised_graphsage(
        num_labels=nl,
        dims=[nf, 128, 128],
        num_samples=ns,
        batch_in=tf_batch_in,
        agg=MeanAggregator
    )

    with tf.Session() as sess:
        # logs
        tf_summ = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(create_log_dir(), sess.graph)

        # initialize variables
        sess.run(tf.global_variables_initializer())
        it = 0

        # Runs for 5 epochs
        for epoch in range(10):
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

