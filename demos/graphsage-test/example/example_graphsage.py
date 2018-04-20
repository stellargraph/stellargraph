import tensorflow as tf
from model.graphsage import supervised_graphsage, MeanAggregator
from graph.redisgraph import RedisGraph
from util.evaluation import calc_f1
from redis import StrictRedis
from util.redisutil import write_to_redis
import time


def print_stats(time, loss, mic, mac):
    print("time={:.5f}, loss={:.5f}, f1_micro={:.5f}, f1_macro={:.5f}".format(
        time, loss, mic, mac
    ))


def main():
    nb, ns, nf = 1000, [25, 10], 50
    graph = RedisGraph(StrictRedis(), nb, ns)
    nl = int(graph.num_labels)

    inp_types = (tf.int32, tf.float32, tf.float32, tf.float32, tf.float32)
    inp_shapes = (
        tf.TensorShape(()),
        tf.TensorShape((None, nl)),
        tf.TensorShape((None, nf)),
        tf.TensorShape((None, nf)),
        tf.TensorShape((None, nf))
    )

    train_ds = tf.data.Dataset.from_generator(graph.train_gen, inp_types, inp_shapes).prefetch(1)
    test_ds = tf.data.Dataset.from_generator(graph.test_gen, inp_types, inp_shapes)

    t_batch_iter = tf.data.Iterator.from_structure(inp_types, inp_shapes)

    train_iter_init_op = t_batch_iter.make_initializer(train_ds)
    test_iter_init_op = t_batch_iter.make_initializer(test_ds)

    batch_in = t_batch_iter.get_next()
    loss, opt_op, y_preds, y_true = supervised_graphsage(
        num_labels=nl,
        dims=[nf, 128, 128],
        num_samples=ns,
        batch_in=batch_in,
        agg=MeanAggregator
    )

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        # Runs for 5 epochs
        for epoch in range(5):
            print("Epoch", epoch)
            sess.run(train_iter_init_op)
            while True:
                try:
                    t = time.time()
                    outs = sess.run([loss, opt_op, y_preds, y_true])
                    print_stats(time.time() - t, outs[0], *calc_f1(outs[3], outs[2]))
                except tf.errors.OutOfRangeError:
                    break

        # Final run with test set
        sess.run(test_iter_init_op)
        outs = sess.run([loss, y_preds, y_true])
        print_stats(-1, outs[0], *calc_f1(outs[2], outs[1]))

    print("Done")


if __name__ == '__main__':
    r = write_to_redis("./data/ppi")
    main()

