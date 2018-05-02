from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import tensorflow as tf
from sklearn.model_selection import KFold
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
tf.logging.set_verbosity(tf.logging.INFO)

#############################   References ##################################################
#  https://www.tensorflow.org/programmers_guide/summaries_and_tensorboard
#############################   References ##################################################


plot = []
FLAGS = None
def DisplayLearningCurve(plot):
    plt.plot(plot)
    plt.interactive(False)
    plt.show(block=True)


# Normalizes Data between 0-> 1
def NormalizeData(data):
    return data / 255.0

# def variable_summaries(var):
#     with tf.name_scope('summaries'):
#         mean = tf.reduce_mean(var)
#         tf.summary.scalar('mean', mean)
#         with tf.name_scope('stddev'):
#             stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
#         tf.summary.scalar('stddev', stddev)
#         tf.summary.scalar('max', tf.reduce_max(var))
#         tf.summary.scalar('min', tf.reduce_min(var))
#         tf.summary.histogram('histogram', var)

# def feed_dict(train):
#   """Make a TensorFlow feed_dict: maps data onto Tensor placeholders."""
#   if train or FLAGS.fake_data:
#     xs, ys = mnist.train.next_batch(100, fake_data=FLAGS.fake_data)
#     k = FLAGS.dropout
#   else:
#     xs, ys = mnist.test.images, mnist.test.labels
#     k = 1.0
#   return {x: xs, y_: ys, keep_prob: k}

# for i in range(FLAGS.max_steps):
#   if i % 10 == 0:  # Record summaries and test-set accuracy
#     summary, acc = sess.run([merged, accuracy], feed_dict=feed_dict(False))
#     test_writer.add_summary(summary, i)
#     print('Accuracy at step %s: %s' % (i, acc))
#   else:  # Record train set summaries, and train
#     summary, _ = sess.run([merged, train_step], feed_dict=feed_dict(True))
#     train_writer.add_summary(summary, i)


def cnn_model_fn(features, labels, mode):
    input_layer = tf.reshape(features["x"], [-1, 28, 28, 1])
    conv1 = tf.layers.conv2d(
        inputs= input_layer,
        filters=32,
        kernel_size=[5,5],
        padding="same",
        activation=tf.nn.relu)

    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2,2], strides=2)
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=64,
        kernel_size=[5,5],
        padding="same",
        activation=tf.nn.relu)
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
    pool2_flat = tf.reshape(pool2,[-1, 7 * 7 * 64])
    dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
    dropout = tf.layers.dropout(inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

    logits = tf.layers.dense(inputs=dropout, units=10)

    predictions = {
        "classes": tf.argmax(input=logits, axis=1),
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)
    sess = tf.InteractiveSession()

    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
    tf.summary.scalar("cross_entropy", loss)
    writer = tf.summary.FileWriter("/tmp/demo", sess.graph)
    # writer.add.graph(sess.graph)

    # tf.scalar_summary("cost", loss)
    # writer = tf.train.
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)

        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())



        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)
    eval_metric_ops= { "accuracy" : tf.metrics.accuracy(
        labels=labels, predictions=predictions["classes"])}

    # # Merge all the summaries and write them out to /tmp/mnist_logs (by default)
    # merged = tf.summary.merge_all()
    # train_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/train',
    #                                      sess.graph)


    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

def main(unused_argv):
    k_fold = 10
    data = np.genfromtxt('MNIST_HW4.csv', delimiter=',', dtype=int, skip_header=1)

    # K Fold for one cycle
    kf = KFold(n_splits=k_fold)
    train_index, test_index = next(kf.split(data))

    # Kfold map
    train_data = np.array(data[train_index])
    test_data = np.array(data[test_index])

    # # Prep training and test labels
    # # Training labels are one hot array
    train_label = train_data[:, 0]
    test_label = test_data[:, 0]

    train_data = train_data[:50]
    train_label = train_label[:50]

    # Prep training and test data
    train_data = NormalizeData(train_data[:, 1:])
    test_data = NormalizeData(test_data[:, 1:])
    # sess.run(tf.global_variables_initializer())
    # writer = tf.summary.FileWriter("/tmp/demo")
    # writer.add.graph(sess.graph)

    mnist_classifier = tf.estimator.Estimator(
        model_fn=cnn_model_fn, model_dir="/tmp/mnist_convnet_model")
    tensors_to_log = {"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=50)

    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": train_data},
        y=train_label,
        batch_size=100,
        num_epochs=None,
        shuffle=True)
    mnist_classifier.train(
        input_fn=train_input_fn,
        steps=20000,
        hooks=[logging_hook])

    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": test_data},
        y=test_label,
        num_epochs=1,
        shuffle=False)
    eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
    print(eval_results)
    # DisplayLearningCurve(plot)


if __name__ == "__main__":
    tf.app.run()