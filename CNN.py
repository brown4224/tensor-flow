# Sean McGlincy
# HW 5

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import tensorflow
from sklearn.model_selection import KFold
tensorflow.logging.set_verbosity(tensorflow.logging.INFO)

#############################   References ##################################################
# https://www.tensorflow.org/tutorials/layers
#  https://www.tensorflow.org/programmers_guide/summaries_and_tensorboard
#############################   References ##################################################


# Normalizes Data between 0-> 1
def NormalizeData(data):
    return data / 255.0


def CNN(features, labels, mode):

    # Sets input layer parm: batch, width, height, depth (ie. greyscale vs rgb)
    input_layer = tensorflow.reshape(features["x"], [-1, 28, 28, 1])

    # Create a convolutional network with relu and the pool the layers with strides
    convolution_1 = tensorflow.layers.conv2d(inputs= input_layer,filters=32,kernel_size=[5,5],padding="same",activation=tensorflow.nn.relu)
    pooling_1 = tensorflow.layers.max_pooling2d(inputs=convolution_1, pool_size=[2, 2], strides=2)

    #  Second layer in our cnn network.  Again apply convolutionaland pooling
    convolution_2 = tensorflow.layers.conv2d(inputs=pooling_1,filters=64,kernel_size=[5,5],padding="same", activation=tensorflow.nn.relu)
    pooling_2 = tensorflow.layers.max_pooling2d(inputs=convolution_2, pool_size=[2, 2], strides=2)

    #  Flatten and create a dense layer for classification, 1024 neurons.
    pooling_flat = tensorflow.reshape(pooling_2, [-1, 7 * 7 * 64])
    dense_layer = tensorflow.layers.dense(inputs=pooling_flat, units=1024, activation=tensorflow.nn.relu)

    # drop out training
    dropout = tensorflow.layers.dropout(inputs=dense_layer, rate=0.4, training=mode == tensorflow.estimator.ModeKeys.TRAIN)
    logits = tensorflow.layers.dense(inputs=dropout, units=10)

    #  Apply softmax on outpul layer
    predictions = {"classes": tensorflow.argmax(input=logits, axis=1),"probabilities": tensorflow.nn.softmax(logits, name="softmax_tensor")}

    # Return the prediction
    if mode == tensorflow.estimator.ModeKeys.PREDICT:
        return tensorflow.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Convert labels to onehot and calculate the loss
    # one_hot = tensorflow.one_hot(indices=tensorflow.cast(labels, tensorflow.int32), depth=10)
    loss = tensorflow.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
    # loss = tensorflow.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
    # tensorflow.summary.scalar("loss", loss)

    # Sets gradient descent
    if mode == tensorflow.estimator.ModeKeys.TRAIN:
        gradient_descent = tensorflow.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = gradient_descent.minimize(loss=loss, global_step=tensorflow.train.get_global_step())
        return tensorflow.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # Metrics for tensor board
    metrics= { "accuracy" : tensorflow.metrics.accuracy(
        labels=labels, predictions=predictions["classes"])}
    return tensorflow.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=metrics)

#############################   Main Program   ##################################################
def main(unused_argv):
    k_fold = 10
    data = np.genfromtxt('MNIST_HW4.csv', delimiter=',', dtype=int, skip_header=1)

    # K Fold for one cycle
    kf = KFold(n_splits=k_fold)
    train_index, test_index = next(kf.split(data))

    # Kfold MAP
    train_data = np.array(data[train_index])
    test_data = np.array(data[test_index])

    # Prep training and test labels
    train_label = train_data[:, 0]
    test_label = test_data[:, 0]

    # Normalize Data
    train_data = NormalizeData(train_data[:, 1:])
    test_data = NormalizeData(test_data[:, 1:])

    #  Saves file for tensorboard.  Logs file while it runs
    cnn_classifier = tensorflow.estimator.Estimator(model_fn=CNN, model_dir="./cnn_tensorboard_model")
    tensors_to_log = {"probabilities": "softmax_tensor"}
    logging = tensorflow.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=50)

    #   Trains data with estimator, this returns values to tensorboard
    train = tensorflow.estimator.inputs.numpy_input_fn(x={"x": train_data}, y=train_label, batch_size=100, num_epochs=None, shuffle=True)
    cnn_classifier.train(input_fn=train, steps=20000, hooks=[logging])

    #  make prediction
    prediction = tensorflow.estimator.inputs.numpy_input_fn(x={"x": test_data}, y=test_label, num_epochs=1, shuffle=False)
    results = cnn_classifier.evaluate(input_fn=prediction)
    print(results)


if __name__ == "__main__":
    tensorflow.app.run()