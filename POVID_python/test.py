import tensorflow as tf
import math
import numpy as np
import os
import argparse

from data_part import load_data_for_epoch
from NN import neural_network_model

parser = argparse.ArgumentParser(fromfile_prefix_chars='@', description='test run with args')
parser.add_argument('--dataset', help='Path to dataset', required=True)
parser.add_argument('--model', help='Path to where to save model', required=True)
parser.add_argument('--batch', help='Numbers of samples for each epoch of training', required=True)


args = parser.parse_args()

keep_prob = tf.placeholder(tf.float32)  # drop out layer value
x = tf.placeholder('float', shape=[None, 64, 64, 3])  # input images
y = tf.placeholder('float', shape=[None, 64, 64])  # output


def shuffle_data(data_x, data_y):  # randomly shuffles given data for epoch
    x_shuffled = np.zeros_like(data_x)
    y_shuffled = np.zeros_like(data_y)

    indexes = np.arange(data_x.__len__())
    np.random.shuffle(indexes)

    i = 0
    for index in indexes:
        x_shuffled[i] = data_x[index]
        y_shuffled[i] = data_y[index]
        i += 1

    return x_shuffled, y_shuffled


def get_data_for_epoch(files):  # loads data for epoch and splits them to validation and test data
    x_data, y_data = load_data_for_epoch(files)
    data_length = x_data.__len__()

    train_length = int(data_length * 0.9)

    x_train_data = x_data[0:train_length]
    y_train_data = y_data[0:train_length]

    x_val_data = x_data[train_length:]
    y_val_data = y_data[train_length:]

    return x_train_data, y_train_data, x_val_data, y_val_data


def train_neural_network(keep_prob):
    graph = tf.Graph()
    with graph.as_default():
        # creates model of neural network
        prediction = neural_network_model(x, keep_prob)

        # setting up parameters of model
        cross_entropy = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=prediction, labels=y)
        )
        train_step = tf.train.FtrlOptimizer(0.2).minimize(cross_entropy)

        validation_cross_entropy = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=prediction, labels=y)
        )
        test_cross_entropy = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=prediction, labels=y)
        )

        # saver for saving trained model
        saver = tf.train.Saver()

        with tf.Session() as sess:

            sess.run(tf.global_variables_initializer())

            end = False

            iteration_count = 0
            all_files = os.listdir(args.dataset)  # get list of all samples in dataset
            num_of_samples = args.batch
            batch_counter = 0
            # training on 90% of data (80% train data, 10% validation data)
            # test on rest 10% of data
            for i in range(0, len(all_files)*0.9):
                iterations = 30  # number of iterations in epoch

                if batch_counter >= len(all_files) * 0.9:
                    break

                x_train_data, y_train_data, x_val_data, y_val_data = get_data_for_epoch(
                    all_files[batch_counter:batch_counter + num_of_samples]
                )

                if len(x_train_data) == 0:
                    continue

                x_train_data, y_train_data = shuffle_data(x_train_data, y_train_data)
                x_val_data, y_val_data = shuffle_data(x_val_data, y_val_data)

                train_data = {
                    x: x_train_data,
                    y: y_train_data,
                    keep_prob: 0.5
                }

                val_data = {
                    x: x_val_data,
                    y: y_val_data,
                    keep_prob: 1.0
                }

                print("start training on series: " + str(batch_counter))
                iteration_loss = 100000

                for count in range(0, iterations):  # start training
                    _, c = sess.run([train_step, cross_entropy], feed_dict=train_data)

                    if math.isnan(c):  # too big loss
                        print("died at: " + str(count))
                        end = True
                        break

                    print("current loss: ", c)

                    val = sess.run(validation_cross_entropy, feed_dict=val_data)
                    print("validation loss: ", val)

                    # if validation loss starts rising, stop training
                    if iteration_loss < val:
                        print("end of trainig, loss starts rising")

                        end = True
                        break
                    iteration_loss = val

                batch_counter += num_of_samples
                print("number of processed samples: " + str(i))

                if end:
                    break

            # saving model
            save_path = saver.save(sess, args.model)
            print("saved in: %s" % save_path)

            # test model on test data
            # load rest 10% of data as test data
            x_test_data, y_test_data = load_data_for_epoch(all_files[batch_counter:])

            test_data = {
                x: x_test_data,
                y: y_test_data,
                keep_prob: 1.0
            }
            test_loss = sess.run(test_cross_entropy, feed_dict=test_data)
            print('test loss: ' + str(test_loss))


train_neural_network(keep_prob)
