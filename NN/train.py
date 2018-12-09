import math
import random
import numpy as np
import tensorflow as tf
import argparse

from nn_model import neural_network_model
from load_data import load_data

parser = argparse.ArgumentParser(fromfile_prefix_chars='@', description='train neural network')
parser.add_argument('--images', help='folder with input images', required=True)
parser.add_argument('--maps', help='folder with saliency maps', required=True)
parser.add_argument('--batch', help='Numbers of samples for each epoch of training', required=True)
parser.add_argument('--epochs', help='Number of epochs', required=True)
parser.add_argument('--model', help='File where final trained model will be saved', required=True)
parser.add_argument('--n', help='Images size n x n', required=True)

args = parser.parse_args()

keep_prob = tf.placeholder(tf.float32)  # drop out layer value
x = tf.placeholder('float', shape=[None, args.n, args.n, 3])  # input images
y = tf.placeholder('float', shape=[None, args.n, args.n])  # output

from tensorflow.contrib.layers import fully_connected


def print_shape(tensor):
    print(tensor.get_shape)


# creates model of NN, data = input image, keep_prob = value on dropout layer

def shuffle_data(images, maps):
    count = len(images)

    ind = list(range(0, count))
    random.shuffle(ind)
    split = int(count * 0.8)

    train_ind = ind[:split]
    valid_ind = ind[split:]

    print(images.shape)
    print(maps.shape)

    # return train_x, train_y, valid_x, valid_y
    return images[train_ind, :, :, :], maps[train_ind, :, :], images[valid_ind, :, :, :], maps[valid_ind, :, :]


def train_neural_network():
    # creates model of neural network
    prediction = neural_network_model(x, keep_prob, int(args.n))

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

    # add variables to log file
    tf.summary.scalar("train_cross_entropy", cross_entropy)
    tf.summary.scalar("validation_cross_entropy", cross_entropy)
    summary = tf.summary.merge_all()


    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())

        # set up directories for log files
        # TODO os.path.join
        summary_writer = tf.summary.FileWriter(args.model + "logs/train", sess.graph)
        validation_writer = tf.summary.FileWriter(args.model + "logs/val/", sess.graph)


        end = False

        images = np.array(load_data(args.images, (int(args.n), int(args.n)), int(args.batch)))
        maps = np.array(load_data(args.maps, (int(args.n), int(args.n)), int(args.batch)))

        train_x, train_y, valid_x, valid_y = shuffle_data(images, maps)

        train_data = {
            x: train_x,
            y: train_y,
            keep_prob: 0.5
        }

        val_data = {
            x: valid_x,
            y: valid_y,
            keep_prob: 1.0
        }

        iteration_loss = 100000
        val = 0

        for i in range(0, int(args.epochs)):
            _, c = sess.run([train_step, cross_entropy], feed_dict=train_data)

            if math.isnan(c):  # too big loss
                print("died at: " + str(i))
                end = True
                break

            # print("current loss: ", c)

            # writing trainig loss to log file
            summary_str = sess.run(summary, feed_dict=train_data)
            summary_writer.add_summary(summary_str, i)
            summary_writer.flush()


            val = sess.run(validation_cross_entropy, feed_dict=val_data)
            # print("validation loss: ", val)

            # writing validation loss to log file
            summary_str = sess.run(summary, feed_dict=val_data)
            validation_writer.add_summary(summary_str, i)
            validation_writer.flush()


            if i != 0 and i%5 == 0:
                #val = sess.run(validation_cross_entropy, feed_dict=val_data)
                print("current loss: ", c)
                print("validation loss: ", val)

            # if validation loss starts rising, stop training
            if iteration_loss < val:
                print("end of trainig, loss starts rising")

                end = True
                break
            iteration_loss = val

        saver.save(sess, args.model+"trained_model.model")

train_neural_network()
