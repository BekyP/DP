from __future__ import division
import tensorflow as tf
import numpy as np
import sys
import os
from scipy.stats import linregress
import metrics  # source codes of metrics from: https://github.com/herrlich10/saliency
from nn_model import neural_network_model
from utils import listdir_fullpath

parser = argparse.ArgumentParser(fromfile_prefix_chars='@', description='train neural network')
parser.add_argument('--images', help='folder with input images to predict', required=True)
parser.add_argument('--model', help='file with trained model', required=True)
parser.add_argument('--save_to', help='folder where predictions will be saved', required=True)


args = parser.parse_args()

x = tf.placeholder('float', shape=[None, 64, 64, 3])
y = tf.placeholder('float', shape=[None, 64, 64])
keep_prob = tf.placeholder(tf.float32)


def make_prediction(prediction, data):
    predicted_heatmap = prediction.eval(data)
    return np.array(predicted_heatmap)

def predict(heatmap_type='binary'):

    # creates model of neural network
    prediction = neural_network_model(x, keep_prob)
    cross_entropy = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=prediction, labels=y)
    )

    images_for_prediction = np.array(load_data(args.images, (64, 64)))
    images_original = sorted(listdir_fullpath(args.images))

    saver = tf.train.Saver()

    with tf.Session() as sess:
        saver.restore(sess, model)  # restores saved model

        predicted_heatmaps = make_prediction(prediction,
                                             {x: images_for_prediction,
                                              keep_prob: 1.0})

        for map, img in zip(predicted_heatmaps, images_original):
            print ("working on: " + str(img.rsplit('/', 1)[1]))
            # saving predicted heatmaps on image
            data_part.vizualize_heatmap(map, img,
                                        args.save_to,
                                        heatmap_type)

predict()