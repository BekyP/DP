from __future__ import division
import tensorflow as tf
import data_part
from scipy.stats import linregress
import argparse
import matplotlib.pyplot as plt
import numpy as np

import metrics  # source codes of metrics from: https://github.com/herrlich10/saliency

from NN import neural_network_model

parser = argparse.ArgumentParser(fromfile_prefix_chars='@', description='test run with args')
parser.add_argument('--dataset', help='Path to dataset', required=True)
parser.add_argument('--model', help='Path to saved model', required=True)

args = parser.parse_args()


def print_shape(tensor):
    print(tensor.get_shape)


def make_prediction(prediction, data):
    predicted_heatmap = prediction.eval(data)
    return np.array(predicted_heatmap)


def correlation_coefficient(x, y):  # computes correlation coefficient of x, y
    cc = str(linregress(x, y))
    cc = cc.rsplit(',', 4)[2]
    cc = cc.rsplit('=', 2)[1]
    return float(cc)


def count_metrics(predicted_heatmaps, orig, num=5):  # computes metrics
    # original_heatmaps = data_part.load_heatmaps(directory + "/heatmaps/", 0, 0, num)

    cc = 0
    auc = 0
    similarity = 0
    nss = 0
    auc_s = 0

    # loads fixations necessary for metrics
    tmp = np.zeros([64, 64])
    tmp[32][32] = 1
    fixations = []
    print(tmp.shape)
    for i in range(0, 5):
        fixations.append(tmp)

    fixations = np.array(fixations)

    for map, fix, original_map in zip(predicted_heatmaps, fixations, orig):
        auc += metrics.AUC_Judd(map, fix, True)
        nss += metrics.NSS(map, fix)
        auc_s += metrics.AUC_shuffled(map, fix, np.zeros([64, 64]))
        cc += correlation_coefficient(map.flatten(), original_map.flatten())
        similarity += metrics.SIM(map, original_map)

    print("final correlation coeficient: " + str(cc / num))
    print("final SIM: " + str(similarity / num))
    print("final NSS: " + str(nss / num))
    print("final AUC: " + str(auc / num))
    print("final shuffled AUC: " + str(auc_s / num))


def predict(model,
            metrics_set=1):
    graph = tf.Graph()
    with graph.as_default():

        x = tf.placeholder('float', shape=[None, 64, 64, 3])
        y = tf.placeholder('float', shape=[None, 64, 64])
        keep_prob = tf.placeholder(tf.float32)

        # creates model of neural network
        prediction = neural_network_model(x, keep_prob)
        cross_entropy = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=prediction, labels=y)
        )

        '''
        config = tf.ConfigProto(
            device_count = {'GPU': 0}
        )
        '''

        # load names of all images in directory
        files = ['map_287_BlackWhite_021.jpeg', 'map_287_BlackWhite_033.jpeg', 'map_287_BlackWhite_035.jpeg',
                 'map_287_BlackWhite_051.jpeg', 'map_287_BlackWhite_055.jpeg']
        print(files)
        x_data, y_data = data_part.load_data_for_epoch(files, args.dataset)
        images_for_prediction = []
        images_original = []

        saver = tf.train.Saver()

        with tf.Session(graph=graph) as sess:
            saver.restore(sess, model)  # restores saved model

            predicted_heatmaps = make_prediction(prediction,
                                                 {x: x_data,
                                                  keep_prob: 1.0})

            if metrics_set == 1:
                # computes metrics
                count_metrics(predicted_heatmaps, y_data, 5)

            for map, file, img in zip(predicted_heatmaps, files, x_data):
                print("should show something")
                fig = plt.figure(frameon=False)

                ax = fig.add_subplot(1, 1, 1)
                plt.axis('off')
                plt.style.use('grayscale')

                plt.imshow(img)
                plt.imshow(map, alpha=0.45)
                plt.savefig(file)


predict("C:\\Users\\bekap\\Desktop\\diplomka\\model\\test.model")
