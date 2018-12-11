from __future__ import division
from nn_model import neural_network_model
import tensorflow as tf
import numpy as np
import sys
import os
import argparse
from scipy.stats import linregress
from scipy.misc import toimage
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

#from nn_model import neural_network_model
from utils import listdir_fullpath
from load_data import load_data
from plot import visualize_heatmap
from binary_map_with_fixations import get_binary_fixation_maps
import metrics

parser = argparse.ArgumentParser(fromfile_prefix_chars='@', description='train neural network')
parser.add_argument('--images', help='folder with input images to predict', required=True)
parser.add_argument('--model', help='file with trained model', required=True)
parser.add_argument('--save_to', help='folder where predictions will be saved', required=True)
parser.add_argument('--binary_maps', help='folder with binary fixations maps', required=True)
parser.add_argument('--maps', help='folder with original maps', required=True)

args = parser.parse_args()

x = tf.placeholder('float', shape=[None, 64, 64, 3])
y = tf.placeholder('float', shape=[None, 64, 64])
keep_prob = tf.placeholder(tf.float32)

def count_metrics(predicted_heatmaps, orig, binary_maps):  # computes metrics
    cc = 0
    auc = 0
    similarity = 0
    nss = 0
    auc_s = 0
    auc_b = 0

    for map, original_map, fix in zip(predicted_heatmaps, orig, binary_maps):
        auc += metrics.AUC_Judd(map, fix, True)
        auc_b += metrics.AUC_Borji(map, fix)
        nss += metrics.NSS(map, fix)
        auc_s += metrics.AUC_shuffled(map, fix, np.zeros([64,64]))
        cc += metrics.CC(map, original_map)
        similarity += metrics.SIM(map, original_map)

    num=len(orig)
    print("final correlation coeficient: " + str(cc / num))
    print("final SIM: " + str(similarity / num))
    print("final NSS: " + str(nss / num))
    print("final judd AUC: " + str(auc / num))
    print("final shuffled AUC: " + str(auc_s / num))
    print("final borji AUC: " + str(auc_s / num))


def make_prediction(prediction, data):
    predicted_heatmap = prediction.eval(data)
    return np.array(predicted_heatmap)

def predict(heatmap_type='jet'):

    # creates model of neural network
    prediction = neural_network_model(x, keep_prob)
    cross_entropy = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=prediction, labels=y)
    )

    images_for_prediction = np.array(load_data(args.images, (64, 64), first=5000))
    images_original = sorted(listdir_fullpath(args.images))[5000:]
    original_maps = np.array(load_data(args.maps, (64, 64), first=5000))
    saver = tf.train.Saver()

    with tf.Session() as sess:
        saver.restore(sess, args.model)  # restores saved model

        predicted_heatmaps = make_prediction(prediction,
                                             {x: images_for_prediction,
                                              keep_prob: 1.0})
        
        binary_maps = np.array(get_binary_fixation_maps(args.binary_maps, size=64, first=5000))

        count_metrics(predicted_heatmaps, original_maps, binary_maps)

        i = 0
        for map, img in zip(predicted_heatmaps, images_original):
            i += 1
            #print ("working on: " + str(img.rsplit('/', 1)[1]))
            # saving predicted heatmaps on image
            visualize_heatmap(map, img, args.save_to, heatmap_type)

            i += 1
            p=map
            print(str(i) + ". saving " + str(img.rsplit('/', 1)[1]))
            plt.imshow(p, cmap='jet')
            toimage(p).save("predicted_maps/"+str(img.rsplit('/', 1)[1]))
            plt.savefig("predicted_maps/plot_"+str(img.rsplit('/', 1)[1]))
            if i == 50:
                break


predict()
