import argparse
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from keras import backend as K
from keras.models import load_model
from scipy.misc import toimage

import metrics  # source codes of metrics from: https://github.com/herrlich10/saliency
from utils import listdir_fullpath
from load_data import load_data
from binary_map_with_fixations import get_binary_fixation_maps

import matplotlib
matplotlib.use('Agg')

parser = argparse.ArgumentParser(fromfile_prefix_chars='@', description='train neural network')
parser.add_argument('--images', help='folder with input images', required=True)
parser.add_argument('--maps', help='folder with heatmaps', required=True)
parser.add_argument('--binary_maps', help='folder with binary fixations maps', required=True)
parser.add_argument('--n', help='images size n x n', type=int, required=True)
parser.add_argument('--device', help='cpu or gpu', default='gpu')

args = parser.parse_args()
n = int(args.n)

if args.device == 'cpu':
    config = tf.ConfigProto(intra_op_parallelism_threads=12,\
        inter_op_parallelism_threads=12, allow_soft_placement=True,\
        device_count = {'CPU' : 1, 'GPU' : 0})
    session = tf.Session(config=config)
    K.set_session(session)

model = load_model('adadelta+binary.save')
#model = load_model('model.save')
model.compile(loss='binary_crossentropy',
              optimizer='adadelta',
              metrics=['accuracy'])

imgs = np.array(load_data(args.images, (n, n)))[5000:]
img_names = sorted(listdir_fullpath(args.images))[5000:]

predicted = model.predict(imgs, verbose=1)

'''
for p, img in zip(predicted, img_names):
    toimage(p).save("predicted_maps/"+str(img.rsplit('/', 1)[1]))
'''

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
        auc_s += metrics.AUC_shuffled(map, fix, np.zeros([n,n]))
        cc += metrics.CC(map, original_map)
        similarity += metrics.SIM(map, original_map)

    num=len(orig)
    print("final correlation coeficient: " + str(cc / num))
    print("final SIM: " + str(similarity / num))
    print("final NSS: " + str(nss / num))
    print("final judd AUC: " + str(auc / num))
    print("final shuffled AUC: " + str(auc_s / num))
    print("final borji AUC: " + str(auc_s / num))

original = np.array(load_data(args.maps, (n, n)))[5000:]
binary_maps = np.array(get_binary_fixation_maps(args.binary_maps, 5000))

count_metrics(predicted, original, binary_maps)
