import tensorflow as tf
import sys
sys.path.append('../') 

import argparse
import matplotlib.pyplot as plt
import numpy as np
import os

from keras import backend as K
from keras.models import load_model
from scipy.misc import toimage

import nn_utils.metrics as metrics # source codes of metrics from: https://github.com/herrlich10/saliency
from nn_utils.utils import listdir_fullpath
from nn_utils.load_data import load_data
from nn_utils.binary_map_with_fixations import get_binary_fixation_maps

import matplotlib
matplotlib.use('Agg')

parser = argparse.ArgumentParser(fromfile_prefix_chars='@', description='train neural network')
parser.add_argument('--images', help='folder with input images', required=True)
parser.add_argument('--maps', help='folder with heatmaps', required=True)
parser.add_argument('--binary_maps', help='folder with binary fixations maps', required=True)
parser.add_argument('--n', help='images size n x n', type=int, required=True)
parser.add_argument('--device', help='cpu or gpu', default='gpu')
parser.add_argument('--model', required=True)

args = parser.parse_args()
n = int(args.n)

if args.device == 'cpu':
    config = tf.ConfigProto(intra_op_parallelism_threads=12,\
        inter_op_parallelism_threads=12, allow_soft_placement=True,\
        device_count = {'CPU' : 1, 'GPU' : 0})
    session = tf.Session(config=config)
    K.set_session(session)

#model = load_model('adadelta+binary.save')
model = load_model(args.model)

optimizer=args.model.rsplit("/", 2)[-1].split('_1',2)[0].split('_',1)[0]
loss=args.model.rsplit("/", 2)[-1].split('_1',2)[0].split('_', 1)[1]

print(optimizer, loss)


model.compile(loss=loss,
              optimizer=optimizer,
              metrics=['accuracy'])

model.summary()

imgs = np.array(load_data(args.images, (n, n),first=5000))
img_names = sorted(listdir_fullpath(args.images))[5000:]

predicted = model.predict(imgs, verbose=1)

i = 0
if not os.path.exists("predicted_maps"):  
    os.makedirs("predicted_maps")
for p, img in zip(predicted, img_names):
    i += 1
    #print(str(i) + ". saving " + str(img.rsplit('/', 1)[1]))
    plt.imshow(p, cmap='jet')
    if not os.path.exists("predicted_maps/"+args.model.rsplit("/", 2)[-1]):
        os.makedirs("predicted_maps/"+args.model.rsplit("/", 2)[-1])
    
    toimage(p).save("predicted_maps/"+args.model.rsplit("/", 2)[-1]+"/"+str(img.rsplit('/', 1)[1]))
    plt.savefig("predicted_maps/"+args.model.rsplit("/", 2)[-1]+"/plot_"+str(img.rsplit('/', 1)[1]))
    if i == 50:
        break

print("saved")
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

original = np.array(load_data(args.maps, (n, n),first=5000))
binary_maps = np.array(get_binary_fixation_maps(args.binary_maps,size=n,first=5000))
print("counting metrics")
count_metrics(predicted, original, binary_maps)
