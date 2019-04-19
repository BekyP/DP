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
from nn_utils.utils import listdir_fullpath, count_metrics
from nn_utils.load_data import load_data
from nn_utils.binary_map_with_fixations import get_binary_fixation_maps

import matplotlib
import cv2
matplotlib.use('Agg')

parser = argparse.ArgumentParser(fromfile_prefix_chars='@', description='train neural network')
parser.add_argument('--images', help='folder with input images', required=True)
parser.add_argument('--maps', help='folder with heatmaps', required=True)
parser.add_argument('--binary_maps', help='folder with binary fixations maps', required=True)
parser.add_argument('--binary_format', help='binary maps format, mat or jpg', default='jpg')
parser.add_argument('--n', help='images size n x n', type=int, default=224)
parser.add_argument('--device', help='cpu or gpu', default='gpu')
parser.add_argument('--model', required=True)
parser.add_argument('--optimizer', default="adadelta")
parser.add_argument('--loss', default="binary_crossentropy")

args = parser.parse_args()
n = int(args.n)

config = tf.ConfigProto()
session = tf.Session(config=config)
config.gpu_options.allow_growth = True
K.set_session(session)

if args.device == 'cpu':
    config = tf.ConfigProto(intra_op_parallelism_threads=12,\
        inter_op_parallelism_threads=12, allow_soft_placement=True,\
        device_count = {'CPU' : 1, 'GPU' : 0})
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)
    K.set_session(session)

#model = load_model('adadelta+binary.save')
model = load_model(args.model)

model.compile(loss=args.loss,
              optimizer=args.optimizer,
              metrics=['accuracy'])

model.summary()

imgs = np.array(load_data(args.images, (n, n), read_flag=cv2.IMREAD_COLOR))
img_names = sorted(listdir_fullpath(args.images))

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

original = np.array(load_data(args.maps, (n, n)))
if args.binary_format == 'mat':
    binary_maps = np.array(get_binary_fixation_maps(args.binary_maps, size=n))
else:
    binary_maps = np.array(load_data(args.binary_maps, (n, n)))

print("counting metrics")
count_metrics(predicted, original, binary_maps)
