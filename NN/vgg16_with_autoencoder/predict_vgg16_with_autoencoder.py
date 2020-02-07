import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model

import argparse
import matplotlib.pyplot as plt
import numpy as np

from scipy.misc import toimage
import cv2

import sys
sys.path.append('../') 

from nn_utils.metrics import count_metrics # source codes of metrics from: https://github.com/herrlich10/saliency
from nn_utils.utils import listdir_fullpath
from nn_utils.load_data import load_data
from nn_utils.binary_map_with_fixations import load_fixation_locs

import matplotlib
matplotlib.use('Agg')

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
K.set_session(session)


parser = argparse.ArgumentParser(fromfile_prefix_chars='@', description='train neural network')
parser.add_argument('--images', help='folder with input images', required=True)
parser.add_argument('--maps', help='folder with heatmaps', required=True)
parser.add_argument('--binary_maps', help='folder with binary fixations maps', required=True)
parser.add_argument('--binary_format', help='binary maps format, mat or jpg', default='jpg')
parser.add_argument('--loss', help='loss function', required=True)
parser.add_argument('--optimizer', help='optimizer', required=True)

args = parser.parse_args()

n = 224

model = load_model("models_all_train/"+args.optimizer+"_"+args.loss+".model")

model.compile(loss=args.loss,
              optimizer=args.optimizer,
              metrics=['accuracy'])

model.summary()

imgs = np.array(load_data(args.images, (n, n), read_flag=cv2.IMREAD_COLOR))
img_names = sorted(listdir_fullpath(args.images))

predicted = model.predict(imgs, verbose=1)

i = 0

for p, img in zip(predicted, img_names):
    i += 1
    print(str(i) + ". saving " + str(img.rsplit('/', 1)[1]))
    plt.imshow(p, cmap='jet')
    toimage(p).save("predicted_maps/"+str(img.rsplit('/', 1)[1]))
    plt.savefig("predicted_maps/plot_"+str(img.rsplit('/', 1)[1]))
    if i == 50:
        break

original = np.array(load_data(args.maps, (n, n)))

print(str(len(original)))

if args.binary_format == 'mat':
    binary_maps = np.array(load_fixation_locs(args.binary_maps, size=(n, n)))
else:
    binary_maps = np.array(load_data(args.binary_maps, (n, n)))

if binary_maps.any()>0:
    print("good")
else:
    print("bad")

count_metrics(predicted, original, binary_maps)
