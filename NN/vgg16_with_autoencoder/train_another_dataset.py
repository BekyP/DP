import tensorflow as tf
from keras import backend as K
from keras.models import load_model

import argparse
import matplotlib.pyplot as plt
import numpy as np

from scipy.misc import toimage
import cv2

import sys
sys.path.append('../')

from nn_utils.utils import listdir_fullpath
from nn_utils.load_data import load_data

parser = argparse.ArgumentParser(fromfile_prefix_chars='@', description='train neural network')
parser.add_argument('--images', help='folder with input images', required=True)
parser.add_argument('--maps', help='folder with heatmaps', required=True)
parser.add_argument('--loss', help='loss function', default='binary_crossentropy')
parser.add_argument('--optimizer', help='optimizer', default='adadelta')
parset.add_argument('--model', help="trained model", required=True)

args=parser.parse_args()

model = load_model(args.model)

model.compile(loss=args.loss,
              optimizer=args.optimizer,
              metrics=['accuracy'])

model.summary()
n = 224
images = np.array(load_data(args.images, (n, n), read_flag=cv2.IMREAD_COLOR))
maps = np.array(load_data(args.maps, (n, n)))

split = int(0.87 * len(images))

train_images = images[:split]
train_maps = maps[:split]

valid_images = images[split:]
valid_maps = maps[split:]

print("memory usage: " + str(get_model_memory_usage(args.batch_size, final)) + " GB")

model_name = args.model + "_fine_tuning"
final.fit(train_images, train_maps, epochs=args.epochs, batch_size=args.batch_size,
                                shuffle=True, validation_data=(valid_images, valid_maps),
                                verbose=5, callbacks=[ModelCheckpoint(model_name,monitor='val_loss', verbose=3, save_best_only=True),
                                                                          TensorBoard(log_dir='logs/', histogram_freq=0, write_graph=True, write_images=True),
                                                                          EarlyStopping(monitor='val_loss', patience=10)])
