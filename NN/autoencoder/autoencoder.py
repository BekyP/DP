import sys
import os
sys.path.append('../')

import argparse

import numpy as np

from keras.layers import Dense, Conv2D, Input, Flatten, Reshape, Convolution2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras import backend as K
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping

from nn_utils.load_data import load_data

parser = argparse.ArgumentParser(fromfile_prefix_chars='@', description='train neural network')
parser.add_argument('--images', help='folder with input images', required=True)
parser.add_argument('--maps', help='folder with saliency maps', required=True)
parser.add_argument('--batch', help='Numbers of samples for each epoch of training', type=int, default=5000)
parser.add_argument('--epoch', help='Numbers of epochs', type=int, default=500)
parser.add_argument('--n', help='Images size n x n', type=int, default=256)
parser.add_argument('--conv_layers', help='number of hidden conv layers with max pooling', type=int, default=3)

parser.add_argument('--conv_encoder_filters', help='number of filter on first', type=int, default=3)
parser.add_argument('--conv_encoder_hidden_filters', type=int, default=3)
parser.add_argument('--conv_decoder_hidden_filters', type=int, default=3)
parser.add_argument('--conv_decoder_filters', type=int, default=3)

parser.add_argument('--optimizer', default="adadelta")
parser.add_argument('--loss', default="binary_crossentropy")
args = parser.parse_args()
n = int(args.n)

input_img = Input(shape=(n, n, 3))  # 1ch=black&white, 28 x 28

x = Conv2D(filters=args.conv_encoder_filters, kernel_size=3, activation='relu', padding='same')(input_img)  # nb_filter, nb_row, nb_col
print("first conv",K.int_shape(x))
x = MaxPooling2D((2, 2), padding='same')(x)
print("first maxpool",K.int_shape(x))

resized_n = n

for i in range(0, int(args.conv_layers)):
    x = Conv2D(filters=args.conv_encoder_hidden_filters, kernel_size=3, activation='relu', padding='same')(x)
    print("hidden conv",K.int_shape(x))
    x = MaxPooling2D((2, 2), padding='same')(x)
    print("hidden maxpool",K.int_shape(x))
    resized_n /= 2

x = Flatten()(x)
resized_n = int(resized_n/2)
print("resized_n", resized_n)
encoded = Dense(resized_n*resized_n)(x)

#print("shape of encoded", K.int_shape(encoded))

x = Reshape((resized_n,resized_n,1))(encoded)
print("reshape", K.int_shape(x))

for i in range(0, int(args.conv_layers)):
    x = Conv2D(filters=args.conv_decoder_hidden_filters, kernel_size=3, activation='relu', padding='same')(x)
    print("shape c", K.int_shape(x))
    x = UpSampling2D((2, 2))(x)
    print("shape u", K.int_shape(x))

x = Conv2D(filters=args.conv_decoder_filters, kernel_size=3, activation='relu', padding='same')(x)
print("shape c", K.int_shape(x))
x = UpSampling2D((2, 2))(x)
print("shape u", K.int_shape(x))

x = Conv2D(filters=1, kernel_size=5, activation='sigmoid', padding='same')(x)
print("shape before decoded", K.int_shape(x))
decoded = Reshape((n, n))(x)
print("shape of decoded", K.int_shape(decoded))

autoencoder = Model(input_img, decoded)
#autoencoder.compile(optimizer='adadelta', loss='mean_squared_error')
autoencoder.compile(optimizer=args.optimizer, loss='mean_squared_error')

autoencoder.summary()

images = np.array(load_data(args.images, (n, n), last=int(args.batch)))
maps = np.array(load_data(args.maps, (n, n), last=int(args.batch)))

split = int(0.9 * len(images))

train_images = images[:split]
train_maps = maps[:split]

valid_images = images[split:]
valid_maps = maps[split:]

model_name = args.optimizer + "_" + args.loss + "_" + str(args.conv_encoder_filters) + "_" + str(args.conv_encoder_hidden_filters) + "_" + str(args.conv_decoder_hidden_filters) + "_" + str(args.conv_decoder_filters) + "_" + ".save"

if not os.path.exists("models"):
    os.makedirs("models")

autoencoder.fit(train_images, train_maps, epochs=int(args.epoch), batch_size=500,
                shuffle=True, validation_data=(valid_images, valid_maps), verbose=5, callbacks=[ModelCheckpoint("models/"+model_name,monitor='val_loss', verbose=3, save_best_only=True), TensorBoard(log_dir='logs', histogram_freq=0, write_graph=True, write_images=True), EarlyStopping(monitor='val_loss', patience=10)])

