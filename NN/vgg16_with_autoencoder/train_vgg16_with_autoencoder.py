import tensorflow as tf
import sys
sys.path.append('../') 
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping
from tensorflow.keras import backend as K

from math import ceil

import argparse

from nn_utils.load_data import load_data, load_images_and_maps
from nn_utils.utils import listdir_fullpath, get_model_memory_usage
import cv2

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
K.set_session(session)

parser = argparse.ArgumentParser(fromfile_prefix_chars='@', description='train neural network')
parser.add_argument('--images', help='folder with input images', required=True)
parser.add_argument('--maps', help='folder with saliency maps', required=True)
parser.add_argument('--loss', help='loss function', required=True)
parser.add_argument('--optimizer', help='optimizer', required=True)
parser.add_argument('--conv_layers', help='number of conv layers', type=int, default=3)
parser.add_argument('--batch_size', help='batch size', type=int, default=10)
parser.add_argument('--epochs', help='number of epochs', type=int, default=500)
parser.add_argument('--samples', help='number of samples', type=int, default=5000)

args = parser.parse_args()

def setup_ds(orig_ds, type="train"):
	ds = orig_ds.cache(filename='./'+type+'-cache.tf-data')
	ds = ds.apply(
	  tf.data.experimental.shuffle_and_repeat(buffer_size=8000))
	ds = ds.batch(args.batch_size).prefetch(1)

	print('image shape: ', ds.output_shapes[0])
	print('label shape: ', ds.output_shapes[1])
	print('types: ', ds.output_types)
	print()
	print(ds)

	return ds


conv_layers = args.conv_layers

input_layer=Input(shape=(224, 224, 3))

model = VGG16(input_tensor=input_layer)

#model.summary()

n = 224

x = Conv2D(filters=3, kernel_size=3, activation='relu', padding='same')(input_layer)
x = MaxPooling2D((2, 2), padding='same')(x)  

resized_n = n 

for i in range(0, conv_layers-1):
    x = Conv2D(filters=3, kernel_size=3, activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    resized_n /= 2

x = Flatten()(x)
resized_n = int(resized_n/2)
encoded = Dense(resized_n*resized_n)(x)
x = Reshape((resized_n,resized_n,1))(encoded)
x = Conv2D(filters=3, kernel_size=3, activation='relu', padding='same')(x)

autoencoder = Model(input_layer, x)

autoencoder.summary()

small_model= Sequential()

vgg16_layers_num = 6+(conv_layers-1)*4

for layer in model.layers[:vgg16_layers_num]:
    small_model.add(layer)

for i in range(0, len(model.layers)-vgg16_layers_num):
	model.layers.pop()

#for layer in small_model.layers:
	#layer.trainable = False

model=None
small_model.summary()

x = Concatenate()([small_model.output, autoencoder.output])

print(K.int_shape(x))

for i in range(0, conv_layers):
    x = Conv2D(filters=3, kernel_size=3, activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x) 

#x = Conv2D(filters=3, kernel_size=3, activation='relu', padding='same')(x)
#x = UpSampling2D((2, 2))(x)

x = Conv2D(filters=1, kernel_size=5, activation='sigmoid', padding='same')(x)
print(K.int_shape(x))

x = Reshape((n, n))(x)

final = Model(input_layer, x) 
small_model = None
autoencoder = None

final.compile(optimizer=args.optimizer, loss=args.loss)
final.summary()

split = int(0.85 * args.samples)

imgs_paths = sorted(listdir_fullpath(args.images))
maps_paths = sorted(listdir_fullpath(args.maps))

train_imgs_paths = imgs_paths[:split]
train_maps_paths = maps_paths[:split]

valid_imgs_paths = imgs_paths[split:]
valid_maps_paths = maps_paths[split:]

print("number of train samples: " + str(len(train_imgs_paths)))
print("number of valid samples: " + str(len(valid_imgs_paths)))

train_img_map_ds = load_images_and_maps(train_imgs_paths, train_maps_paths, (n, n))
valid_img_map_ds = load_images_and_maps(valid_imgs_paths, valid_maps_paths, (n, n))

train_ds = setup_ds(train_img_map_ds, "train")
valid_ds = setup_ds(valid_img_map_ds, "valid")

#train_image_batch, train_map_batch = next(iter(train_ds))
#valid_image_batch, valid_map_batch = next(iter(valid_ds))

steps_per_epoch=ceil(len(train_imgs_paths)/args.batch_size)

print("steps_per_epoch: " + str(steps_per_epoch))

#print("memory usage: " + str(get_model_memory_usage(args.batch_size, final)) + " GB")

model_name = args.optimizer + "_" + args.loss + ".model"
final.fit(train_ds, epochs=args.epochs, steps_per_epoch=steps_per_epoch, validation_data=valid_ds, 
				verbose=5, callbacks=[ModelCheckpoint("models_all_train/"+model_name,monitor='val_loss', verbose=3, save_best_only=True), 
									  TensorBoard(log_dir='logs/'+model_name, histogram_freq=0, write_graph=True, write_images=True), 
									  EarlyStopping(monitor='val_loss', patience=10)])
