import tensorflow as tf
import sys
sys.path.append('../') 
from keras.applications.vgg16 import VGG16
from keras.layers import *
from keras.models import Model, Sequential
from keras.utils import plot_model
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping

from keras import backend as K
import argparse

from nn_utils.load_data import load_data 
from nn_utils.utils import get_model_memory_usage
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

for layer in small_model.layers:
	layer.trainable = False

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


#plot_model(final, to_file='model.png', show_shapes=True)

#print("saved in model.png")

images = np.array(load_data(args.images, (n, n), last=args.samples, read_flag=cv2.IMREAD_COLOR))
maps = np.array(load_data(args.maps, (n, n), last=args.samples))

split = int(0.85 * len(images))

train_images = images[:split]
train_maps = maps[:split]

valid_images = images[split:]
valid_maps = maps[split:]

print("memory usage: " + str(get_model_memory_usage(args.batch_size, final)) + " GB")

model_name = args.optimizer + "_" + args.loss + ".model"
final.fit(train_images, train_maps, epochs=args.epochs, batch_size=args.batch_size,
				shuffle=True, validation_data=(valid_images, valid_maps), 
				verbose=5, callbacks=[ModelCheckpoint("models/"+model_name,monitor='val_loss', verbose=3, save_best_only=True), 
									  TensorBoard(log_dir='logs', histogram_freq=0, write_graph=True, write_images=True), 
									  EarlyStopping(monitor='val_loss', patience=10)])
