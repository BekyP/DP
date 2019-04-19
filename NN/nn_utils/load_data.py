from __future__ import absolute_import, division, print_function
import tensorflow as tf

import sys
sys.path.append('../') 

import cv2
import numpy as np

from nn_utils.utils import listdir_fullpath

AUTOTUNE = tf.data.experimental.AUTOTUNE

def load_data(folder, first=0, last=None, read_flag=cv2.IMREAD_UNCHANGED):
	return list(map(lambda x: cv2.imread(x, read_flag) / 255, sorted(listdir_fullpath(folder))[first:last]))


def load_data(folder, size, first=0, last=None, read_flag=cv2.IMREAD_UNCHANGED):
	return list(map(lambda x: cv2.resize(cv2.imread(x, read_flag), size) / 255,
					sorted(listdir_fullpath(folder))[first:last]))

def preprocess_image(image, channels, size=(224, 224)):
	image = tf.image.decode_jpeg(image, channels=channels)
	image = tf.image.resize_images(image, size)
	image /= 255.0  # normalize to [0,1] range

	if channels == 1:
		image = tf.reshape(image, size)

	return image

def load_and_preprocess_image(path, channels, size=(224, 224)):
	image = tf.read_file(path)
	return preprocess_image(image, channels)

def create_img_map_pairs(img_path, map_path, size=(224, 224)):
	return load_and_preprocess_image(img_path, 3), load_and_preprocess_image(map_path, 1)

def load_images_and_maps(imgs_paths, maps_paths, split, size=(224, 224)):
	ds = tf.data.Dataset.from_tensor_slices((imgs_paths, maps_paths))

	img_map_ds = ds.map(lambda img_path, map_path: create_img_map_pairs(img_path, map_path, size), 
		num_parallel_calls=AUTOTUNE)

	return img_map_ds