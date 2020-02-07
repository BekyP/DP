import sys
sys.path.append('./')

from keras.applications import VGG16
from vis.utils import utils
from keras import activations
from keras.utils import CustomObjectScope
from keras.initializers import glorot_uniform

n = 224

# Build the VGG16 network with ImageNet weights
model = VGG16(weights='imagenet', include_top=True)

# Utility to search for layer index by name. 
# Alternatively we can specify this as -1 since it corresponds to the last layer.
layer_idx = utils.find_layer_idx(model, 'predictions')

# Swap softmax with linear
model.layers[layer_idx].activation = activations.linear

with CustomObjectScope({'GlorotUniform': glorot_uniform()}):
    model = utils.apply_modifications(model)


from matplotlib import pyplot as plt
from vis.visualization import visualize_saliency, overlay
import numpy as np
import matplotlib.cm as cm
from vis.visualization import visualize_cam
import argparse
import cv2
from scipy.misc import toimage


import nn_utils.metrics as metrics
from nn_utils.utils import listdir_fullpath
from nn_utils.load_data import load_data
from nn_utils.binary_map_with_fixations import get_binary_fixation_maps

parser = argparse.ArgumentParser(fromfile_prefix_chars='@', description='train neural network')
parser.add_argument('--images', help='folder with input images', required=True)
parser.add_argument('--maps', help='folder with heatmaps', required=True)
parser.add_argument('--binary_maps', help='folder with binary fixations maps', required=True)
parser.add_argument('--binary_format', help='binary maps format, mat or jpg', default='jpg')

args = parser.parse_args()

imgs = np.array(load_data(args.images, (n, n), read_flag=cv2.IMREAD_COLOR))
img_names = sorted(listdir_fullpath(args.images))

def rgb2gray(rgb):

    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray

predicted = []

for modifier in ['relu']: #[None, 'guided', 'relu']:
    i = 0
    print(modifier)
    for img, img_name in zip(imgs, img_names):    
        # 20 is the imagenet index corresponding to `ouzel`
        grads = visualize_cam(model, layer_idx, filter_indices=20, 
                              seed_input=img, backprop_modifier=modifier)      
        grayscale = rgb2gray(grads)*(-1)

        if i < 50:
            toimage(grayscale).save("predicted_maps/backprop_change/"+str(modifier)+"/"+str(img_name.rsplit('/', 1)[1]))
        i += 1

        if modifier == 'relu':
            predicted.append(grayscale)

        if i >300:
            break

print("done with heatmaps")

def count_metrics(predicted_heatmaps, orig, binary_maps):  # computes metrics
    cc = 0
    auc = 0
    similarity = 0
    nss = 0
    auc_s = 0
    auc_b = 0

    for map, original_map, fix in zip(predicted_heatmaps, orig, binary_maps):
        auc += metrics.AUC_Judd(map, fix, True)
        #auc_b += metrics.AUC_Borji(map, fix)
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

original = np.array(load_data(args.maps, (n, n)))

if args.binary_format == 'mat':
    binary_maps = np.array(get_binary_fixation_maps(args.binary_maps, size=n))
else:
    binary_maps = np.array(load_data(args.binary_maps, (n, n)))

print("counting metrics")
count_metrics(np.array(predicted[:100]), original[:100], binary_maps[:100])
