import sys
sys.path.append('./')

from keras.applications import VGG16
from vis.utils import utils
from keras import activations
from keras.utils import CustomObjectScope
from keras.initializers import glorot_uniform

from nn_utils.load_data import load_data

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

import nn_utils.metrics as metrics

parser = argparse.ArgumentParser(fromfile_prefix_chars='@', description='train neural network')
parser.add_argument('--images', help='folder with input images', required=True)
parser.add_argument('--maps', help='folder with heatmaps', required=True)
parser.add_argument('--binary_maps', help='folder with binary fixations maps', required=True)
parser.add_argument('--binary_format', help='binary maps format, mat or jpg', default='jpg')

args = parser.parse_args()

imgs = np.array(load_data(args.images, (n, n), read_flag=cv2.IMREAD_COLOR))
img_names = sorted(listdir_fullpath(args.images))

for modifier in [None, 'guided', 'relu']:
    for img, img_name in zip(imgs[:50], img_names[:50]):    
        # 20 is the imagenet index corresponding to `ouzel`
        grads = visualize_cam(model, layer_idx, filter_indices=20, 
                              seed_input=img, backprop_modifier=modifier)        
        # Lets overlay the heatmap onto original image.    
        jet_heatmap = np.uint8(cm.jet(grads)[..., :3] * 255)

        plt.imshow(overlay(jet_heatmap, img))
        toimage(jet_heatmap).save("predicted_maps/backprop_change/"+str(img_name.rsplit('/', 1)[1]))
        plt.savefig("predicted_maps/backprop_change/overlay_"+str(img_name.rsplit('/', 1)[1]))

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

count_metrics(predicted, original, binary_maps)
