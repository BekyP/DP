import os
import numpy as np
from skimage import img_as_float
from skimage import exposure

def get_model_memory_usage(batch_size, model):
    import numpy as np
    from keras import backend as K

    shapes_mem_count = 0
    for l in model.layers:
        single_layer_mem = 1
        for s in l.output_shape:
            if s is None:
                continue
            single_layer_mem *= s
        shapes_mem_count += single_layer_mem

    trainable_count = np.sum([K.count_params(p) for p in set(model.trainable_weights)])
    non_trainable_count = np.sum([K.count_params(p) for p in set(model.non_trainable_weights)])

    number_size = 4.0
    if K.floatx() == 'float16':
         number_size = 2.0
    if K.floatx() == 'float64':
         number_size = 8.0

    total_memory = number_size*(batch_size*shapes_mem_count + trainable_count + non_trainable_count)
    gbytes = np.round(total_memory / (1024.0 ** 3), 3)
    return gbytes

def listdir_fullpath(d):
    return [os.path.join(d, f) for f in os.listdir(d)]

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

    return cc, sim, nss, auc, auc_s, auc_b

"""
source codes from https://github.com/herrlich10/saliency
"""

# -*- coding: utf-8 -*-
'''
Helper functions
'''

def normalize(x, method='standard', axis=None):
    '''Normalizes the input with specified method.

    Parameters
    ----------
    x : array-like
    method : string, optional
        Valid values for method are:
        - 'standard': mean=0, std=1
        - 'range': min=0, max=1
        - 'sum': sum=1
    axis : int, optional
        Axis perpendicular to which array is sliced and normalized.
        If None, array is flattened and normalized.

    Returns
    -------
    res : numpy.ndarray
        Normalized array.
    '''
    # TODO: Prevent divided by zero if the map is flat
    x = np.array(x, copy=False)
    if axis is not None:
        y = np.rollaxis(x, axis).reshape([x.shape[axis], -1])
        shape = np.ones(len(x.shape))
        shape[axis] = x.shape[axis]
        if method == 'standard':
            res = (x - np.mean(y, axis=1).reshape(shape)) / np.std(y, axis=1).reshape(shape)
        elif method == 'range':
            res = (x - np.min(y, axis=1).reshape(shape)) / (np.max(y, axis=1) - np.min(y, axis=1)).reshape(shape)
        elif method == 'sum':
            res = x / np.float_(np.sum(y, axis=1).reshape(shape))
        else:
            raise ValueError('method not in {"standard", "range", "sum"}')
    else:
        if method == 'standard':
            res = (x - np.mean(x)) / np.std(x)
        elif method == 'range':
            res = (x - np.min(x)) / (np.max(x) - np.min(x))
        elif method == 'sum':
            res = x / float(np.sum(x))
        else:
            raise ValueError('method not in {"standard", "range", "sum"}')
    return res


def match_hist(image, cdf, bin_centers, nbins=256):
    '''Modify pixels of input image so that its histogram matches target image histogram, specified by:
    cdf, bin_centers = cumulative_distribution(target_image)

    Parameters
    ----------
    image : array
        Image to be transformed.
    cdf : 1D array
        Values of cumulative distribution function of the target histogram.
    bin_centers ; 1D array
        Centers of bins of the target histogram.
    nbins : int, optional
        Number of bins for image histogram.

    Returns
    -------
    out : float array
        Image array after histogram matching.

    References
    ----------
    [1] Matlab implementation histoMatch(MTX, N, X) by Simoncelli, 7/96.
    '''
    image = img_as_float(image)
    old_cdf, old_bin = exposure.cumulative_distribution(image, nbins) # Unlike [1], we didn't add small positive number to the histogram
    new_bin = np.interp(old_cdf, cdf, bin_centers)
    out = np.interp(image.ravel(), old_bin, new_bin)
    return out.reshape(image.shape)
