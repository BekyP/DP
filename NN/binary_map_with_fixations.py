import argparse
import os
import numpy as np

from scipy.io import loadmat
from scipy.misc import toimage
from cv2 import GaussianBlur
from utils import listdir_fullpath


def get_size_and_fix(data, users=[1, 2, 3, 4, 5]):
    size = data[0, 0:2]

    user_fixations = {}

    for user in users:
        user_fixations['user_' + str(user)] = data[np.where(data[:, 2] == user), 0:2][0]

    return {'size': size, **user_fixations}


def apply_fixations(original_size, fixations, map):

    for fix in fixations:
        x = int(fix[0]/original_size[0]*len(map))
        y = int(fix[1]/original_size[1]*len(map))

        map[x][y] = 1

    return map


def get_binary_fixation_maps(folder, size=256, first=0, last=None):
    ret_maps = []

    extracted_data = list(
        map(lambda x: {'file': x.rsplit('/', 1)[1].replace(".mat", ".jpg"), **get_size_and_fix(loadmat(x)['s'])},
            sorted(listdir_fullpath(folder))[first:last]))

    for data in extracted_data:
        #print("working on file: " + data['file'])
        final_map = np.zeros([size,size])

        for i in range(1, 6):
            user_key = 'user_' + str(i)
            user_fix = data[user_key]
            final_map = apply_fixations(data['size'], data[user_key], final_map)

        ret_maps.append(final_map)

    return ret_maps
