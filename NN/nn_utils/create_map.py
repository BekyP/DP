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


def apply_fixations(size, fixations, map=None):
    if map is None:
        map = np.zeros(size)

    for fix in fixations:
        x = fix[0]
        y = fix[1]

        map[x][y] = 1

    return map


def sum(x, a, b):
    return a + b


parser = argparse.ArgumentParser(fromfile_prefix_chars='@', description='args for creating saliency map')
parser.add_argument('--dataset', help='Path to fixations in .mat files', required=True)
parser.add_argument('--output', help='Path to folder where maps should be stored', required=True)

args = parser.parse_args()

print("loading .mat files from: " + args.dataset)

extracted_data = list(
    map(lambda x: {'file': x.rsplit('/', 1)[1].replace(".mat", ".fix.png"), **get_size_and_fix(loadmat(x)['s'])},
        listdir_fullpath(args.dataset)))

for data in extracted_data:
    print("working on file: " + data['file'])
    final_map = np.zeros(data['size'])

    for i in range(1, 6):
        user_key = 'user_' + str(i)
        user_fix = data[user_key]
        final_map = apply_fixations(data['size'], data[user_key], final_map)

    print(final_map.shape)
    final_map = GaussianBlur(final_map, (95, 95), 0)
    toimage(final_map.T).save(os.path.join(args.output, data['file']))
