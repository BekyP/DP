import cv2
import numpy as np

from utils import listdir_fullpath


def load_data(folder, first=0, last=None, read_flag=cv2.IMREAD_UNCHANGED):
    return list(map(lambda x: cv2.imread(x, read_flag) / 255, sorted(listdir_fullpath(folder))[first:last]))


def load_data(folder, size, first=0, last=None, read_flag=cv2.IMREAD_UNCHANGED):
    return list(map(lambda x: cv2.resize(cv2.imread(x, read_flag), size) / 255,
                    sorted(listdir_fullpath(folder))[first:last]))
