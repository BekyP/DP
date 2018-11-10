import cv2

from utils import listdir_fullpath


def load_data(folder, num_of_samples=None, read_flag=cv2.IMREAD_UNCHANGED):
    return list(map(lambda x: cv2.imread(x, read_flag) / 255, sorted(listdir_fullpath(folder))[0:num_of_samples]))
