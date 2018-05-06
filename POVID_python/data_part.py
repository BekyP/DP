import numpy as np
import cv2
import scipy.io
import os


def find_1(fixations):
    for y in fixations:
        for x in y:
            if x != 0:
                print("yes" + str(x))


def load_fixation_locs(folder):
    subfolders = os.listdir(folder)
    all_fixations = {}
    for subfolder in subfolders:
        path_to_maps = os.path.join(folder, subfolder)
        print("fixation locs, working on: " + subfolder)
        for fix_map in os.listdir(path_to_maps):
            if os.path.isdir(os.path.join(path_to_maps, fix_map)):
                continue
            map_fixations = scipy.io.loadmat(os.path.join(path_to_maps, fix_map))['fixLocs']
            all_fixations[subfolder + '_' + fix_map.split('.')[0]] = np.where(
                map_fixations > 0)
    return all_fixations


def load_fixation_maps(folder):
    subfolders = os.listdir(folder)
    maps = {}
    for subfolder in subfolders:
        path_to_maps = os.path.join(folder, subfolder)
        print("fixation maps, working on: " + subfolder)
        for fix_map in os.listdir(path_to_maps):
            if os.path.isdir(os.path.join(path_to_maps, fix_map)):
                continue
            maps[subfolder + '_' + fix_map.split('.')[0]] = cv2.imread(os.path.join(path_to_maps, fix_map))
    return maps


def load_stimuli(folder):
    subfolders = os.listdir(folder)
    images = {}
    for subfolder in subfolders:
        path_to_maps = os.path.join(folder, subfolder)
        print("stimuli, working on: " + subfolder)
        for img in os.listdir(path_to_maps):
            if os.path.isdir(os.path.join(path_to_maps, img)):
                continue
            images[subfolder + '_' + img.split('.')[0]] = cv2.imread(os.path.join(path_to_maps, img))
    return images


def load_data_for_predictions(folder):
    x = []
    y = []

    subfolders = os.listdir(folder)
    for subfolder in subfolders:
        path_to_maps = os.path.join(folder, subfolder)
        print("working on: " + subfolder)
        for img_name in os.listdir(path_to_maps):
            if 'map' in img_name:
                img = cv2.imread(os.path.join(path_to_maps, img_name), 0)
            else:
                img = cv2.imread(os.path.join(path_to_maps, img_name))
            if img is None:
                print("corrupted data, throwing away: " + str(img_name))
                continue

            if not (str(img.shape) == "(64, 64)" or str(img.shape) == "(64, 64, 3)"):
                print("corrupted data, throwing away: " + str(img_name))
                continue
            if 'map' in img_name:
                y.append(np.array(img / 255))  # normalization
            else:
                x.append(np.array(img / 255))  # normalization
    return np.stack(x), np.stack(y)


def load_data_for_epoch(files, folder="C:\\Users\\bekap\\Desktop\\diplomka\\datasets\\CAT2000\\all_data"):
    x = []
    y = []

    for map_name in files:
        name = str(map_name.split('map')[1])
        print("loading: " + str(name))
        if not 'map' in map_name:
            continue
        map = cv2.imread(os.path.join(folder, map_name), 0)
        img = cv2.imread(os.path.join(folder, "stimuli" + name))
        if img is None or map is None:
            print("corrupted data, throwing away: " + str(name))
            continue

        if not (str(map.shape) == "(64, 64)" or str(img.shape) == "(64, 64, 3)"):
            print("corrupted data, throwing away: " + str(name))
            continue

        y.append(np.array(map / 255))  # normalization
        x.append(np.array(img / 255))  # normalization

    if len(x) == 0:
        return [], []

    return np.stack(x), np.stack(y)
