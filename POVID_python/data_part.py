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
        print("working on: " + subfolder)
        for fix_map in os.listdir(path_to_maps):
            if os.path.isdir(os.path.join(path_to_maps, fix_map)):
                continue
            # print("\t\t\t\t" + fix_map)
            map_fixations = scipy.io.loadmat(os.path.join(path_to_maps, fix_map))['fixLocs']
            all_fixations[subfolder + '_' + fix_map.split('.')[0]] = np.where(
                map_fixations > 0)  # TODO zistit v akom poradi to vracia
    return all_fixations


def load_fixation_maps(folder):
    subfolders = os.listdir(folder)
    maps = {}
    for subfolder in subfolders:
        path_to_maps = os.path.join(folder, subfolder)
        print("working on: " + subfolder)
        for fix_map in os.listdir(path_to_maps):
            if os.path.isdir(os.path.join(path_to_maps, fix_map)):
                continue
            # print("\t\t\t\t" + fix_map)
            maps[subfolder + '_' + fix_map.split('.')[0]] = cv2.imread(os.path.join(path_to_maps, fix_map))
    return maps


def load_stimuli(folder):
    subfolders = os.listdir(folder)
    images = {}
    for subfolder in subfolders:
        path_to_maps = os.path.join(folder, subfolder)
        print("working on: " + subfolder)
        for img in os.listdir(path_to_maps):
            if os.path.isdir(os.path.join(path_to_maps, img)):
                continue
            # print("\t\t\t\t" + img)
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
            print(img_name)
            if 'map' in img_name:
                img = cv2.imread(os.path.join(path_to_maps, img_name), 0)
            else:
                img = cv2.imread(os.path.join(path_to_maps, img_name))
            if img is None:
                continue

            if not (str(img.shape) == "(64, 64)" or str(img.shape) == "(64, 64, 3)"):
                print("wrong")
                continue
            if 'map' in img_name:
                y.append(np.array(img / 255))
            else:
                x.append(np.array(img / 255))
        # break  # TODO dat do pice
    return np.stack(x), np.stack(y)


def load_data_for_epoch(files, folder="C:\\Users\\bekap\\Desktop\\diplomka\\datasets\\CAT2000\\all_data"):
    x = []
    y = []

    for map_name in files:
        #print(map_name)
        if not 'map' in map_name:
            continue
        map = cv2.imread(os.path.join(folder, map_name), 0)
        img = cv2.imread(os.path.join(folder, "stimuli"+map_name.split('map')[1]))
        if img is None or map is None:
            continue

        if not (str(map.shape) == "(64, 64)" or str(img.shape) == "(64, 64, 3)"):
            print("wrong")
            continue

        y.append(np.array(map / 255))
        x.append(np.array(img / 255))

    if len(x) == 0:
        return [], []

    return np.stack(x), np.stack(y)

# x, y = load_data_for_predictions("C:\\Users\\bekap\\Desktop\\diplomka\\datasets\\CAT2000\\roi")
'''
all_files = os.listdir("C:\\Users\\bekap\\Desktop\\diplomka\\datasets\\CAT2000\\all_data")
print("map" in all_files[0:100])
x, y = load_data_for_epoch(all_files[0:100])
print(x.shape)
print(y.shape)

exit()
'''
# print(np.array(y).shape)

'''
locs = load_fixation_locs("C:\\Users\\bekap\\Desktop\\diplomka\\datasets\\CAT2000\\FIXATIONLOCS")
maps = load_fixation_maps("C:\\Users\\bekap\\Desktop\\diplomka\\datasets\\CAT2000\\FIXATIONMAPS")
stimuli = load_stimuli("C:\\Users\\bekap\\Desktop\\diplomka\\datasets\\CAT2000\\Stimuli")

roi_maps = {}
roi_stimuli = {}

for key, values in locs.items():
    roi_maps[key] = []
    roi_stimuli[key] = []
    for i in range(0, len(values[0])):
        roi_maps[key].append(maps[key][values[0][i] - 32:values[0][i] + 32, values[1][i] - 32:values[1][i] + 32])
        roi_stimuli[key].append(stimuli[key][values[0][i] - 32:values[0][i] + 32, values[1][i] - 32:values[1][i] + 32])

for key in roi_maps.keys():
    print(key)
    i = 0
    for stimuli, map in zip(roi_stimuli[key], roi_maps[key]):
        i += 1
        dir = "C:\\Users\\bekap\\Desktop\\diplomka\\datasets\\CAT2000\\all_data\\"
        if not os.path.exists(dir):
            os.makedirs(dir)
        map_path = os.path.join(dir, "map_" + str(i) + "_" + key + ".jpeg")
        if not os.path.exists(map_path):
            cv2.imwrite(map_path, map)
        stimuli_path = os.path.join(dir, "stimuli_" + str(i) + "_" + key + '.jpeg')
        if not os.path.exists(stimuli_path):
            cv2.imwrite(stimuli_path, stimuli)
'''