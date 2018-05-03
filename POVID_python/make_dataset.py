import os
import argparse
import cv2

from data_part import load_stimuli, load_fixation_maps, load_fixation_locs

parser = argparse.ArgumentParser(fromfile_prefix_chars='@', description='test run with args')
parser.add_argument('--dataset', help='Path to dataset', required=True)

args = parser.parse_args()

# loading data
locs = load_fixation_locs(os.path.join(args.dataset, "FIXATIONLOCS"))
maps = load_fixation_maps(os.path.join(args.dataset, "FIXATIONMAPS"))
stimuli = load_stimuli(os.path.join(args.dataset, "Stimuli"))

roi_maps = {}
roi_stimuli = {}

for key, values in locs.items():
    roi_maps[key] = []
    roi_stimuli[key] = []
    for i in range(0, len(values[0])):
        # getting region of interests (ROI) around fixations
        roi_maps[key].append(maps[key][values[0][i] - 32:values[0][i] + 32, values[1][i] - 32:values[1][i] + 32])
        roi_stimuli[key].append(stimuli[key][values[0][i] - 32:values[0][i] + 32, values[1][i] - 32:values[1][i] + 32])

# directory where created dataset will be stored
dir = os.path.join(args.dataset, "all_data")

for key in roi_maps.keys():
    print("storing roi for: "+str(key))
    i = 0
    for stimuli, map in zip(roi_stimuli[key], roi_maps[key]):
        i += 1
        if not os.path.exists(dir):
            os.makedirs(dir)
        map_path = os.path.join(dir, "map_" + str(i) + "_" + key + ".jpeg")
        if not os.path.exists(map_path):
            cv2.imwrite(map_path, map)
        stimuli_path = os.path.join(dir, "stimuli_" + str(i) + "_" + key + '.jpeg')
        if not os.path.exists(stimuli_path):
            cv2.imwrite(stimuli_path, stimuli)
