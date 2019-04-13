from salicon import SALICON
import argparse
import numpy as np
from scipy.misc import toimage

parser = argparse.ArgumentParser(fromfile_prefix_chars='@', description='train neural network')
parser.add_argument('--file', help = "salicon annotation file", required=True)
parser.add_argument('--out', help = "output directory for binary maps", required=True)

args = parser.parse_args()

salicon = SALICON(args.file)

imgIds = salicon.getImgIds();

for imgId in imgIds:
    print(imgId)
    annIds = salicon.getAnnIds(imgId)
    anns = salicon.loadAnns(annIds)
    binary_map = salicon.buildFixMap(anns, doBlur=False)
    #print(binary_map)
    toimage(binary_map).save(args.out + "/COCO_val2014_"+ "{:012d}".format(imgId) + ".jpg")
