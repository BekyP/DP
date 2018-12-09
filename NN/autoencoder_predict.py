import argparse
import matplotlib.pyplot as plt
import numpy as np

from keras.models import load_model

from load_data import load_data

import matplotlib
matplotlib.use('Agg')

parser = argparse.ArgumentParser(fromfile_prefix_chars='@', description='train neural network')
parser.add_argument('--images', help='folder with input images', required=True)
parser.add_argument('--n', help='images size n x n', required=True)

args = parser.parse_args()
n = int(args.n)

#model = load_model('adadelta+binary.save')
model = load_model('model.save')
model.compile(loss='binary_crossentropy',
              optimizer='adadelta',
              metrics=['accuracy'])

imgs = np.array(load_data(args.images, (n, n), 5100))[5050:]

predicted = model.predict(imgs, verbose=1)

i=0
for p in predicted:
    plt.axis('off')

    plt.imshow(p, cmap='jet')
    plt.xlim(0,n)
    plt.ylim(0,n)
    plt.savefig("predicted_maps/"+str(i)+".png")  # saving small heatmap

    i +=1
