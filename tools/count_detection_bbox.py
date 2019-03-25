import sys
import os.path as osp
import os
import pickle as pkl
import numpy as np

detFile = sys.argv[1]

d = pkl.load(open(detFile,'rb'))

b = d['all_boxes']

numClass = len(b)

numImages = len(b[1])

print ("numClass = {}, numImages = {}".format(numClass, numImages))

count = 0
clsCount = np.zeros((numClass,), dtype=int)

for c in range(numClass):
    for ix in range(numImages):
        count += len(b[c][ix])
        clsCount[c] += len(b[c][ix])

print ("clsCount = {}".format(clsCount))
print ("Count = {}".format(count))

