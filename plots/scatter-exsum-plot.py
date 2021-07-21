#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
import sys
import csv
import statistics
import scipy.stats as stats
from adjustText import adjust_text
import numpy as np
from numpy.random import *

FILEPATH_PERF = "../acda_data/3D-exsum-double-performance.csv"
FILEPATH_SPACE = "../acda_data/3D-exsum-double-space.csv"

plt.rcParams["figure.figsize"] = (10, 8)

plt.rcParams.update({'font.size': 20})

ax = plt.gca()

ax.set_facecolor('whitesmoke')

size_of_doubles = 8

multiplier = 1000

def divide(y_data, x_data):
    out = []
    for i in range(len(x_data)):
        out.append(y_data[i]/ float(x_data[i]))
    return out

with open(FILEPATH_PERF, 'r') as csvfile:
    data = csv.reader(csvfile, delimiter=',')
    next(data)
    x = [float(i) for i in next(data)[1:]]
    perf = []
    for row in data:
        perf.append([float(i) * multiplier for i in row[1:]])

with open(FILEPATH_SPACE, 'r') as csvfile:
    data = csv.reader(csvfile, delimiter=',')
    next(data)
    x = [float(i) for i in next(data)[1:]]
    space = []
    for row in data:
        space.append([float(i) for i in row[1:]])


print (perf)
print (space)

assert len(space) == len(perf)

normalized_space = []
normalized_perf = []

byte_size = [size_of_doubles] * len(x)

for alg in perf:
    normalized_perf.append(divide(alg, x))
for alg in space:
    normed = divide(alg, x)
    normalized_space.append(divide(normed, byte_size))

plt.rcParams.update({'lines.markersize': 20})
#plt.xscale('log')
#plt.yscale('log')
plt.grid()

# x index
index = 5


NAIVE = [normalized_space[0][index], normalized_perf[0][index]]
SUMMED_AREA = [normalized_space[1][index], normalized_perf[1][index]]
CORNERS_SPINE = [normalized_space[2][index], normalized_perf[2][index]]
CORNERS_LEAF_1 = [normalized_space[3][index], normalized_perf[3][index]]
CORNERS_LEAF_2 = [normalized_space[4][index], normalized_perf[4][index]]
CORNERS_LEAF_4 = [normalized_space[5][index], normalized_perf[5][index]]
CORNERS_LEAF_8 = [normalized_space[6][index], normalized_perf[6][index]]
BOX_COMPLEMENT = [normalized_space[7][index], normalized_perf[7][index]]

colors = []
markers = []
texts = []

#plt.scatter(NAIVE[0], NAIVE[1],  c = 'y', marker = '^', label = 'NAIVE')
#texts.append(plt.text(NAIVE[0], NAIVE[1], 'Naive', ha='center', va='center'))

#plt.scatter(SUMMED_AREA[0], SUMMED_AREA[1],  c = 'b', marker = 'p', label = 'SUMMED_AREA')
#texts.append(plt.text(SUMMED_AREA[0], SUMMED_AREA[1], 'Summed_Area', ha='center', va='center'))

plt.scatter(CORNERS_SPINE[0], CORNERS_SPINE[1],  c = 'orange', marker = 'P', label = 'CORNERS_SPINE')
texts.append(plt.text(CORNERS_SPINE[0], CORNERS_SPINE[1], 'Corners spine', ha='center', va='center'))

plt.scatter(CORNERS_LEAF_1[0], CORNERS_LEAF_1[1],  c = 'maroon', marker = "X", label = 'CORNERS_LEAF_1')
texts.append(plt.text(CORNERS_LEAF_1[0], CORNERS_LEAF_1[1], 'Corners (1)', ha='center', va='center'))

plt.scatter(CORNERS_LEAF_2[0], CORNERS_LEAF_2[1],  c = 'red', marker = "X", label = 'CORNERS_LEAF_2')
texts.append(plt.text(CORNERS_LEAF_2[0], CORNERS_LEAF_2[1], 'Corners (2)', ha='center', va='center'))

plt.scatter(CORNERS_LEAF_4[0], CORNERS_LEAF_4[1],  c = 'tomato', marker = "X", label = 'CORNERS_LEAF_4')
texts.append(plt.text(CORNERS_LEAF_4[0], CORNERS_LEAF_4[1], 'Corners (4)', ha='center', va='center'))

plt.scatter(CORNERS_LEAF_8[0], CORNERS_LEAF_8[1],  c = 'lightcoral', marker = 'X', label = 'CORNERS_LEAF_8')
texts.append(plt.text(CORNERS_LEAF_8[0], CORNERS_LEAF_8[1], 'Corners (8)', ha='center', va='center'))

plt.scatter(BOX_COMPLEMENT[0], BOX_COMPLEMENT[1],  c = 'darkgreen', marker = 's', label = 'BOX_COMPLEMENT')
texts.append(plt.text(BOX_COMPLEMENT[0], BOX_COMPLEMENT[1], 'Box-complement', ha='center', va='center'))

adjust_text(texts)

plt.xlabel('Space per element (in number of elements)')
plt.ylabel('Time per element (in microseconds)')

# plt.legend()
#

ax.set_ylim(ymin=0)
ax.set_xlim(xmin=0)

plt.show()
