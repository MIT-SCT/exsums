#!/usr/bin/env python3
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import sys
import csv
import scipy.stats as stats

FILEPATH = sys.argv[1]

def divide(y_data, x_data):
    out = []
    for i in range(len(x_data)):
        out.append(y_data[i]/ float(x_data[i]))
    return out

with open(FILEPATH, 'r') as csvfile:
    data = csv.reader(csvfile, delimiter=',')
    next(data)
    N = [float(i) for i in next(data)[1:]]
    algs = []
    for row in data:
        algs.append([float(i) for i in row])

def transpose_2d_list(inp):
    return [list(x) for x in zip(*inp)]

# transpose data to get input size in first row (x's), and rows be diff algs

print ("Approx Input Size: {}".format(N))

transposed_input = transpose_2d_list(algs)
x_data = transposed_input[0]
times = transposed_input[1]
algs = transposed_input[2:]

print(x_data)
print(times)
print(algs)

plt.rcParams.update({'font.size': 20})
fig = plt.figure(figsize=(10,8))

ax = fig.add_subplot(111)

ax.set_facecolor('whitesmoke')

ax.set_ylabel("Time per element (in microseconds)")
ax.set_xlabel("Dimension")
#ax.set_yscale('log')

normalized_algs = []
for alg in algs:
    alg = divide(alg, [0.001] * len(times))
    normalized_algs.append(divide(alg, times))

# ingore zeroes
normalized_algs = [np.array(alg) for alg in normalized_algs]

np_algs = []
for alg in normalized_algs:
    a = np.array(alg)
    a = np.ma.masked_where(a == 0, a)
    np_algs.append(a)

print (np_algs)

ax.set_yscale('log')
#ax.set_xticks([1000, 10000, 100000, 1000000, 10000000])
#ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
#plt.ticklabel_format(style="plain", axis='x', scilimits=(0,0))

ax.plot(x_data, np_algs[0],
        label = 'Naive included sum complement', marker = 'o', color = 'tab:purple')
ax.plot(x_data, np_algs[1],
        label = 'SATC', marker = '^', color = 'tab:olive')
ax.plot(x_data, np_algs[2],
        label = 'BDBSC', marker = 'D', color = 'tab:blue')
ax.plot(x_data, np_algs[3],
        label = 'Box-complement', marker = 's', color = 'darkgreen')

plt.legend()
plt.grid()
plt.show()

