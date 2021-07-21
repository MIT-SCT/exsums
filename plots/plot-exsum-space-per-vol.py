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
    x = [float(i) for i in next(data)[2:]]
    algs = []
    for row in data:
        algs.append([float(i) for i in row[2:]])

print(x)
print(algs)

plt.rcParams.update({'font.size': 20})
fig = plt.figure(figsize=(10,10))

ax = fig.add_subplot(111)

ax.set_ylabel("Space per element (in number of elements)")
ax.set_xlabel("Input Size (N)")
ax.set_xscale('log')
ax.set_facecolor('whitesmoke')

doubles_size_in_bytes = [8] * len(x)

div_algs = []
for y in algs:
    processed = divide(y,x)
    processed_in_num_doubles = divide(processed, doubles_size_in_bytes)
    div_algs.append(processed_in_num_doubles)
    print("======")
    print(y)
    print(x)
    print(processed)
    print(processed_in_num_doubles)

# ax.set_yscale('log')
#ax.set_xticks([1000, 10000, 100000, 1000000, 10000000])
#ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
#plt.ticklabel_format(style="plain", axis='x', scilimits=(0,0))

#ax.plot(x, div_algs[0], label = 'Naive', marker = '^', color = 'y')
#ax.plot(x, div_algs[1], label = 'Summed-Area Table', marker = 's', color = 'b')
ax.plot(x, div_algs[2], label = 'Corners Spine', marker = 'P', color = 'orange')
ax.plot(x, div_algs[3], label = 'Corners (1)', marker = 'X', color = 'maroon')
ax.plot(x, div_algs[4], label = 'Corners (2)', marker = 'X', color = 'red')
ax.plot(x, div_algs[5], label = 'Corners (4)', marker = 'X', color = 'tomato')
ax.plot(x, div_algs[6], label = 'Corners (8)', marker = 'X', color = 'lightcoral')
ax.plot(x, div_algs[7], label = 'Box-Complement', marker = 's', color = 'darkgreen')
#ax.plot(x, div_algs[8], label = 'BDBS', marker = 'X', color = 'm')
plt.legend()
plt.grid()
plt.show()
plt.savefig(sys.argv[1][:-4] + ".png")
