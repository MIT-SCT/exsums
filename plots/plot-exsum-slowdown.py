#!/usr/bin/env python3
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import sys
import csv
import scipy.stats as stats
from collections import defaultdict

def fib(n):
    if n <= 1:
        return n
    else:
        return fib(n-1) + fib(n-2)

def preprocess(start_fib, end_fib, increment):
  algs = defaultdict(list)
  x_data = []
  fib_times = []
  input_size = []
  for fib_arg in range(start_fib, end_fib + increment, increment):
    inp_file = "../acda_data/3D-exsum-double-performance-{}.csv".format(fib_arg)
    x_data.append(float(fib_arg))
    with open(inp_file, 'r') as csvfile:
        data = csv.reader(csvfile, delimiter=',')
        next(data)
        second_row = next(data)
        x = float(second_row[1])
        fib_time = float(second_row[2])
        input_size.append(x)
        fib_times.append(fib_time * 1000 * 1000)
        for row in data:
          key = row[0]
          value = float(row[1]) * 1000 * 1000
          algs[key].append(value)
  return algs, x_data, input_size, fib_times


algs, x_data, input_size, fib_times = preprocess(2, 20, 2)

print (algs)
print(x_data)
print(input_size)
print ("fib times: =============")
print(fib_times)

def divide(y_data, x_data):
    out = []
    for i in range(len(x_data)):
        out.append(y_data[i]/ float(x_data[i]))
    return out

plt.rcParams.update({'font.size': 20})
fig = plt.figure(figsize=(10, 8))

plt.ylim(2**4, 2**9)
plt.xlim(2**3, 2**17)

ax = fig.add_subplot(111)

ax.set_ylabel("Time per element / Time per $\oplus$")
ax.set_xlabel("Time per $\oplus$ (in ns)")
ax.set_yscale('log', basey=2)
ax.set_xscale('log', basex=2)

ax.set_facecolor('whitesmoke')

normalization = [fib(x) for x in x_data]
print ("normalization")
print (normalization)

div_algs = []
for y in algs:
    print(algs[y])
    algs[y] = divide(algs[y], fib_times)
    div_algs.append(divide(algs[y], input_size))

#ax.set_xticks([2**3, 2**5, 2**7, 2**9, 2**11, 2**13, 2**15, 2**17])
#ax.set_yticks([2**4, 2**5, 2**6, 2**7, 2**8, 2**9])
#ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
#ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
#plt.ticklabel_format(style="sci", axis='x', scilimits=(0,0))
#plt.ticklabel_format(style="sci", axis='y', scilimits=(0,0))

#ax.plot(x_data, div_algs[0], label = 'Naive', marker = '^', color = 'y')
#ax.plot(x_data, div_algs[1], label = 'Summed-Area Table', marker = 's', color = 'b')
ax.plot(fib_times, div_algs[2], label = 'Corners spine', marker = 'P', color = 'orange')
ax.plot(fib_times, div_algs[3], label = 'Corners (1)', marker = 'o', color = 'maroon')
ax.plot(fib_times, div_algs[4], label = 'Corners (2)', marker = '^', color = 'red')
ax.plot(fib_times, div_algs[5], label = 'Corners (4)', marker = 'd', color = 'tomato')
ax.plot(fib_times, div_algs[6], label = 'Corners (8)', marker = 'X', color = 'lightcoral')
ax.plot(fib_times, div_algs[7], label = 'Box-complement', marker = 's', color = 'darkgreen')
plt.legend()
plt.grid()
plt.show()

