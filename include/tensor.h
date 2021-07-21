/**
 * Copyright (c) 2020 MIT License by Helen Xu, Sean Fraser
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 **/
#pragma once
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include <cmath>
#include <iostream>
#include <random>
#include <type_traits>
#include <vector>

#include "cpu_workload.h"

using namespace std;

namespace exsum_tensor {

// Based on the 4th Edition C++ Programming Language Book
class Rand_int {
 public:
  Rand_int(int lo, int hi, unsigned int seed = 5489u)
      : re(seed), dist(lo, hi) {}
  int operator()() { return dist(re); }

 private:
  mt19937 re;
  uniform_int_distribution<> dist;
};

// float or double
template <typename T>
class Rand_real {
 public:
  Rand_real(T lo, T hi, unsigned int seed = 5489u) : re(seed), dist(lo, hi) {}
  T operator()() { return dist(re); }

 private:
  mt19937 re;
  uniform_real_distribution<T> dist;
};

// float or double
template <typename T>
class Rand_real_exp {
 public:
  Rand_real_exp(T lambda, unsigned int seed = 5489u) : re(seed), dist(lambda) {}
  T operator()() { return dist(re); }

 private:
  mt19937 re;
  exponential_distribution<T> dist;
};

// float or double
template <typename T>
class Rand_real_normal {
 public:
  Rand_real_normal(T mu, T sigma, unsigned int seed = 5489u)
      : re(seed), dist(mu, sigma) {}
  T operator()() { return dist(re); }

 private:
  mt19937 re;
  normal_distribution<T> dist;
};

// type can be either int, float, or double. N is the number of dimensions
template <typename T, size_t N>
class Tensor {
 public:
  static constexpr size_t order = N;
  typedef const vector<size_t> index_t;
  typedef vector<size_t> mod_index_t;

  Tensor() = default;
  Tensor(const Tensor&) = default;  // copy ctor
  Tensor(Tensor&&) = default;       // move ctor
  Tensor& operator=(const Tensor&) = default;
  ~Tensor() = default;

  static constexpr size_t max_dims = 10;
  // note: if you want more dims, run gray_code.py to get the two arrays and
  // put the number of dims as the arg
  size_t gray_1d[2] = {0, 0};
  size_t gray_2d[4] = {0, 0, 1, 0};
  size_t gray_3d[8] = {0, 0, 1, 0, 2, 0, 1, 0};
  size_t gray_4d[16] = {0, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0};
  size_t gray_5d[32] = {0, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0,
                        4, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0};
  size_t gray_6d[64] = {0, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0,
                        4, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0,
                        5, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0,
                        4, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0};
  size_t gray_7d[128] = {
      0, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 4, 0, 1, 0, 2, 0,
      1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 5, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0,
      2, 0, 1, 0, 4, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 6, 0,
      1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 4, 0, 1, 0, 2, 0, 1, 0,
      3, 0, 1, 0, 2, 0, 1, 0, 5, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0,
      1, 0, 4, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0};
  size_t gray_8d[256] = {
      0, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 4, 0, 1, 0, 2, 0, 1, 0,
      3, 0, 1, 0, 2, 0, 1, 0, 5, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0,
      4, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 6, 0, 1, 0, 2, 0, 1, 0,
      3, 0, 1, 0, 2, 0, 1, 0, 4, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0,
      5, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 4, 0, 1, 0, 2, 0, 1, 0,
      3, 0, 1, 0, 2, 0, 1, 0, 7, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0,
      4, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 5, 0, 1, 0, 2, 0, 1, 0,
      3, 0, 1, 0, 2, 0, 1, 0, 4, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0,
      6, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 4, 0, 1, 0, 2, 0, 1, 0,
      3, 0, 1, 0, 2, 0, 1, 0, 5, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0,
      4, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0};
  size_t gray_9d[512] = {
      0, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 4, 0, 1, 0, 2, 0, 1, 0, 3,
      0, 1, 0, 2, 0, 1, 0, 5, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 4, 0,
      1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 6, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1,
      0, 2, 0, 1, 0, 4, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 5, 0, 1, 0,
      2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 4, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2,
      0, 1, 0, 7, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 4, 0, 1, 0, 2, 0,
      1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 5, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1,
      0, 4, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 6, 0, 1, 0, 2, 0, 1, 0,
      3, 0, 1, 0, 2, 0, 1, 0, 4, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 5,
      0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 4, 0, 1, 0, 2, 0, 1, 0, 3, 0,
      1, 0, 2, 0, 1, 0, 8, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 4, 0, 1,
      0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 5, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0,
      2, 0, 1, 0, 4, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 6, 0, 1, 0, 2,
      0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 4, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0,
      1, 0, 5, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 4, 0, 1, 0, 2, 0, 1,
      0, 3, 0, 1, 0, 2, 0, 1, 0, 7, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0,
      4, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 5, 0, 1, 0, 2, 0, 1, 0, 3,
      0, 1, 0, 2, 0, 1, 0, 4, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 6, 0,
      1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 4, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1,
      0, 2, 0, 1, 0, 5, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 4, 0, 1, 0,
      2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0};
  size_t gray_10d[1024] = {
      0, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 4, 0, 1, 0, 2, 0, 1, 0, 3,
      0, 1, 0, 2, 0, 1, 0, 5, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 4, 0,
      1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 6, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1,
      0, 2, 0, 1, 0, 4, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 5, 0, 1, 0,
      2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 4, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2,
      0, 1, 0, 7, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 4, 0, 1, 0, 2, 0,
      1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 5, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1,
      0, 4, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 6, 0, 1, 0, 2, 0, 1, 0,
      3, 0, 1, 0, 2, 0, 1, 0, 4, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 5,
      0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 4, 0, 1, 0, 2, 0, 1, 0, 3, 0,
      1, 0, 2, 0, 1, 0, 8, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 4, 0, 1,
      0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 5, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0,
      2, 0, 1, 0, 4, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 6, 0, 1, 0, 2,
      0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 4, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0,
      1, 0, 5, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 4, 0, 1, 0, 2, 0, 1,
      0, 3, 0, 1, 0, 2, 0, 1, 0, 7, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0,
      4, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 5, 0, 1, 0, 2, 0, 1, 0, 3,
      0, 1, 0, 2, 0, 1, 0, 4, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 6, 0,
      1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 4, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1,
      0, 2, 0, 1, 0, 5, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 4, 0, 1, 0,
      2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 9, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2,
      0, 1, 0, 4, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 5, 0, 1, 0, 2, 0,
      1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 4, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1,
      0, 6, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 4, 0, 1, 0, 2, 0, 1, 0,
      3, 0, 1, 0, 2, 0, 1, 0, 5, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 4,
      0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 7, 0, 1, 0, 2, 0, 1, 0, 3, 0,
      1, 0, 2, 0, 1, 0, 4, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 5, 0, 1,
      0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 4, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0,
      2, 0, 1, 0, 6, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 4, 0, 1, 0, 2,
      0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 5, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0,
      1, 0, 4, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 8, 0, 1, 0, 2, 0, 1,
      0, 3, 0, 1, 0, 2, 0, 1, 0, 4, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0,
      5, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 4, 0, 1, 0, 2, 0, 1, 0, 3,
      0, 1, 0, 2, 0, 1, 0, 6, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 4, 0,
      1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 5, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1,
      0, 2, 0, 1, 0, 4, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 7, 0, 1, 0,
      2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 4, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2,
      0, 1, 0, 5, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 4, 0, 1, 0, 2, 0,
      1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 6, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1,
      0, 4, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 5, 0, 1, 0, 2, 0, 1, 0,
      3, 0, 1, 0, 2, 0, 1, 0, 4, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0};

  size_t* gray_diffs[max_dims] = {gray_1d, gray_2d, gray_3d, gray_4d, gray_5d,
                                  gray_6d, gray_7d, gray_8d, gray_9d, gray_10d};

  size_t add_or_subtract_1d[2] = {0, 1};
  size_t add_or_subtract_2d[4] = {0, 1, 1, 0};
  size_t add_or_subtract_3d[8] = {0, 1, 1, 0, 1, 1, 0, 0};
  size_t add_or_subtract_4d[16] = {0, 1, 1, 0, 1, 1, 0, 0,
                                   1, 1, 1, 0, 0, 1, 0, 0};
  size_t add_or_subtract_5d[32] = {0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1,
                                   0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 1,
                                   0, 0, 0, 1, 1, 0, 0, 1, 0, 0};
  size_t add_or_subtract_6d[64] = {
      0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 1,
      0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0,
      0, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0};
  size_t add_or_subtract_7d[128] = {
      0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 1,
      0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0,
      0, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1,
      1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0,
      0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1,
      0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0};
  size_t add_or_subtract_8d[256] = {
      0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0,
      0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0,
      0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0,
      1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0,
      0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0,
      0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0,
      1, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0,
      1, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0,
      0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0,
      0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0,
      0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0};
  size_t add_or_subtract_9d[512] = {
      0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0,
      1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 1,
      1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1,
      0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0,
      1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0,
      1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 1,
      0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0,
      0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0,
      1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0,
      1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1,
      1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1,
      0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0,
      0, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1,
      1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1,
      0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0,
      0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0,
      1, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1,
      1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1,
      1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1,
      0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0,
      1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0};

  size_t add_or_subtract_10d[1024] = {
      0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0,
      1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 1,
      1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1,
      0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0,
      1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0,
      1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 1,
      0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0,
      0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0,
      1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0,
      1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1,
      1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1,
      0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0,
      0, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1,
      1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1,
      0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0,
      0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0,
      1, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1,
      1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1,
      1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1,
      0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0,
      1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0,
      1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 1,
      0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0,
      0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0,
      0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0,
      1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1,
      1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1,
      0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0,
      0, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1,
      1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1,
      0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0,
      0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0,
      1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0,
      1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1,
      1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1,
      0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0,
      1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0,
      1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 1,
      0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0,
      0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0,
      1, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0};
  size_t* add_or_subtract[max_dims] = {add_or_subtract_1d, add_or_subtract_2d,
                                       add_or_subtract_3d, add_or_subtract_4d,
                                       add_or_subtract_5d, add_or_subtract_6d,
                                       add_or_subtract_7d, add_or_subtract_8d,
                                       add_or_subtract_9d, add_or_subtract_10d};

  explicit Tensor(const vector<size_t>& side_lens) {
    dim_lens = side_lens;
    size_t s = side_lens[0];
    for (size_t n = 1; n < N; ++n) {
      s *= dim_lens[n];
    }
    elems = vector<T>(s);
  }

  // 0: uniform(hi, lo), 1: exp(lambda), 2: normal(mu, sigma)
  void RandFill(T lo, T hi, int distr_flag = 0) {
    if constexpr (is_same<T, int>::value) {
      Rand_int ri{static_cast<int>(lo), static_cast<int>(hi)};
      for (int i = 0; i < size(); ++i) {
        elems[i] = ri();
      }
    } else {
      if (distr_flag == 0) {
        Rand_real<T> rr{static_cast<T>(lo), static_cast<T>(hi)};
        for (int i = 0; i < size(); ++i) {
          elems[i] = rr();
        }
      } else if (distr_flag == 1) {
        Rand_real_exp<T> rr{static_cast<T>(lo)};
        for (int i = 0; i < size(); ++i) {
          elems[i] = rr();
        }
      } else if (distr_flag == 2) {
        Rand_real_normal<T> rr{static_cast<T>(lo), static_cast<T>(hi)};
        for (int i = 0; i < size(); ++i) {
          elems[i] = rr();
        }
      }
    }
  }

  void ZeroFill() {
    for (int i = 0; i < size(); ++i) {
      elems[i] = 0;
    }
  }

  void Fill(T val) {
    for (int i = 0; i < size(); ++i) {
      elems[i] = val;
    }
  }

  T GetElt(const vector<size_t>& indices) {
    // convert n-dimensional indices to one dimension - row major
    size_t index = indices[0];
    for (size_t d = 1; d < N; ++d) {
      index = index * dim_lens[d] + indices[d];
    }
    return elems[index];
  }

  size_t getAddress(const vector<size_t>& indices) {
    // convert n-dimensional indices to one dimension - row major
    size_t index = indices[0];
    for (size_t d = 1; d < N; ++d) {
      index = index * dim_lens[d] + indices[d];
    }
    return index;
  }

  const vector<size_t> getMaxIndex() {
    vector<size_t> max_index = dim_lens;
    for (auto& element : max_index) element -= 1;
    return max_index;
  }

  const vector<size_t> getMinIndex() {
    vector<size_t> min_index(order);
    return min_index;
  }

  void SetElt(const vector<size_t>& indices, T val) {
    // convert n-dimensional indices to one dimension - row major
    int index = indices[0];
    for (size_t d = 1; d < N; ++d) {
      index = index * dim_lens[d] + indices[d];
    }
    elems[index] = val;
  }

  void AppendElt(const vector<size_t>& indices, T val) {
    // convert n-dimensional indices to one dimension - row major
    int index = indices[0];
    for (size_t d = 1; d < N; ++d) {
      index = index * dim_lens[d] + indices[d];
    }
#ifdef DELAY
    volatile long x = fib(DELAY);
#endif
    elems[index] += val;
  }

  // prints a 1D or 2D array to an output stream
  // prints to cout by default
  void Print(vector<size_t> indices = vector<size_t>(N, 0), size_t curr_dim = 1,
             ostream& o = cout) {
    // base case
    if (curr_dim == N) {
      size_t dim_len = dim_lens[curr_dim - 1];
      vector<size_t> index = indices;
      for (size_t i = 0; i < dim_len; ++i) {
        index[curr_dim - 1] = i;
        o << GetElt(index) << " ";
      }
      o << endl;
      return;
    }
    size_t dim_len = dim_lens[curr_dim - 1];
    vector<size_t> index = indices;
    for (size_t i = 0; i < dim_len; ++i) {
      index[curr_dim - 1] = i;
      Print(index, curr_dim + 1);
    }
    o << endl;
  }

  T total() const {
    T sum = 0;
    for (const auto& e : elems) {
      sum += e;
#ifdef DELAY
      volatile long x = fib(DELAY);
#endif
    }
    return sum;
  }

  size_t size() const { return elems.size(); }

  T* data() { return elems.data(); }  // C array style access for testing
  const T* data() const { return elems.data(); }

  bool valid_index(index_t index) {
    for (size_t d = 0; d < order; ++d) {
      if (index[d] < 0 || (index[d] >= dim_lens[d])) return false;
    }
    return true;
  }

  void prefix_linear(size_t start, size_t end, size_t stride) {
    T sum = 0;
    for (size_t i = start; i <= end; i += stride) {
#ifdef DELAY
      volatile long x = fib(DELAY);
#endif
      sum += elems[i];
      elems[i] = sum;
    }
  }

  void suffix_linear(int start, int end, int stride) {
    T sum = 0;
    for (int i = end; i >= start; i -= stride) {
#ifdef DELAY
      volatile long x = fib(DELAY);
#endif
      sum += elems[i];
      elems[i] = sum;
    }
  }

  void suffix_along_dim(size_t dim) {
    // iterate over all possible coordinates fixing first dim dimensions
    if (dim > order - 1 || dim < 0) return;
    vector<size_t> index = getMaxIndex();
    size_t stride = 1;
    for (size_t d = dim + 1; d < order; ++d) {
      index[d] = 0;
      stride *= dim_lens[d];
    }
    suffix_along_dim_helper(dim, index, stride, dim + 1);
  }

  void suffix_along_dim_helper(size_t dim, vector<size_t> index, size_t stride,
                               size_t curr_dim) {
    if (curr_dim >= order) {
      size_t dim_len = dim_lens[dim];
      vector<size_t> start_index = index;
      vector<size_t> end_index = index;
      start_index[dim] = 0;
      end_index[dim] = dim_len - 1;
      suffix_linear(getAddress(start_index), getAddress(end_index), stride);
    } else {
      vector<size_t> new_index = index;
      for (size_t i = 0; i < dim_lens[curr_dim]; ++i) {
        new_index[curr_dim] = i;
        suffix_along_dim_helper(dim, new_index, stride, curr_dim + 1);
      }
    }
  }

  void suffix_through_dim(size_t dim, size_t curr_dim = 0) {
    // iterate over all possible coordinates fixing first dim dimensions
    if (dim > order - 1 || dim < 0) return;
    vector<size_t> index = getMaxIndex();
    size_t stride = 1;
    for (size_t d = dim + 1; d < order; ++d) {
      index[d] = 0;
      stride *= dim_lens[d];
    }
    suffix_through_dim_helper(dim, index, stride, curr_dim);
  }

  void suffix_through_dim_helper(size_t dim, vector<size_t> index,
                                 size_t stride, size_t curr_dim) {
    if (curr_dim >= order) {
      size_t dim_len = dim_lens[dim];
      vector<size_t> start_index = index;
      vector<size_t> end_index = index;
      start_index[dim] = 0;
      end_index[dim] = dim_len - 1;
      suffix_linear(getAddress(start_index), getAddress(end_index), stride);
    } else {
      vector<size_t> new_index = index;
      if (dim != curr_dim) {
        for (size_t i = 0; i < dim_lens[curr_dim]; ++i) {
          new_index[curr_dim] = i;
          suffix_through_dim_helper(dim, new_index, stride, curr_dim + 1);
        }
      } else {
        suffix_through_dim_helper(dim, new_index, stride, curr_dim + 1);
      }
    }
  }

  void prefix_along_dim(size_t dim) {
    // iterate over all possible coordinates fixing first dim dimensions
    if (dim > order - 1 || dim < 0) return;
    vector<size_t> index = getMaxIndex();
    size_t stride = 1;
    for (size_t d = dim + 1; d < order; ++d) {
      index[d] = 0;
      stride *= dim_lens[d];
    }
    prefix_along_dim_helper(dim, index, stride, dim + 1);
  }

  void prefix_along_dim_helper(size_t dim, vector<size_t> index, size_t stride,
                               size_t curr_dim) {
    if (curr_dim >= order) {
      size_t dim_len = dim_lens[dim];
      vector<size_t> start_index = index;
      vector<size_t> end_index = index;
      start_index[dim] = 0;
      end_index[dim] = dim_len - 1;
      prefix_linear(getAddress(start_index), getAddress(end_index), stride);
    } else {
      vector<size_t> new_index = index;
      for (size_t i = 0; i < dim_lens[curr_dim]; ++i) {
        new_index[curr_dim] = i;
        prefix_along_dim_helper(dim, new_index, stride, curr_dim + 1);
      }
    }
  }

  void prefix_through_dim(size_t dim, size_t curr_dim = 0) {
    // iterate over all possible coordinates fixing first dim dimensions
    if (dim > order - 1 || dim < 0) return;
    vector<size_t> index = getMaxIndex();
    size_t stride = 1;
    for (size_t d = dim + 1; d < order; ++d) {
      index[d] = 0;
      stride *= dim_lens[d];
    }
    prefix_through_dim_helper(dim, index, stride, curr_dim);
  }

  void prefix_through_dim_helper(size_t dim, vector<size_t> index,
                                 size_t stride, size_t curr_dim) {
    if (curr_dim >= order) {
      size_t dim_len = dim_lens[dim];
      vector<size_t> start_index = index;
      vector<size_t> end_index = index;
      start_index[dim] = 0;
      end_index[dim] = dim_len - 1;
      prefix_linear(getAddress(start_index), getAddress(end_index), stride);
    } else {
      vector<size_t> new_index = index;
      if (dim != curr_dim) {
        for (size_t i = 0; i < dim_lens[curr_dim]; ++i) {
          new_index[curr_dim] = i;
          prefix_through_dim_helper(dim, new_index, stride, curr_dim + 1);
        }
      } else {
        prefix_through_dim_helper(dim, new_index, stride, curr_dim + 1);
      }
    }
  }

  void incsum_along_dim(size_t dim, index_t& box_lens, size_t curr_dim = 0) {
    // iterate over all possible coordinates fixing first dim dimensions
    if (dim > order - 1 || dim < 0) return;
    Tensor<T, N> prefix_temp(*this);
    Tensor<T, N> suffix_temp(*this);
    vector<size_t> index = getMaxIndex();
    size_t stride = 1;
    for (size_t d = dim + 1; d < order; ++d) {
      index[d] = 0;
      stride *= dim_lens[d];
    }
    incsum_along_dim_helper(dim, index, stride, curr_dim, prefix_temp,
                            suffix_temp, box_lens);
  }

  void incsum_along_dim_helper(size_t dim, vector<size_t> index, size_t stride,
                               size_t curr_dim, Tensor<T, N>& prefix_temp,
                               Tensor<T, N>& suffix_temp, index_t& box_lens) {
    if (curr_dim >= order) {
      // incsum linear
      for (size_t i = 0; i < dim_lens[dim] / box_lens[dim]; ++i) {
        vector<size_t> start_index = index;
        vector<size_t> end_index = index;
        start_index[dim] = i * box_lens[dim];
        end_index[dim] = i * box_lens[dim] + box_lens[dim] - 1;
        prefix_temp.prefix_linear(getAddress(start_index),
                                  getAddress(end_index), stride);
        suffix_temp.suffix_linear(getAddress(start_index),
                                  getAddress(end_index), stride);
      }
      for (size_t i = 0; i < dim_lens[dim]; ++i) {
        vector<size_t> new_index = index;
        new_index[dim] = i;
        T suffix = suffix_temp.GetElt(new_index);
        T prefix = 0;
        if (i % box_lens[dim] == 0) {
          SetElt(new_index, suffix);
        } else {
          vector<size_t> pref_index = index;
          pref_index[dim] = i + box_lens[dim] - 1;
          if (valid_index(pref_index)) {
            prefix = prefix_temp.GetElt(pref_index);
          }
#ifdef DELAY
          volatile long x = fib(DELAY);
#endif
          SetElt(new_index, suffix + prefix);
        }
      }
    } else {
      vector<size_t> new_index = index;
      if (dim != curr_dim) {
        for (size_t i = 0; i < dim_lens[curr_dim]; ++i) {
          new_index[curr_dim] = i;
          incsum_along_dim_helper(dim, new_index, stride, curr_dim + 1,
                                  prefix_temp, suffix_temp, box_lens);
        }
      } else {
        incsum_along_dim_helper(dim, new_index, stride, curr_dim + 1,
                                prefix_temp, suffix_temp, box_lens);
      }
    }
  }

  void add_contribution(Tensor<T, N>& A, size_t dim, size_t offset) {
    // iterate over all possible coordinates fixing first dim dimensions
    if (dim > order - 1 || dim < 0) return;
    vector<size_t> index(order);
    add_contribution_helper(A, dim, offset, index, 0);
  }

  void add_contribution_helper(Tensor<T, N>& A, size_t dim, size_t offset,
                               vector<size_t> index, size_t curr_dim) {
    if (curr_dim >= order) {
      vector<size_t> set_index = index;
      vector<size_t> get_index = getMaxIndex();
      get_index[dim] = set_index[dim] + offset;
      for (int i = dim + 1; i < order; i++) {
        get_index[i] = set_index[i];
      }
      if (valid_index(get_index)) {
        AppendElt(set_index, A.GetElt(get_index));
      }
    } else {
      vector<size_t> new_index = index;
      for (size_t i = 0; i < dim_lens[curr_dim]; ++i) {
        new_index[curr_dim] = i;
        add_contribution_helper(A, dim, offset, new_index, curr_dim + 1);
      }
    }
  }

  void Incsum(index_t& box_lens) {
    for (size_t j = 0; j < order; ++j) {
      incsum_along_dim(j, box_lens);
    }
  }

  void BoxComplementExsum(index_t& box_lens) {
    Tensor<T, N> prefix_temp(*this);
    Tensor<T, N> suffix_temp(dim_lens);
    Tensor<T, N> A_prime(*this);
    // zero output (in place)
    ZeroFill();
    for (size_t i = 0; i < order; ++i) {
      // prefix step
      A_prime = prefix_temp;
      suffix_temp = prefix_temp;
      A_prime.prefix_along_dim(i);
      prefix_temp = A_prime;
      for (size_t j = i + 1; j < order; ++j) {
        A_prime.incsum_along_dim(j, box_lens, i);
      }
      add_contribution(A_prime, i, -1);
      // suffix step
      A_prime = suffix_temp;
      A_prime.suffix_along_dim(i);
      for (size_t j = i + 1; j < order; ++j) {
        A_prime.incsum_along_dim(j, box_lens, i);
      }
      add_contribution(A_prime, i, box_lens[i]);
    }
  }

  void ExsumIncsumSubtraction(index_t& box_lens) {
    Tensor<T, N> incsum_temp(*this);
    incsum_temp.Incsum(box_lens);
    // negate incsum
    for (size_t i = 0; i < incsum_temp.size(); ++i) {
#ifdef DEBUG
      printf("incsum[%lu] = %f\n", i, incsum_temp.data()[i]);
#endif
      incsum_temp.data()[i] = -incsum_temp.data()[i];
    }
#ifdef DEBUG
    printf("total sum %f\n", total());
#endif
    Fill(total());
    add_contribution(incsum_temp, 0, 0);
#ifdef DEBUG
    printf("after subtracting total\n");
    for (size_t i = 0; i < incsum_temp.size(); ++i) {
      printf("exsum[%lu] = %f\n", i, data()[i]);
    }
#endif
  }

  // naive alg that does incsum with nested loops then subtract
  void NaiveExsum(index_t& box_lens) {

    T tensor_sum = total();
    Tensor<T, N> temp(*this); // keep the input to do naive sums

    // zero output (in place)
    this->ZeroFill();

    // total of entire tensor to do subtraction from
#ifdef DEBUG
    printf("tensor sum %f\n", tensor_sum);
#endif
    vector<size_t> index(order);
    naive_helper(temp, box_lens, index, 0, tensor_sum);
  }

  void naive_helper(Tensor<T, N>& temp, index_t& box_lens, vector<size_t> index, size_t curr_dim, T tensor_sum) {
    if (curr_dim >= order) {
      // do nested loops
      #ifdef DEBUG
      assert(index.size() == order);
      #endif
      T box_total = 0;

      // keep dest idx
      std::vector<size_t> set_index;
      for (size_t i = 0; i < order; i++) {
        set_index.push_back(index[i]);
      }

      #ifdef DEBUG
      if (order == 1) {printf("set index = %lu\n", index[0]); }
      else if (order == 2) {
        printf("set idx (%lu, %lu)\n", set_index[0], set_index[1]);
      }
      #endif

      // naive sum in one dim
      if (order == 1) {
        size_t start = set_index[0];
        size_t end = std::min(dim_lens[0], set_index[0] + box_lens[0]);
        
        for(size_t i1 = start; i1 < end; i1++) {
          index[0] = i1;
          box_total += temp.GetElt(index);
          
          #ifdef DEBUG
          assert(valid_index(index));
          printf("\tadd index %lu, box total %f\n", i1, box_total);
          #endif
        }
      }
      else if (order == 2) { // naive sum in 2 dims
        for(size_t i1 = set_index[0]; i1 < std::min(dim_lens[0], set_index[0] + box_lens[0]); i1++) {
          for(size_t i2 = set_index[1]; i2 < std::min(dim_lens[1], set_index[1] + box_lens[1]); i2++) {
            index[0] = i1;
            index[1] = i2;

            #ifdef DEBUG
            assert(valid_index(index));
            printf("\t add idx (%lu, %lu), box total %f\n", index[0], index[1], box_total);
            #endif

            box_total += temp.GetElt(index);
          }
        }
      }
      else if (order == 3) { // naive sum in 3 dims
        for(size_t i1 = set_index[0]; i1 < std::min(dim_lens[0], set_index[0] + box_lens[0]); i1++) {
          for(size_t i2 = set_index[1]; i2 < std::min(dim_lens[1], set_index[1] + box_lens[1]); i2++) {
            for(size_t i3 = set_index[2]; i3 < std::min(dim_lens[2], set_index[2] + box_lens[2]); i3++) {
              index[0] = i1;
              index[1] = i2;
              index[2] = i3;
              #ifdef DEBUG
              assert(valid_index(index));
              printf("\t add idx (%lu, %lu), box total %f\n", index[0], index[1], box_total);
              #endif

              box_total += temp.GetElt(index);
            }
          }
        }
      }
      else if (order == 4) { // naive sum in 4 dims
        for(size_t i1 = set_index[0]; i1 < std::min(dim_lens[0], set_index[0] + box_lens[0]); i1++) {
          for(size_t i2 = set_index[1]; i2 < std::min(dim_lens[1], set_index[1] + box_lens[1]); i2++) {
            for(size_t i3 = set_index[2]; i3 < std::min(dim_lens[2], set_index[2] + box_lens[2]); i3++) {
              for(size_t i4 = set_index[3]; i4 < std::min(dim_lens[3], set_index[3] + box_lens[3]); i4++) {
                index[0] = i1;
                index[1] = i2;
                index[2] = i3;
                index[3] = i4;

                box_total += temp.GetElt(index);
                #ifdef DEBUG

                assert(valid_index(index));
                printf("\t add idx (%lu, %lu), box total %f\n", index[0], index[1], box_total);
                #endif
              }
            }
          }
        }
      }
      else if (order == 5) { // naive sum in 4 dims
        for(size_t i1 = set_index[0]; i1 < std::min(dim_lens[0], set_index[0] + box_lens[0]); i1++) {
          for(size_t i2 = set_index[1]; i2 < std::min(dim_lens[1], set_index[1] + box_lens[1]); i2++) {
            for(size_t i3 = set_index[2]; i3 < std::min(dim_lens[2], set_index[2] + box_lens[2]); i3++) {
              for(size_t i4 = set_index[3]; i4 < std::min(dim_lens[3], set_index[3] + box_lens[3]); i4++) {
                for(size_t i5 = set_index[4]; i5 < std::min(dim_lens[4], set_index[4] + box_lens[4]); i5++) {
                  index[0] = i1;
                  index[1] = i2;
                  index[2] = i3;
                  index[3] = i4;
                  index[4] = i5;

                  box_total += temp.GetElt(index);
                  #ifdef DEBUG

                  assert(valid_index(index));
                  printf("\t add idx (%lu, %lu), box total %f\n", index[0], index[1], box_total);
                  #endif
                }
              }
            }
          }
        }
      }

      // excluded sum is total - box
      T exsum = tensor_sum - box_total;
      #ifdef DEBUG
      printf("\texsum %f, total = %f, box_total = %f\n", exsum, tensor_sum, box_total);
      #endif

      SetElt(set_index, exsum);

      #ifdef DEBUG
      assert(GetElt(set_index) == exsum);
      #endif
    } else {  // recursive case
      vector<size_t> new_index = index;
      for (size_t i = 0; i < dim_lens[curr_dim]; ++i) {
        new_index[curr_dim] = i;
        naive_helper(temp, box_lens, new_index, curr_dim + 1, tensor_sum);
      }
    }
  }



  // summed area table
  void SummedAreaTable(index_t& box_lens) {
    Tensor<T, N> prefix_temp(*this);

    // zero output (in place)
    this->ZeroFill();

    // preproc summed area table
    // do prefix along each dimension
    for (size_t i = 0; i < this->order; ++i) {
      prefix_temp.prefix_through_dim(i);
    }

    index_t max_idx = prefix_temp.getMaxIndex();

    // total of entire tensor to do subtraction from
    T tensor_sum = prefix_temp.GetElt(max_idx);
#ifdef DEBUG
    printf("tensor sum %f\n", tensor_sum);
#endif
    vector<size_t> index(order);
    size_t num_corners = (1 << order);
    SAT_helper(prefix_temp, box_lens, index, max_idx, 0, tensor_sum,
               num_corners);

#ifdef DEBUG
    for (size_t i = 0; i <= getAddress(getMaxIndex()); ++i) {
      printf("output idx %lu, got %f\n", i, data()[i]);
    }
#endif
  }

  void SAT_helper(Tensor<T, N>& prefix_temp, index_t& box_lens,
                  vector<size_t> index, index_t max_idx, size_t curr_dim,
                  T tensor_sum, size_t num_corners) {

    // do gray codes
    if (curr_dim >= order) {

// offset index by -1
#ifdef DEBUG
      assert(index.size() == order);
      printf("\nstart idx before offset\n");
      for (size_t i = 0; i < order; i++) {
        printf("\tidx[%lu] = %lu\n", i, index[i]);
      }
#endif
      std::vector<size_t> base_index;
      std::vector<size_t> set_index;
      for (size_t i = 0; i < order; i++) {
        set_index.push_back(index[i]);
        index[i]--;
        base_index.push_back(index[i]);
      }

      T box_total = 0;
#ifdef DEBUG
      printf("base idx (%lu, %lu, %lu)\n", base_index[0], base_index[1],
             base_index[2]);
#endif

      size_t num_bits_set = 1;  // init at 1 bc -1 at 0..0
      // goal is to get the benefit of minimally changing idx
      // to do this i think you have to keep the actual scalar idx?
      // in higher dimensions, goes up to 2^d
      for (size_t gray_idx = 0; gray_idx < num_corners; ++gray_idx) {
        size_t bit_to_change = gray_diffs[order - 1][gray_idx];

        if (add_or_subtract[order - 1][gray_idx]) {
          index[bit_to_change] =
              std::min(max_idx[bit_to_change],
                       base_index[bit_to_change] + box_lens[bit_to_change]);
#ifdef DEBUG
          printf("\t\tMAX: set index[%lu] = %lu\n", bit_to_change,
                 index[bit_to_change]);
#endif
          num_bits_set++;
        } else {
          index[bit_to_change] = base_index[bit_to_change];

#ifdef DEBUG
          printf("\t\tMIN:set index[%lu] = %lu, base[%lu] = %lu\n",
                 bit_to_change, index[bit_to_change], bit_to_change,
                 base_index[bit_to_change]);
#endif
          num_bits_set--;
        }
        #ifdef DEBUG
        assert(num_bits_set <= this->order);
        #endif
        index_t& const_index = index;
        size_t parity = (this->order - num_bits_set) % 2;
#ifdef DEBUG
        printf("\t\t\tsource idx (%lu, %lu, %lu), parity %lu\n", const_index[0],
               const_index[1], const_index[2], parity);
#endif

        if (parity) {  // subtract (-1)^parity
          if (this->valid_index(const_index)) {
#ifdef DELAY
            volatile long x = fib(DELAY);
#endif
            box_total -= prefix_temp.GetElt(const_index);
#ifdef DEBUG
            printf("\t\t\t\tSUBTRACTED %f\n", prefix_temp.GetElt(const_index));
#endif
          }
        } else {  // add (-1)^parity
          if (this->valid_index(const_index)) {
#ifdef DELAY
            volatile long x = fib(DELAY);
#endif
            box_total += prefix_temp.GetElt(const_index);
#ifdef DEBUG
            printf("\t\t\t\tADDED %f\n", prefix_temp.GetElt(const_index));
#endif
          }
        }
      }
      // excluded sum is total - box
#ifdef DELAY
      volatile long x = fib(DELAY);
#endif
      T exsum = tensor_sum - box_total;

      SetElt(set_index, exsum);

#ifdef DEBUG
      printf("\texsum = %f, output got %f\n", exsum, GetElt(set_index));
      assert(GetElt(set_index) == exsum);
#endif

    } else {  // recursive case
      vector<size_t> new_index = index;
      for (size_t i = 0; i < dim_lens[curr_dim]; ++i) {
        new_index[curr_dim] = i;

        SAT_helper(prefix_temp, box_lens, new_index, max_idx, curr_dim + 1,
                   tensor_sum, num_corners);
      }
    }
  }

 protected:
  vector<T> elems;  // the tensor data itself (unpadded)
  vector<size_t> dim_lens;
};

}  // namespace exsum_tensor
