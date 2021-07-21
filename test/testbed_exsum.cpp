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

#include <stdio.h>
#include <stdlib.h>

#include <algorithm>
#include <boost/multiprecision/cpp_bin_float.hpp>
#include <cmath>
#include <fstream>
#include <iostream>
#include <iterator>
#include <string>
#include <numeric>  // for inclusive_scan
#include <random>
#include <type_traits>
#include <vector>

#include "fasttime.h"
#include "fmm_2d.h"
#include "fmm_3d.h"
#include "gtest/gtest.h"
#include "tensor.h"

typedef boost::multiprecision::cpp_bin_float_100 mp_100;

#define OUTDATED __cplusplus <= 201402L;

// #define DEBUG
#define CHECK_RESULT

namespace excluded_sum {
namespace {

template <typename T>
void omp_scan(T* a, const int N) {
  T scan_a = 0;
#pragma omp simd reduction(inscan, + : scan_a)
  for (int i = 0; i < N; i++) {
    scan_a += a[i];
#pragma omp scan inclusive(scan_a);
    a[i] = scan_a;
  }
}

int roundUp(int numToRound, int multiple) {
  if (multiple == 0) return numToRound;

  int remainder = numToRound % multiple;
  if (remainder == 0) return numToRound;

  return numToRound + multiple - remainder;
}

template <typename T>
void cpp_inclusive_scan(T* arr, const int n) {
#ifdef OUTDATED
  omp_scan(arr, n);
#else
  std::inclusive_scan(arr, arr + n, arr);
#endif
}

template <typename T>
void cpp_naive_suffix_scan(T* arr, const int n) {
  std::reverse(arr, arr + n);
  cpp_inclusive_scan(arr, n);
  std::reverse(arr, arr + n);
}

// Uses GTest framework to set up a variety of tests for arrays of INC/EXSUM
// that fit in RAM
// Testing space partitions for prefix sum are:
//      - Array size
//      - Array values
//      - Array types
//      - TODO PADDING
//      - TODO later (operation)
class AccuracyTest : public ::testing::Test {
 protected:
  AccuracyTest() {
    // You can do set-up work for each test here.
  }

  ~AccuracyTest() override {
    // You can do clean-up work that doesn't throw exceptions here.
  }

  // If the constructor and destructor are not enough for setting up
  // and cleaning up each test, you can define the following methods:

  void SetUp() override {
    // Code here will be called immediately after the constructor (right
    // before each test).
  }

  void TearDown() override {
    // Code here will be called immediately after each test (right
    // before the destructor).
  }

  // Class members declared here can be used by all tests in the test suite
  // for Foo.

  // For floating-point types, gets relative error between x and y with
  // with reference to the minimum of the two.
  template <typename T>
  const T RelativeError(T x, T y) {
    T xAbs = abs(x);
    T yAbs = abs(y);
    T diff = abs(xAbs - yAbs);
    return diff / std::min(xAbs, yAbs);
  }

  template<typename T, typename T1> 
  const T RMSE(T* ref, T1* arr, size_t N) {
    T rms_rel_err(0);
    /*
    mp_100 max_rel_err(0);
    mp_100 min_rel_err(1);
    mp_100 avg_rel_err(0);
    mp_100 rms_rel_err(0);
    */

    for (size_t i = 1; i < N; ++i) {
      //EXPECT_NEAR(tensor.data()[i], static_cast<float>(array_ref[i]), 1e3);
      T err = RelativeError(ref[i], T(arr[i]));
      /*
      if (err > max_rel_err) {
        max_rel_err = err;
      }
      if (err < min_rel_err) {
        min_rel_err = err;
      }
      avg_rel_err += err;
      */
      rms_rel_err += err * err;
    }
    // avg_rel_err = avg_rel_err / N;
    rms_rel_err = sqrt(rms_rel_err / N);
    // printf("avg %f, rmse %f, min %f, max %f\n", avg_rel_err, rms_rel_err, min_rel_err, max_rel_err);
    return rms_rel_err;
  }


  template <size_t N>
  void prefix_sum_mp_100(mp_100* arr, const size_t increment) {
    mp_100 sum(0);
    for (size_t i = 0; i < N; i += increment) {
      sum += mp_100(arr[i]);
      arr[i] = mp_100(sum);
    }
  }

  template <typename T1, typename T2, size_t N>
  std::vector<T1> copy_array(T2* arr) {
    std::vector<T1> copy;
    for (size_t i = 0; i < N; ++i) {
      copy.emplace_back(T1(arr[i]));
    }
    return copy;
  }

  template <typename T>
  void print_vector(std::vector<T> const& a) {
    for (int i = 0; i < a.size(); i++) {
      std::cout << a.at(i) << " ";
    }
    std::cout << std::endl;
  }

// Kahan Summation algorithm for floats or doubles. Follows same prefix
// sum API as rest of the implementations.
// Optimization by the compiler is turned off to ensure the numerical
// stability and accuracy of the algorithm. Otherwise some parts would be
// optimized out and it would lose its compensated summation.
#pragma clang optimize off
  template <typename T, size_t N>
  void KahanSummation(T* arr, const size_t increment) {
    T sum = 0;
    T c = 0;
    for (size_t i = 0; i < N; i += increment) {
      T y = arr[i] - c;
      T t = sum + y;
      c = (t - sum) - y;
      sum = t;
      arr[i] = sum;
    }
  }
#pragma clang optimize on

  template <typename T, size_t N>
  void naive_serial(T* arr, const size_t increment) {
    T sum = 0;
    for (size_t i = 0; i < N; i += increment) {
      sum += arr[i];
      arr[i] = sum;
    }
  }
};

TEST_F(AccuracyTest, Float) {
  // initialize type and size
  typedef double dtype;
  typedef vector<size_t> index_t;

  std::ofstream f("ND-exsum-double-accuracy.csv");

  int increment = 4;

  int lower = 1 << 20;
  int upper = (1 << 20) + increment;

  /*
  int lower = 1 << 26;
  int upper = (1 << 26) + increment;
  */
  // size_t box_len = 2;
  size_t box_len = 8;

  // lines separate the different algorithms
  // columns are the CSV values
  if (f.is_open()) {
    f << "Algorithm Num";
    for (int p = lower; p < upper; p += increment) {
      const size_t N = p * box_len;
      f << ",runtime(ms) for N = ";
      f << N;
    }
    f << std::endl;
    f << "X-data";
    for (int p = lower; p < upper; p += increment) {
      const size_t N = p * box_len;
      f << "," << N;
    }
    f << std::endl;
  }

  f.precision(10);
  
  // dim = 1
  {
    printf("1d\n");
    f << 1;
    constexpr size_t dim = 1;
    std::cout << "\nAlgorithm " << dim << std::endl;
    for (int p = lower; p < upper; p += increment) {
      size_t N;
#ifdef DEBUG
      N = 4;  // 4 ^ 1
#else
      N = p * box_len;
#endif

      size_t n = roundUp(std::floor(std::pow(N, 1. / dim)), box_len);
      N = std::pow(n, dim);
      index_t side_lens(dim);
      index_t box_lens(dim);
      std::fill(side_lens.begin(), side_lens.end(), n);
      std::cout << "n = " << n << std::endl;
      std::cout << "N = " << N << std::endl;

      exsum_tensor::Tensor<dtype, dim> tensor(side_lens);
      tensor.RandFill(0, 1);

      exsum_tensor::Tensor<dtype, dim> tensor_copy(tensor);

      std::fill(box_lens.begin(), box_lens.end(), box_len);

      exsum_tensor::Tensor<mp_100, dim> tensor_naive(side_lens);
      exsum_tensor::Tensor<dtype, dim> tensor_ref(tensor_copy);
      exsum_tensor::Tensor<dtype, dim> tensor_test(tensor_copy);
      exsum_tensor::Tensor<dtype, dim> tensor_test_2(tensor_copy);

      std::cout << "size = " << tensor_test.size() << std::endl;
  
      printf("BDBS\n");
      // time incsum + subtraction
      tensor_ref = tensor_copy;
      tensor_ref.ExsumIncsumSubtraction(box_lens);

      printf("BOXCOMP\n");

      // time box complement
      tensor_test = tensor_copy;
      tensor_test.BoxComplementExsum(box_lens);

      // time summed area table + subtraction
      tensor_test_2 = tensor_copy;
      tensor_test_2.SummedAreaTable(box_lens);

      printf("naive\n");
      // time naive
      mp_100* data = tensor_naive.data();

      // copy and convert to extended precision
      for (size_t i = 0; i <= tensor.getAddress(tensor.getMaxIndex()); ++i) {
        data[i] = mp_100(tensor.data()[i]);
      }
      tensor_naive.NaiveExsum(box_lens);

      printf("compute rmse\n");
    //  const T RMSE(T* ref, T1* arr, size_t N) {
      mp_100 rmse_bdbs = RMSE(tensor_naive.data(), tensor_ref.data(), N);
      mp_100 rmse_boxcomp = RMSE(tensor_naive.data(), tensor_test.data(), N);
      mp_100 rmse_sat = RMSE(tensor_naive.data(), tensor_test_2.data(), N);

      if (f.is_open()) {
        f << N << ",";
        f << rmse_sat << ",";
        f << rmse_bdbs << ",";
        f << rmse_boxcomp;
      }
    }
    f << std::endl;
  }
  // dim = 2
  {
    printf("2d\n");
    f << 2;
    constexpr size_t dim = 2;
    std::cout << "\nAlgorithm " << dim << std::endl;
    for (int p = lower; p < upper; p += increment) {
      size_t N;
#ifdef DEBUG
      N = 16;  // 4 ^ 2
#else
      N = p * box_len;
#endif

      size_t n = roundUp(std::floor(std::pow(N, 1. / dim)), box_len);
      N = std::pow(n, dim);
      index_t side_lens(dim);
      index_t box_lens(dim);
      std::fill(side_lens.begin(), side_lens.end(), n);
      std::cout << "n = " << n << std::endl;
      std::cout << "N = " << N << std::endl;

      exsum_tensor::Tensor<dtype, dim> tensor(side_lens);
      tensor.RandFill(0, 1);

      exsum_tensor::Tensor<dtype, dim> tensor_copy(tensor);

      std::fill(box_lens.begin(), box_lens.end(), box_len);

      exsum_tensor::Tensor<mp_100, dim> tensor_naive(side_lens);
      exsum_tensor::Tensor<dtype, dim> tensor_ref(tensor_copy);
      exsum_tensor::Tensor<dtype, dim> tensor_test(tensor_copy);
      exsum_tensor::Tensor<dtype, dim> tensor_test_2(tensor_copy);

      std::cout << "size = " << tensor_test.size() << std::endl;
  
      printf("BDBS\n");
      // time incsum + subtraction
      tensor_ref = tensor_copy;
      tensor_ref.ExsumIncsumSubtraction(box_lens);

      printf("BOXCOMP\n");

      // time box complement
      tensor_test = tensor_copy;
      tensor_test.BoxComplementExsum(box_lens);

      // time summed area table + subtraction
      tensor_test_2 = tensor_copy;
      tensor_test_2.SummedAreaTable(box_lens);

      printf("naive\n");
      // time naive
      mp_100* data = tensor_naive.data();

      // copy and convert to extended precision
      for (size_t i = 0; i <= tensor.getAddress(tensor.getMaxIndex()); ++i) {
        data[i] = mp_100(tensor.data()[i]);
      }
      tensor_naive.NaiveExsum(box_lens);

      printf("compute rmse\n");
    //  const T RMSE(T* ref, T1* arr, size_t N) {
      mp_100 rmse_bdbs = RMSE(tensor_naive.data(), tensor_ref.data(), N);
      mp_100 rmse_boxcomp = RMSE(tensor_naive.data(), tensor_test.data(), N);
      mp_100 rmse_sat = RMSE(tensor_naive.data(), tensor_test_2.data(), N);

      if (f.is_open()) {
        f << N << ",";
        f << rmse_sat << ",";
        f << rmse_bdbs << ",";
        f << rmse_boxcomp;
      }
    }
    f << std::endl;
  }
  // dim = 3
  {
    printf("3d\n");
    f << 3;
    constexpr size_t dim = 3;
    std::cout << "\nAlgorithm " << dim << std::endl;
    for (int p = lower; p < upper; p += increment) {
      size_t N;
#ifdef DEBUG
      N = 64;
#else
      N = p * box_len;
#endif

      size_t n = roundUp(std::floor(std::pow(N, 1. / dim)), box_len);
      N = std::pow(n, dim);
      index_t side_lens(dim);
      index_t box_lens(dim);
      std::fill(side_lens.begin(), side_lens.end(), n);
      std::cout << "n = " << n << std::endl;
      std::cout << "N = " << N << std::endl;

      exsum_tensor::Tensor<dtype, dim> tensor(side_lens);
      tensor.RandFill(0, 1);

      exsum_tensor::Tensor<dtype, dim> tensor_copy(tensor);

      std::fill(box_lens.begin(), box_lens.end(), box_len);

      exsum_tensor::Tensor<mp_100, dim> tensor_naive(side_lens);
      exsum_tensor::Tensor<dtype, dim> tensor_ref(tensor_copy);
      exsum_tensor::Tensor<dtype, dim> tensor_test(tensor_copy);
      exsum_tensor::Tensor<dtype, dim> tensor_test_2(tensor_copy);

      std::cout << "size = " << tensor_test.size() << std::endl;
  
      printf("BDBS\n");
      // time incsum + subtraction
      tensor_ref = tensor_copy;
      tensor_ref.ExsumIncsumSubtraction(box_lens);

      printf("BOXCOMP\n");

      // time box complement
      tensor_test = tensor_copy;
      tensor_test.BoxComplementExsum(box_lens);

      // time summed area table + subtraction
      tensor_test_2 = tensor_copy;
      tensor_test_2.SummedAreaTable(box_lens);

      printf("naive\n");
      // time naive
      mp_100* data = tensor_naive.data();

      // copy and convert to extended precision
      for (size_t i = 0; i <= tensor.getAddress(tensor.getMaxIndex()); ++i) {
        data[i] = mp_100(tensor.data()[i]);
      }
      tensor_naive.NaiveExsum(box_lens);

      printf("compute rmse\n");
    //  const T RMSE(T* ref, T1* arr, size_t N) {
      mp_100 rmse_bdbs = RMSE(tensor_naive.data(), tensor_ref.data(), N);
      mp_100 rmse_boxcomp = RMSE(tensor_naive.data(), tensor_test.data(), N);
      mp_100 rmse_sat = RMSE(tensor_naive.data(), tensor_test_2.data(), N);

      if (f.is_open()) {
        f << N << ",";
        f << rmse_sat << ",";
        f << rmse_bdbs << ",";
        f << rmse_boxcomp;
      }
    }
    f << std::endl;
  }
  // dim = 4
  {
    printf("4d\n");
    f << 4;
    constexpr size_t dim = 4;
    std::cout << "\nAlgorithm " << dim << std::endl;
    for (int p = lower; p < upper; p += increment) {
      size_t N;
#ifdef DEBUG
      N = 256;
#else
      N = p * box_len;
#endif

      size_t n = roundUp(std::floor(std::pow(N, 1. / dim)), box_len);
      N = std::pow(n, dim);
      index_t side_lens(dim);
      index_t box_lens(dim);
      std::fill(side_lens.begin(), side_lens.end(), n);
      std::cout << "n = " << n << std::endl;
      std::cout << "N = " << N << std::endl;

      exsum_tensor::Tensor<dtype, dim> tensor(side_lens);
      tensor.RandFill(0, 1);

      exsum_tensor::Tensor<dtype, dim> tensor_copy(tensor);

      std::fill(box_lens.begin(), box_lens.end(), box_len);

      exsum_tensor::Tensor<mp_100, dim> tensor_naive(side_lens);
      exsum_tensor::Tensor<dtype, dim> tensor_ref(tensor_copy);
      exsum_tensor::Tensor<dtype, dim> tensor_test(tensor_copy);
      exsum_tensor::Tensor<dtype, dim> tensor_test_2(tensor_copy);

      std::cout << "size = " << tensor_test.size() << std::endl;
  
      printf("BDBS\n");
      // time incsum + subtraction
      tensor_ref = tensor_copy;
      tensor_ref.ExsumIncsumSubtraction(box_lens);

      printf("BOXCOMP\n");

      // time box complement
      tensor_test = tensor_copy;
      tensor_test.BoxComplementExsum(box_lens);

      // time summed area table + subtraction
      tensor_test_2 = tensor_copy;
      tensor_test_2.SummedAreaTable(box_lens);

      printf("naive\n");
      // time naive
      mp_100* data = tensor_naive.data();

      // copy and convert to extended precision
      for (size_t i = 0; i <= tensor.getAddress(tensor.getMaxIndex()); ++i) {
        data[i] = mp_100(tensor.data()[i]);
      }
      tensor_naive.NaiveExsum(box_lens);

      printf("compute rmse\n");
    //  const T RMSE(T* ref, T1* arr, size_t N) {
      mp_100 rmse_bdbs = RMSE(tensor_naive.data(), tensor_ref.data(), N);
      mp_100 rmse_boxcomp = RMSE(tensor_naive.data(), tensor_test.data(), N);
      mp_100 rmse_sat = RMSE(tensor_naive.data(), tensor_test_2.data(), N);

      if (f.is_open()) {
        f << N << ",";
        f << rmse_sat << ",";
        f << rmse_bdbs << ",";
        f << rmse_boxcomp;
      }
    }
    f << std::endl;
  }
  // dim = 5
  {
    printf("5d\n");
    f << 5;
    constexpr size_t dim = 5;
    std::cout << "\nAlgorithm " << dim << std::endl;
    for (int p = lower; p < upper; p += increment) {
      size_t N;
#ifdef DEBUG
      N = 1024;
#else
      N = p * box_len;
#endif

      size_t n = roundUp(std::floor(std::pow(N, 1. / dim)), box_len);
      N = std::pow(n, dim);
      index_t side_lens(dim);
      index_t box_lens(dim);
      std::fill(side_lens.begin(), side_lens.end(), n);
      std::cout << "n = " << n << std::endl;
      std::cout << "N = " << N << std::endl;

      exsum_tensor::Tensor<dtype, dim> tensor(side_lens);
      tensor.RandFill(0, 1);

      exsum_tensor::Tensor<dtype, dim> tensor_copy(tensor);

      std::fill(box_lens.begin(), box_lens.end(), box_len);

      exsum_tensor::Tensor<mp_100, dim> tensor_naive(side_lens);
      exsum_tensor::Tensor<dtype, dim> tensor_ref(tensor_copy);
      exsum_tensor::Tensor<dtype, dim> tensor_test(tensor_copy);
      exsum_tensor::Tensor<dtype, dim> tensor_test_2(tensor_copy);

      std::cout << "size = " << tensor_test.size() << std::endl;
  
      printf("BDBS\n");
      // time incsum + subtraction
      tensor_ref = tensor_copy;
      tensor_ref.ExsumIncsumSubtraction(box_lens);

      printf("BOXCOMP\n");

      // time box complement
      tensor_test = tensor_copy;
      tensor_test.BoxComplementExsum(box_lens);

      // time summed area table + subtraction
      tensor_test_2 = tensor_copy;
      tensor_test_2.SummedAreaTable(box_lens);

      printf("naive\n");
      // time naive
      mp_100* data = tensor_naive.data();

      // copy and convert to extended precision
      for (size_t i = 0; i <= tensor.getAddress(tensor.getMaxIndex()); ++i) {
        data[i] = mp_100(tensor.data()[i]);
      }
      tensor_naive.NaiveExsum(box_lens);

      printf("compute rmse\n");
    //  const T RMSE(T* ref, T1* arr, size_t N) {
      mp_100 rmse_bdbs = RMSE(tensor_naive.data(), tensor_ref.data(), N);
      mp_100 rmse_boxcomp = RMSE(tensor_naive.data(), tensor_test.data(), N);
      mp_100 rmse_sat = RMSE(tensor_naive.data(), tensor_test_2.data(), N);

      if (f.is_open()) {
        f << N << ",";
        f << rmse_sat << ",";
        f << rmse_bdbs << ",";
        f << rmse_boxcomp;
      }
    }
    f << std::endl;
  }

}

class PerfTest : public ::testing::Test {
 protected:
  PerfTest() {
    // You can do set-up work for each test here.
  }

  ~PerfTest() override {
    // You can do clean-up work that doesn't throw exceptions here.
  }

  // If the constructor and destructor are not enough for setting up
  // and cleaning up each test, you can define the following methods:

  void SetUp() override {
    // Code here will be called immediately after the constructor (right
    // before each test).
  }

  void TearDown() override {
    // Code here will be called immediately after each test (right
    // before the destructor).
  }

  // Class members declared here can be used by all tests in the test suite
  // for Foo.
  template <typename T1, typename T2>
  std::vector<T1> copy_array(T2* arr, const size_t N) {
    std::vector<T1> copy;
    for (size_t i = 0; i < N; ++i) {
      copy.emplace_back(T1(arr[i]));
    }
    return copy;
  }
};

/*
// size 2^24 was previously used for timing results
TEST_F(AccuracyTest, Float) {
  // initialize type and size
  typedef float dtype;
  const size_t N = 1 << 2;

  typedef const vector<size_t> index_t;

  index_t side_lens {N, N + 2};

  exsum_tensor::Tensor_2D<dtype> tensor(side_lens);

  //tensor.RandFill(1, 1);

  //tensor.Print();

  tensor.SetElt(index_t {0, 0}, 1);
  tensor.SetElt(index_t {0, 1}, 7);
  tensor.SetElt(index_t {0, 2}, 3);
  tensor.SetElt(index_t {0, 3}, 2);
  tensor.SetElt(index_t {0, 4}, 4);
  tensor.SetElt(index_t {0, 5}, 2);
  tensor.SetElt(index_t {1, 0}, 6);
  tensor.SetElt(index_t {1, 1}, 3);
  tensor.SetElt(index_t {1, 2}, 0);
  tensor.SetElt(index_t {1, 3}, 9);
  tensor.SetElt(index_t {1, 4}, 3);
  tensor.SetElt(index_t {1, 5}, 6);
  tensor.SetElt(index_t {2, 0}, 8);
  tensor.SetElt(index_t {2, 1}, 4);
  tensor.SetElt(index_t {2, 2}, 4);
  tensor.SetElt(index_t {2, 3}, 1);
  tensor.SetElt(index_t {2, 4}, 9);
  tensor.SetElt(index_t {2, 5}, 5);
  tensor.SetElt(index_t {3, 0}, 3);
  tensor.SetElt(index_t {3, 1}, 7);
  tensor.SetElt(index_t {3, 2}, 6);
  tensor.SetElt(index_t {3, 3}, 1);
  tensor.SetElt(index_t {3, 4}, 4);
  tensor.SetElt(index_t {3, 5}, 3);

  tensor.Print();

  index_t box_lens {2 , 3};

  exsum_tensor::Tensor_2D<dtype> incsum_copy(tensor);
  exsum_tensor::Tensor_2D<dtype> exsum_copy(tensor);
  exsum_tensor::Tensor_2D<dtype> exsum_copy2(tensor);
  exsum_tensor::Tensor_2D<dtype> corners(tensor);
  exsum_tensor::Tensor_2D<dtype> incsum(tensor);
  exsum_tensor::Tensor_2D<dtype> box_comp(tensor);

  incsum_copy.IncsumCheck(box_lens);
  exsum_copy.ExsumCheckSubtraction(box_lens);
  exsum_copy2.ExsumCheckNaive(box_lens);
  corners.CornersExsum(box_lens);
  incsum.Incsum(box_lens);
  box_comp.BoxComplementExsum(box_lens);


  std::cout << "=================== Results ==========================" <<
std::endl;

  tensor.Print();
  //incsum_copy.Print();
  //exsum_copy.Print();
  exsum_copy2.Print();
  corners.Print();

  //incsum.Print();
  box_comp.Print();
}
*/

TEST_F(PerfTest, Float) {
  // initialize type and size
  typedef float dtype;

  int trials = 1;
  int lower_pow2 = 3;
  int upper_pow2 = 4;

  std::ofstream f("2D-exsum-float-performance.csv");
  // lines separate the different algorithms
  // columns are the CSV values
  if (f.is_open()) {
    f << "Algorithm Num";
    for (int p = lower_pow2; p < upper_pow2; p++) {
      const size_t N = 1 << p;
      f << ",runtime(ms) for ";
      f << N << " elems";
    }
    f << std::endl;
    f << "X-data";
    for (int p = lower_pow2; p < upper_pow2; p++) {
      const size_t N = 1 << p;
      f << "," << N;
    }
    f << std::endl;
  }

  f.precision(10);

  for (int j = 0; j < 1; j++) {
    f << j;
    for (int p = lower_pow2; p < upper_pow2; p++) {
      const size_t N = 1 << p;
      typedef const vector<size_t> index_t;

      index_t side_lens {N, N};

      exsum_tensor::Tensor_2D<dtype> tensor(side_lens);

      tensor.RandFill(0, 1);

      exsum_tensor::Tensor_2D<dtype> tensor_copy(tensor);
      index_t box_lens {4 , 4};

      exsum_tensor::Tensor_2D<dtype> box_comp(tensor_copy);

      std::vector<double> elapsed_times(trials);

      for (int k = 0; k < trials; k++) {
        box_comp = tensor_copy;
        fasttime_t start = gettime();
        box_comp.BoxComplementExsum(box_lens);
        fasttime_t end = gettime();
        double elapsed = tdiff(start, end);
        elapsed_times.emplace_back(elapsed);
      }

      tensor.ExsumCheckSubtraction(box_lens);

      std::nth_element(elapsed_times.begin(),
                       elapsed_times.begin() + elapsed_times.size()/2,
                       elapsed_times.end());
      double median_time = elapsed_times[elapsed_times.size()/2];


      for (size_t i = 0; i <= tensor.getAddress(tensor.getMaxIndex()); ++i) {
        EXPECT_NEAR(tensor.data()[i], box_comp.data()[i], N >> 3);
      }
      f << "," << median_time * 1000;
    }
    f << std::endl;
  }
  f.close();
}

/*
TEST_F(AccuracyTest, Double) {
  // initialize type and size
  typedef double dtype;
  const size_t N = 1 << 2;

  typedef const vector<size_t> index_t;

  index_t side_lens {N, N + 2, N};

  exsum_tensor::Tensor_3D<dtype> tensor(side_lens);

  tensor.RandFill(1, 1);

  tensor.SetElt(index_t {1, 1, 2}, 3);

  //tensor.Print();

  exsum_tensor::Tensor_3D<dtype> exsum_test(tensor);

  index_t box_lens {2, 3, 2};

  tensor.Print();

  fasttime_t start = gettime();
  tensor.ExsumCheckNaive(box_lens);
  fasttime_t end = gettime();
  double elapsed = tdiff(start, end);

  std::cout << "=====" << std::endl;
  std::cout << elapsed * 1000 << std::endl;

  tensor.Print();

  start = gettime();
  // exsum_test.BoxComplementExsum(box_lens);
  // exsum_test.BoxComplementExsum_space(box_lens);
  exsum_test.BoxComplementExsum_space_new(box_lens);
  // exsum_test.CornersExsum_dN(box_lens);
  // exsum_test.CornersExsum_leaf(box_lens, 3);

  end = gettime();
  elapsed = tdiff(start, end);

  std::cout << "=====" << std::endl;
  std::cout << elapsed * 1000 << std::endl;

  exsum_test.Print();

  for (size_t i = 0; i <= tensor.getAddress(tensor.getMaxIndex()); ++i) {
    EXPECT_DOUBLE_EQ(tensor.data()[i], exsum_test.data()[i]);
  }

}
*/

TEST_F(PerfTest, Double) {
  // initialize type and size
  typedef double dtype;
  typedef const vector<size_t> index_t;

  int trials;
#ifdef DEBUG
  trials = 1;
#else
  trials = 3;
#endif

  int lower = 2;

  int upper;
#ifdef DEBUG
  upper = lower + 1;
#else
  upper = 24;
#endif

#ifdef DELAY
  lower = 4;
  upper = 8;
#endif

  int increment = 4;

  size_t box_len;
#ifdef DEBUG
  box_len = 2;
#else
  box_len = 4;
#endif

#ifdef DELAY
  std::ofstream f("3D-exsum-double-performance-" +
    std::to_string(DELAY) + ".csv");
#else
  std::ofstream f("3D-exsum-double-performance.csv");
#endif

  typedef void (exsum_tensor::Tensor_3D<dtype>::*fn)(index_t&);

  static fn funcs[] = {
      &exsum_tensor::Tensor_3D<dtype>::ExsumCheckSubtraction,
      &exsum_tensor::Tensor_3D<dtype>::SummedAreaTable,
      &exsum_tensor::Tensor_3D<dtype>::CornersExsum_dN,      // spine
      &exsum_tensor::Tensor_3D<dtype>::CornersExsum_Space,   // space = 1
      &exsum_tensor::Tensor_3D<dtype>::CornersExsum_leaf_2,  // space = 2
      &exsum_tensor::Tensor_3D<dtype>::CornersExsum_leaf,    // space = 4
      &exsum_tensor::Tensor_3D<dtype>::CornersExsum,         // space = 8
      &exsum_tensor::Tensor_3D<dtype>::BoxComplementExsum,
  };

#ifdef DELAY
  fasttime_t t1 = gettime();
  for (int i = 0; i < trials * 10000; i++) {
    volatile long x = fib(DELAY);
  }
  fasttime_t t2 = gettime();
  double elapsed_time = tdiff(t1, t2);
  double time_ms = (elapsed_time * 1000) / (trials * 10000);
  std::cout << "fib time (ms): " << time_ms << std::endl;

#endif

  // lines separate the different algorithms
  // columns are the CSV values
  if (f.is_open()) {
    f << "Algorithm Num";
    for (int p = lower; p < upper; p += increment) {
      const size_t N = p * box_len;
      f << ",runtime(ms) for ";
      f << N * N * N << " elems";
    }
#ifdef DELAY
    f << ", fibtime(ms)";
#endif
    f << std::endl;
    f << "X-data";
    for (int p = lower; p < upper; p += increment) {
      const size_t N = p * box_len;
      f << "," << N * N * N;
    }
#ifdef DELAY
    f << "," << time_ms;
#endif
    f << std::endl;
  }

  f.precision(10);

  int len = sizeof(funcs) / sizeof(funcs[0]);


  for (int j = 0; j < len; j++) {
    f << j;
    std::cout << "Algorithm " << j << std::endl;
    for (int p = lower; p < upper; p += increment) {
      const size_t N = p * box_len;
      index_t side_lens{N, N, N};
      std::cout << "N = " << N * N * N << std::endl;

      exsum_tensor::Tensor_3D<dtype> tensor(side_lens);

      tensor.RandFill(0, 1);

      exsum_tensor::Tensor_3D<dtype> tensor_copy(tensor);
      index_t box_lens{box_len, box_len, box_len};

      exsum_tensor::Tensor_3D<dtype> tensor_test(tensor_copy);

      std::vector<double> elapsed_times(trials);

      for (int k = 0; k < trials; k++) {
        tensor_test = tensor_copy;
        fasttime_t start = gettime();
        (tensor_test.*funcs[j])(box_lens);
        fasttime_t end = gettime();
        double elapsed = tdiff(start, end);
        elapsed_times.emplace_back(elapsed);
      }

      tensor.ExsumCheckSubtraction(box_lens);

      std::nth_element(elapsed_times.begin(),
                       elapsed_times.begin() + elapsed_times.size() / 2,
                       elapsed_times.end());
      double median_time = elapsed_times[elapsed_times.size() / 2];

#ifdef CHECK_RESULT
      for (size_t i = 0; i <= tensor.getAddress(tensor.getMaxIndex()); ++i) {
        // N>>3 was 0 in debugging, so just making it something tiny
        EXPECT_NEAR(tensor.data()[i], tensor_test.data()[i],
                    (double)1 / 10000000);  // N >> 3);
      }
#endif
      f << "," << median_time * 1000;
    }
    f << std::endl;
  }
  f.close();
}

TEST_F(PerfTest, ArbitraryDims) {
  // initialize type and size
  typedef double dtype;
  typedef vector<size_t> index_t;

  std::ofstream f("ND-exsum-double-performance.csv");

  int trials;
  #ifdef DEBUG
    trials = 1;
  #else
    trials = 3;
  #endif
  int increment = 4;


  int lower = 1 << 20;
  int upper = (1 << 20) + increment;

  /*
  int lower = 1 << 26;
  int upper = (1 << 26) + increment;
  */
  // size_t box_len = 2;
  double debug_diff = (double) 1/100;
  size_t debug_shift = 3;
  size_t box_len = 8;

  // lines separate the different algorithms
  // columns are the CSV values
  if (f.is_open()) {
    f << "Algorithm Num";
    for (int p = lower; p < upper; p += increment) {
      const size_t N = p * box_len;
      f << ",runtime(ms) for N = ";
      f << N;
    }
    f << std::endl;
    f << "X-data";
    for (int p = lower; p < upper; p += increment) {
      const size_t N = p * box_len;
      f << "," << N;
    }
    f << std::endl;
  }

  f.precision(10);
  // dim = 1
  {
    f << 1;
    constexpr size_t dim = 1;
    std::cout << "\nAlgorithm " << dim << std::endl;
    for (int p = lower; p < upper; p += increment) {
      size_t N;
#ifdef DEBUG
      N = 4;  // 4 ^ 1
#else
      N = p * box_len;
#endif

      size_t n = roundUp(std::floor(std::pow(N, 1. / dim)), box_len);
      N = std::pow(n, dim);
      index_t side_lens(dim);
      index_t box_lens(dim);
      std::fill(side_lens.begin(), side_lens.end(), n);
      std::cout << "n = " << n << std::endl;
      std::cout << "N = " << N << std::endl;

      exsum_tensor::Tensor<dtype, dim> tensor(side_lens);
      tensor.RandFill(0, 1);

      exsum_tensor::Tensor<dtype, dim> tensor_copy(tensor);

      std::fill(box_lens.begin(), box_lens.end(), box_len);

      exsum_tensor::Tensor<dtype, dim> tensor_naive(tensor_copy);
      exsum_tensor::Tensor<dtype, dim> tensor_ref(tensor_copy);
      exsum_tensor::Tensor<dtype, dim> tensor_test(tensor_copy);
      exsum_tensor::Tensor<dtype, dim> tensor_test_2(tensor_copy);

      std::cout << "size = " << tensor_test.size() << std::endl;

      // time incsum + subtraction
      std::vector<double> elapsed_times_incsum(trials);
      for (int k = 0; k < trials; k++) {
        tensor_ref = tensor_copy;
        fasttime_t start = gettime();
        tensor_ref.ExsumIncsumSubtraction(box_lens);
        fasttime_t end = gettime();
        double elapsed = tdiff(start, end);
        printf("\tBDBS trial %d, time %f\n", k, elapsed);
        elapsed_times_incsum[k] = elapsed;
      }
      // time box complement
      std::vector<double> elapsed_times(trials);
      for (int k = 0; k < trials; k++) {
        tensor_test = tensor_copy;
        fasttime_t start = gettime();
        tensor_test.BoxComplementExsum(box_lens);
        fasttime_t end = gettime();
        double elapsed = tdiff(start, end);
        printf("\tBOXCOMP trial %d, time %f\n", k, elapsed);
        elapsed_times[k] = elapsed;
      }

#ifdef DEBUG
      for (size_t i = 0; i <= tensor.getAddress(tensor.getMaxIndex()); ++i) {
        EXPECT_NEAR(tensor_ref.data()[i], tensor_test.data()[i], debug_diff); // tiny diff check
      }
#endif

      // time summed area table + subtraction
      std::vector<double> elapsed_times_sat(trials);
      for (int k = 0; k < trials; k++) {
        tensor_test_2 = tensor_copy;
        fasttime_t start = gettime();
        tensor_test_2.SummedAreaTable(box_lens);
        fasttime_t end = gettime();
        double elapsed = tdiff(start, end);
        printf("\tSAT trial %d, time %f\n", k, elapsed);
        elapsed_times_sat[k] = elapsed;
      }

#ifdef DEBUG
      for (size_t i = 0; i <= tensor.getAddress(tensor.getMaxIndex()); ++i) {
        EXPECT_NEAR(tensor_ref.data()[i], tensor_test_2.data()[i], debug_diff);
      }
#endif

      // time naive
      std::vector<double> elapsed_times_naive(trials);
      for (int k = 0; k < trials; k++) {
        tensor_naive = tensor_copy;
        fasttime_t start = gettime();
        tensor_naive.NaiveExsum(box_lens);
        fasttime_t end = gettime();
        double elapsed = tdiff(start, end);
        printf("\tNAIVE trial %d, time %f\n", k, elapsed);
        elapsed_times_naive[k] = elapsed;
      }
#ifdef CHECK_RESULT
      // check naive
      for (size_t i = 0; i <= tensor.getAddress(tensor.getMaxIndex()); ++i) {
        EXPECT_NEAR(tensor_ref.data()[i], tensor_naive.data()[i], debug_diff); // tiny diff check
      }
#endif

      int med_idx = trials / 2;

      // sort the times
      std::nth_element(elapsed_times.begin(),
                       elapsed_times.begin() + elapsed_times.size() / 2,
                       elapsed_times.end());
      std::nth_element(elapsed_times_sat.begin(),
                       elapsed_times_sat.begin() + elapsed_times_sat.size() / 2,
                       elapsed_times_sat.end());
      std::nth_element(
          elapsed_times_incsum.begin(),
          elapsed_times_incsum.begin() + elapsed_times_incsum.size() / 2,
          elapsed_times_incsum.end());
      std::nth_element(
          elapsed_times_naive.begin(),
          elapsed_times_naive.begin() + elapsed_times_naive.size() / 2,
          elapsed_times_naive.end());

      double sat_med = elapsed_times_sat[med_idx] * 1000;
      double incsum_med = elapsed_times_incsum[med_idx] * 1000;
      double boxcomp_med = elapsed_times[med_idx] * 1000;
      double naive_med = elapsed_times_naive[med_idx] * 1000;
      // dim, vol, SAT + subtraction time, BDBS + subtraction time, BOXCOMP
      // time
      f << "," << N << "," << naive_med << "," << sat_med << "," << incsum_med << ","
        << boxcomp_med;

      std::cout << "NAIVE Time = " << naive_med << std::endl;
      std::cout << "SAT Time = " << sat_med << std::endl;
      std::cout << "INCSUM Time = " << incsum_med << std::endl;
      std::cout << "BOXCOMP Time = " << boxcomp_med << std::endl;
    }
    f << std::endl;
  }

  // dim = 2
  {
    f << 2;
    constexpr size_t dim = 2;
    std::cout << "\nAlgorithm " << dim << std::endl;
    for (int p = lower; p < upper; p += increment) {
      size_t N;
#ifdef DEBUG
      N = 16;  // 4 ^ 2
#else
      N = p * box_len;
#endif

      size_t n = roundUp(std::floor(std::pow(N, 1. / dim)), box_len);
      N = std::pow(n, dim);
      index_t side_lens(dim);
      index_t box_lens(dim);
      std::fill(side_lens.begin(), side_lens.end(), n);
      std::cout << "n = " << n << std::endl;
      std::cout << "N = " << N << std::endl;

      exsum_tensor::Tensor<dtype, dim> tensor(side_lens);
      tensor.RandFill(0, 1);

      exsum_tensor::Tensor<dtype, dim> tensor_copy(tensor);

      std::fill(box_lens.begin(), box_lens.end(), box_len);

      exsum_tensor::Tensor<dtype, dim> tensor_naive(tensor_copy);
      exsum_tensor::Tensor<dtype, dim> tensor_ref(tensor_copy);
      exsum_tensor::Tensor<dtype, dim> tensor_test(tensor_copy);
      exsum_tensor::Tensor<dtype, dim> tensor_test_2(tensor_copy);

      std::cout << "size = " << tensor_test.size() << std::endl;

      // time box complement
      std::vector<double> elapsed_times(trials);
      for (int k = 0; k < trials; k++) {
        tensor_test = tensor_copy;
        fasttime_t start = gettime();
        tensor_test.BoxComplementExsum(box_lens);
        fasttime_t end = gettime();
        double elapsed = tdiff(start, end);
        // elapsed_times.emplace_back(elapsed);
        elapsed_times[k] = elapsed;
      }

      // time summed area table + subtraction
      std::vector<double> elapsed_times_sat(trials);
      for (int k = 0; k < trials; k++) {
        tensor_test_2 = tensor_copy;
        fasttime_t start = gettime();
        tensor_test_2.SummedAreaTable(box_lens);
        fasttime_t end = gettime();
        double elapsed = tdiff(start, end);
        elapsed_times_sat[k] = elapsed;
      }

      // time incsum + subtraction
      std::vector<double> elapsed_times_incsum(trials);
      for (int k = 0; k < trials; k++) {
        tensor_ref = tensor_copy;
        fasttime_t start = gettime();
        tensor_ref.ExsumIncsumSubtraction(box_lens);
        fasttime_t end = gettime();
        double elapsed = tdiff(start, end);

        elapsed_times_incsum[k] = elapsed;
      }
      // time naive
      std::vector<double> elapsed_times_naive(trials);
      for (int k = 0; k < trials; k++) {
        tensor_naive = tensor_copy;
        fasttime_t start = gettime();
        tensor_naive.NaiveExsum(box_lens);
        fasttime_t end = gettime();
        double elapsed = tdiff(start, end);
        elapsed_times_naive[k] = elapsed;
      }


      int med_idx = trials / 2;

      // sort the times
      std::nth_element(elapsed_times.begin(),
                       elapsed_times.begin() + elapsed_times.size() / 2,
                       elapsed_times.end());
      std::nth_element(elapsed_times_sat.begin(),
                       elapsed_times_sat.begin() + elapsed_times_sat.size() / 2,
                       elapsed_times_sat.end());
      std::nth_element(
          elapsed_times_incsum.begin(),
          elapsed_times_incsum.begin() + elapsed_times_incsum.size() / 2,
          elapsed_times_incsum.end());
      std::nth_element(
          elapsed_times_naive.begin(),
          elapsed_times_naive.begin() + elapsed_times_naive.size() / 2,
          elapsed_times_naive.end());


#ifdef CHECK_RESULT
      for (size_t i = 0; i <= tensor.getAddress(tensor.getMaxIndex()); ++i) {
        EXPECT_NEAR(tensor_ref.data()[i], tensor_test.data()[i], N >> debug_shift);
      }
      for (size_t i = 0; i <= tensor.getAddress(tensor.getMaxIndex()); ++i) {
        EXPECT_NEAR(tensor_ref.data()[i], tensor_test_2.data()[i], N >> debug_shift);
      }
      // check naive
      for (size_t i = 0; i <= tensor.getAddress(tensor.getMaxIndex()); ++i) {
        // printf("checking naive idx %lu\n", i);
        EXPECT_NEAR(tensor_ref.data()[i], tensor_naive.data()[i], N >> debug_shift);
      }
#endif

      double naive_med = elapsed_times_naive[med_idx] * 1000;
      double sat_med = elapsed_times_sat[med_idx] * 1000;
      double incsum_med = elapsed_times_incsum[med_idx] * 1000;
      double boxcomp_med = elapsed_times[med_idx] * 1000;
      // dim, vol, SAT + subtraction time, BDBS + subtraction time, BOXCOMP
      // time
      // f << "," << sat_med << "," << incsum_med << "," << boxcomp_med ;

      f << "," << N << "," << naive_med << "," << sat_med << "," << incsum_med << ","
        << boxcomp_med;

      std::cout << "NAIVE Time = " << naive_med << std::endl;
      std::cout << "SAT Time = " << sat_med << std::endl;
      std::cout << "INCSUM Time = " << incsum_med << std::endl;
      std::cout << "BOXCOMP Time = " << boxcomp_med << std::endl;
    }
    f << std::endl;
  }

  // dim = 3
  {
    f << 3;
    constexpr size_t dim = 3;
    std::cout << "\nAlgorithm " << dim << std::endl;

    for (int p = lower; p < upper; p += increment) {
      size_t N;
#ifdef DEBUG
      N = 64;  // 4 ^ 3
#else
      N = p * box_len;
#endif

      size_t n = roundUp(std::floor(std::pow(N, 1. / dim)), box_len);
      N = std::pow(n, dim);
      index_t side_lens(dim);
      index_t box_lens(dim);
      std::fill(side_lens.begin(), side_lens.end(), n);
      std::cout << "n = " << n << std::endl;
      std::cout << "N = " << N << std::endl;

      exsum_tensor::Tensor<dtype, dim> tensor(side_lens);
      tensor.RandFill(0, 1);

      exsum_tensor::Tensor<dtype, dim> tensor_copy(tensor);
      std::fill(box_lens.begin(), box_lens.end(), box_len);

      // setup output tensors
      exsum_tensor::Tensor<dtype, dim> tensor_naive(tensor_copy);
      exsum_tensor::Tensor<dtype, dim> tensor_ref(tensor_copy);
      exsum_tensor::Tensor<dtype, dim> tensor_test(tensor_copy);
      exsum_tensor::Tensor<dtype, dim> tensor_test_2(tensor_copy);

      std::cout << "size = " << tensor_test.size() << std::endl;

      // time box complement
      std::vector<double> elapsed_times(trials);
      for (int k = 0; k < trials; k++) {
        tensor_test = tensor_copy;
        fasttime_t start = gettime();
        tensor_test.BoxComplementExsum(box_lens);
        fasttime_t end = gettime();
        double elapsed = tdiff(start, end);
        // elapsed_times.emplace_back(elapsed);
        elapsed_times[k] = elapsed;
      }

      // time summed area table + subtraction
      std::vector<double> elapsed_times_sat(trials);
      for (int k = 0; k < trials; k++) {
        tensor_test_2 = tensor_copy;
        fasttime_t start = gettime();
        tensor_test_2.SummedAreaTable(box_lens);
        fasttime_t end = gettime();
        double elapsed = tdiff(start, end);
        elapsed_times_sat[k] = elapsed;
      }

      // time incsum + subtraction
      std::vector<double> elapsed_times_incsum(trials);
      for (int k = 0; k < trials; k++) {
        tensor_ref = tensor_copy;
        fasttime_t start = gettime();
        tensor_ref.ExsumIncsumSubtraction(box_lens);
        fasttime_t end = gettime();
        double elapsed = tdiff(start, end);
        elapsed_times_incsum[k] = elapsed;
      }

      // time naive
      std::vector<double> elapsed_times_naive(trials);
      for (int k = 0; k < trials; k++) {
        tensor_naive = tensor_copy;
        fasttime_t start = gettime();
        tensor_naive.NaiveExsum(box_lens);
        fasttime_t end = gettime();
        double elapsed = tdiff(start, end);
        elapsed_times_naive[k] = elapsed;
      }

      int med_idx = trials / 2;
      // sort the times
      std::nth_element(elapsed_times.begin(),
                       elapsed_times.begin() + elapsed_times.size() / 2,
                       elapsed_times.end());
      std::nth_element(elapsed_times_sat.begin(),
                       elapsed_times_sat.begin() + elapsed_times_sat.size() / 2,
                       elapsed_times_sat.end());
      std::nth_element(
          elapsed_times_incsum.begin(),
          elapsed_times_incsum.begin() + elapsed_times_incsum.size() / 2,
          elapsed_times_incsum.end());
      std::nth_element(
          elapsed_times_naive.begin(),
          elapsed_times_naive.begin() + elapsed_times_naive.size() / 2,
          elapsed_times_naive.end());

// verify result
#ifdef CHECK_RESULT
      for (size_t i = 0; i <= tensor.getAddress(tensor.getMaxIndex()); ++i) {
        EXPECT_NEAR(tensor_ref.data()[i], tensor_test.data()[i], N >> debug_shift);
      }
      for (size_t i = 0; i <= tensor.getAddress(tensor.getMaxIndex()); ++i) {
        EXPECT_NEAR(tensor_ref.data()[i], tensor_test_2.data()[i], N >> debug_shift);
      }
      // check naive
      for (size_t i = 0; i <= tensor.getAddress(tensor.getMaxIndex()); ++i) {
        EXPECT_NEAR(tensor_ref.data()[i], tensor_naive.data()[i], N>>debug_shift); // tiny diff check
      }
#endif

      double sat_med = elapsed_times_sat[med_idx] * 1000;
      double incsum_med = elapsed_times_incsum[med_idx] * 1000;
      double boxcomp_med = elapsed_times[med_idx] * 1000;
      double naive_med = elapsed_times_naive[med_idx] * 1000;
      // dim, vol, SAT + subtraction time, BDBS + subtraction time, BOXCOMP
      // time
      f << "," << N << "," << naive_med << "," << sat_med << "," << incsum_med << ","
        << boxcomp_med;

      std::cout << "NAIVE Time = " << naive_med << std::endl;
      std::cout << "SAT Time = " << sat_med << std::endl;
      std::cout << "INCSUM Time = " << incsum_med << std::endl;
      std::cout << "BOXCOMP Time = " << boxcomp_med << std::endl;
    }
    f << std::endl;
  }

  // dim = 4
  {
    f << 4;
    constexpr size_t dim = 4;
    std::cout << "\nAlgorithm " << dim << std::endl;
    for (int p = lower; p < upper; p += increment) {
      size_t N;
#ifdef DEBUG
      N = 256;  // 4 ^ 4
#else
      N = p * box_len;
#endif
      size_t n = roundUp(std::floor(std::pow(N, 1. / dim)), box_len);
      N = std::pow(n, dim);
      index_t side_lens(dim);
      index_t box_lens(dim);
      std::fill(side_lens.begin(), side_lens.end(), n);
      std::cout << "n = " << n << std::endl;
      std::cout << "N = " << N << std::endl;

      exsum_tensor::Tensor<dtype, dim> tensor(side_lens);
      tensor.RandFill(0, 1);

      exsum_tensor::Tensor<dtype, dim> tensor_copy(tensor);
      std::fill(box_lens.begin(), box_lens.end(), box_len);

      // setup output tensors
      exsum_tensor::Tensor<dtype, dim> tensor_naive(tensor_copy);
      exsum_tensor::Tensor<dtype, dim> tensor_ref(tensor_copy);
      exsum_tensor::Tensor<dtype, dim> tensor_test(tensor_copy);
      exsum_tensor::Tensor<dtype, dim> tensor_test_2(tensor_copy);

      std::cout << "size = " << tensor_test.size() << std::endl;

      // time box complement
      std::vector<double> elapsed_times(trials);
      for (int k = 0; k < trials; k++) {
        tensor_test = tensor_copy;
        fasttime_t start = gettime();
        tensor_test.BoxComplementExsum(box_lens);
        fasttime_t end = gettime();
        double elapsed = tdiff(start, end);
        // elapsed_times.emplace_back(elapsed);
        elapsed_times[k] = elapsed;
      }

      // time summed area table + subtraction
      std::vector<double> elapsed_times_sat(trials);
      for (int k = 0; k < trials; k++) {
        tensor_test_2 = tensor_copy;
        fasttime_t start = gettime();
        tensor_test_2.SummedAreaTable(box_lens);
        fasttime_t end = gettime();
        double elapsed = tdiff(start, end);
        elapsed_times_sat[k] = elapsed;
      }

      // time incsum + subtraction
      std::vector<double> elapsed_times_incsum(trials);
      for (int k = 0; k < trials; k++) {
        tensor_ref = tensor_copy;
        fasttime_t start = gettime();
        tensor_ref.ExsumIncsumSubtraction(box_lens);
        fasttime_t end = gettime();
        double elapsed = tdiff(start, end);

        // elapsed_times_incsum.emplace_back(elapsed);
        elapsed_times_incsum[k] = elapsed;
      }

      // time naive
      std::vector<double> elapsed_times_naive(trials);
      for (int k = 0; k < trials; k++) {
        tensor_naive = tensor_copy;
        fasttime_t start = gettime();
        tensor_naive.NaiveExsum(box_lens);
        fasttime_t end = gettime();
        double elapsed = tdiff(start, end);
        elapsed_times_naive[k] = elapsed;
      }

      int med_idx = trials / 2;

      // sort the times
      std::nth_element(elapsed_times.begin(),
                       elapsed_times.begin() + elapsed_times.size() / 2,
                       elapsed_times.end());
      std::nth_element(elapsed_times_sat.begin(),
                       elapsed_times_sat.begin() + elapsed_times_sat.size() / 2,
                       elapsed_times_sat.end());
      std::nth_element(
          elapsed_times_incsum.begin(),
          elapsed_times_incsum.begin() + elapsed_times_incsum.size() / 2,
          elapsed_times_incsum.end());
      std::nth_element(
          elapsed_times_naive.begin(),
          elapsed_times_naive.begin() + elapsed_times_naive.size() / 2,
          elapsed_times_naive.end());

// verify result
#ifdef CHECK_RESULT
      for (size_t i = 0; i <= tensor.getAddress(tensor.getMaxIndex()); ++i) {
        EXPECT_NEAR(tensor_ref.data()[i], tensor_test.data()[i], N >> debug_shift);
      }
      for (size_t i = 0; i <= tensor.getAddress(tensor.getMaxIndex()); ++i) {
        EXPECT_NEAR(tensor_ref.data()[i], tensor_test_2.data()[i], N >> debug_shift);
      }
      // check naive
      for (size_t i = 0; i <= tensor.getAddress(tensor.getMaxIndex()); ++i) {
        EXPECT_NEAR(tensor_ref.data()[i], tensor_naive.data()[i], N>>debug_shift); // tiny diff check
      }
#endif
      double sat_med = elapsed_times_sat[med_idx] * 1000;
      double incsum_med = elapsed_times_incsum[med_idx] * 1000;
      double boxcomp_med = elapsed_times[med_idx] * 1000;
      double naive_med = elapsed_times_naive[med_idx] * 1000;
      // dim, vol, SAT + subtraction time, BDBS + subtraction time, BOXCOMP
      // time
      f << "," << N << "," << naive_med << "," << sat_med << "," << incsum_med << ","
        << boxcomp_med;

      std::cout << "NAIVE Time = " << naive_med << std::endl;
      std::cout << "SAT Time = " << sat_med << std::endl;
      std::cout << "INCSUM Time = " << incsum_med << std::endl;
      std::cout << "BOXCOMP Time = " << boxcomp_med << std::endl;
    }
    f << std::endl;
  }

  // dim = 5
  {
    f << 5;
    constexpr size_t dim = 5;
    std::cout << "\nAlgorithm " << dim << std::endl;
    for (int p = lower; p < upper; p += increment) {
      size_t N;
#ifdef DEBUG
      N = 1024;  // 4 ^ 5
#else
      N = p * box_len;
#endif

      size_t n = roundUp(std::floor(std::pow(N, 1. / dim)), box_len);
      N = std::pow(n, dim);
      index_t side_lens(dim);
      index_t box_lens(dim);
      std::fill(side_lens.begin(), side_lens.end(), n);
      std::cout << "n = " << n << std::endl;
      std::cout << "N = " << N << std::endl;

      exsum_tensor::Tensor<dtype, dim> tensor(side_lens);
      tensor.RandFill(0, 1);

      exsum_tensor::Tensor<dtype, dim> tensor_copy(tensor);
      std::fill(box_lens.begin(), box_lens.end(), box_len);
      // setup output tensors
      exsum_tensor::Tensor<dtype, dim> tensor_naive(tensor_copy);

      exsum_tensor::Tensor<dtype, dim> tensor_ref(tensor_copy);
      exsum_tensor::Tensor<dtype, dim> tensor_test(tensor_copy);
      exsum_tensor::Tensor<dtype, dim> tensor_test_2(tensor_copy);

      std::cout << "size = " << tensor_test.size() << std::endl;

      // time box complement
      std::vector<double> elapsed_times(trials);
      for (int k = 0; k < trials; k++) {
        tensor_test = tensor_copy;
        fasttime_t start = gettime();
        tensor_test.BoxComplementExsum(box_lens);
        fasttime_t end = gettime();
        double elapsed = tdiff(start, end);
        elapsed_times[k] = elapsed;
      }

      // time summed area table + subtraction
      std::vector<double> elapsed_times_sat(trials);
      for (int k = 0; k < trials; k++) {
        tensor_test_2 = tensor_copy;
        fasttime_t start = gettime();
        tensor_test_2.SummedAreaTable(box_lens);
        fasttime_t end = gettime();
        double elapsed = tdiff(start, end);
        elapsed_times_sat[k] = elapsed;
      }

      // time incsum + subtraction
      std::vector<double> elapsed_times_incsum(trials);
      for (int k = 0; k < trials; k++) {
        tensor_ref = tensor_copy;
        fasttime_t start = gettime();
        tensor_ref.ExsumIncsumSubtraction(box_lens);
        fasttime_t end = gettime();
        double elapsed = tdiff(start, end);
        elapsed_times_incsum[k] = elapsed;
      }
      // time naive
      std::vector<double> elapsed_times_naive(trials);
      for (int k = 0; k < trials; k++) {
        tensor_naive = tensor_copy;
        fasttime_t start = gettime();
        tensor_naive.NaiveExsum(box_lens);
        fasttime_t end = gettime();
        double elapsed = tdiff(start, end);
        elapsed_times_naive[k] = elapsed;
      }


      int med_idx = trials / 2;
      // sort the times
      std::nth_element(elapsed_times.begin(),
                       elapsed_times.begin() + elapsed_times.size() / 2,
                       elapsed_times.end());
      std::nth_element(elapsed_times_sat.begin(),
                       elapsed_times_sat.begin() + elapsed_times_sat.size() / 2,
                       elapsed_times_sat.end());
      std::nth_element(
          elapsed_times_incsum.begin(),
          elapsed_times_incsum.begin() + elapsed_times_incsum.size() / 2,
          elapsed_times_incsum.end());
      std::nth_element(
          elapsed_times_naive.begin(),
          elapsed_times_naive.begin() + elapsed_times_naive.size() / 2,
          elapsed_times_naive.end());


// verify result
#ifdef CHECK_RESULT
      for (size_t i = 0; i <= tensor.getAddress(tensor.getMaxIndex()); ++i) {
        EXPECT_NEAR(tensor_ref.data()[i], tensor_test.data()[i], N >> debug_shift);
      }
      for (size_t i = 0; i <= tensor.getAddress(tensor.getMaxIndex()); ++i) {
        EXPECT_NEAR(tensor_ref.data()[i], tensor_test_2.data()[i], N >> debug_shift);
      }
      // check naive
      for (size_t i = 0; i <= tensor.getAddress(tensor.getMaxIndex()); ++i) {
        EXPECT_NEAR(tensor_ref.data()[i], tensor_naive.data()[i], N>>debug_shift); // tiny diff check
      }

#endif

      // write median to file
      double sat_med = elapsed_times_sat[med_idx] * 1000;
      double incsum_med = elapsed_times_incsum[med_idx] * 1000;
      double boxcomp_med = elapsed_times[med_idx] * 1000;
      double naive_med = elapsed_times_naive[med_idx] * 1000;
      // dim, vol, SAT + subtraction time, BDBS + subtraction time, BOXCOMP
      // time
      f << "," << N << "," << naive_med << "," << sat_med << "," << incsum_med << ","
        << boxcomp_med;

      std::cout << "NAIVE Time = " << naive_med << std::endl;

      std::cout << "SAT Time = " << sat_med << std::endl;
      std::cout << "INCSUM Time = " << incsum_med << std::endl;
      std::cout << "BOXCOMP Time = " << boxcomp_med << std::endl;
    }
    f << std::endl;
  }

  return;
  // dim = 6
  {
    f << 6;

    constexpr size_t dim = 6;
    std::cout << "\nAlgorithm " << dim << std::endl;
    for (int p = lower; p < upper; p += increment) {
      size_t N;
#ifdef DEBUG
      N = 4096;  // 4 ^ 6
#else
      N = p * box_len;
#endif

      size_t n = roundUp(std::floor(std::pow(N, 1. / dim)), box_len);
      N = std::pow(n, dim);

      index_t side_lens(dim);
      index_t box_lens(dim);
      std::fill(side_lens.begin(), side_lens.end(), n);
      std::cout << "n = " << n << std::endl;
      std::cout << "N = " << N << std::endl;

      exsum_tensor::Tensor<dtype, dim> tensor(side_lens);
      tensor.RandFill(0, 1);

      exsum_tensor::Tensor<dtype, dim> tensor_copy(tensor);
      std::fill(box_lens.begin(), box_lens.end(), box_len);

      exsum_tensor::Tensor<dtype, dim> tensor_ref(tensor_copy);
      exsum_tensor::Tensor<dtype, dim> tensor_test(tensor_copy);
      exsum_tensor::Tensor<dtype, dim> tensor_test_2(tensor_copy);

      std::cout << "size = " << tensor_test.size() << std::endl;

      // time box complement
      std::vector<double> elapsed_times(trials);
      for (int k = 0; k < trials; k++) {
        tensor_test = tensor_copy;
        fasttime_t start = gettime();
        tensor_test.BoxComplementExsum(box_lens);
        fasttime_t end = gettime();
        double elapsed = tdiff(start, end);
        // elapsed_times.emplace_back(elapsed);
        elapsed_times[k] = elapsed;
      }

      // time summed area table + subtraction
      std::vector<double> elapsed_times_sat(trials);
      for (int k = 0; k < trials; k++) {
        tensor_test_2 = tensor_copy;
        fasttime_t start = gettime();
        tensor_test_2.SummedAreaTable(box_lens);
        fasttime_t end = gettime();
        double elapsed = tdiff(start, end);
        elapsed_times_sat[k] = elapsed;
      }

      // time incsum + subtraction
      std::vector<double> elapsed_times_incsum(trials);
      for (int k = 0; k < trials; k++) {
        tensor_ref = tensor_copy;
        fasttime_t start = gettime();
        tensor_ref.ExsumIncsumSubtraction(box_lens);
        fasttime_t end = gettime();
        double elapsed = tdiff(start, end);

        // elapsed_times_incsum.emplace_back(elapsed);
        elapsed_times_incsum[k] = elapsed;
      }

      int med_idx = trials / 2;
      // sort the times
      std::nth_element(elapsed_times.begin(),
                       elapsed_times.begin() + elapsed_times.size() / 2,
                       elapsed_times.end());
      std::nth_element(elapsed_times_sat.begin(),
                       elapsed_times_sat.begin() + elapsed_times_sat.size() / 2,
                       elapsed_times_sat.end());
      std::nth_element(
          elapsed_times_incsum.begin(),
          elapsed_times_incsum.begin() + elapsed_times_incsum.size() / 2,
          elapsed_times_incsum.end());

// verify result
#ifdef CHECK_RESULT
      for (size_t i = 0; i <= tensor.getAddress(tensor.getMaxIndex()); ++i) {
        EXPECT_NEAR(tensor_ref.data()[i], tensor_test.data()[i], N >> debug_shift);
      }
      for (size_t i = 0; i <= tensor.getAddress(tensor.getMaxIndex()); ++i) {
        EXPECT_NEAR(tensor_ref.data()[i], tensor_test_2.data()[i], N >> debug_shift);
      }
#endif

      // write median to file
      double sat_med = elapsed_times_sat[med_idx] * 1000;
      double incsum_med = elapsed_times_incsum[med_idx] * 1000;
      double boxcomp_med = elapsed_times[med_idx] * 1000;

      // dim, vol, SAT + subtraction time, BDBS + subtraction time, BOXCOMP
      // time
      /*
      f << "," << N << "," << sat_med << "," << incsum_med << ","
        << boxcomp_med;
      */
      f << "," << N << ",0," << sat_med << "," << incsum_med << ","
        << boxcomp_med;
      std::cout << "SAT Time = " << sat_med << std::endl;
      std::cout << "INCSUM Time = " << incsum_med << std::endl;
      std::cout << "BOXCOMP Time = " << boxcomp_med << std::endl;
    }
    f << std::endl;
  }

  // dim = 7
  {
    f << 7;
    constexpr size_t dim = 7;
    std::cout << "\nAlgorithm " << dim << std::endl;
    for (int p = lower; p < upper; p += increment) {
      size_t N;
#ifdef DEBUG
      N = 16384;  // 4 ^ 7
#else
      N = p * box_len;
#endif

      size_t n = roundUp(std::floor(std::pow(N, 1. / dim)), box_len);
      N = std::pow(n, dim);
      index_t side_lens(dim);
      index_t box_lens(dim);
      std::fill(side_lens.begin(), side_lens.end(), n);
      std::cout << "n = " << n << std::endl;
      std::cout << "N = " << N << std::endl;

      exsum_tensor::Tensor<dtype, dim> tensor(side_lens);
      tensor.RandFill(0, 1);

      exsum_tensor::Tensor<dtype, dim> tensor_copy(tensor);
      std::fill(box_lens.begin(), box_lens.end(), box_len);

      exsum_tensor::Tensor<dtype, dim> tensor_ref(tensor_copy);
      exsum_tensor::Tensor<dtype, dim> tensor_test(tensor_copy);
      exsum_tensor::Tensor<dtype, dim> tensor_test_2(tensor_copy);

      std::cout << "size = " << tensor_test.size() << std::endl;

      // time box complement
      std::vector<double> elapsed_times(trials);
      for (int k = 0; k < trials; k++) {
        tensor_test = tensor_copy;
        fasttime_t start = gettime();
        tensor_test.BoxComplementExsum(box_lens);
        fasttime_t end = gettime();
        double elapsed = tdiff(start, end);
        // elapsed_times.emplace_back(elapsed);
        elapsed_times[k] = elapsed;
      }

      // time summed area table + subtraction
      std::vector<double> elapsed_times_sat(trials);
      for (int k = 0; k < trials; k++) {
        tensor_test_2 = tensor_copy;
        fasttime_t start = gettime();
        tensor_test_2.SummedAreaTable(box_lens);
        fasttime_t end = gettime();
        double elapsed = tdiff(start, end);
        elapsed_times_sat[k] = elapsed;
      }

      // time incsum + subtraction
      std::vector<double> elapsed_times_incsum(trials);
      for (int k = 0; k < trials; k++) {
        tensor_ref = tensor_copy;
        fasttime_t start = gettime();
        tensor_ref.ExsumIncsumSubtraction(box_lens);
        fasttime_t end = gettime();
        double elapsed = tdiff(start, end);

        // elapsed_times_incsum.emplace_back(elapsed);
        elapsed_times_incsum[k] = elapsed;
      }

      int med_idx = trials / 2;
      // sort the times
      std::nth_element(elapsed_times.begin(),
                       elapsed_times.begin() + elapsed_times.size() / 2,
                       elapsed_times.end());
      std::nth_element(elapsed_times_sat.begin(),
                       elapsed_times_sat.begin() + elapsed_times_sat.size() / 2,
                       elapsed_times_sat.end());
      std::nth_element(
          elapsed_times_incsum.begin(),
          elapsed_times_incsum.begin() + elapsed_times_incsum.size() / 2,
          elapsed_times_incsum.end());

// verify result
#ifdef CHECK_RESULT
      for (size_t i = 0; i <= tensor.getAddress(tensor.getMaxIndex()); ++i) {
        EXPECT_NEAR(tensor_ref.data()[i], tensor_test.data()[i], N >> debug_shift);
      }
      for (size_t i = 0; i <= tensor.getAddress(tensor.getMaxIndex()); ++i) {
        EXPECT_NEAR(tensor_ref.data()[i], tensor_test_2.data()[i], N >> debug_shift);
      }
#endif

      // write median to file
      double sat_med = elapsed_times_sat[med_idx] * 1000;
      double incsum_med = elapsed_times_incsum[med_idx] * 1000;
      double boxcomp_med = elapsed_times[med_idx] * 1000;

      // dim, vol, SAT + subtraction time, BDBS + subtraction time, BOXCOMP
      // time
      /*
      f << "," << N << "," << sat_med << "," << incsum_med << ","
        << boxcomp_med;
      */
      f << "," << N << ",0," << sat_med << "," << incsum_med << ","
        << boxcomp_med;
      std::cout << "SAT Time = " << sat_med << std::endl;
      std::cout << "INCSUM Time = " << incsum_med << std::endl;
      std::cout << "BOXCOMP Time = " << boxcomp_med << std::endl;
    }

    f << std::endl;
  }
  // dim = 8
  {
    f << 8;
    constexpr size_t dim = 8;
    std::cout << "\nAlgorithm " << dim << std::endl;
    for (int p = lower; p < upper; p += increment) {
      size_t N;
#ifdef DEBUG
      N = 65536;  // 4 ^ 8
#else
      N = p * box_len;
#endif

      size_t n = roundUp(std::floor(std::pow(N, 1. / dim)), box_len);
      N = std::pow(n, dim);
      index_t side_lens(dim);
      index_t box_lens(dim);
      std::fill(side_lens.begin(), side_lens.end(), n);
      std::cout << "n = " << n << std::endl;
      std::cout << "N = " << N << std::endl;

      exsum_tensor::Tensor<dtype, dim> tensor(side_lens);
      tensor.RandFill(0, 1);

      exsum_tensor::Tensor<dtype, dim> tensor_copy(tensor);
      std::fill(box_lens.begin(), box_lens.end(), box_len);

      exsum_tensor::Tensor<dtype, dim> tensor_ref(tensor_copy);
      exsum_tensor::Tensor<dtype, dim> tensor_test(tensor_copy);
      exsum_tensor::Tensor<dtype, dim> tensor_test_2(tensor_copy);

      std::cout << "size = " << tensor_test.size() << std::endl;

      // time box complement
      std::vector<double> elapsed_times(trials);
      for (int k = 0; k < trials; k++) {
        tensor_test = tensor_copy;
        fasttime_t start = gettime();
        tensor_test.BoxComplementExsum(box_lens);
        fasttime_t end = gettime();
        double elapsed = tdiff(start, end);
        // elapsed_times.emplace_back(elapsed);
        elapsed_times[k] = elapsed;
      }

      // time summed area table + subtraction
      std::vector<double> elapsed_times_sat(trials);
      for (int k = 0; k < trials; k++) {
        tensor_test_2 = tensor_copy;
        fasttime_t start = gettime();
        tensor_test_2.SummedAreaTable(box_lens);
        fasttime_t end = gettime();
        double elapsed = tdiff(start, end);
        elapsed_times_sat[k] = elapsed;
      }

      // time incsum + subtraction
      std::vector<double> elapsed_times_incsum(trials);
      for (int k = 0; k < trials; k++) {
        tensor_ref = tensor_copy;
        fasttime_t start = gettime();
        tensor_ref.ExsumIncsumSubtraction(box_lens);
        fasttime_t end = gettime();
        double elapsed = tdiff(start, end);

        // elapsed_times_incsum.emplace_back(elapsed);
        elapsed_times_incsum[k] = elapsed;
      }

      int med_idx = trials / 2;
      // sort the times
      std::nth_element(elapsed_times.begin(),
                       elapsed_times.begin() + elapsed_times.size() / 2,
                       elapsed_times.end());
      std::nth_element(elapsed_times_sat.begin(),
                       elapsed_times_sat.begin() + elapsed_times_sat.size() / 2,
                       elapsed_times_sat.end());
      std::nth_element(
          elapsed_times_incsum.begin(),
          elapsed_times_incsum.begin() + elapsed_times_incsum.size() / 2,
          elapsed_times_incsum.end());

// verify result
#ifdef CHECK_RESULT
      for (size_t i = 0; i <= tensor.getAddress(tensor.getMaxIndex()); ++i) {
        EXPECT_NEAR(tensor_ref.data()[i], tensor_test.data()[i], N >> debug_shift);
      }
      for (size_t i = 0; i <= tensor.getAddress(tensor.getMaxIndex()); ++i) {
        EXPECT_NEAR(tensor_ref.data()[i], tensor_test_2.data()[i], N >> debug_shift);
      }
#endif

      // write median to file
      double sat_med = elapsed_times_sat[med_idx] * 1000;
      double incsum_med = elapsed_times_incsum[med_idx] * 1000;
      double boxcomp_med = elapsed_times[med_idx] * 1000;

      // dim, vol, SAT + subtraction time, BDBS + subtraction time, BOXCOMP
      // time
      /*
      f << "," << N << "," << sat_med << "," << incsum_med << ","
        << boxcomp_med;
      */
      f << "," << N << ",0," << sat_med << "," << incsum_med << ","
        << boxcomp_med;
      std::cout << "SAT Time = " << sat_med << std::endl;
      std::cout << "INCSUM Time = " << incsum_med << std::endl;
      std::cout << "BOXCOMP Time = " << boxcomp_med << std::endl;
    }

    f << std::endl;
  }
  // dim = 9
  {
    f << 9;
    constexpr size_t dim = 9;
    std::cout << "\nAlgorithm " << dim << std::endl;
    for (int p = lower; p < upper; p += increment) {
      size_t N;
#ifdef DEBUG
      N = 262144;  // 4 ^ 9
#else
      N = p * box_len;
#endif

      size_t n = roundUp(std::floor(std::pow(N, 1. / dim)), box_len);
      N = std::pow(n, dim);
      index_t side_lens(dim);
      index_t box_lens(dim);
      std::fill(side_lens.begin(), side_lens.end(), n);
      std::cout << "n = " << n << std::endl;
      std::cout << "N = " << N << std::endl;

      exsum_tensor::Tensor<dtype, dim> tensor(side_lens);
      tensor.RandFill(0, 1);

      exsum_tensor::Tensor<dtype, dim> tensor_copy(tensor);
      std::fill(box_lens.begin(), box_lens.end(), box_len);

      exsum_tensor::Tensor<dtype, dim> tensor_ref(tensor_copy);
      exsum_tensor::Tensor<dtype, dim> tensor_test(tensor_copy);
      exsum_tensor::Tensor<dtype, dim> tensor_test_2(tensor_copy);

      std::cout << "size = " << tensor_test.size() << std::endl;

      // time box complement
      std::vector<double> elapsed_times(trials);
      for (int k = 0; k < trials; k++) {
        tensor_test = tensor_copy;
        fasttime_t start = gettime();
        tensor_test.BoxComplementExsum(box_lens);
        fasttime_t end = gettime();
        double elapsed = tdiff(start, end);
        // elapsed_times.emplace_back(elapsed);
        elapsed_times[k] = elapsed;
      }

      // time summed area table + subtraction
      std::vector<double> elapsed_times_sat(trials);
      for (int k = 0; k < trials; k++) {
        tensor_test_2 = tensor_copy;
        fasttime_t start = gettime();
        tensor_test_2.SummedAreaTable(box_lens);
        fasttime_t end = gettime();
        double elapsed = tdiff(start, end);
        elapsed_times_sat[k] = elapsed;
      }

      // time incsum + subtraction
      std::vector<double> elapsed_times_incsum(trials);
      for (int k = 0; k < trials; k++) {
        tensor_ref = tensor_copy;
        fasttime_t start = gettime();
        tensor_ref.ExsumIncsumSubtraction(box_lens);
        fasttime_t end = gettime();
        double elapsed = tdiff(start, end);

        // elapsed_times_incsum.emplace_back(elapsed);
        elapsed_times_incsum[k] = elapsed;
      }

      int med_idx = trials / 2;
      // sort the times
      std::nth_element(elapsed_times.begin(),
                       elapsed_times.begin() + elapsed_times.size() / 2,
                       elapsed_times.end());
      std::nth_element(elapsed_times_sat.begin(),
                       elapsed_times_sat.begin() + elapsed_times_sat.size() / 2,
                       elapsed_times_sat.end());
      std::nth_element(
          elapsed_times_incsum.begin(),
          elapsed_times_incsum.begin() + elapsed_times_incsum.size() / 2,
          elapsed_times_incsum.end());

// verify result
#ifdef CHECK_RESULT
      for (size_t i = 0; i <= tensor.getAddress(tensor.getMaxIndex()); ++i) {
        EXPECT_NEAR(tensor_ref.data()[i], tensor_test.data()[i], N >> debug_shift);
      }
      for (size_t i = 0; i <= tensor.getAddress(tensor.getMaxIndex()); ++i) {
        EXPECT_NEAR(tensor_ref.data()[i], tensor_test_2.data()[i], N >> debug_shift);
      }
#endif

      // write median to file
      double sat_med = elapsed_times_sat[med_idx] * 1000;
      double incsum_med = elapsed_times_incsum[med_idx] * 1000;
      double boxcomp_med = elapsed_times[med_idx] * 1000;

      // dim, vol, SAT + subtraction time, BDBS + subtraction time, BOXCOMP
      // time
      /*
      f << "," << N << "," << sat_med << "," << incsum_med << ","
        << boxcomp_med;
      */
      f << "," << N << ",0," << sat_med << "," << incsum_med << ","
        << boxcomp_med;
      std::cout << "SAT Time = " << sat_med << std::endl;
      std::cout << "INCSUM Time = " << incsum_med << std::endl;
      std::cout << "BOXCOMP Time = " << boxcomp_med << std::endl;
    }

    f << std::endl;
  }
  // TODO: make this a function? not sure how to do it in gtest
  // dim = 10
  {
    f << 10;
    constexpr size_t dim = 10;
    std::cout << "\nAlgorithm " << dim << std::endl;
    for (int p = lower; p < upper; p += increment) {
      size_t N;
#ifdef DEBUG
      N = 1048576;  // 4 ^ 10
#else
      N = p * box_len;
#endif

      size_t n = roundUp(std::floor(std::pow(N, 1. / dim)), box_len);
      N = std::pow(n, dim);
      index_t side_lens(dim);
      index_t box_lens(dim);
      std::fill(side_lens.begin(), side_lens.end(), n);
      std::cout << "n = " << n << std::endl;
      std::cout << "N = " << N << std::endl;

      exsum_tensor::Tensor<dtype, dim> tensor(side_lens);
      tensor.RandFill(0, 1);

      exsum_tensor::Tensor<dtype, dim> tensor_copy(tensor);
      std::fill(box_lens.begin(), box_lens.end(), box_len);

      exsum_tensor::Tensor<dtype, dim> tensor_ref(tensor_copy);
      exsum_tensor::Tensor<dtype, dim> tensor_test(tensor_copy);
      exsum_tensor::Tensor<dtype, dim> tensor_test_2(tensor_copy);

      std::cout << "size = " << tensor_test.size() << std::endl;

      // time box complement
      std::vector<double> elapsed_times(trials);
      for (int k = 0; k < trials; k++) {
        tensor_test = tensor_copy;
        fasttime_t start = gettime();
        tensor_test.BoxComplementExsum(box_lens);
        fasttime_t end = gettime();
        double elapsed = tdiff(start, end);
        // elapsed_times.emplace_back(elapsed);
        elapsed_times[k] = elapsed;
      }

      // time summed area table + subtraction
      std::vector<double> elapsed_times_sat(trials);
      for (int k = 0; k < trials; k++) {
        tensor_test_2 = tensor_copy;
        fasttime_t start = gettime();
        tensor_test_2.SummedAreaTable(box_lens);
        fasttime_t end = gettime();
        double elapsed = tdiff(start, end);
        elapsed_times_sat[k] = elapsed;
      }

      // time incsum + subtraction
      std::vector<double> elapsed_times_incsum(trials);
      for (int k = 0; k < trials; k++) {
        tensor_ref = tensor_copy;
        fasttime_t start = gettime();
        tensor_ref.ExsumIncsumSubtraction(box_lens);
        fasttime_t end = gettime();
        double elapsed = tdiff(start, end);

        // elapsed_times_incsum.emplace_back(elapsed);
        elapsed_times_incsum[k] = elapsed;
      }

      int med_idx = trials / 2;
      // sort the times
      std::nth_element(elapsed_times.begin(),
                       elapsed_times.begin() + elapsed_times.size() / 2,
                       elapsed_times.end());
      std::nth_element(elapsed_times_sat.begin(),
                       elapsed_times_sat.begin() + elapsed_times_sat.size() / 2,
                       elapsed_times_sat.end());
      std::nth_element(
          elapsed_times_incsum.begin(),
          elapsed_times_incsum.begin() + elapsed_times_incsum.size() / 2,
          elapsed_times_incsum.end());

// verify result
#ifdef CHECK_RESULT
      for (size_t i = 0; i <= tensor.getAddress(tensor.getMaxIndex()); ++i) {
        EXPECT_NEAR(tensor_ref.data()[i], tensor_test.data()[i], N >> debug_shift);
      }
      for (size_t i = 0; i <= tensor.getAddress(tensor.getMaxIndex()); ++i) {
        EXPECT_NEAR(tensor_ref.data()[i], tensor_test_2.data()[i], N >> debug_shift);
      }
#endif

      // write median to file
      double sat_med = elapsed_times_sat[med_idx] * 1000;
      double incsum_med = elapsed_times_incsum[med_idx] * 1000;
      double boxcomp_med = elapsed_times[med_idx] * 1000;

      // dim, vol, SAT + subtraction time, BDBS + subtraction time, BOXCOMP
      // time
      /*
      f << "," << N << "," << sat_med << "," << incsum_med << ","
        << boxcomp_med;
      */
      f << "," << N << ",0," << sat_med << "," << incsum_med << ","
        << boxcomp_med;
      std::cout << "SAT Time = " << sat_med << std::endl;
      std::cout << "INCSUM Time = " << incsum_med << std::endl;
      std::cout << "BOXCOMP Time = " << boxcomp_med << std::endl;
    }

    f << std::endl;
  }
  f.close();
}

}  // namespace
}  // namespace excluded_sum

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
