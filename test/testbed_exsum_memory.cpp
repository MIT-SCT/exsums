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

#include "fmm_2d.h"
#include "fmm_3d.h"
#include "fasttime.h"
#include "tensor.h"
#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <iterator>
#include <numeric>  // for inclusive_scan
#include <random>
#include <type_traits>
#include <vector>

#include <map>

std::size_t allocated = 0;

// max memory usage
size_t max_memory_usage = 0;

template<typename T>
struct track_alloc : std::allocator<T> {
    typedef typename std::allocator<T>::pointer pointer;
    typedef typename std::allocator<T>::size_type size_type;

    template<typename U>
    struct rebind {
        typedef track_alloc<U> other;
    };

    track_alloc() {}

    template<typename U>
    track_alloc(track_alloc<U> const& u)
        :std::allocator<T>(u) {}

    pointer allocate(size_type size, 
                     std::allocator<void>::const_pointer = 0) {
        void * p = std::malloc(size * sizeof(T));
        if(p == 0) {
            throw std::bad_alloc();
        }
        return static_cast<pointer>(p);
    }

    void deallocate(pointer p, size_type) {
        std::free(p);
    }
};

typedef std::map< void*, std::size_t, std::less<void*>, 
                  track_alloc< std::pair<void* const, std::size_t> > > track_type;

struct track_printer {
    track_type * track;
    track_printer(track_type * track):track(track) {}
    ~track_printer() {
        track_type::const_iterator it = track->begin();
        while(it != track->end()) {
            std::cerr << "TRACK: leaked at " << it->first << ", "
                      << it->second << " bytes\n";
            ++it;
        }
    }
};

track_type * get_map() {
    // don't use normal new to avoid infinite recursion.
    static track_type * track = new (std::malloc(sizeof *track)) 
        track_type;
    static track_printer printer(track);
    return track;
}

void * operator new(std::size_t size) noexcept(false) { //(std::bad_alloc) {
    // we are required to return non-null
    void * mem = std::malloc(size == 0 ? 1 : size);

    if(mem == 0) {
        throw std::bad_alloc();
    }

    (*get_map())[mem] = size;

    allocated += size;
    if (allocated > max_memory_usage) {
      max_memory_usage = allocated;
    }

    return mem;
}

void operator delete(void * mem) throw() {
    allocated -= (*get_map())[mem];
    if(get_map()->erase(mem) == 0) {
        // this indicates a serious bug
        std::cerr << "bug: memory at " 
                  << mem << " wasn't allocated by us\n";
    }
    std::free(mem);
}


int main(int argc, char** argv) {
  // initialize type and size
  typedef double dtype;
  typedef const vector<size_t> index_t;

  int lower = 2;

  int upper;
#ifdef DEBUG
  upper = lower + 1;
#else
  upper = 24;
#endif

  int increment = 4;

  size_t box_len;
#ifdef DEBUG
  box_len = 2;
#else
  box_len = 4;
#endif

  std::ofstream f("3D-exsum-double-space.csv");

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

  // lines separate the different algorithms
  // columns are the CSV values
  if (f.is_open()) {
    f << "Algorithm Num";
    for (int p = lower; p < upper; p += increment) {
      const size_t N = p * box_len;
      f << ",space usage for ";
      f << N * N * N << " elems";
    }
    f << std::endl;
    f << "X-data";
    for (int p = lower; p < upper; p += increment) {
      const size_t N = p * box_len;
      f << "," << N * N * N;
    }
    f << std::endl;
  }

  f.precision(10);

  int len = sizeof(funcs) / sizeof(funcs[0]);

  for (int j = 0; j < len; j++) {
    f << j;
    std::cout << "Algorithm " << j << std::endl;
    for (int p = lower; p < upper; p += increment) {
      const size_t N = p * box_len;
      index_t side_lens {N, N, N};
      std::cout << "N = " << N*N*N << std::endl;

      exsum_tensor::Tensor_3D<dtype> tensor(side_lens);

      tensor.RandFill(0, 1);

      exsum_tensor::Tensor_3D<dtype> tensor_copy(tensor);
      index_t box_lens {box_len, box_len, box_len};

      max_memory_usage = 0;
      allocated = 0;

      // run the test
      (tensor_copy.*funcs[j])(box_lens);

      std::printf("max memory for alg %d at size %lu= %zu\n", j, N*N*N, max_memory_usage);

      f << "," << max_memory_usage;

    }
    f << std::endl;
  }
  f.close();
  return 0;
}
