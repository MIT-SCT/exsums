/**
 * Copyright (c) 2020 MIT License by CSAIL Supertech Group
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
#include <fcntl.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include "tbassert.h"
#include "tensor.h"

namespace exsum_tensor {

template <typename T>
class Tensor_3D : public Tensor<T, 3> {
  typedef const vector<size_t> index_t;
  typedef vector<size_t> mod_index_t;

 public:
  Tensor_3D() = default;
  Tensor_3D(const Tensor_3D&) = default;
  Tensor_3D(Tensor_3D&&) = default;
  Tensor_3D& operator=(const Tensor_3D&) = default;
  ~Tensor_3D() = default;

  // gray code delta encoded for 3D
  // original gray code in binary:
  // 000 001 011 010 110 111 101 100
  // TODO: need to gen this for arbitrary dims

  // we want the position of the bit changed
  // size_t gray_diff_3D[8] = {1, 1, 2, 1, 4, 1, 2, 1};
  size_t gray_diff_3D[8] = {0, 0, 1, 0, 2, 0, 1, 0};
  size_t add_or_subtract[8] = {0, 1, 1, 0, 1, 1, 0, 0};

  explicit Tensor_3D(index_t& side_lens) {
    this->dim_lens = side_lens;
    padded_side_lens = side_lens;
    size_t s = side_lens[0];
    for (size_t n = 1; n < this->order; ++n) {
      s *= this->dim_lens[n];
    }
    this->elems = vector<T>(s);
  }

  // naive included sums
  void IncsumCheck(index_t& box_lens) {
    for (size_t i = 0; i < this->dim_lens[0]; ++i) {
      for (size_t j = 0; j < this->dim_lens[1]; ++j) {
        for (size_t k = 0; k < this->dim_lens[2]; ++k) {
          T box_total = 0;
          index_t set_index = {i, j, k};
          for (size_t ki = 0; ki < box_lens[0]; ++ki) {
            for (size_t kj = 0; kj < box_lens[1]; ++kj) {
              for (size_t kk = 0; kk < box_lens[2]; ++kk) {
                if (i + ki < this->padded_side_lens[0] &&
                    j + kj < this->padded_side_lens[1] &&
                    k + kk < this->padded_side_lens[2]) {
                  index_t get_index = {i + ki, j + kj, k + kk};
#ifdef DELAY
                  volatile long x = fib(DELAY);
#endif
                  box_total += this->GetElt(get_index);
                }
              }
            }
          }
          this->SetElt(set_index, box_total);
        }
      }
    }
  }

  // naive excluded sums (with subtraction)
  void ExsumCheckSubtraction(index_t& box_lens) {
    T entire_total = 0;
    for (size_t i = 0; i < this->dim_lens[0]; ++i) {
      for (size_t j = 0; j < this->dim_lens[1]; ++j) {
        for (size_t k = 0; k < this->dim_lens[2]; ++k) {
          index_t get_index = {i, j, k};
#ifdef DELAY
          volatile long x = fib(DELAY);
#endif
#ifdef DEBUG
          printf("val at idx (%lu, %lu, %lu) = %f\n", i, j, k,
                 this->GetElt(get_index));
#endif
          entire_total += this->GetElt(get_index);
        }
      }
    }
    Tensor_3D<T> copy(*this);
    copy.IncsumCheck(box_lens);
    for (size_t i = 0; i < this->dim_lens[0]; ++i) {
      for (size_t j = 0; j < this->dim_lens[1]; ++j) {
        for (size_t k = 0; k < this->dim_lens[2]; ++k) {
          index_t index = {i, j, k};
#ifdef DELAY
          volatile long x = fib(DELAY);
#endif
          this->SetElt(index, entire_total - copy.GetElt(index));
        }
      }
    }
  }

  // naive excluded sums (without subtraction)
  void ExsumCheckNaive(index_t& box_lens) {
    Tensor_3D<T> copy(*this);
    for (size_t i = 0; i < this->dim_lens[0]; ++i) {
      for (size_t j = 0; j < this->dim_lens[1]; ++j) {
        for (size_t k = 0; k < this->dim_lens[2]; ++k) {
          T box_total = 0;
          index_t set_index = {i, j, k};
          for (size_t ki = 0; ki < this->dim_lens[0]; ++ki) {
            for (size_t kj = 0; kj < this->dim_lens[1]; ++kj) {
              for (size_t kk = 0; kk < this->dim_lens[2]; ++kk) {
                // check if index is in the box. Only add if not in box
                if ((ki >= i && ki < (i + box_lens[0])) &&
                    (kj >= j && kj < (j + box_lens[1])) &&
                    (kk >= k && kk < (k + box_lens[2]))) {
                  continue;
                }
                index_t get_index = {ki, kj, kk};
#ifdef DELAY
                volatile long x = fib(DELAY);
#endif
                box_total += copy.GetElt(get_index);
              }
            }
          }
          this->SetElt(set_index, box_total);
        }
      }
    }
  }

  // This implementation is the dN space variant implementation
  void CornersExsum_dN(index_t& box_lens) {
    std::vector<Tensor_3D<T>> dims;
    for (size_t i = 0; i < this->order; i++) {
      dims.emplace_back(Tensor_3D<T>(padded_side_lens));
    }
    Tensor_3D<T> output(padded_side_lens);
    // zero output
    output.ZeroFill();
    size_t prev = 1 << this->order - 1;
    for (size_t leaf = 0; leaf < (1 << this->order); leaf++) {
      size_t diff_index = log2(leaf ^ prev);
      // this is the index in the bitstring from which we need to recompute
      diff_index = this->order - 1 - diff_index;
      prev = leaf;
      for (size_t i = diff_index; i < this->order; i++) {
        if (i == 0) {
          dims[i] = Tensor_3D<T>(*this);
        } else {
          dims[i] = Tensor_3D<T>(dims[i - 1]);
        }
        size_t mask = 1 << (this->order - 1 - i);
        // if this bit is not set, then do a prefix, otherwise suffix
        if ((leaf & mask) == 0) {
          dims[i].prefix_through_dim(i);
        } else {
          dims[i].suffix_through_dim(i);
        }
      }
      // at end of each leaf bitstring, now add contribution using map from
      // bitstring to coordinate
      for (size_t i = 0; i < this->dim_lens[0]; ++i) {
        for (size_t j = 0; j < this->dim_lens[1]; ++j) {
          for (size_t k = 0; k < this->dim_lens[2]; ++k) {
            size_t ki = box_lens[0], kj = box_lens[1], kk = box_lens[2];
            size_t max_i = this->dim_lens[0] - 1;
            size_t max_j = this->dim_lens[1] - 1;
            size_t max_k = this->dim_lens[2] - 1;
            index_t index = {i, j, k};
            index_t PPP_index = {i - 1, j - 1, k - 1};
            index_t PPS_index = {std::min(i + ki - 1, max_i), j - 1, k};
            index_t PSP_index = {i - 1, j, std::min(k + kk - 1, max_k)};
            index_t PSS_index = {std::min(i + ki - 1, max_i), j, k + kk};
            index_t SPP_index = {i, std::min(j + kj - 1, max_j), k - 1};
            index_t SPS_index = {i + ki, std::min(j + kj - 1, max_j), k};
            index_t SSP_index = {i, j + kj, std::min(k + kk - 1, max_k)};
            index_t SSS_index = {i + ki, j + kj, k + kk};
            vector<vector<size_t>> indices;
            indices.push_back(PPP_index);
            indices.push_back(PPS_index);
            indices.push_back(PSP_index);
            indices.push_back(PSS_index);
            indices.push_back(SPP_index);
            indices.push_back(SPS_index);
            indices.push_back(SSP_index);
            indices.push_back(SSS_index);
            T val = 0;
            if (this->valid_index(indices[leaf]))
              val = dims[this->order - 1].GetElt(indices[leaf]);
            output.AppendElt(index, val);
          }
        }
      }
    }
    *this = output;
  }

  // uses extra_space factor to save as many leaves as possible at once,
  // then gathers them into the output
  void CornersExsum_leaf(index_t& box_lens) {
    size_t extra_space = 4;
    // initialize extra space
    std::vector<Tensor_3D<T>> leaf_store;
    for (size_t i = 0; i < extra_space; i++) {
      leaf_store.emplace_back(Tensor_3D<T>(padded_side_lens));
    }
    Tensor_3D<T> output(padded_side_lens);
    // zero output
    output.ZeroFill();

    // how many times do we need to collect the input into the output
    size_t num_leaves = 1 << this->order;
    size_t num_rounds = num_leaves / extra_space;
    if (num_leaves % extra_space != 0) {
      num_rounds++;
    }

    // for each round
    for (size_t round = 0; round < num_rounds; round++) {
      size_t start = round * extra_space;
      size_t end = std::min((round + 1) * extra_space, num_leaves);

      // compute as many leaves as possible
      for (size_t level = start; level < end; level++) {
        size_t ind = level - start;  // where are we in the extra space
        leaf_store[ind] = *this;     // start from original input
        // along dimension 0
        if (level < 4) {
          leaf_store[ind].prefix_through_dim(0);
        } else {
          leaf_store[ind].suffix_through_dim(0);
        }
        // along dimension 1
        if (level % 4 < 2) {
          leaf_store[ind].prefix_through_dim(1);
        } else {
          leaf_store[ind].suffix_through_dim(1);
        }
        // along dimension 2
        if (level % 2 == 0) {
          leaf_store[ind].prefix_through_dim(2);
        } else {
          leaf_store[ind].suffix_through_dim(2);
        }
      }

      // at end of round, do add contribution
      for (size_t i = 0; i < this->dim_lens[0]; ++i) {
        for (size_t j = 0; j < this->dim_lens[1]; ++j) {
          for (size_t k = 0; k < this->dim_lens[2]; ++k) {
            size_t ki = box_lens[0];
            size_t kj = box_lens[1];
            size_t kk = box_lens[2];
            size_t max_i = this->dim_lens[0] - 1;
            size_t max_j = this->dim_lens[1] - 1;
            size_t max_k = this->dim_lens[2] - 1;

            // TODO: this could be a function
            index_t index = {i, j, k};
            index_t PPP_index = {i - 1, j - 1, k - 1};
            index_t PPS_index = {std::min(i + ki - 1, max_i), j - 1, k};
            index_t PSP_index = {i - 1, j, std::min(k + kk - 1, max_k)};
            index_t PSS_index = {std::min(i + ki - 1, max_i), j, k + kk};
            index_t SPP_index = {i, std::min(j + kj - 1, max_j), k - 1};
            index_t SPS_index = {i + ki, std::min(j + kj - 1, max_j), k};
            index_t SSP_index = {i, j + kj, std::min(k + kk - 1, max_k)};
            index_t SSS_index = {i + ki, j + kj, k + kk};
            vector<vector<size_t>> indices;
            indices.push_back(PPP_index);
            indices.push_back(PPS_index);
            indices.push_back(PSP_index);
            indices.push_back(PSS_index);
            indices.push_back(SPP_index);
            indices.push_back(SPS_index);
            indices.push_back(SSP_index);
            indices.push_back(SSS_index);

            // collect all inputs
            for (size_t level = start; level < end; level++) {
              T val = 0;
              if (this->valid_index(indices[level]))
                val = leaf_store[level - start].GetElt(indices[level]);
              output.AppendElt(index, val);
            }
          }
        }
      }
    }
    *this = output;
  }

  // uses extra_space factor to save as many leaves as possible at once,
  // then gathers them into the output
  void CornersExsum_leaf_2(index_t& box_lens) {
    size_t extra_space = 2;
    // initialize extra space
    std::vector<Tensor_3D<T>> leaf_store;
    for (size_t i = 0; i < extra_space; i++) {
      leaf_store.emplace_back(Tensor_3D<T>(padded_side_lens));
    }
    Tensor_3D<T> output(padded_side_lens);
    // zero output
    output.ZeroFill();

    // how many times do we need to collect the input into the output
    size_t num_leaves = 1 << this->order;
    size_t num_rounds = num_leaves / extra_space;
    if (num_leaves % extra_space != 0) {
      num_rounds++;
    }

    // for each round
    for (size_t round = 0; round < num_rounds; round++) {
      size_t start = round * extra_space;
      size_t end = std::min((round + 1) * extra_space, num_leaves);

      // compute as many leaves as possible
      for (size_t level = start; level < end; level++) {
        size_t ind = level - start;  // where are we in the extra space
        leaf_store[ind] = *this;     // start from original input
        // along dimension 0
        if (level < 4) {
          leaf_store[ind].prefix_through_dim(0);
        } else {
          leaf_store[ind].suffix_through_dim(0);
        }
        // along dimension 1
        if (level % 4 < 2) {
          leaf_store[ind].prefix_through_dim(1);
        } else {
          leaf_store[ind].suffix_through_dim(1);
        }
        // along dimension 2
        if (level % 2 == 0) {
          leaf_store[ind].prefix_through_dim(2);
        } else {
          leaf_store[ind].suffix_through_dim(2);
        }
      }

      // at end of round, do add contribution
      for (size_t i = 0; i < this->dim_lens[0]; ++i) {
        for (size_t j = 0; j < this->dim_lens[1]; ++j) {
          for (size_t k = 0; k < this->dim_lens[2]; ++k) {
            size_t ki = box_lens[0];
            size_t kj = box_lens[1];
            size_t kk = box_lens[2];
            size_t max_i = this->dim_lens[0] - 1;
            size_t max_j = this->dim_lens[1] - 1;
            size_t max_k = this->dim_lens[2] - 1;

            // TODO: this could be a function
            index_t index = {i, j, k};
            index_t PPP_index = {i - 1, j - 1, k - 1};
            index_t PPS_index = {std::min(i + ki - 1, max_i), j - 1, k};
            index_t PSP_index = {i - 1, j, std::min(k + kk - 1, max_k)};
            index_t PSS_index = {std::min(i + ki - 1, max_i), j, k + kk};
            index_t SPP_index = {i, std::min(j + kj - 1, max_j), k - 1};
            index_t SPS_index = {i + ki, std::min(j + kj - 1, max_j), k};
            index_t SSP_index = {i, j + kj, std::min(k + kk - 1, max_k)};
            index_t SSS_index = {i + ki, j + kj, k + kk};
            vector<vector<size_t>> indices;
            indices.push_back(PPP_index);
            indices.push_back(PPS_index);
            indices.push_back(PSP_index);
            indices.push_back(PSS_index);
            indices.push_back(SPP_index);
            indices.push_back(SPS_index);
            indices.push_back(SSP_index);
            indices.push_back(SSS_index);

            // collect all inputs
            for (size_t level = start; level < end; level++) {
              T val = 0;
              if (this->valid_index(indices[level]))
                val = leaf_store[level - start].GetElt(indices[level]);
              output.AppendElt(index, val);
            }
          }
        }
      }
    }
    *this = output;
  }

  // This implementation uses minimum space for maximum work
  void CornersExsum_Space(index_t& box_lens) {
    Tensor_3D<T> dim(*this);
    Tensor_3D<T> output(padded_side_lens);
    // zero output
    output.ZeroFill();
    for (size_t level = 0; level < 8; level++) {
      dim = *this;
      // along dimension 0
      if (level < 4) {
        dim.prefix_through_dim(0);
      } else {
        dim.suffix_through_dim(0);
      }
      // along dimension 1
      if (level % 4 < 2) {
        dim.prefix_through_dim(1);
      } else {
        dim.suffix_through_dim(1);
      }
      // along dimension 2
      if (level % 2 == 0) {
        dim.prefix_through_dim(2);
      } else {
        dim.suffix_through_dim(2);
      }
      // add vertices contribution step
      for (size_t i = 0; i < this->dim_lens[0]; ++i) {
        for (size_t j = 0; j < this->dim_lens[1]; ++j) {
          for (size_t k = 0; k < this->dim_lens[2]; ++k) {
            size_t ki = box_lens[0];
            size_t kj = box_lens[1];
            size_t kk = box_lens[2];
            size_t max_i = this->dim_lens[0] - 1;
            size_t max_j = this->dim_lens[1] - 1;
            size_t max_k = this->dim_lens[2] - 1;
            index_t index = {i, j, k};
            index_t PPP_index = {i - 1, j - 1, k - 1};
            index_t PPS_index = {std::min(i + ki - 1, max_i), j - 1, k};
            index_t PSP_index = {i - 1, j, std::min(k + kk - 1, max_k)};
            index_t PSS_index = {std::min(i + ki - 1, max_i), j, k + kk};
            index_t SPP_index = {i, std::min(j + kj - 1, max_j), k - 1};
            index_t SPS_index = {i + ki, std::min(j + kj - 1, max_j), k};
            index_t SSP_index = {i, j + kj, std::min(k + kk - 1, max_k)};
            index_t SSS_index = {i + ki, j + kj, k + kk};
            vector<vector<size_t>> indices;
            indices.push_back(PPP_index);
            indices.push_back(PPS_index);
            indices.push_back(PSP_index);
            indices.push_back(PSS_index);
            indices.push_back(SPP_index);
            indices.push_back(SPS_index);
            indices.push_back(SSP_index);
            indices.push_back(SSS_index);
            T val = 0;
            if (this->valid_index(indices[level]))
              val = dim.GetElt(indices[level]);
            output.AppendElt(index, val);
          }
        }
      }
    }
    *this = output;
  }

  // Demaine et al. algorithm --> 2^d
  // Implementation notes:
  //  Space vs time tradeoff: this implementation uses maximum space for minimum
  // work
  void CornersExsum(index_t& box_lens) {
    Tensor_3D<T> PPP(*this);
    Tensor_3D<T> SSS(*this);
    Tensor_3D<T> output(padded_side_lens);
    // zero output
    output.ZeroFill();
    // along dimension 0
    PPP.prefix_through_dim(0);
    SSS.suffix_through_dim(0);
    Tensor_3D<T> PSP(PPP);
    Tensor_3D<T> SPP(SSS);
    // along dimension 1
    PPP.prefix_through_dim(1);
    PSP.suffix_through_dim(1);
    SPP.prefix_through_dim(1);
    SSS.suffix_through_dim(1);
    Tensor_3D<T> PPS(PPP);
    Tensor_3D<T> PSS(PSP);
    Tensor_3D<T> SPS(SPP);
    Tensor_3D<T> SSP(SSS);
    // along dimension 2
    PPP.prefix_through_dim(2);
    PPS.suffix_through_dim(2);
    PSP.prefix_through_dim(2);
    PSS.suffix_through_dim(2);
    SPP.prefix_through_dim(2);
    SPS.suffix_through_dim(2);
    SSP.prefix_through_dim(2);
    SSS.suffix_through_dim(2);
    // add vertices contribution step
    for (size_t i = 0; i < this->dim_lens[0]; ++i) {
      for (size_t j = 0; j < this->dim_lens[1]; ++j) {
        for (size_t k = 0; k < this->dim_lens[2]; ++k) {
          size_t ki = box_lens[0];
          size_t kj = box_lens[1];
          size_t kk = box_lens[2];
          size_t max_i = this->dim_lens[0] - 1;
          size_t max_j = this->dim_lens[1] - 1;
          size_t max_k = this->dim_lens[2] - 1;
          index_t index = {i, j, k};
          index_t PPP_index = {i - 1, j - 1, k - 1};
          index_t PPS_index = {std::min(i + ki - 1, max_i), j - 1, k};
          index_t PSP_index = {i - 1, j, std::min(k + kk - 1, max_k)};
          index_t PSS_index = {std::min(i + ki - 1, max_i), j, k + kk};
          index_t SPP_index = {i, std::min(j + kj - 1, max_j), k - 1};
          index_t SPS_index = {i + ki, std::min(j + kj - 1, max_j), k};
          index_t SSP_index = {i, j + kj, std::min(k + kk - 1, max_k)};
          index_t SSS_index = {i + ki, j + kj, k + kk};
          T PPP_val = 0;
          T PPS_val = 0;
          T PSP_val = 0;
          T PSS_val = 0;
          T SPP_val = 0;
          T SPS_val = 0;
          T SSP_val = 0;
          T SSS_val = 0;
          if (this->valid_index(PPP_index)) PPP_val = PPP.GetElt(PPP_index);
          if (this->valid_index(PPS_index)) PPS_val = PPS.GetElt(PPS_index);
          if (this->valid_index(PSP_index)) PSP_val = PSP.GetElt(PSP_index);
          if (this->valid_index(PSS_index)) PSS_val = PSS.GetElt(PSS_index);
          if (this->valid_index(SPP_index)) SPP_val = SPP.GetElt(SPP_index);
          if (this->valid_index(SPS_index)) SPS_val = SPS.GetElt(SPS_index);
          if (this->valid_index(SSP_index)) SSP_val = SSP.GetElt(SSP_index);
          if (this->valid_index(SSS_index)) SSS_val = SSS.GetElt(SSS_index);
#ifdef DELAY
          for (size_t a = 0; a < 7; a++) {
            volatile long x = fib(DELAY);
          }
#endif
          output.SetElt(index, PPP_val + PPS_val + PSP_val + PSS_val + SPP_val +
                                   SPS_val + SSP_val + SSS_val);
        }
      }
    }
    *this = output;
  }

  void Incsum_Archive(index_t& box_lens) {
    // along dim 0
    size_t dim = 0;
    Tensor_3D<T> prefix_temp(*this);
    Tensor_3D<T> suffix_temp(*this);
    for (size_t j = 0; j < padded_side_lens[1]; j++) {
      for (size_t k = 0; k < padded_side_lens[2]; k++) {
        size_t stride = this->dim_lens[1] * this->dim_lens[2];
        index_t index = {0, j, k};
        for (size_t i = 0; i < padded_side_lens[dim] / box_lens[dim]; ++i) {
          vector<size_t> start_index = index;
          vector<size_t> end_index = index;
          start_index[dim] = i * box_lens[dim];
          end_index[dim] = i * box_lens[dim] + box_lens[dim] - 1;
          prefix_temp.prefix_linear(this->getAddress(start_index),
                                    this->getAddress(end_index), stride);
          suffix_temp.suffix_linear(this->getAddress(start_index),
                                    this->getAddress(end_index), stride);
        }
        for (size_t i = 0; i < padded_side_lens[dim]; ++i) {
          vector<size_t> new_index = index;
          new_index[dim] = i;
          T suffix = suffix_temp.GetElt(new_index);
          T prefix = 0;
          if (i % box_lens[dim] == 0) {
            this->SetElt(new_index, suffix);
          } else {
            vector<size_t> pref_index = index;
            pref_index[dim] = i + box_lens[dim] - 1;
            if (this->valid_index(pref_index)) {
              prefix = prefix_temp.GetElt(pref_index);
            }
            this->SetElt(new_index, suffix + prefix);
          }
        }
      }
    }

    // along dim 1
    dim = 1;
    prefix_temp = *this;
    suffix_temp = *this;
    for (size_t i = 0; i < padded_side_lens[0]; i++) {
      for (size_t k = 0; k < padded_side_lens[2]; k++) {
        size_t stride = this->dim_lens[2];
        index_t index = {i, 0, k};
        for (size_t i = 0; i < padded_side_lens[dim] / box_lens[dim]; ++i) {
          vector<size_t> start_index = index;
          vector<size_t> end_index = index;
          start_index[dim] = i * box_lens[dim];
          end_index[dim] = i * box_lens[dim] + box_lens[dim] - 1;
          prefix_temp.prefix_linear(this->getAddress(start_index),
                                    this->getAddress(end_index), stride);
          suffix_temp.suffix_linear(this->getAddress(start_index),
                                    this->getAddress(end_index), stride);
        }
        for (size_t i = 0; i < padded_side_lens[dim]; ++i) {
          vector<size_t> new_index = index;
          new_index[dim] = i;
          T suffix = suffix_temp.GetElt(new_index);
          T prefix = 0;
          if (i % box_lens[dim] == 0) {
            this->SetElt(new_index, suffix);
          } else {
            vector<size_t> pref_index = index;
            pref_index[dim] = i + box_lens[dim] - 1;
            if (this->valid_index(pref_index)) {
              prefix = prefix_temp.GetElt(pref_index);
            }
            this->SetElt(new_index, suffix + prefix);
          }
        }
      }
    }

    // along dim 2
    dim = 2;
    prefix_temp = *this;
    suffix_temp = *this;
    for (size_t i = 0; i < padded_side_lens[0]; i++) {
      for (size_t j = 0; j < padded_side_lens[1]; j++) {
        size_t stride = 1;
        index_t index = {i, j, 0};
        for (size_t i = 0; i < padded_side_lens[dim] / box_lens[dim]; ++i) {
          vector<size_t> start_index = index;
          vector<size_t> end_index = index;
          start_index[dim] = i * box_lens[dim];
          end_index[dim] = i * box_lens[dim] + box_lens[dim] - 1;
          prefix_temp.prefix_linear(this->getAddress(start_index),
                                    this->getAddress(end_index), stride);
          suffix_temp.suffix_linear(this->getAddress(start_index),
                                    this->getAddress(end_index), stride);
        }
        for (size_t i = 0; i < padded_side_lens[dim]; ++i) {
          vector<size_t> new_index = index;
          new_index[dim] = i;
          T suffix = suffix_temp.GetElt(new_index);
          T prefix = 0;
          if (i % box_lens[dim] == 0) {
            this->SetElt(new_index, suffix);
          } else {
            vector<size_t> pref_index = index;
            pref_index[dim] = i + box_lens[dim] - 1;
            if (this->valid_index(pref_index)) {
              prefix = prefix_temp.GetElt(pref_index);
            }
            this->SetElt(new_index, suffix + prefix);
          }
        }
      }
    }
  }

  // TODO: generalize this for arbitrary dimensions
  // currently only for 3 dims
  void add_contributions(std::vector<Tensor_3D<T>> temp, index_t& box_lens) {
    assert(temp.size() == 6);
    for (size_t i = 0; i < this->dim_lens[0]; ++i) {
      for (size_t j = 0; j < this->dim_lens[1]; ++j) {
        for (size_t k = 0; k < this->dim_lens[2]; ++k) {
          index_t set_index = {i, j, k};
          // fill in indices to gather from
          vector<vector<size_t>> indices;
          // dim = 0
          index_t prefix_0 = {i - 1, j, k};
          index_t suffix_0 = {i + box_lens[0], j, k};
          // dim = 1
          index_t prefix_1 = {this->dim_lens[0] - 1, j - 1, k};
          index_t suffix_1 = {this->dim_lens[0] - 1, j + box_lens[1], k};
          // dim = 2
          index_t prefix_2 = {this->dim_lens[0] - 1, this->dim_lens[1] - 1,
                              k - 1};
          index_t suffix_2 = {this->dim_lens[0] - 1, this->dim_lens[1] - 1,
                              k + box_lens[2]};

          indices.push_back(prefix_0);
          indices.push_back(suffix_0);
          indices.push_back(prefix_1);
          indices.push_back(suffix_1);
          indices.push_back(prefix_2);
          indices.push_back(suffix_2);
          T sum = 0;
          for (size_t i = 0; i < 6; i++) {
            if (this->valid_index(indices[i])) {
#ifdef DELAY
              volatile long x = fib(DELAY);
#endif
              sum += temp[i].GetElt(indices[i]);
            }
          }
          this->SetElt(set_index, sum);
        }
      }
    }
  }

  // TODO: generalize this for arbitrary dimensions
  // currently only for 3 dims
  void add_contributions_new(std::vector<Tensor_3D<T>> temp,
                             index_t& box_lens) {
    assert(temp.size() == 6);
    for (size_t i = 0; i < this->dim_lens[0]; ++i) {
      for (size_t j = 0; j < this->dim_lens[1]; ++j) {
        for (size_t k = 0; k < this->dim_lens[2]; ++k) {
          index_t set_index = {i, j, k};
          // fill in indices to gather from
          vector<vector<size_t>> indices;
          // dim = 0
          index_t prefix_0 = {i - 1, j, k};
          index_t suffix_0 = {i + box_lens[0], j, k};
          // dim = 1
          index_t prefix_1 = {0, j - 1, k};
          index_t suffix_1 = {0, j + box_lens[1], k};
          // dim = 2
          index_t prefix_2 = {0, 0, k - 1};
          index_t suffix_2 = {0, 0, k + box_lens[2]};

          indices.push_back(prefix_0);
          indices.push_back(suffix_0);
          indices.push_back(prefix_1);
          indices.push_back(suffix_1);
          indices.push_back(prefix_2);
          indices.push_back(suffix_2);
          T sum = 0;
          for (size_t i = 0; i < 6; i++) {
            if (this->valid_index(indices[i])) {
#ifdef DELAY
              volatile long x = fib(DELAY);
#endif
              sum += temp[i].GetElt(indices[i]);
            }
          }
          this->SetElt(set_index, sum);
        }
      }
    }
  }

  // with extra space
  void BoxComplementExsum_space(index_t& box_lens) {
    std::vector<Tensor_3D<T>> temp;
    Tensor_3D<T> prefix_temp(*this);
    Tensor_3D<T> suffix_temp(padded_side_lens);
    Tensor_3D<T> A_prime(*this);
    // zero output (in place)
    this->ZeroFill();

    for (size_t i = 0; i < this->order; ++i) {
      // prefix step
      A_prime = prefix_temp;
      suffix_temp = prefix_temp;
      A_prime.prefix_along_dim(i);
      prefix_temp = A_prime;
      for (size_t j = i + 1; j < this->order; ++j) {
        A_prime.incsum_along_dim(j, box_lens, i);
      }
      // TODO: we only need the full intermediate tensor in the first dim
      // in later dims, copy the dimension reduction only
      // requires updating add_contributions
      temp.push_back(A_prime);

      // suffix step
      A_prime = suffix_temp;
      A_prime.suffix_along_dim(i);
      for (size_t j = i + 1; j < this->order; ++j) {
        A_prime.incsum_along_dim(j, box_lens, i);
      }
      temp.push_back(A_prime);
    }
    // add contributions all at the end
    this->add_contributions(temp, box_lens);
  }

  Tensor_3D<T> get_reduced_temp(Tensor_3D<T> A, size_t reduced_dims) {
    if (reduced_dims == 0) {
      return A;
    } else if (reduced_dims == 1) {
      Tensor_3D<T> output({1, this->dim_lens[1], this->dim_lens[2]});
      // copy last 2d slice into output

      for (size_t i = 0; i < this->dim_lens[1]; ++i) {
        for (size_t j = 0; j < this->dim_lens[2]; ++j) {
          output.SetElt({0, i, j}, A.GetElt({this->dim_lens[0] - 1, i, j}));
        }
      }
      return output;
    } else {
      Tensor_3D<T> output({1, 1, this->dim_lens[2]});

      for (size_t i = 0; i < this->dim_lens[2]; ++i) {
        output.SetElt({0, 0, i}, A.GetElt({this->dim_lens[0] - 1,
                                           this->dim_lens[1] - 1, i}));
      }
      return output;
    }
  }

  // with extra space + dimension reduction on space
  void BoxComplementExsum_space_new(index_t& box_lens) {
    std::vector<Tensor_3D<T>> temp;
    Tensor_3D<T> prefix_temp(*this);
    Tensor_3D<T> suffix_temp(padded_side_lens);
    Tensor_3D<T> A_prime(*this);
    // zero output (in place)
    this->ZeroFill();

    for (size_t i = 0; i < this->order; ++i) {
      // prefix step
      A_prime = prefix_temp;
      suffix_temp = prefix_temp;
      A_prime.prefix_along_dim(i);
      prefix_temp = A_prime;
      for (size_t j = i + 1; j < this->order; ++j) {
        A_prime.incsum_along_dim(j, box_lens, i);
      }
      Tensor_3D<T> prefix_reduced = this->get_reduced_temp(A_prime, i);
      temp.push_back(prefix_reduced);

      // suffix step
      A_prime = suffix_temp;
      A_prime.suffix_along_dim(i);
      for (size_t j = i + 1; j < this->order; ++j) {
        A_prime.incsum_along_dim(j, box_lens, i);
      }
      Tensor_3D<T> suffix_reduced = this->get_reduced_temp(A_prime, i);
      temp.push_back(suffix_reduced);
    }
    // add contributions all at the end
    this->add_contributions_new(temp, box_lens);
  }

  // summed area table
  void SummedAreaTable_Archive(index_t& box_lens) {
    // size_t gray_diff_3D[8] = {0, 0, 1, 0, 2, 0, 1, 0};
    // size_t add_or_subtract[8] = {0, 1, 1, 0, 1, 1, 0, 0};
    Tensor_3D<T> prefix_temp(*this);
    // zero output (in place)
    this->ZeroFill();

// precompute strides
/*
vector<size_t> strides;
size_t stride = 1;
for (size_t d = 0; d < order; ++d) {
  strides[d] = stride;
  stride *= dim_lens[d];
}
*/

// preproc summed area table
// do prefix along each dimension
#ifdef DEBUG
    index_t test_idx = {1, 1, 1};
#endif
    for (size_t i = 0; i < this->order; ++i) {
      prefix_temp.prefix_through_dim(i);
#ifdef DEBUG
      printf("prefix along dim %lu, prefix[test_idx] = %f\n", i,
             prefix_temp.GetElt(test_idx));
#endif
    }

    index_t max_idx = prefix_temp.getMaxIndex();

    // total of entire tensor to do subtraction from
    T tensor_sum = prefix_temp.GetElt(max_idx);
#ifdef DEBUG
    printf("tensor sum %f\n", tensor_sum);
#endif

    // go through all the points
    for (size_t i = 0; i < this->dim_lens[0]; ++i) {
      for (size_t j = 0; j < this->dim_lens[1]; ++j) {
        for (size_t k = 0; k < this->dim_lens[2]; ++k) {
          T box_total = 0;
          index_t set_index = {i, j, k};
          mod_index_t index = {i - 1, j - 1, k - 1};
          index_t base_index = {i - 1, j - 1, k - 1};
#ifdef DEBUG
          printf("\nSAT (%lu, %lu, %lu)\n", i, j, k);
#endif
          size_t num_bits_set = 1;  // init at 1 bc -1 at 0..0
          // goal is to get the benefit of minimally changing idx
          // to do this i think you have to keep the actual scalar idx?
          // in higher dimensions, goes up to 2^d
          for (size_t gray_idx = 0; gray_idx < 8; ++gray_idx) {
            size_t bit_to_change = gray_diff_3D[gray_idx];
#ifdef DEBUG
            printf("\tgray idx %lu, bit to change %lu, add_or_subtract %lu\n",
                   gray_idx, bit_to_change, add_or_subtract[gray_idx]);
#endif
            if (add_or_subtract[gray_idx]) {
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
              printf("\t\tMIN:set index[%lu] = %lu\n", bit_to_change,
                     index[bit_to_change]);
#endif
              num_bits_set--;
            }
            assert(num_bits_set <= this->order);

            index_t& const_index = index;
            size_t parity = (this->order - num_bits_set) % 2;
#ifdef DEBUG
            printf("\t\tidx (%lu, %lu, %lu), parity %lu\n", const_index[0],
                   const_index[1], const_index[2], parity);
#endif
            if (parity) {  // subtract (-1)^parity
              if (this->valid_index(const_index)) {
#ifdef DEBUG
                printf("\t\t\tVALID IDX, subtracting %f\n",
                       prefix_temp.GetElt(const_index));
#endif
                box_total -= prefix_temp.GetElt(const_index);
              }
            } else {  // add (-1)^parity
              if (this->valid_index(const_index)) {
#ifdef DEBUG
                printf("\t\t\tVALID IDX, adding %f\n",
                       prefix_temp.GetElt(const_index));
#endif
                box_total += prefix_temp.GetElt(const_index);
              }
            }
          }
          assert(box_total > 0);

          // excluded sum is total - box
          T exsum = tensor_sum - box_total;

#ifdef DEBUG
          printf("SAT (%lu, %lu, %lu): incsum = %f, exsum = %f\n", i, j, k,
                 box_total, exsum);
#endif
          this->SetElt(set_index, exsum);
        }
      }
    }
  }

 protected:
  // TODO(sfraser) padding - but for now assume all inputs are already padded
  // or that the box side lengths divide each tensor side length
  vector<size_t> padded_side_lens;
};

}  // namespace exsum_tensor
