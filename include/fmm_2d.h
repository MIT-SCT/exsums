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

template<typename T>
class Tensor_2D : public Tensor<T, 2> {
  typedef const vector<size_t> index_t;
 public:
  Tensor_2D() = default;
  Tensor_2D(const Tensor_2D &) = default;
  Tensor_2D(Tensor_2D &&) = default;
  Tensor_2D & operator=(const Tensor_2D &) = default;
  ~Tensor_2D() = default;

  explicit Tensor_2D(index_t& side_lens) {
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
        T box_total = 0;
        index_t set_index = {i, j};
        for (size_t ki = 0; ki < box_lens[0]; ++ki) {
          for (size_t kj = 0; kj < box_lens[1]; ++kj) {
            if (i + ki < this->padded_side_lens[0] &&
                j + kj < this->padded_side_lens[1]) {
              index_t get_index = {i + ki, j + kj};
              box_total += this->GetElt(get_index);
            }
          }
        }
        this->SetElt(set_index, box_total);
      }
    }
  }

  // naive excluded sums (with subtraction)
  void ExsumCheckSubtraction(index_t& box_lens) {
    T entire_total = 0;
    for (size_t i = 0; i < this->dim_lens[0]; ++i) {
      for (size_t j = 0; j < this->dim_lens[1]; ++j) {
        index_t get_index = {i, j};
        entire_total += this->GetElt(get_index);
      }
    }
    Tensor_2D<T> copy(*this);
    copy.IncsumCheck(box_lens);
    for (size_t i = 0; i < this->dim_lens[0]; ++i) {
      for (size_t j = 0; j < this->dim_lens[1]; ++j) {
        index_t index = {i, j};
        this->SetElt(index, entire_total - copy.GetElt(index));
      }
    }
  }

  // naive excluded sums (without subtraction)
  void ExsumCheckNaive(index_t& box_lens) {
    Tensor_2D<T> copy(*this);
    for (size_t i = 0; i < this->dim_lens[0]; ++i) {
      for (size_t j = 0; j < this->dim_lens[1]; ++j) {
        T box_total = 0;
        index_t set_index = {i, j};
        for (size_t ki = 0; ki < this->dim_lens[0]; ++ki) {
          for (size_t kj = 0; kj < this->dim_lens[1]; ++kj) {
            // check if index is in the box. Only add if not in box
            if ((ki >= i && ki < (i + box_lens[0])) &&
                (kj >= j && kj < (j + box_lens[1]))) {
              continue;
            }
            index_t get_index = {ki, kj};
            box_total += copy.GetElt(get_index);
          }
        }
        this->SetElt(set_index, box_total);
      }
    }
  }

  // Demaine et al. algorithm --> 2^d
  // Implementation notes:
  //  Space vs time tradeoff: this implementation uses TODO overhead
  void CornersExsum(index_t& box_lens) {
    Tensor_2D<T> prefix_temp(*this);
    Tensor_2D<T> output(padded_side_lens);
    // zero output
    output.ZeroFill();
    // downwards prefix sum (prefix along dimension 1)
    for (size_t column = 0; column < padded_side_lens[1]; column++) {
      size_t stride = this->dim_lens[1];
      index_t start_index = {0, column};
      index_t end_index = {this->dim_lens[0] - 1, column};
      prefix_temp.prefix_linear(this->getAddress(start_index),
                                this->getAddress(end_index), stride);
    }
    Tensor_2D<T> suffix_temp(prefix_temp);
    // prefix/suffix sum along dimension 0
    for (size_t row = 0; row < padded_side_lens[0]; row++) {
      size_t stride = 1;
      index_t start_index = {row, 0};
      index_t end_index = {row, this->dim_lens[1] - 1};
      // stores PP
      prefix_temp.prefix_linear(this->getAddress(start_index),
                                this->getAddress(end_index), stride);
      // stores PS
      suffix_temp.suffix_linear(this->getAddress(start_index),
                                this->getAddress(end_index), stride);
    }

    // add corners contribution step from PP and PS
    for (size_t i = 0; i < this->dim_lens[0]; ++i) {
      for (size_t j = 0; j < this->dim_lens[1]; ++j) {
        index_t index = {i, j};
        index_t PP_index = {i - 1,
                        std::min(j + box_lens[1] - 1, this->dim_lens[1] - 1)};
        index_t PS_index = {std::min(i + box_lens[0] - 1, this->dim_lens[0] - 1)
                            , j + box_lens[1]};
        T PP_val = 0;
        T PS_val = 0;
        if (this->valid_index(PP_index))
          PP_val = prefix_temp.GetElt(PP_index);
        if (this->valid_index(PS_index))
          PS_val = suffix_temp.GetElt(PS_index);
        output.SetElt(index, output.GetElt(index) + PS_val + PP_val);
      }
    }

    suffix_temp = *this;
    // upwards suffix sum (suffix along dimension 1)
    for (size_t column = 0; column < padded_side_lens[1]; column++) {
      size_t stride = this->dim_lens[1];
      index_t start_index = {0, column};
      index_t end_index = {this->dim_lens[0] - 1, column};
      suffix_temp.suffix_linear(this->getAddress(start_index),
                                this->getAddress(end_index), stride);
    }
    prefix_temp = suffix_temp;
    // suffix/prefix sum along dimension 0
    for (size_t row = 0; row < padded_side_lens[0]; row++) {
      size_t stride = 1;
      index_t start_index = {row, 0};
      index_t end_index = {row, this->dim_lens[1] - 1};
      // stores SP
      prefix_temp.prefix_linear(this->getAddress(start_index),
                                this->getAddress(end_index), stride);
      // stores SS
      suffix_temp.suffix_linear(this->getAddress(start_index),
                                this->getAddress(end_index), stride);
    }
    // add corners contribution step from SP and SS
    for (size_t i = 0; i < this->dim_lens[0]; ++i) {
      for (size_t j = 0; j < this->dim_lens[1]; ++j) {
        index_t index = {i, j};
        index_t SP_index = {i, j - 1};
        index_t SS_index = {i + box_lens[0], j};
        T SP_val = 0;
        T SS_val = 0;
        if (this->valid_index(SP_index))
          SP_val = prefix_temp.GetElt(SP_index);
        if (this->valid_index(SS_index))
          SS_val = suffix_temp.GetElt(SS_index);
        output.SetElt(index, output.GetElt(index) + SS_val + SP_val);
      }
    }
    *this = output;
  }

  // These functions are left here from the old manual implementation of 2D
  void Incsum_Archive(index_t& box_lens) {
    Tensor_2D<T> prefix_temp(*this);
    Tensor_2D<T> suffix_temp(*this);
    // zero output (in place)
    this->ZeroFill();
    for (size_t column = 0; column < padded_side_lens[1]; column++) {
      for (size_t i = 0; i < padded_side_lens[0] / box_lens[0]; ++i) {
        size_t stride = this->dim_lens[1];
        index_t start_index = {i * box_lens[0], column};
        index_t end_index = {i * box_lens[0] + box_lens[0] - 1, column};
        prefix_temp.prefix_linear(this->getAddress(start_index),
                                  this->getAddress(end_index), stride);
        suffix_temp.suffix_linear(this->getAddress(start_index),
                                  this->getAddress(end_index), stride);
      }
      for (size_t i = 0; i < padded_side_lens[0]; ++i) {
        index_t index = {i, column};
        T suffix = suffix_temp.GetElt(index);
        T prefix = 0;
        if (i % box_lens[0] == 0) {
          this->SetElt(index, suffix);
        } else {
          index_t pref_index = {i + box_lens[0] - 1, column};
          if (this->valid_index(pref_index)) {
            prefix = prefix_temp.GetElt(pref_index);
          }
          this->SetElt(index, suffix + prefix);
        }
      }
    }
    prefix_temp = *this;
    suffix_temp = *this;
    for (size_t row = 0; row < padded_side_lens[0]; row++) {
      for (size_t j = 0; j < padded_side_lens[1] / box_lens[1]; ++j) {
        size_t stride = 1;
        index_t start_index = {row, j * box_lens[1]};
        index_t end_index = {row, j * box_lens[1] + box_lens[1] - 1};
        prefix_temp.prefix_linear(this->getAddress(start_index),
                                  this->getAddress(end_index), stride);
        suffix_temp.suffix_linear(this->getAddress(start_index),
                                  this->getAddress(end_index), stride);
      }
      for (size_t j = 0; j < padded_side_lens[1]; ++j) {
        index_t index = {row, j};
        T suffix = suffix_temp.GetElt(index);
        T prefix = 0;
        if (j % box_lens[1] == 0) {
          this->SetElt(index, suffix);
        } else {
          index_t pref_index = {row, j + box_lens[1] - 1};
          if (this->valid_index(pref_index)) {
            prefix = prefix_temp.GetElt(pref_index);
          }
          this->SetElt(index, suffix + prefix);
        }
      }
    }
  }

  // These functions are left here from the old manual implementation of 2D
  void BoxComplementExsum_Archive(index_t& box_lens) {
    Tensor_2D<T> prefix_temp(*this);
    Tensor_2D<T> output(padded_side_lens);
    // zero output
    output.ZeroFill();
    index_t reduced_dims = {1, padded_side_lens[1]};
    Tensor_2D<T> reduced(reduced_dims);
    // prefix sum along dim 0 (downwards prefix sum)
    for (size_t column = 0; column < padded_side_lens[1]; column++) {
      size_t stride = this->dim_lens[1];
      index_t start_index = {0, column};
      index_t end_index = {padded_side_lens[0] - 1, column};
      prefix_temp.prefix_linear(this->getAddress(start_index),
                                this->getAddress(end_index), stride);
      index_t reduced_index = {0, column};
      reduced.SetElt(reduced_index, prefix_temp.GetElt(end_index));
    }
    Tensor_2D<T> suffix_temp(prefix_temp);
    // INCSUM along dim 1
    for (size_t row = 0; row < padded_side_lens[0]; row++) {
      for (size_t j = 0; j < padded_side_lens[1] / box_lens[1]; ++j) {
        size_t stride = 1;
        index_t start_index = {row, j * box_lens[1]};
        index_t end_index = {row, j * box_lens[1] + box_lens[1] - 1};
        prefix_temp.prefix_linear(this->getAddress(start_index),
                                  this->getAddress(end_index), stride);
        suffix_temp.suffix_linear(this->getAddress(start_index),
                                  this->getAddress(end_index), stride);
      }
      for (size_t j = 0; j < padded_side_lens[1]; ++j) {
        index_t index = {row, j};
        // note the +1
        index_t output_index = {row + 1, j};
        T suffix = suffix_temp.GetElt(index);
        T prefix = 0;
        if (j % box_lens[1] == 0) {
          if (this->valid_index(output_index)) {
            output.AppendElt(output_index, suffix);
          }
        } else {
          index_t pref_index = {row, j + box_lens[1] - 1};
          if (this->valid_index(pref_index)) {
            prefix = prefix_temp.GetElt(pref_index);
          }
          if (this->valid_index(output_index)) {
            output.AppendElt(output_index, suffix + prefix);
          }
        }
      }
    }
    prefix_temp = *this;
    // suffix along dim 0
    for (size_t column = 0; column < padded_side_lens[1]; column++) {
      size_t stride = this->dim_lens[1];
      index_t start_index = {0, column};
      index_t end_index = {padded_side_lens[0] - 1, column};
      prefix_temp.suffix_linear(this->getAddress(start_index),
                                this->getAddress(end_index), stride);
    }
    suffix_temp = prefix_temp;
    // INCSUM along dim 1
    for (size_t row = 0; row < padded_side_lens[0]; row++) {
      for (size_t j = 0; j < padded_side_lens[1] / box_lens[1]; ++j) {
        size_t stride = 1;
        index_t start_index = {row, j * box_lens[1]};
        index_t end_index = {row, j * box_lens[1] + box_lens[1] - 1};
        prefix_temp.prefix_linear(this->getAddress(start_index),
                                  this->getAddress(end_index), stride);
        suffix_temp.suffix_linear(this->getAddress(start_index),
                                  this->getAddress(end_index), stride);
      }
      for (size_t j = 0; j < padded_side_lens[1]; ++j) {
        index_t index = {row, j};
        T suffix = suffix_temp.GetElt(index);
        T prefix = 0;
        index_t output_index = {row - box_lens[0], j};
        if (j % box_lens[1] == 0) {
          if (this->valid_index(output_index)) {
            output.AppendElt(output_index, suffix);
          }
        } else {
          index_t pref_index = {row, j + box_lens[1] - 1};
          if (this->valid_index(pref_index)) {
            prefix = prefix_temp.GetElt(pref_index);
          }
          if (this->valid_index(output_index)) {
            output.AppendElt(output_index, suffix + prefix);
          }
        }
      }
    }
    Tensor_2D<T> reduced_prefix(reduced);
    Tensor_2D<T> reduced_suffix(reduced);
    size_t stride = 1;
    index_t start_index = {0, 0};
    index_t end_index = {0, padded_side_lens[1] - 1};
    reduced_prefix.prefix_linear(this->getAddress(start_index),
                          this->getAddress(end_index), stride);
    reduced_suffix.suffix_linear(this->getAddress(start_index),
                          this->getAddress(end_index), stride);
    for (size_t i = 0; i < padded_side_lens[0]; ++i) {
      for (size_t j = 0; j < padded_side_lens[1]; ++j) {
        index_t index = {i, j};
        index_t pref_index = {0, j - 1};
        index_t suf_index = {0, j + box_lens[1]};
        T prefix = 0;
        T suffix = 0;
        if (this->valid_index(pref_index)) {
          prefix = reduced_prefix.GetElt(pref_index);
        }
        if (this->valid_index(suf_index)) {
          suffix = reduced_suffix.GetElt(suf_index);
        }
        output.AppendElt(index, suffix + prefix);
      }
    }
    *this = output;
  }

 protected:
  // TODO(sfraser) padding - but for now assume all inputs are already padded
  vector<size_t> padded_side_lens;
};

}  // namespace exsum_tensor

