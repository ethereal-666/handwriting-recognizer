#include "softmax.hpp"
#include <cassert>
#include <cmath>

Matrix softmax::forward(const Matrix &input) const {
  int inputsize{input.getRows()};
  assert((inputsize > 0 && input.getCols() == 1) && "softmax input size error");
  // 防止溢出求 exp(input[i]-max) 再求和计算
  float max{input.getVal(0, 0)};
  for (int i = 0; i < inputsize; i++) {
    float tmp{input.getVal(i, 0)};
    if (tmp > 0 && tmp > max) {
      max = tmp;
    }
  }
  Matrix result{inputsize, 1};
  float sum{0.0f};
  for (int i = 0; i < inputsize; i++) {
    float tmp{std::exp(input.getVal(i, 0) - max)};
    result.setVal(i, 0, tmp);
    sum += tmp;
  }
  for (int i = 0; i < inputsize; i++) {
    result.setVal(i, 0, result.getVal(i, 0) / sum);
  }

  return result;
}