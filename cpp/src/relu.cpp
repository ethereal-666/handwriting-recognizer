#include "relu.hpp"
#include "matrix.hpp"
#include <cassert>

Matrix ReLU::forward(const Matrix &input) {
  this->lastInput = input;
  Matrix result{input.getRows(), input.getCols()};
  for (int i = 0; i < input.getRows(); i++) {
    for (int j = 0; j < input.getCols(); j++) {
      float val{input.getVal(i, j)};
      result.setVal(i, j, val > 0 ? val : 0);
    }
  }
  return result;
}

Matrix ReLU::backward(const Matrix &gradOutput) const {
  assert((gradOutput.getRows() == lastInput.getRows() &&
          gradOutput.getCols() == lastInput.getCols()) &&
         "ReLU backward error");
  Matrix result{gradOutput.getRows(), gradOutput.getCols()};
  for (int i = 0; i < gradOutput.getRows(); i++) {
    for (int j = 0; j < gradOutput.getCols(); j++) {
      // 只有forward时候>0的时候才能反向传递去更新
      result.setVal(i, j,
                    lastInput.getVal(i, j) > 0 ? gradOutput.getVal(i, j) : 0);
    }
  }
  return result;
}