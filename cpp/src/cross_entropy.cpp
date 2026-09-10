#include "cross_entropy.hpp"
#include <algorithm>
#include <cassert>
#include <cmath>

float CrossEntropy::forward(const Matrix &probabilities, int label) const {
  assert((label >= 0 && label < probabilities.getRows()) &&
         "CrossEntropy label error");
  assert((probabilities.getCols() == 1) && "probabilities cols error");
  float p{std::max(probabilities.getVal(label, 0), 1e-7f)}; // 防止出现log(0)
  return -std::log(p);
}

Matrix CrossEntropy::backward(const Matrix &probabilities, int label) const {
  assert((label >= 0 && label < probabilities.getRows()) &&
         "CrossEntropy label error");
  assert((probabilities.getCols() == 1) && "probabilities cols error");
  int rows = probabilities.getRows();
  Matrix result{rows, probabilities.getCols()};
  for (int i = 0; i < rows; i++) {
    if (i == label) {
      result.setVal(i, 0, probabilities.getVal(i, 0) - 1);
      continue;
    }
    result.setVal(i, 0, probabilities.getVal(i, 0));
  }
  return result;
}