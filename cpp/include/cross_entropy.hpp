#pragma once
#include "matrix.hpp"

class CrossEntropy {
public:
  float forward(const Matrix &probabilities, int label) const; // 计算交叉熵
  Matrix backward(const Matrix &probabilities,
                  int label) const; // 计算dz用于更新
};