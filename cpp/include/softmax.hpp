#pragma once
#include "matrix.hpp"
class softmax {
public:
  Matrix forward(const Matrix &input) const; // softmax处理
};