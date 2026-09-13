#pragma once
#include "matrix.hpp"

class ReLU {
private:
  Matrix lastInput{0, 0};

public:
  Matrix forward(const Matrix &input);             // 根据输入计算
  Matrix backward(const Matrix &gradOutput) const; // 根据后向输入计算
};